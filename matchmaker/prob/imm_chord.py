"""Chord-level IMM score follower: steady and maneuvering log-tempo Kalman filters,
a hold mode for fermatas, and a measure graph for structural jumps.

Position is a discrete chord with an age (frames), as in Jiang & Raphael's switching
state-space model. Tempo is u = log(seconds per whole note), so a chord of notated length
l is expected to last d = l exp(u) seconds, observed with timing noise
Var = JITTER^2 + (SPREAD d)^2. Each ``(chord, age, route)`` hypothesis runs an IMM over
two tempo dynamics, as a maneuvering-target tracker does over motion models:

- STEADY: the tempo barely drifts, so the follower trusts it and is conservative;
- MANEUVER: the tempo drifts fast, so the follower follows the durations it hears.

Both the tempo drift and the switching between modes run on notated time, as a performer's
tempo changes over beats and phrases rather than per note: a dense run of short chords
offers as much room for a tempo change as one long chord of the same notated length, so a
wrong position hypothesis cannot re-fit the tempo chord by chord. At each chord onset the
modes are mixed over the chain T(l) = expm(G l) for the chord's notated length l (IMM
interaction), while the chord sounds each mode is weighed by the survival of its predicted
duration and by the audio, and when it ends each mode's Kalman filter is updated with the
observed duration. On a fermata chord either mode may instead HOLD: a memoryless pause of
about one beat that leaves the tempo untouched. All noise parameters are
maximum-likelihood fits on the validation split (see the constants).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import partitura as pt
from scipy.special import logsumexp, ndtr

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.graph.score_graph import EdgeKind, ScoreGraph
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.prob.chord_position import estimate_chord_position
from scipy.linalg import expm

from matchmaker.prob.imm import interact, kalman_update, score_activity
from matchmaker.prob.skf import MAX_HYPOTHESES, PROB_FLOOR, build_chord_sequence
from matchmaker.utils.misc import set_latency_stats

MODE_NAMES = ("steady", "maneuver", "hold")
N_CHROMA = 12
SILENCE_PEAKINESS = 2.0

# Maximum-likelihood fits on the validation split's annotated chord durations plus the
# follower's own sensor noise: it does not see annotated onsets but its frame-quantised
# transition times, whose duration error is white with a robust sd of 37 ms on the
# validation split. JITTER in seconds, SPREAD relative to the duration; per mode, DIFFUSION
# is the variance of u per whole note of notated time and RESIDENCE the mean stay in whole notes.
IMM_FIT = dict(jitter=0.04, spread=0.15, diffusion=(3e-4, 0.3), residence=(64.0, 8.0))
SINGLE_FIT = dict(jitter=0.04, spread=0.2, diffusion=(0.03,), residence=(np.inf,))
# a hypothesis may land several chords on within this many standard deviations of its
# tempo estimate (the usual Gaussian validation gate)
GATE_SIGMAS = 3.0
# log ratio of the performed to the marked opening tempo on the validation split
TEMPO_PRIOR_MEAN, TEMPO_PRIOR_SD = 0.15, 0.38
# probability that a fermata chord is held (maximum likelihood on the validation split's
# fermata chords); notated rests are timed like other chords (their fitted hold prior is 0)
HOLD_PRIOR = 0.5
# Likelihood ratio of a chord onset in the previous frame given this frame's relative
# spectral flux (ChromaOnsetProcessor), measured on the validation split's annotated
# onsets: log ratio at the median log flux of each background-quantile bin. The end
# bins bound it, so a missed attack cannot veto a true onset outright.
ONSET_LOG_FLUX = [-3.025, -2.545, -2.198, -1.824, -1.419, -1.090, -0.805, -0.586, -0.380]
ONSET_LOG_LR = [-4.386, -3.169, -1.997, -0.702, 0.529, 1.397, 2.059, 2.446, 2.641]


class IMMChordFollower(OnlineAlignment):

    def __init__(
        self,
        reference_features: np.ndarray,
        score_positions: Optional[np.ndarray] = None,
        queue: Optional[RECVQueue] = None,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: Optional[np.ndarray] = None,
        note_array: Any = None,
        score_graph: Optional[ScoreGraph] = None,
        score_part: Any = None,
        tempo: float = 120.0,
        max_hypotheses: int = MAX_HYPOTHESES,
        modes: Tuple[str, ...] = ("steady", "hold"),
        fit: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(reference_features=reference_features, score_positions=score_positions, queue=queue)
        if not set(modes) <= set(MODE_NAMES) or "steady" not in modes:
            raise ValueError("modes must include 'steady' and be drawn from %s" % (MODE_NAMES,))
        fit = fit or (IMM_FIT if "maneuver" in modes else SINGLE_FIT)
        self.jitter, self.spread = fit["jitter"], fit["spread"]
        self.diffusion, residence = np.array(fit["diffusion"]), np.array(fit["residence"], dtype=float)
        self.D = len(self.diffusion)
        # generator of the mode chain in notated time (whole notes)
        self.generator = (np.ones((self.D, self.D)) - self.D * np.eye(self.D)) / (residence[:, None] * max(self.D - 1, 1))
        self.stationary = residence / residence.sum() if np.isfinite(residence).all() else np.full(self.D, 1.0 / self.D)
        self._transitions: Dict[float, np.ndarray] = {}
        self.hold_enabled = "hold" in modes
        self.max_hypotheses = max_hypotheses
        self.delta = 1.0 / frame_rate

        self.chords, self.lengths, self.onset_beats = build_chord_sequence(note_array)
        self.K = len(self.chords)
        self.score_positions = self.onset_beats
        reference = np.asarray(reference_features, dtype=float)
        self.onset_evidence = reference.shape[1] == N_CHROMA + 1
        reference = np.maximum(reference[:, :N_CHROMA], PROB_FLOOR)
        self.log_reference = np.log(reference / reference.sum(axis=1, keepdims=True))
        beats = np.asarray(ref_frame_to_beat, dtype=float)
        self.onset_frame = np.clip(np.searchsorted(beats, self.onset_beats), 0, len(beats) - 1)
        self.last_frame = np.r_[np.maximum(self.onset_frame[1:] - 1, self.onset_frame[:-1]), len(beats) - 1]
        sounding = score_activity(score_part, beats, len(reference))
        self.log_rest = self.log_reference[~sounding] if not sounding.all() else np.full(
            (1, N_CHROMA), -np.log(N_CHROMA))

        fermata = np.zeros(self.K, dtype=bool)
        for f in score_part.iter_all(pt.score.Fermata) if score_part is not None else ():
            if getattr(f.ref, "start", None) is not None:
                fermata |= np.isclose(self.onset_beats, float(score_part.beat_map(f.ref.start.t)))
        self.hold_prior = np.where(fermata & self.hold_enabled, HOLD_PRIOR, 0.0)

        self.marked_tempo = 240.0 / tempo
        self.beat_seconds = 60.0 / tempo
        marks = sorted((float(score_part.beat_map(o.start.t)), float(o.bpm))
                       for o in score_part.iter_all(pt.score.Tempo) if o.bpm) if score_part is not None else []
        self.mark_bpm = np.full(self.K, tempo, dtype=float)
        for beat, bpm in marks:
            self.mark_bpm[self.onset_beats >= beat - 1e-6] = bpm
        self.jumps = self._chord_jumps(score_graph)
        self._jump_chords = np.array(sorted(self.jumps), dtype=np.int64)
        self._cumsum_lengths = np.r_[0.0, np.cumsum(self.lengths)]
        self.routes: List[Tuple[str, ...]] = [()]
        self.route_ids: Dict[Tuple[str, ...], int] = {(): 0}

        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats = {"total_latency": 0, "total_frames": 0, "max_latency": 0, "min_latency": float("inf")}
        self.reset()

    def _chord_jumps(self, graph: Optional[ScoreGraph]) -> Dict[int, Tuple[float, list]]:
        """Last chord of a measure -> (linear share, [(target chord, share, edge key)])."""
        jumps: Dict[int, Tuple[float, list]] = {}
        if graph is None or not graph.nodes:
            return jumps
        nodes = sorted(graph.nodes.values(), key=lambda n: n.score_beat)
        starts = np.array([n.score_beat for n in nodes])
        first_chord = dict(zip((n.node_id for n in nodes), np.searchsorted(self.onset_beats, starts - 1e-6)))
        node_of_chord = np.searchsorted(starts, self.onset_beats + 1e-6) - 1
        for i, node in enumerate(nodes):
            transitions = graph.transition_distribution(node.node_id)
            edges = [(e, p) for e, p in transitions if e.kind not in (EdgeKind.LINEAR, EdgeKind.STAY)]
            chords = np.flatnonzero(node_of_chord == i)
            if not edges or not len(chords):
                continue
            linear = sum(p for e, p in transitions if e.kind in (EdgeKind.LINEAR, EdgeKind.STAY))
            targets = [(int(first_chord[e.target]), p, f"{node.node_id}->{e.target}")
                       for e, p in edges if first_chord[e.target] < self.K]
            if targets:
                jumps[int(chords[-1])] = (linear, targets)
        return jumps

    def _chord_modes(self, dynamics, chord):
        """Mode weights of a chord that starts: each dynamics mode, then its held twin."""
        h = self.hold_prior[chord][:, None]
        return np.hstack([dynamics * (1 - h), dynamics * h])

    def reset(self) -> None:
        self.k = np.zeros(1, dtype=np.int64)
        self.a = np.ones(1, dtype=np.int64)
        self.r = np.zeros(1, dtype=np.int64)
        self.p = np.ones(1)
        self.u = np.full((1, self.D), np.log(self.marked_tempo) + TEMPO_PRIOR_MEAN)
        self.P = np.full((1, self.D), TEMPO_PRIOR_SD ** 2)
        self.w = self._chord_modes(self.stationary[None, :], self.k)
        self.input_index = 0
        self.current_index = 0
        self._music_started = False
        self.current_route: Tuple[str, ...] = ()
        self._position = float(self.onset_beats[0])
        self._alignment_path = []

    def _route_id(self, route: Tuple[str, ...]) -> int:
        if route not in self.route_ids:
            self.route_ids[route] = len(self.routes)
            self.routes.append(route)
        return self.route_ids[route]

    def _log_frame_likelihoods(self, chroma: np.ndarray) -> np.ndarray:
        """Multinomial chroma log-likelihood of every rendered score frame and, last, of a rest."""
        y = np.maximum(chroma, 0.0)
        y = y / y.sum() if y.any() else np.full_like(y, 1.0 / len(y))
        return np.r_[self.log_reference @ y, logsumexp(self.log_rest @ y) - np.log(len(self.log_rest))]

    def _heard(self, log_frames, frame):
        """Softmin-pooled log-likelihood of the rendered frame and its neighbours."""
        window = np.stack([log_frames[np.clip(frame + d, 0, len(log_frames) - 2)] for d in (-1, 0, 1)])
        return logsumexp(window, axis=0) - np.log(3.0)

    def _duration_noise(self, expected):
        return self.jitter ** 2 + (self.spread * expected) ** 2

    def _survival(self, expected, P, tempo, age):
        """P(the chord lasts beyond `age` frames): each dynamics mode, then each held twin."""
        t = age[:, None] * self.delta
        z = (t - expected) / np.sqrt(expected ** 2 * P + self._duration_noise(expected))
        hold_mean = self.beat_seconds * tempo / self.marked_tempo     # one beat at the tempo
        return np.hstack([1.0 - ndtr(z), np.exp(-t / hold_mean)])

    def _skip_boundary(self, k):
        """Nearest chord after `k` with an outgoing structural jump (it must be visited, not
        skipped, for the jump to fire), else the last chord."""
        if not len(self._jump_chords):
            return np.full(len(k), self.K - 1)
        idx = np.searchsorted(self._jump_chords, k + 1)
        return np.where(idx < len(self._jump_chords), self._jump_chords[np.minimum(idx, len(self._jump_chords) - 1)], self.K - 1)

    def _landings(self, parent, end, log_onset):
        """Where each ending hypothesis lands: the duration hazard of a run of chords depends
        only on its total notated length, so a hypothesis overdue on an ambiguous run of
        short chords may land several chords on in one frame. The run is gated around the
        chord its elapsed time reaches at its tempo, weighted by the run's survival and by the
        emission of the chords it passes, and never passes a chord with a structural jump.
        Returns (row -> parent index, chords advanced, each mode's mass)."""
        k0, D = self.k[parent], self.D
        mu = end[parent, :D] + end[parent, D:]
        mu = mu / mu.sum(axis=1, keepdims=True)
        u_hat, sd = (mu * self.u[parent]).sum(axis=1), np.sqrt((mu * self.P[parent]).sum(axis=1))
        elapsed = self._cumsum_lengths[k0] + self.a[parent] * self.delta / np.exp(u_hat)
        m_star = np.maximum(np.searchsorted(self._cumsum_lengths, elapsed) - k0, 1)
        reach = np.clip(np.minimum(np.ceil(m_star * (1 + GATE_SIGMAS * sd)), self._skip_boundary(k0) - k0),
                        1, self.max_hypotheses).astype(int)
        row = np.repeat(np.arange(len(parent)), reach)
        m = np.arange(len(row)) - np.repeat(np.cumsum(reach) - reach, reach) + 1
        span = lambda n: self._cumsum_lengths[np.minimum(k0[row] + n, self.K)] - self._cumsum_lengths[k0[row]]
        u, P, age = self.u[parent[row]], self.P[parent[row]], self.a[parent[row]]
        S_m, S_next = (self._survival(span(n)[:, None] * np.exp(u), P, np.exp(u), age)[:, :D] for n in (m, m + 1))
        last = m == reach[row]
        zone = np.where(last[:, None], 1.0 - S_m, np.clip(S_next - S_m, 0.0, None))  # exactly m, or at least m at the gate
        zone = np.hstack([zone, np.repeat((m == 1)[:, None], D, axis=1).astype(float)])  # a hold never skips
        onset_cumsum = np.r_[0.0, np.cumsum(log_onset)]
        passed = onset_cumsum[np.minimum(k0[row] + m, self.K)] - onset_cumsum[np.minimum(k0[row] + 1, self.K)]
        passed -= np.maximum.reduceat(passed, np.cumsum(reach) - reach)[row]
        weight = zone * np.exp(passed)[:, None]
        total = np.zeros((len(parent), 2 * D))
        np.add.at(total, row, weight)
        return row, m, end[parent[row]] * weight / np.maximum(total[row], PROB_FLOOR)

    def _end_chord(self, u, P, end, length, duration):
        """Mode-matched Kalman updates by the observed duration of the chord that ends.
        `end` is each mode's mass for ending now (dynamics modes, then held twins); returns
        each dynamics mode's posterior tempo and the posterior mode probabilities."""
        expected = length[:, None] * np.exp(u)
        u_play, P_play = kalman_update(u, P, duration[:, None] - expected, expected, self._duration_noise(expected))
        played, held = end[:, :self.D], end[:, self.D:]
        mode_mass = np.maximum(played + held, PROB_FLOOR)
        u_end = (played * u_play + held * u) / mode_mass                # a hold leaves u untouched
        P_end = (played * (P_play + (u_play - u_end) ** 2) + held * (P + (u - u_end) ** 2)) / mode_mass
        return u_end, P_end, mode_mass / mode_mass.sum(axis=1, keepdims=True)

    def _start_chord(self, u, P, mu, length):
        """IMM interaction and mode-matched prediction over the notated length of the chord
        that starts: the mode chain and the tempo drift both advance by that length."""
        for l in set(length.tolist()) - self._transitions.keys():
            self._transitions[l] = expm(self.generator * l)
        transition = np.stack([self._transitions[l] for l in length.tolist()])
        c, u0, P0 = interact(mu, transition, u, P)
        return u0, P0 + self.diffusion * length[:, None], c

    def step(self, features: np.ndarray) -> None:
        frame = np.asarray(features, dtype=float).reshape(-1, N_CHROMA + self.onset_evidence)[-1]
        chroma = frame[:N_CHROMA]
        if not self._music_started:
            if np.abs(chroma).max() < SILENCE_PEAKINESS * (np.abs(chroma).mean() + 1e-10):
                return
            self._music_started = True
        log_frames = self._log_frame_likelihoods(chroma)
        p = self.p
        if self.onset_evidence:
            # an attack shows in the frame after its onset: it confirms or refutes the
            # hypotheses that entered their chord in the previous frame (age 1)
            log_lr = np.interp(np.log(max(frame[N_CHROMA], 1e-9)), ONSET_LOG_FLUX, ONSET_LOG_LR)
            p = np.where(self.a == 1, p * np.exp(log_lr), p)

        length = self.lengths[self.k]
        tempo = np.exp(self.u)
        expected = length[:, None] * tempo
        rendered = np.minimum(self.onset_frame[self.k][:, None] + np.rint(self.a[:, None] * self.marked_tempo / tempo).astype(int),
                              self.last_frame[self.k][:, None])
        log_play = self._heard(log_frames, rendered)
        log_hold = np.logaddexp(self._heard(log_frames, self.last_frame[self.k]), log_frames[-1]) - np.log(2.0)
        log_stay = np.hstack([log_play, np.repeat(log_hold[:, None], self.D, axis=1)])
        log_onset = self._heard(log_frames, self.onset_frame)
        scale = max(log_stay.max(), log_onset[np.minimum(self.k + 1, self.K - 1)].max())
        acoustic = np.exp(log_stay - scale)
        onset_liks = np.exp(log_onset - scale)

        survive = self._survival(expected, self.P, tempo, self.a)
        survive_next = self._survival(expected, self.P, tempo, self.a + 1)
        stay_m = np.where(survive > 0, survive_next / np.where(survive > 0, survive, 1.0), 0.0)
        can_advance = (self.k + 1 < self.K) | np.isin(self.k, list(self.jumps))
        stay_m = np.where(can_advance[:, None], stay_m, 1.0)

        rows = [(np.stack([self.k, self.a + 1, self.r], 1), p[:, None] * self.w * acoustic * stay_m, self.u, self.P)]

        end = self.w * (1.0 - stay_m)
        parent = np.flatnonzero(can_advance & (end.sum(axis=1) > 0))
        if len(parent):
            row, m, mode_end = self._landings(parent, end, log_onset[: self.K])
            k0 = self.k[parent[row]]
            share, route = np.ones(len(row)), self.r[parent[row]].copy()
            extra = []
            for n in np.flatnonzero(np.isin(self.k[parent], list(self.jumps))):
                linear, targets = self.jumps[int(self.k[parent[n]])]
                share[row == n] = linear
                for t, jump_share, key in targets:
                    extra.append((n, t, jump_share, self._route_id((*self.routes[self.r[parent[n]]], key))))
            src, target, played = row, k0 + m, self._cumsum_lengths[np.minimum(k0 + m, self.K)] - self._cumsum_lengths[k0]
            if extra:
                e = np.array(extra, dtype=object)
                j = e[:, 0].astype(int)
                src, target = np.r_[src, j], np.r_[target, e[:, 1].astype(int)]
                share, route = np.r_[share, e[:, 2].astype(float)], np.r_[route, e[:, 3].astype(int)]
                played = np.r_[played, length[parent[j]]]
                mode_end = np.vstack([mode_end, end[parent[j]]])
            origin = self.k[parent[src]]
            ended = target >= self.K
            landing = np.where(ended, origin, target)
            u_end, P_end, mu_end = self._end_chord(self.u[parent[src]], self.P[parent[src]], mode_end, played,
                                                   self.a[parent[src]] * self.delta)
            u_land, P_land, dyn = self._start_chord(u_end, P_end, mu_end, self.lengths[landing])
            # a new tempo marking is a known input: shift the tempo by the marked ratio and
            # reopen its uncertainty to that of an opening tempo relative to its marking
            changed = self.mark_bpm[origin] != self.mark_bpm[landing]
            u_land[changed] += np.log(self.mark_bpm[origin][changed] / self.mark_bpm[landing][changed])[:, None]
            P_land[changed] += TEMPO_PRIOR_SD ** 2
            mass = p[parent[src]] * share * mode_end.sum(axis=1) * onset_liks[landing]
            rows.append((np.stack([landing, np.where(ended, self.a[parent[src]] + 1, 1), route], 1),
                         mass[:, None] * self._chord_modes(dyn, landing), u_land, P_land))

        key, w_rows, uu, PP = (np.concatenate(parts) for parts in zip(*rows))
        prob = w_rows.sum(axis=1)
        live = prob > 0
        key, w_rows, uu, PP, prob = key[live], w_rows[live], uu[live], PP[live], prob[live]
        key, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        m = len(key)
        p_sum = np.zeros(m)
        np.add.at(p_sum, inverse, prob)
        w_sum = np.zeros((m, 2 * self.D))
        np.add.at(w_sum, inverse, w_rows)
        # moment-match each dynamics mode's tempo over merged rows, weighted by that mode's mass
        dm = w_rows[:, :self.D] + w_rows[:, self.D:]
        dm_sum = np.zeros((m, self.D))
        np.add.at(dm_sum, inverse, dm)
        weight = np.where(dm_sum[inverse] > 0, dm, prob[:, None])
        norm = np.zeros((m, self.D))
        np.add.at(norm, inverse, weight)
        u_mean = np.zeros((m, self.D))
        np.add.at(u_mean, inverse, weight * uu)
        u_mean /= norm
        P_mean = np.zeros((m, self.D))
        np.add.at(P_mean, inverse, weight * (PP + (uu - u_mean[inverse]) ** 2))
        P_mean /= norm

        keep = np.argsort(p_sum)[::-1][: self.max_hypotheses]
        self.k, self.a, self.r = key[keep, 0], key[keep, 1], key[keep, 2]
        self.p = p_sum[keep] / p_sum[keep].sum()
        self.u, self.P = u_mean[keep], P_mean[keep]
        self.w = w_sum[keep] / p_sum[keep, None]

        tempos = np.exp(self.u)
        self._position, route = estimate_chord_position(
            self.onset_beats, self.lengths, self.k, self.a, self.r, self.p, self.w,
            np.hstack([tempos, tempos]), self.delta,
            hold_mode=list(range(self.D, 2 * self.D)) if self.hold_enabled else None,
        )
        self.current_route = self.routes[route]
        self.current_index = int(np.clip(np.searchsorted(self.onset_beats, self._position, side="right") - 1,
                                         0, self.K - 1))
        self.input_index += 1

    def get_current_position(self) -> float:
        return self._position

    def is_still_following(self) -> bool:
        return True

    def __call__(self, observation, perf_time: float) -> float:
        t0 = time.time()
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(time.time() - t0, self.latency_stats, self.input_index)
        return beat
