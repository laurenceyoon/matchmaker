"""Chord-level IMM score follower: a mean-reverting log-tempo Kalman filter whose dynamics
switch between musical modes, and a measure graph for repeats.

Position is a discrete chord with an age (frames), as in Jiang & Raphael's switching
state-space model. The log tempo (seconds per whole note) of each ``(chord, age, route)``
hypothesis is u = b + e: a base tempo b that drifts slowly, and a rubato deviation e that is
pulled back to zero (an Ornstein-Uhlenbeck process in notated time), as a performer's borrowed
time is paid back. A chord of notated length l is expected to last l exp(u) seconds, with
timing noise JITTER^2 + (SPREAD d)^2.

The IMM modes (Blom & Bar-Shalom) are musical:

- CV: the deviation reverts to zero, so the follower keeps its tempo through dense passages
  whose chords the audio cannot tell apart;
- CA: a tempo change, the deviation drawn toward a faster or a slower target (accelerando,
  ritardando) with CV's variance;
- ZV: a fermata chord held, a memoryless pause of about one beat that leaves the tempo
  untouched;
- JUMP: a new section at a new tempo. As in a variable-structure IMM on a road map, it is only
  reachable at junctions the notation marks (double bars, key, time or tempo changes, tempo
  words, after a fermata), where it reopens the base.

At each chord onset the modes are mixed and predicted over the chord's notated length (IMM
interaction); while it sounds each mode is weighed by the survival of its predicted duration
and by the audio; when it ends each mode is updated by the observed duration. The update is
on log duration, log d = log l + b + e, which is linear in the state (an exact Kalman step),
and, as a clutter hypothesis in probabilistic data association, a duration far from its
prediction (an agogic stretch, a missed onset) is left out of the update with its posterior
outlier probability. Neither widens the duration law that also times the advance, so wrong
position hypotheses stay as discriminated as before. Before the first chord the follower
rests, hearing silence as pitchless chroma. All noise parameters are fits on the validation
split (see the constants).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import partitura as pt
from scipy.linalg import expm
from scipy.special import logsumexp, ndtr

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.graph.score_graph import EdgeKind, ScoreGraph
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.prob.chord_position import estimate_chord_position
from matchmaker.prob.imm import interact, kalman_update, score_activity
from matchmaker.prob.skf import MAX_HYPOTHESES, PROB_FLOOR, build_chord_sequence
from matchmaker.utils.misc import set_latency_stats

MODE_NAMES = ("cv", "ca", "zv", "jump")
BLOCKS = 2      # each dynamics mode plays or holds (ZV)
N_CHROMA = 12
TEMPO = np.array([1.0, 1.0])     # u = b + e: the observation direction of the state

# Maximum-likelihood fits on the validation split's annotated chord durations plus the
# follower's own sensor noise (its frame-quantised transition times: duration error white,
# robust sd 37 ms), under the log-duration measurement. JITTER in seconds, SPREAD relative
# to the duration, BASE_DRIFT the variance of the base per whole note, REVERSION the
# deviation's reversion length (whole notes) and DEVIATION its stationary variance.
JITTER, SPREAD, BASE_DRIFT, REVERSION, DEVIATION = 0.05, 0.2, 1e-4, 0.5, 0.01
# CA targets and the mean residences (whole notes) of CV and of a tempo change (CA, and
# JUMP, which shares CA's), fitted on the same durations
CA_TARGET = 0.1
RESIDENCE = {"cv": 32.0, "ca": 2.0, "jump": 2.0}
# Across the validation split's junctions the log tempo ratio of the measures after and before
# has a robust sd of 0.175 (0.062 at other barlines), and 46% change tempo by more than 15%
JUMP_SD, JUMP_PRIOR = 0.175, 0.46
# a hypothesis may land several chords on within this many standard deviations of its tempo
# estimate (the usual Gaussian validation gate)
GATE_SIGMAS = 3.0
# log ratio of the performed to the marked opening tempo on the validation split
TEMPO_PRIOR_MEAN, TEMPO_PRIOR_SD = 0.15, 0.38
# probability that a fermata chord is held (maximum likelihood on the validation split's
# fermata chords); notated rests are timed like other chords (their fitted hold prior is 0)
HOLD_PRIOR = 0.5
# Likelihood ratio of a chord onset in the previous frame given this frame's relative
# spectral flux (ChromaOnsetProcessor), measured on the validation split's annotated onsets:
# log ratio at the median log flux of each background-quantile bin. The end bins bound it, so
# a missed attack cannot veto a true onset outright.
ONSET_LOG_FLUX = [-3.025, -2.545, -2.198, -1.824, -1.419, -1.090, -0.805, -0.586, -0.380]
ONSET_LOG_LR = [-4.386, -3.169, -1.997, -0.702, 0.529, 1.397, 2.059, 2.446, 2.641]
# prior and duration spread of an outlier chord (agogic stretch, arpeggio, missed onset)
OUTLIER_PRIOR, OUTLIER_SPREAD = 0.03, 1.5
SECTION_BARLINES = ("light-light", "light-heavy", "heavy-light", "heavy-heavy")


def junctions(score_part, onset_beats):
    """Chords where the notation opens a section whose tempo may differ: after a double bar,
    at a key, time signature or tempo change, at a tempo word (rit., accel., a tempo, ...),
    and after a fermata."""
    junction = np.zeros(len(onset_beats), dtype=bool)
    if score_part is None:
        return junction
    beat = lambda t: float(score_part.beat_map(t))
    at = lambda b: min(int(np.searchsorted(onset_beats, b - 1e-6)), len(onset_beats) - 1)
    marks = [o.start.t for o in score_part.iter_all(pt.score.Barline) if getattr(o, "style", None) in SECTION_BARLINES]
    for cls in (pt.score.KeySignature, pt.score.TimeSignature, pt.score.Tempo):
        marks += [o.start.t for o in score_part.iter_all(cls)]
    marks += [o.start.t for o in score_part.iter_all(pt.score.TempoDirection, include_subclasses=True)]
    for t in marks:
        b = beat(t)
        if b > onset_beats[0] + 1e-6:
            junction[at(b)] = True
    for f in score_part.iter_all(pt.score.Fermata):
        if getattr(f.ref, "start", None) is not None:
            junction[min(at(beat(f.ref.start.t)) + 1, len(onset_beats) - 1)] = True
    return junction


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
        modes: Tuple[str, ...] = MODE_NAMES,
        **kwargs,
    ):
        super().__init__(reference_features=reference_features, score_positions=score_positions, queue=queue)
        if not set(modes) <= set(MODE_NAMES) or "cv" not in modes:
            raise ValueError("modes must include 'cv' and be drawn from %s" % (MODE_NAMES,))
        # dynamics modes: CV, CA toward a slower and a faster target, JUMP last
        names = ["cv"] + ["ca", "ca"] * ("ca" in modes) + ["jump"] * ("jump" in modes)
        self.D = len(names)
        self.target = np.zeros(self.D)
        if "ca" in modes:
            self.target[1:3] = -CA_TARGET, CA_TARGET
        if self.D > 1:
            residence = np.array([RESIDENCE[n] for n in names])
            self.generator = (np.ones((self.D, self.D)) - self.D * np.eye(self.D)) / (residence[:, None] * (self.D - 1))
            self.stationary = residence / residence.sum()
        else:
            self.generator, self.stationary = np.zeros((1, 1)), np.ones(1)
        self._prediction: Dict[float, Tuple[np.ndarray, ...]] = {}
        self.hold_enabled = "zv" in modes
        self.jump = "jump" in modes
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
        self.junction = junctions(score_part, self.onset_beats) & self.jump

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
        """Mode weights of a chord that starts: each dynamics mode playing or holding (ZV)."""
        h = self.hold_prior[chord][:, None]
        return np.hstack([dynamics * (1 - h), dynamics * h])

    def _opening(self):
        """Tempo state of a performance that starts: the marked tempo with its prior spread."""
        x = np.zeros((1, self.D, 2))
        x[..., 0] = np.log(self.marked_tempo) + TEMPO_PRIOR_MEAN
        P = np.zeros((1, self.D, 2, 2))
        P[..., 0, 0] = TEMPO_PRIOR_SD ** 2
        P[..., 1, 1] = DEVIATION
        return x, P

    def reset(self) -> None:
        self.k = np.zeros(0, dtype=np.int64)
        self.a = np.ones(0, dtype=np.int64)
        self.r = np.zeros(0, dtype=np.int64)
        self.p = np.ones(0)
        self.x, self.P = (v[:0] for v in self._opening())
        self.w = np.zeros((0, BLOCKS * self.D))
        # before the first chord the performer rests: this mass hears silence as pitchless
        # (flat) chroma and enters the first chord with the hold hazard, so room noise before
        # the music is not timed as chords
        self.waiting = 1.0
        self.input_index = 0
        self.current_index = 0
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

    @staticmethod
    def _duration_noise(expected):
        return JITTER ** 2 + (SPREAD * expected) ** 2

    def _survival(self, length, x, P, age):
        """P(a run of notated `length` lasts beyond `age` frames): each dynamics mode, then
        each held twin. x, P: (rows, modes, 2[, 2])."""
        tempo = np.exp(x @ TEMPO)
        expected = length[:, None] * tempo
        tempo_var = np.einsum("a,hmab,b->hm", TEMPO, P, TEMPO)
        t = age[:, None] * self.delta
        z = (t - expected) / np.sqrt(expected ** 2 * tempo_var + self._duration_noise(expected))
        hold_mean = self.beat_seconds * tempo / self.marked_tempo     # one beat at the tempo
        return np.hstack([1.0 - ndtr(z), np.exp(-t / hold_mean)])

    def _prediction_model(self, length):
        """Mode transition, state transition, input and process noise over `length` whole
        notes: the base drifts, the deviation reverts toward its mode's target."""
        if length not in self._prediction:
            phi = np.exp(-length / REVERSION)
            F = np.zeros((self.D, 2, 2))
            F[:, 0, 0], F[:, 1, 1] = 1.0, phi
            B = np.zeros((self.D, 2))
            B[:, 1] = (1 - phi) * self.target
            Q = np.zeros((self.D, 2, 2))
            Q[:, 0, 0] = BASE_DRIFT * length
            Q[:, 1, 1] = DEVIATION * (1 - phi ** 2)
            self._prediction[length] = (expm(self.generator * length), F, B, Q)
        return self._prediction[length]

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
        mu = end[parent].reshape(len(parent), BLOCKS, D).sum(axis=1)
        mu = mu / mu.sum(axis=1, keepdims=True)
        u = self.x[parent] @ TEMPO
        u_hat = (mu * u).sum(axis=1)
        sd = np.sqrt((mu * np.einsum("a,hmab,b->hm", TEMPO, self.P[parent], TEMPO)).sum(axis=1))
        elapsed = self._cumsum_lengths[k0] + self.a[parent] * self.delta / np.exp(u_hat)
        m_star = np.maximum(np.searchsorted(self._cumsum_lengths, elapsed) - k0, 1)
        reach = np.clip(np.minimum(np.ceil(m_star * (1 + GATE_SIGMAS * sd)), self._skip_boundary(k0) - k0),
                        1, self.max_hypotheses).astype(int)
        row = np.repeat(np.arange(len(parent)), reach)
        m = np.arange(len(row)) - np.repeat(np.cumsum(reach) - reach, reach) + 1
        span = lambda n: self._cumsum_lengths[np.minimum(k0[row] + n, self.K)] - self._cumsum_lengths[k0[row]]
        x, P, age = self.x[parent[row]], self.P[parent[row]], self.a[parent[row]]
        S_m, S_next = (self._survival(span(n), x, P, age)[:, :D] for n in (m, m + 1))
        last = m == reach[row]
        zone = np.where(last[:, None], 1.0 - S_m, np.clip(S_next - S_m, 0.0, None))  # exactly m, or at least m at the gate
        zone = np.hstack([zone, np.repeat((m == 1)[:, None], D, axis=1).astype(float)])  # holds do not skip
        onset_cumsum = np.r_[0.0, np.cumsum(log_onset)]
        passed = onset_cumsum[np.minimum(k0[row] + m, self.K)] - onset_cumsum[np.minimum(k0[row] + 1, self.K)]
        passed -= np.maximum.reduceat(passed, np.cumsum(reach) - reach)[row]
        weight = zone * np.exp(passed)[:, None]
        total = np.zeros((len(parent), BLOCKS * D))
        np.add.at(total, row, weight)
        return row, m, end[parent[row]] * weight / np.maximum(total[row], PROB_FLOOR)

    def _end_chord(self, x, P, end, length, duration):
        """Mode-matched Kalman updates by the observed log duration of the chord (run) that
        ends. `end` is each mode's mass for ending now (playing, held); a hold says nothing
        about the tempo, nor does an outlier duration. Returns each dynamics mode's posterior
        and the posterior mode probabilities."""
        expected = length[:, None] * np.exp(x @ TEMPO)
        R = self._duration_noise(expected) / expected ** 2       # timing noise on the log scale
        residual = np.log(duration[:, None] / expected)
        x_play, P_play = kalman_update(x, P, residual, np.broadcast_to(TEMPO, x.shape), R)
        played, held = end[:, :self.D, None], end[:, self.D:, None]
        tempo_var = np.einsum("a,hmab,b->hm", TEMPO, P, TEMPO)
        S, S_wide = tempo_var + R, tempo_var + (JITTER / expected) ** 2 + OUTLIER_SPREAD ** 2
        normal = (1 - OUTLIER_PRIOR) * np.exp(-0.5 * residual ** 2 / S) / np.sqrt(S)
        wide = OUTLIER_PRIOR * np.exp(-0.5 * residual ** 2 / S_wide) / np.sqrt(S_wide)
        outlier = (wide / np.maximum(normal + wide, PROB_FLOOR))[..., None]
        played, held = played * (1 - outlier), held + played * outlier
        mass = np.maximum(played + held, PROB_FLOOR)
        x_end = (played * x_play + held * x) / mass
        d_play, d_hold = x_play - x_end, x - x_end
        P_end = (played[..., None] * (P_play + d_play[..., :, None] * d_play[..., None, :])
                 + held[..., None] * (P + d_hold[..., :, None] * d_hold[..., None, :])) / mass[..., None]
        mu = mass[..., 0] / mass[..., 0].sum(axis=1, keepdims=True)
        return x_end, P_end, mu

    def _start_chord(self, x, P, mu, chords):
        """IMM interaction over the modes available on the chords that start, then each mode's
        prediction over its notated length."""
        models = [self._prediction_model(l) for l in self.lengths[chords].tolist()]
        transition, F, B, Q = (np.stack(parts) for parts in zip(*models))
        if self.jump:
            # JUMP is only reachable at a junction, where every mode enters it with JUMP_PRIOR
            # and it reopens the base
            J, at = self.D - 1, self.junction[chords]
            closed = transition.copy()
            closed[..., 0] += closed[..., J]
            closed[..., J] = 0.0
            closed[:, J, :] = transition[:, J, :]
            opened = (1 - JUMP_PRIOR) * closed
            opened[..., J] += JUMP_PRIOR
            transition = np.where(at[:, None, None], opened, closed)
            Q = Q.copy()
            Q[:, J, 0, 0] += np.where(at, JUMP_SD ** 2, 0.0)
        c, x0, P0 = interact(mu, transition, x, P)
        return np.einsum("hjab,hjb->hja", F, x0) + B, np.einsum("hjab,hjbc,hjdc->hjad", F, P0, F) + Q, c

    def step(self, features: np.ndarray) -> None:
        frame = np.asarray(features, dtype=float).reshape(-1, N_CHROMA + self.onset_evidence)[-1]
        log_frames = self._log_frame_likelihoods(frame[:N_CHROMA])
        p = self.p
        if self.onset_evidence:
            # an attack shows in the frame after its onset: it confirms or refutes the
            # hypotheses that entered their chord in the previous frame (age 1)
            log_lr = np.interp(np.log(max(frame[N_CHROMA], 1e-9)), ONSET_LOG_FLUX, ONSET_LOG_LR)
            p = np.where(self.a == 1, p * np.exp(log_lr), p)

        length = self.lengths[self.k]
        tempo = np.exp(self.x @ TEMPO)
        rendered = np.minimum(self.onset_frame[self.k][:, None] + np.rint(self.a[:, None] * self.marked_tempo / tempo).astype(int),
                              self.last_frame[self.k][:, None])
        log_play = self._heard(log_frames, rendered)
        log_hold = np.logaddexp(self._heard(log_frames, self.last_frame[self.k]), log_frames[-1]) - np.log(2.0)
        log_stay = np.hstack([log_play, np.repeat(log_hold[:, None], self.D, axis=1)])
        log_onset = self._heard(log_frames, self.onset_frame)
        log_silence = -np.log(N_CHROMA)                     # silence carries no pitch: flat chroma
        scale = max(log_stay.max(initial=-np.inf), log_onset[np.minimum(self.k + 1, self.K - 1)].max(initial=-np.inf))
        if self.waiting > 0:
            scale = max(scale, log_silence, log_onset[0])
        acoustic = np.exp(log_stay - scale)
        onset_liks = np.exp(log_onset - scale)

        survive = self._survival(length, self.x, self.P, self.a)
        survive_next = self._survival(length, self.x, self.P, self.a + 1)
        stay_m = np.where(survive > 0, survive_next / np.where(survive > 0, survive, 1.0), 0.0)
        can_advance = (self.k + 1 < self.K) | np.isin(self.k, list(self.jumps))
        stay_m = np.where(can_advance[:, None], stay_m, 1.0)

        rows = [(np.stack([self.k, self.a + 1, self.r], 1), p[:, None] * self.w * acoustic * stay_m, self.x, self.P)]

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
            x_end, P_end, mu_end = self._end_chord(self.x[parent[src]], self.P[parent[src]], mode_end, played,
                                                   self.a[parent[src]] * self.delta)
            x_land, P_land, dyn = self._start_chord(x_end, P_end, mu_end, landing)
            # a new tempo marking is a known input: shift the base by the marked ratio and
            # reopen its uncertainty to that of an opening tempo relative to its marking
            changed = self.mark_bpm[origin] != self.mark_bpm[landing]
            x_land[changed, :, 0] += np.log(self.mark_bpm[origin][changed] / self.mark_bpm[landing][changed])[:, None]
            P_land[changed, :, 0, 0] += TEMPO_PRIOR_SD ** 2
            mass = p[parent[src]] * share * mode_end.sum(axis=1) * onset_liks[landing]
            rows.append((np.stack([landing, np.where(ended, self.a[parent[src]] + 1, 1), route], 1),
                         mass[:, None] * self._chord_modes(dyn, landing), x_land, P_land))

        if self.waiting > 0:
            hold = np.exp(-self.delta / self.beat_seconds)
            x0, P0 = self._opening()
            first = np.zeros(1, dtype=np.int64)
            rows.append((np.array([[0, 1, 0]]),
                         self.waiting * (1.0 - hold) * onset_liks[0] * self._chord_modes(self.stationary[None, :], first), x0, P0))
            self.waiting *= hold * np.exp(log_silence - scale)

        key, w_rows, xx, PP = (np.concatenate(parts) for parts in zip(*rows))
        prob = w_rows.sum(axis=1)
        live = prob > 0
        key, w_rows, xx, PP, prob = key[live], w_rows[live], xx[live], PP[live], prob[live]
        key, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        n = len(key)
        p_sum = np.zeros(n)
        np.add.at(p_sum, inverse, prob)
        w_sum = np.zeros((n, BLOCKS * self.D))
        np.add.at(w_sum, inverse, w_rows)
        # moment-match each dynamics mode's state over merged rows, weighted by that mode's mass
        dm = w_rows.reshape(len(w_rows), BLOCKS, self.D).sum(axis=1)
        dm_sum = np.zeros((n, self.D))
        np.add.at(dm_sum, inverse, dm)
        weight = np.where(dm_sum[inverse] > 0, dm, prob[:, None])
        norm = np.zeros((n, self.D))
        np.add.at(norm, inverse, weight)
        x_mean = np.zeros((n, self.D, 2))
        np.add.at(x_mean, inverse, weight[..., None] * xx)
        x_mean /= norm[..., None]
        spread = xx - x_mean[inverse]
        P_mean = np.zeros((n, self.D, 2, 2))
        np.add.at(P_mean, inverse, weight[..., None, None] * (PP + spread[..., :, None] * spread[..., None, :]))
        P_mean /= norm[..., None, None]

        keep = np.argsort(p_sum)[::-1][: self.max_hypotheses]
        self.k, self.a, self.r = key[keep, 0], key[keep, 1], key[keep, 2]
        total = p_sum[keep].sum() + self.waiting
        self.p, self.waiting = p_sum[keep] / total, self.waiting / total
        if 0 < self.waiting < 0.5:
            # the music has more likely started than not: commit to it, so that music the
            # chroma model explains poorly is not re-read as the silence before it
            self.p, self.waiting = self.p / self.p.sum(), 0.0
        self.x, self.P = x_mean[keep], P_mean[keep]
        self.w = w_sum[keep] / p_sum[keep, None]
        self.input_index += 1
        if not len(self.p):
            return

        tempos = np.exp(self.x @ TEMPO)
        self._position, route = estimate_chord_position(
            self.onset_beats, self.lengths, self.k, self.a, self.r, self.p, self.w,
            np.hstack([tempos] * BLOCKS), self.delta,
            hold_mode=list(range(self.D, 2 * self.D)) if self.hold_enabled else None,
        )
        self.current_route = self.routes[route]
        self.current_index = int(np.clip(np.searchsorted(self.onset_beats, self._position, side="right") - 1,
                                         0, self.K - 1))

    def get_current_position(self) -> float:
        return self._position

    def is_still_following(self) -> bool:
        return True

    def __call__(self, observation, perf_time: float) -> float:
        t0 = time.time()
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(time.time() - t0, self.latency_stats, self.input_index)
        return beat
