"""Chord-state switching Kalman follower with IMM tempo modes and a measure graph.

Position is a discrete chord state with an age, as in Jiang & Raphael's
switching state-space model (``skf.py``): a chord ends by its duration model
under the tempo of the hypothesis. The tempo of every ``(chord, age, route)``
hypothesis is tracked by the CV/CA/ZV mode bank (``IMMMotionModels``); the
mode-conditioned duration hazard is the innovation likelihood that updates the
mode probabilities at every chord transition. A chord is heard through the
chroma frames of the synthesised score audio, time-warped by the hypothesis'
own tempo so that its age selects the rendered frame it should be hearing; at
measure ends the directed score graph distributes the advance over structural
jumps.
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
from matchmaker.prob.imm import IMMMotionModels, score_activity
from matchmaker.prob.skf import MAX_HYPOTHESES, PROB_FLOOR, SIGMA_EPS_SCALE, SIGMA_ETA_SCALE, build_chord_sequence
from matchmaker.utils.misc import set_latency_stats

CV, CA, ZV = 0, 1, 2
# +-10% initial tempo uncertainty, as in the original switching model (skf.py);
# preferred over the validation split's own spread (0.23) on the validation split
TEMPO_PRIOR = 0.1
# the music starts at the first frame whose chroma peak exceeds this multiple of its mean
SILENCE_PEAKINESS = 2.0


class IMMGraphFollowerV1(OnlineAlignment):

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
        sigma_eps_scale: float = SIGMA_EPS_SCALE,
        sigma_eta_scale: float = SIGMA_ETA_SCALE,
        tempo_prior: float = TEMPO_PRIOR,
        modes: Tuple[str, ...] = ("cv", "ca", "zv"),
        **kwargs,
    ):
        super().__init__(reference_features=reference_features, score_positions=score_positions, queue=queue)
        self.max_hypotheses = max_hypotheses
        self.sigma_eps_scale = sigma_eps_scale
        self.sigma_eta_scale = sigma_eta_scale
        self.delta = 1.0 / frame_rate

        self.chords, self.lengths, self.onset_beats = build_chord_sequence(note_array)
        self.K = len(self.chords)
        self.score_positions = self.onset_beats
        reference = np.maximum(np.asarray(reference_features, dtype=float), PROB_FLOOR)
        self.log_reference = np.log(reference / reference.sum(axis=1, keepdims=True))
        beats = np.asarray(ref_frame_to_beat, dtype=float)
        onset_frame = np.clip(np.searchsorted(beats, self.onset_beats), 0, len(beats) - 1)
        self.onset_frame = onset_frame
        self.last_frame = np.r_[np.maximum(onset_frame[1:] - 1, onset_frame[:-1]), len(beats) - 1]
        sounding = score_activity(score_part, beats, len(reference))
        self.log_rest = self.log_reference[~sounding] if not sounding.all() else np.full(
            (1, reference.shape[1]), -np.log(reference.shape[1])
        )

        self.motion = IMMMotionModels(score_part=score_part, tempo=tempo, frame_rate=frame_rate, modes=modes)
        ends = np.r_[self.onset_beats[1:], np.inf]
        self.paused = np.zeros(self.K, dtype=bool)
        for start, end in self.motion.pause_ranges:
            self.paused |= (self.onset_beats < end) & (ends > start)
        self.init_tempo = 240.0 / tempo
        self.init_tempo_var = (tempo_prior * self.init_tempo) ** 2
        marks = sorted((float(score_part.beat_map(o.start.t)), float(o.bpm))
                       for o in score_part.iter_all(pt.score.Tempo) if o.bpm) if score_part is not None else []
        self.marked_tempo = np.full(self.K, tempo, dtype=float)
        for beat, bpm in marks:
            self.marked_tempo[self.onset_beats >= beat - 1e-6] = bpm
        self.jumps = self._chord_jumps(score_graph)
        self._jump_chords = np.array(sorted(self.jumps), dtype=np.int64)
        self._cumsum_lengths = np.concatenate([[0.0], np.cumsum(self.lengths)])
        self.routes: List[Tuple[str, ...]] = [()]
        self.route_ids: Dict[Tuple[str, ...], int] = {(): 0}

        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats = {
            "total_latency": 0, "total_frames": 0,
            "max_latency": 0, "min_latency": float("inf"),
        }
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

    def _cumlen(self, k, m):
        """Cumulative nominal length (whole notes) landing `m` chords after chord `k`."""
        idx = np.minimum(k + m, self.K)
        return self._cumsum_lengths[idx] - self._cumsum_lengths[k]

    def _skip_boundary(self, k):
        """Nearest chord index after `k` with an outgoing structural jump, else the last chord.

        A multi-chord skip must not bypass a jump chord unvisited, or a repeat could
        be silently skipped over instead of taken.
        """
        if len(self._jump_chords) == 0:
            return np.full(len(k), self.K - 1)
        idx = np.searchsorted(self._jump_chords, k + 1)
        has_next = idx < len(self._jump_chords)
        return np.where(has_next, self._jump_chords[np.minimum(idx, len(self._jump_chords) - 1)], self.K - 1)

    def reset(self) -> None:
        self.k = np.zeros(1, dtype=np.int64)
        self.a = np.ones(1, dtype=np.int64)
        self.r = np.zeros(1, dtype=np.int64)
        self.p = np.ones(1)
        self.x = np.zeros((1, 3, 2))
        self.x[:, :, 0] = self.init_tempo
        self.P = np.zeros((1, 3, 2, 2))
        self.P[:, :, 0, 0] = self.init_tempo_var
        self.w = self.motion.initial_probabilities[None, :].copy()
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

    def _log_frame_likelihoods(self, features: np.ndarray) -> np.ndarray:
        """Multinomial chroma log-likelihood of every rendered score frame and, last, of a rest."""
        y = np.maximum(np.asarray(features, dtype=float).reshape(-1, self.log_reference.shape[1])[-1], 0.0)
        y = y / y.sum() if y.any() else np.full_like(y, 1.0 / len(y))
        return np.r_[self.log_reference @ y, logsumexp(self.log_rest @ y) - np.log(len(self.log_rest))]

    def _heard(self, log_frames, frame):
        """Softmin-pooled log-likelihood of the rendered frame and its neighbours."""
        window = np.stack([log_frames[np.clip(frame + d, 0, len(log_frames) - 2)] for d in (-1, 0, 1)])
        return logsumexp(window, axis=0) - np.log(3.0)

    def _mix(self):
        """IMM interaction over one frame; a pause is reachable where the score allows one."""
        transition = np.where(self.paused[self.k][:, None, None], self.motion.M_pause, self.motion.M_play)
        c = np.einsum("hi,hij->hj", self.w, transition)
        mixing = np.divide(
            self.w[:, :, None] * transition, c[:, None, :],
            out=np.tile(np.eye(3), (len(c), 1, 1)), where=c[:, None, :] > 0,
        )
        x = np.einsum("hij,hid->hjd", mixing, self.x)
        residual = self.x[:, :, None, :] - x[:, None, :, :]
        P = np.einsum("hij,hikl->hjkl", mixing, self.P)
        P += np.einsum("hij,hijk,hijl->hjkl", mixing, residual, residual)
        return c, x, P

    def _survival(self, length, x, P, age):
        """P(chord lasts beyond `age` frames) per mode."""
        tempo = x[:, :, 0]
        mean = length[:, None] * tempo
        std = np.sqrt((self.sigma_eps_scale * tempo) ** 2 + length[:, None] ** 2 * P[:, :, 0, 0])
        survival = 1.0 - ndtr((age[:, None] * self.delta - mean) / std)
        if self.motion.enabled[ZV]:
            survival[:, ZV] = np.exp(-age * self.delta / self.motion.beat_seconds)
        return survival

    def _advance(self, x, P, length, duration):
        """Kalman update of tempo from the observed chord duration, then predict."""
        tempo = x[:, :, 0]
        l = length[:, None]
        S = l**2 * P[:, :, 0, 0] + (self.sigma_eps_scale * tempo) ** 2
        gain = P[:, :, :, 0] * l[:, :, None] / S[:, :, None]
        if self.motion.enabled[ZV]:
            gain[:, ZV] = 0.0
        x = x + gain * (duration[:, None] - l * tempo)[:, :, None]
        P = P - gain[:, :, :, None] * gain[:, :, None, :] * S[:, :, None, None]

        F = np.tile(np.eye(2), (len(x), 3, 1, 1))
        F[:, CA, 0, 1] = length
        F[:, CV, 1, 1] = F[:, ZV, 1, 1] = 0.0
        eta = (self.sigma_eta_scale * l * x[:, :, 0]) ** 2
        Q = np.zeros_like(P)
        Q[:, :, 0, 0] = eta
        Q[:, CA] = eta[:, CA, None, None] * np.stack(
            [np.stack([l[:, 0] ** 2 / 3, l[:, 0] / 2], -1),
             np.stack([l[:, 0] / 2, np.ones(len(x))], -1)], -2)
        x = np.einsum("hmij,hmj->hmi", F, x)
        P = np.einsum("hmij,hmjk,hmlk->hmil", F, P, F) + Q
        return x, P

    def step(self, features: np.ndarray) -> None:
        feature = np.abs(np.asarray(features, dtype=float)).reshape(-1)
        if not self._music_started:
            if feature.max() < SILENCE_PEAKINESS * (feature.mean() + 1e-10):
                return
            self._music_started = True
        log_frames = self._log_frame_likelihoods(features)
        c, x, P = self._mix()
        length = self.lengths[self.k]

        rendered = self.onset_frame[self.k][:, None] + np.rint(self.a[:, None] * self.init_tempo / x[:, :, 0]).astype(int)
        rendered = np.minimum(rendered, self.last_frame[self.k][:, None])
        log_stay = self._heard(log_frames, rendered)
        if self.motion.enabled[ZV]:
            held = self._heard(log_frames, self.last_frame[self.k])
            log_stay[:, ZV] = np.logaddexp(held, log_frames[-1]) - np.log(2.0)
        log_onset = self._heard(log_frames, self.onset_frame)
        scale = max(log_stay.max(), log_onset[np.minimum(self.k + 1, self.K - 1)].max())
        acoustic = np.exp(log_stay - scale)
        onset_liks = np.exp(log_onset - scale)

        survive = self._survival(length, x, P, self.a)
        survive_next = self._survival(length, x, P, self.a + 1)
        alive = survive > 0
        stay_m = np.where(alive, survive_next / np.where(alive, survive, 1.0), 0.0)
        can_advance = (self.k + 1 < self.K) | np.isin(self.k, list(self.jumps))
        stay_m = np.where(can_advance[:, None], stay_m, 1.0)

        rows = [(np.stack([self.k, self.a + 1, self.r], 1), self.p[:, None] * acoustic * c * stay_m, x, P)]

        advance = c * (1.0 - stay_m)
        parent = np.flatnonzero(can_advance & (advance.sum(axis=1) > 0))
        share = np.ones(len(parent))
        route = self.r[parent].copy()
        extra = []
        for n in np.flatnonzero(np.isin(self.k[parent], list(self.jumps))):
            linear, targets = self.jumps[int(self.k[parent[n]])]
            share[n] = linear
            for t, jump_share, key in targets:
                extra.append((parent[n], t, jump_share, self._route_id((*self.routes[route[n]], key))))

        # The duration hazard generalizes to a multi-chord span through its cumulative
        # nominal length alone, so a hypothesis stuck on an acoustically ambiguous run
        # of very short chords can land several chords ahead in one frame instead of
        # only ever advancing by one. Where that lands is estimated the same way the
        # score-following literature estimates any point position from elapsed time and
        # tempo: divide elapsed time by tempo for the expected chord reached, and widen
        # by a standard Gaussian confidence gate (GATE_SIGMAS) scaled by the tempo
        # estimate's own relative uncertainty, which is what actually keeps growing this
        # far out, rather than searching outward until a probability decays to some
        # floor -- a tempo estimate's relative uncertainty does not vanish with distance,
        # so that cumulative "not finished yet" probability can plateau above any fixed
        # floor instead of ever reaching it. The range never crosses a chord with an
        # outgoing structural jump, which must still be visited unskipped to fire, and
        # never exceeds max_hypotheses candidates, the beam's own capacity.
        # Landing weight also carries the emission of every chord it skips past against
        # the current frame, exactly as if each were separately checked: a span whose
        # skipped chords do not even resemble what is currently heard is disfavoured
        # the same way a single low-likelihood advance already is.
        # ZV's hazard is a memoryless per-beat sojourn, independent of chord length, so
        # it never has grounds to skip and must not be the reason the range below grows.
        GATE_SIGMAS = 3.0
        skippable = self.motion.enabled.copy()
        skippable[ZV] = False
        k0 = self.k[parent]
        M = 1
        if len(parent) and skippable.any():
            xp, Pp, ap = x[parent], P[parent], self.a[parent]
            boundary = self._skip_boundary(k0)
            cp = c[parent][:, skippable]
            cp = cp / np.maximum(cp.sum(axis=1, keepdims=True), PROB_FLOOR)
            tempo_hat = np.maximum(np.einsum("hm,hm->h", cp, xp[:, skippable, 0]), PROB_FLOOR)
            relative_std = np.sqrt(np.einsum("hm,hm->h", cp, Pp[:, skippable, 0, 0])) / tempo_hat
            elapsed_length = self._cumsum_lengths[k0] + ap * self.delta / tempo_hat
            m_star = np.clip(np.searchsorted(self._cumsum_lengths, elapsed_length) - k0, 1, None)
            reach = np.nan_to_num(np.ceil(m_star * (1.0 + GATE_SIGMAS * relative_std)), nan=1.0)
            M = int(np.clip(np.minimum(reach, boundary - k0).max(), 1, min(self.K, self.max_hypotheses)))

        if M == 1:
            mode_advance = advance[parent]
            landing_length = length[parent]
            target = k0 + 1
        else:
            ms = np.arange(1, M + 1)
            S = np.stack([self._survival(self._cumlen(k0, m), xp, Pp, ap) for m in range(1, M + 2)])
            # a disabled mode's zero-variance survival is 0/0 = nan; it carries no mass
            # (advance is already 0 there), so it must not poison the shared normalization
            exact = np.nan_to_num(np.clip(S[1:] - S[:-1], 0.0, None))    # landing exactly m chords on
            absorb = np.nan_to_num(np.clip(1.0 - S[:-1], 0.0, None))    # at least m, when m is as far as reachable
            max_m = np.clip(boundary - k0, 1, M)
            is_last = ms[:, None] == max_m[None, :]
            zone = np.where(is_last[:, :, None], absorb, exact) * (ms[:, None, None] <= max_m[None, :, None])
            if self.motion.enabled[ZV]:
                zone[:, :, ZV] = 0.0
                zone[0, :, ZV] = 1.0
            onset_cumsum = np.concatenate([[0.0], np.cumsum(log_onset[: self.K])])
            skip_idx = np.minimum(k0[None, :] + ms[:, None], self.K)
            skipped = onset_cumsum[skip_idx] - onset_cumsum[np.minimum(k0[None, :] + 1, self.K)]
            weight = np.nan_to_num(zone * np.exp(skipped - skipped.max(axis=0, keepdims=True))[:, :, None])
            frac = weight / np.maximum(weight.sum(axis=0, keepdims=True), PROB_FLOOR)
            mode_advance = (advance[parent] * frac).reshape(-1, 3)
            landing_length = self._cumlen(k0, ms[:, None]).reshape(-1)
            target = (k0[None, :] + ms[:, None]).reshape(-1)
            parent = np.tile(parent, M)
            share = np.tile(share, M)
            route = np.tile(route, M)

        if extra:
            e = np.array(extra, dtype=object)
            j_parent = e[:, 0].astype(int)
            parent = np.r_[parent, j_parent]
            target = np.r_[target, e[:, 1].astype(int)]
            share = np.r_[share, e[:, 2].astype(float)]
            route = np.r_[route, e[:, 3].astype(int)]
            mode_advance = np.concatenate([mode_advance, advance[j_parent]])
            landing_length = np.r_[landing_length, length[j_parent]]

        xa, Pa = self._advance(x[parent], P[parent], landing_length, self.a[parent] * self.delta)
        landing = np.minimum(target, self.K - 1)
        changed = self.marked_tempo[self.k[parent]] != self.marked_tempo[landing]
        Pa[changed, :, 0, 0] = xa[changed, :, 0] ** 2
        ended = target >= self.K
        target = np.where(ended, self.k[parent], target)
        age = np.where(ended, self.a[parent] + 1, 1)
        rows.append((np.stack([target, age, route], 1),
                     (self.p[parent] * share * onset_liks[target])[:, None] * mode_advance, xa, Pa))

        key, u, x, P = (np.concatenate(parts) for parts in zip(*rows))
        prob = u.sum(axis=1)
        live = prob > 0
        key, u, x, P, prob = key[live], u[live], x[live], P[live], prob[live]

        key, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        m = len(key)
        p_sum = np.zeros(m)
        np.add.at(p_sum, inverse, prob)
        w_sum = np.zeros((m, 3))
        np.add.at(w_sum, inverse, u)
        weight = np.where(w_sum[inverse] > 0, u, prob[:, None])
        mass = np.where(w_sum > 0, w_sum, p_sum[:, None])
        w = w_sum / p_sum[:, None]
        x_sum = np.zeros((m, 3, 2))
        np.add.at(x_sum, inverse, weight[:, :, None] * x)
        second = np.zeros((m, 3, 2, 2))
        np.add.at(second, inverse, weight[:, :, None, None] * (P + x[:, :, :, None] * x[:, :, None, :]))
        x = x_sum / mass[:, :, None]
        P = second / mass[:, :, None, None] - x[:, :, :, None] * x[:, :, None, :]

        keep = np.argsort(p_sum)[::-1][: self.max_hypotheses]
        self.k, self.a, self.r = key[keep, 0], key[keep, 1], key[keep, 2]
        self.p = p_sum[keep] / p_sum[keep].sum()
        self.x, self.P, self.w = x[keep], P[keep], w[keep]

        self._position, route = estimate_chord_position(
            self.onset_beats, self.lengths, self.k, self.a, self.r,
            self.p, self.w, self.x[:, :, 0], self.delta,
            hold_mode=ZV if self.motion.enabled[ZV] else None,
        )
        self.current_route = self.routes[route]
        self.current_index = int(np.clip(
            np.searchsorted(self.onset_beats, self._position, side="right") - 1,
            0, self.K - 1,
        ))
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
