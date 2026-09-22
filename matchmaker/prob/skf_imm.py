"""Switching Kalman filter score following with a synthesised-score chroma
emission, IMM tempo modes and a measure graph.

Position is a discrete chord state with an age, as in Jiang & Raphael's
switching state-space model (``skf.py``): a chord ends by its duration model
under the tempo of the hypothesis. Three extensions: the tempo of every
``(chord, age, route)`` hypothesis is tracked by the CV/CA/ZV mode bank that
steers SoftOLTW (``IMMMotionModels``); a chord is heard through the chroma
frames of the synthesised score audio that belong to it (multinomial
likelihood of the normalised live chroma, softmin-pooled over the frames), so
the score-to-audio gap is bridged by rendering rather than by static spectral
templates; and at measure
ends the directed score graph distributes the advance over structural jumps.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp, ndtr

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.graph.score_graph import EdgeKind, ScoreGraph
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.dp.oltw_soft import SILENCE_PEAKINESS
from matchmaker.prob.imm import IMMMotionModels, score_activity
from matchmaker.prob.skf import MAX_HYPOTHESES, PROB_FLOOR, SIGMA_EPS_SCALE, SIGMA_ETA_SCALE, build_chord_sequence
from matchmaker.utils.misc import set_latency_stats

CV, CA, ZV = 0, 1, 2


class IMMSwitchingKalmanFollower(OnlineAlignment):

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
        # every score frame belongs to the chord sounding at it, and every chord
        # owns at least the frame at its onset (chords may be shorter than a frame)
        chord_of_frame = np.clip(np.searchsorted(self.onset_beats, beats, side="right") - 1, 0, self.K - 1)
        onset_frame = np.clip(np.searchsorted(beats, self.onset_beats), 0, len(beats) - 1)
        pairs = np.unique(np.concatenate([
            np.stack([np.arange(len(beats)), chord_of_frame], 1),
            np.stack([onset_frame, np.arange(self.K)], 1),
        ]), axis=0)
        self.frame_index, self.chord_of_frame = pairs[:, 0], pairs[:, 1]
        # a rest is heard as the score's own silent frames, or as a flat chroma
        sounding = score_activity(score_part, beats, len(reference))
        self.log_rest = self.log_reference[~sounding] if not sounding.all() else np.full(
            (1, reference.shape[1]), -np.log(reference.shape[1])
        )

        self.motion = IMMMotionModels(
            score_part=score_part, tempo=tempo, frame_rate=frame_rate, modes=modes
        )
        ends = np.r_[self.onset_beats[1:], np.inf]
        self.paused = np.zeros(self.K, dtype=bool)
        for start, end in self.motion.pause_ranges:
            self.paused |= (self.onset_beats < end) & (ends > start)
        self.init_tempo = 240.0 / tempo
        self.init_tempo_var = (self.init_tempo * 0.1) ** 2
        self.jumps = self._chord_jumps(score_graph)
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
        first_chord = dict(zip(
            (n.node_id for n in nodes),
            np.searchsorted(self.onset_beats, starts - 1e-6),
        ))
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

    def reset(self) -> None:
        _, _, probabilities = self.motion.initial_state()
        self.k = np.zeros(1, dtype=np.int64)
        self.a = np.ones(1, dtype=np.int64)
        self.r = np.zeros(1, dtype=np.int64)
        self.p = np.ones(1)
        self.x = np.zeros((1, 3, 2))
        self.x[:, :, 0] = self.init_tempo
        self.P = np.zeros((1, 3, 2, 2))
        self.P[:, :, 0, 0] = self.init_tempo_var
        self.w = probabilities[None, :].copy()
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

    def _log_likelihoods(self, features: np.ndarray) -> np.ndarray:
        """Multinomial chroma log-likelihood of every chord and, last, of a rest, softmin-pooled over frames."""
        y = np.maximum(np.asarray(features, dtype=float).reshape(-1, self.log_reference.shape[1])[-1], 0.0)
        y = y / y.sum() if y.any() else np.full_like(y, 1.0 / len(y))
        log_frames = (self.log_reference @ y)[self.frame_index]
        pooled = np.full(self.K + 1, -np.inf)
        np.maximum.at(pooled, self.chord_of_frame, log_frames)
        shifted = np.exp(log_frames - pooled[self.chord_of_frame])
        counts = np.bincount(self.chord_of_frame, minlength=self.K)
        sums = np.bincount(self.chord_of_frame, shifted, minlength=self.K)
        pooled[:-1] += np.log(sums / counts)
        pooled[-1] = logsumexp(self.log_rest @ y) - np.log(len(self.log_rest))
        return pooled

    def _mix(self):
        """IMM interaction over one frame; a pause is reachable where the score allows one."""
        transition = np.where(
            self.paused[self.k][:, None, None], self.motion.M_pause, self.motion.M_play
        )
        c = np.einsum("hi,hij->hj", self.w, transition)
        # modes without mass keep their own estimate instead of a 0/0 mixture
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
            # a held chord or rest ends by the zero-velocity sojourn, decided acoustically
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
        log_liks = self._log_likelihoods(features)
        c, x, P = self._mix()
        length = self.lengths[self.k]
        # likelihoods are relative to the chords the beam can reach this frame
        jump_targets = [t for k in self.k if int(k) in self.jumps for t, _, _ in self.jumps[int(k)][1]]
        reachable = np.concatenate([self.k, np.minimum(self.k + 1, self.K - 1), [self.K], jump_targets]).astype(int)
        liks = np.exp(log_liks - log_liks[reachable].max())

        survive = self._survival(length, x, P, self.a)
        survive_next = self._survival(length, x, P, self.a + 1)
        alive = survive > 0
        stay_m = np.where(alive, survive_next / np.where(alive, survive, 1.0), 0.0)
        can_advance = (self.k + 1 < self.K) | np.isin(self.k, list(self.jumps))
        stay_m = np.where(can_advance[:, None], stay_m, 1.0)

        # moving modes hear the chord; a held chord or rest is heard as the chord or as silence
        acoustic = np.repeat(liks[self.k][:, None], 3, axis=1)
        acoustic[:, ZV] = (liks[self.k] + liks[-1]) / 2.0
        rows = [(np.stack([self.k, self.a + 1, self.r], 1),
                 self.p[:, None] * acoustic * c * stay_m, x, P)]

        advance = c * (1.0 - stay_m)
        parent = np.flatnonzero(can_advance & (advance.sum(axis=1) > 0))
        target = self.k[parent] + 1
        share = np.ones(len(parent))
        route = self.r[parent].copy()
        extra = []
        for n in np.flatnonzero(np.isin(self.k[parent], list(self.jumps))):
            linear, targets = self.jumps[int(self.k[parent[n]])]
            share[n] = linear
            for t, jump_share, key in targets:
                extra.append((parent[n], t, jump_share, self._route_id((*self.routes[route[n]], key))))
        if extra:
            e = np.array(extra, dtype=object)
            parent = np.r_[parent, e[:, 0].astype(int)]
            target = np.r_[target, e[:, 1].astype(int)]
            share = np.r_[share, e[:, 2].astype(float)]
            route = np.r_[route, e[:, 3].astype(int)]
        xa, Pa = self._advance(x[parent], P[parent], length[parent], self.a[parent] * self.delta)
        # the end of the score is absorbing: advancing past it stays on the last chord
        ended = target >= self.K
        target = np.where(ended, self.k[parent], target)
        age = np.where(ended, self.a[parent] + 1, 1)
        rows.append((np.stack([target, age, route], 1),
                     (self.p[parent] * share * liks[target])[:, None] * advance[parent], xa, Pa))

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
        # a mode without mass inherits the hypothesis estimate
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

        chord_p = np.zeros(self.K)
        np.add.at(chord_p, self.k, self.p)
        best = int(np.argmax(chord_p))
        sel = self.k == best
        pb = self.p[sel] / chord_p[best]
        age = pb @ self.a[sel]
        tempo = pb @ np.einsum("hm,hm->h", self.w[sel], self.x[sel, :, 0])
        self.current_index = best
        self.current_route = self.routes[int(self.r[sel][np.argmax(pb)])]
        self._position = float(self.onset_beats[best])
        if best + 1 < self.K:
            fraction = min((age - 1) * self.delta / (self.lengths[best] * tempo), 1.0)
            self._position += fraction * (self.onset_beats[best + 1] - self.onset_beats[best])
        self.input_index += 1

    def get_current_position(self) -> float:
        return self._position

    def is_still_following(self) -> bool:
        # reaching the last chord may precede a structural jump back
        return True

    def __call__(self, observation, perf_time: float) -> float:
        t0 = time.time()
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(time.time() - t0, self.latency_stats, self.input_index)
        return beat
