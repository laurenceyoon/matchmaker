from typing import Any, Optional, Tuple

import numpy as np
import partitura as pt
from numpy.typing import NDArray
from scipy.linalg import expm

from matchmaker.features.audio import FRAME_RATE

DEFAULT_OBS_VAR = 5.0


class ScoreInformedIMM:
    def __init__(
        self,
        score_part: Any = None,
        ref_frame_to_beat: Optional[NDArray] = None,
        init_position: float = 0.0,
        init_tempo: float = 1.0,
        obs_var: float = DEFAULT_OBS_VAR,
        tempo: float = 120.0,
        frame_rate: int = FRAME_RATE,
        modes: Tuple[str, ...] = ("cv", "ca", "zv"),
    ):
        if obs_var <= 0 or tempo <= 0 or frame_rate <= 0:
            raise ValueError("Observation variance, tempo and frame rate must be positive")
        self.enabled = np.array([mode in modes for mode in ("cv", "ca", "zv")])
        if not self.enabled[:2].any():
            raise ValueError("At least one moving mode is required")
        self.R = float(obs_var)
        self.H = np.array([1.0, 0.0, 0.0, 1.0])
        self.score_part = score_part
        self.r2b = ref_frame_to_beat
        self.tempo_bpm = float(tempo)
        self.frame_rate = float(frame_rate)
        self.dt = 1.0 / self.frame_rate
        self.beat_seconds = 60.0 / self.tempo_bpm
        if self.r2b is not None and len(self.r2b) > 1:
            increments = np.diff(self.r2b)
            positive = increments[increments > 0]
            if len(positive):
                self.beat_seconds = self.dt / float(np.median(positive))
        self.pause_ranges = self._score_pause_ranges(score_part)
        measure_beats = []
        if score_part is not None:
            for measure in score_part.iter_all(pt.score.Measure):
                if measure.start is not None and measure.end is not None:
                    duration = float(score_part.beat_map(measure.end.t) - score_part.beat_map(measure.start.t))
                    if duration > 0:
                        measure_beats.append(duration)
        self.measure_seconds = self.beat_seconds * (float(np.median(measure_beats)) if measure_beats else 1.0)

        dt = self.dt
        self.F = np.array([
            [[1, dt, 0], [0, 1, 0], [0, 0, 0]],
            [[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        ], dtype=float)

        q_ca = 20 * self.R / self.beat_seconds**5
        self.Q = np.array([
            np.zeros((3, 3)),
            q_ca * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                             [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                             [dt**3 / 6, dt**2 / 2, dt]]),
            np.zeros((3, 3)),
        ])
        self.F = np.pad(self.F, ((0, 0), (0, 1), (0, 1)))
        self.Q = np.pad(self.Q, ((0, 0), (0, 1), (0, 1)))
        self.M_play = self._transition_matrix(allow_pause=False)
        self.M_pause = self._transition_matrix(allow_pause=True)
        self.reset(init_position, init_tempo)

    @staticmethod
    def _score_pause_ranges(score_part: Any) -> list:
        if score_part is None:
            return []
        ranges = []
        for fermata in score_part.iter_all(pt.score.Fermata):
            ref = fermata.ref
            if getattr(ref, "start", None) is not None and getattr(ref, "end", None) is not None:
                ranges.append((float(score_part.beat_map(ref.start.t)),
                               float(score_part.beat_map(ref.end.t))))

        notes = score_part.note_array()
        if len(notes):
            intervals = sorted((float(n["onset_beat"]),
                                float(n["onset_beat"] + n["duration_beat"]))
                               for n in notes)
            end = intervals[0][1]
            for start, stop in intervals[1:]:
                if start > end:
                    ranges.append((end, start))
                end = max(end, stop)
        return sorted(set(ranges))

    def _transition_matrix(self, allow_pause: bool) -> np.ndarray:
        active = np.flatnonzero(self.enabled & np.array([True, True, allow_pause]))
        n = len(active)
        durations = np.array([self.measure_seconds, self.beat_seconds, self.beat_seconds])[active]
        matrix = np.zeros((3, 3))
        matrix[:, active] = 1 / n
        if n > 1:
            generator = np.ones((n, n)) / (durations[:, None] * (n - 1))
            np.fill_diagonal(generator, -1 / durations)
            matrix[np.ix_(active, active)] = expm(generator * self.dt)
        return matrix

    def _combine(self, probabilities: np.ndarray) -> None:
        self.state = probabilities @ self.states
        residuals = self.states - self.state
        self.P = np.einsum("i,ijk->jk", probabilities, self.P_matrices)
        self.P += np.einsum("i,ij,ik->jk", probabilities, residuals, residuals)

    @property
    def position(self) -> float:
        return float(self.state[0])

    @property
    def tempo(self) -> float:
        """Reference frames per input frame, matching the SoftOLTW interface."""
        return float(self.state[1] * self.dt)

    @property
    def position_uncertainty(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0)))

    @property
    def mode_probabilities(self) -> Tuple[float, float, float]:
        return tuple(float(p) for p in self.mu)

    def reset(self, position: float = 0.0, tempo: float = 1.0) -> None:
        velocity = tempo / self.dt

        self.mu = np.zeros(3)
        self.mu[np.flatnonzero(self.enabled[:2])[0]] = 1.0
        self.c_bar = self.mu.copy()
        self.states = np.array([[position, velocity, 0.0],
                                [position, velocity, 0.0], [position, 0.0, 0.0]])

        self.P_matrices = np.array([
            np.diag([self.R, velocity**2, 0]),
            np.diag([self.R, velocity**2, (velocity / self.beat_seconds)**2]),
            np.diag([self.R, 0, 0]),
        ])
        self.states = np.pad(self.states, ((0, 0), (0, 1)))
        self.P_matrices = np.pad(self.P_matrices, ((0, 0), (0, 1), (0, 1)))
        self.mu_history = [self.mode_probabilities]
        self._combine(self.mu)

    def set_observation_error(self, variance: float, initialize: bool = False) -> None:
        duration_frames = np.sqrt(12 * variance)
        correlation = np.exp(-1 / duration_frames) if duration_frames > 0 else 0.0
        self.F[:, 3, 3] = correlation
        self.Q[:, 3, 3] = variance * (1 - correlation**2)
        if initialize:
            self.P_matrices[:, 3, 3] = variance

    def predict(self, is_silent: Optional[bool] = None) -> np.ndarray:
        beat = self.position
        if self.r2b is not None and len(self.r2b):
            beat = float(np.interp(self.position, np.arange(len(self.r2b)), self.r2b))
        allow_pause = any(start <= beat < end for start, end in self.pause_ranges)

        if self.score_part is None and is_silent is True:
            allow_pause = True
        transition = self.M_pause if allow_pause else self.M_play

        self.c_bar = self.mu @ transition
        mixing = np.divide(
            self.mu[:, None] * transition, self.c_bar[None, :],
            out=np.zeros_like(transition), where=self.c_bar[None, :] > 0,
        )
        for j in np.flatnonzero(self.c_bar == 0):
            mixing[j, j] = 1.0
        mixed_states = mixing.T @ self.states
        residuals = self.states[:, None, :] - mixed_states[None, :, :]
        mixed_covariances = np.einsum("ij,ikl->jkl", mixing, self.P_matrices)
        mixed_covariances += np.einsum("ij,ijk,ijl->jkl", mixing, residuals, residuals)

        self.states = np.einsum("ijk,ik->ij", self.F, mixed_states)
        self.P_matrices = self.F @ mixed_covariances @ self.F.transpose(0, 2, 1) + self.Q
        self._combine(self.c_bar)
        return self.state.copy()

    def update(self, z: Optional[float], obs_var: Optional[float] = None) -> np.ndarray:
        if z is None:
            self.mu = self.c_bar.copy()
            self.mu_history.append(self.mode_probabilities)
            return self.state.copy()
        observation_variance = self.R if obs_var is None else float(obs_var)
        innovations = float(z) - self.states @ self.H
        cross_covariances = self.P_matrices @ self.H
        variances = cross_covariances @ self.H + observation_variance
        gains = cross_covariances / variances[:, None]
        self.states += gains * innovations[:, None]
        residual = np.eye(len(self.H))[None, :, :] - gains[:, :, None] * self.H
        self.P_matrices = (
            residual @ self.P_matrices @ residual.transpose(0, 2, 1)
            + observation_variance * gains[:, :, None] * gains[:, None, :]
        )
        log_weights = np.full(3, -np.inf)
        active = self.c_bar > 0
        log_weights[active] = np.log(self.c_bar[active]) - 0.5 * (
            np.log(2 * np.pi * variances[active])
            + innovations[active]**2 / variances[active]
        )
        weights = np.exp(log_weights - np.max(log_weights))
        self.mu = weights / weights.sum()

        self._combine(self.mu)
        self.mu_history.append(self.mode_probabilities)
        return self.state.copy()


def score_position_variance(score_part, ref_frame_to_beat, n_frames):
    frames = np.arange(n_frames, dtype=float)
    variance = np.zeros(n_frames)
    if score_part is None or ref_frame_to_beat is None:
        return variance
    events = {}
    for note in score_part.note_array():
        start, duration = float(note["onset_beat"]), float(note["duration_beat"])
        if duration <= 0:
            continue
        pc = int(note["pitch"]) % 12
        for beat, delta in ((start, 1), (start + duration, -1)):
            events.setdefault(beat, []).append((pc, delta))
    counts = np.zeros(12, dtype=int)
    signature, run_start, spans = None, None, []
    for beat, changes in sorted(events.items()):
        for pc, delta in changes:
            counts[pc] += delta
        updated = tuple(np.flatnonzero(counts > 0))
        if updated != signature:
            if run_start is not None:
                spans.append((run_start, beat))
            run_start, signature = beat, updated
    beats = ref_frame_to_beat
    if run_start is not None and run_start < beats[-1]:
        spans.append((run_start, float(beats[-1])))
    for start, end in spans:
        left, right = np.searchsorted(beats, [start, end])
        start_frame, end_frame = np.interp([start, end], beats, frames)
        variance[left:right] = (end_frame - start_frame)**2 / 12
    return variance
