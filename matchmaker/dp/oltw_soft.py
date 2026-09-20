#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Soft Online Time Warping (Soft-OLTW) with IMM Tempo Tracking.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numba
import numpy as np
import partitura as pt
from numpy.typing import NDArray
from scipy.linalg import expm

import time

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
    distances,
)
from matchmaker.utils.distances import Metric, vdist
from matchmaker.utils.errors import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
)
from matchmaker.utils.misc import set_latency_stats

DEFAULT_GAMMA: float = 0.05
DEFAULT_W_HORIZONTAL: float = 10.0
DEFAULT_OBS_VAR: float = 5.0
DEFAULT_TEMPO_LOOKBACK_SEC: float = 1.0
DEFAULT_VELOCITY_SCALE: float = 0.35
DEFAULT_MIN_HISTORY_SEC: float = 0.33


@dataclass(frozen=True)
class SoftOLTWStepDiagnostics:
    """Read-only diagnostics produced by a Soft-OLTW update step."""

    input_index: int
    score_position: int
    score_beat: float
    window_start: int
    window_end: int
    predicted_position: Optional[float]
    position_uncertainty: float
    normalized_path_cost: Optional[float]
    selected_local_cost: Optional[float]
    best_local_cost: Optional[float]
    median_local_cost: Optional[float]
    local_cost_margin: Optional[float]
    informativeness: float
    effective_horizontal_weight: float
    expected_advance_rate: float
    is_informative: bool
    mode_probabilities: Optional[Tuple[float, float, float]] = None


@numba.njit(cache=True)
def softmin(a: float, b: float, c: float, gamma: float) -> float:
    """Soft minimum via log-sum-exp."""
    min_val = min(a, b, c)
    if min_val == np.inf:
        return np.inf
    exp_sum = (
        np.exp(-(a - min_val) / gamma)
        + np.exp(-(b - min_val) / gamma)
        + np.exp(-(c - min_val) / gamma)
    )
    return min_val - gamma * np.log(exp_sum + 1e-10)


@numba.njit(cache=True)
def weighted_soft_oltw_loop(
    global_cost_matrix: np.ndarray,
    window_cost: np.ndarray,
    window_start: int,
    window_end: int,
    input_index: int,
    min_costs: float,
    min_index: int,
    gamma: float = DEFAULT_GAMMA,
    w_vertical: float = 1.0,
    w_horizontal: float = 1.0,
    w_diagonal: float = 1.0,
):
    """Directionally weighted Soft-OLTW dynamic programming inner loop."""
    N = global_cost_matrix.shape[0]
    idx = 0
    score_index = window_start

    if score_index == input_index == 0:
        global_cost_matrix[1, 1] = np.sum(window_cost)
        min_costs = global_cost_matrix[1, 1]
        min_index = 0

    while score_index < window_end:
        if not (score_index == input_index == 0):
            local_dist = window_cost[idx]
            dist1 = global_cost_matrix[score_index, 1] + w_vertical * local_dist
            dist2 = global_cost_matrix[score_index + 1, 0] + w_horizontal * local_dist
            dist3 = global_cost_matrix[score_index, 0] + w_diagonal * local_dist
            soft_min_dist = softmin(dist1, dist2, dist3, gamma)
            global_cost_matrix[score_index + 1, 1] = soft_min_dist
            norm_cost = soft_min_dist / (input_index + score_index + 1.0)
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index
        idx += 1
        score_index += 1

    for i in range(N):
        global_cost_matrix[i, 0] = global_cost_matrix[i, 1]
        global_cost_matrix[i, 1] = np.inf

    return global_cost_matrix, min_index, min_costs


@numba.njit(cache=True)
def multi_path_soft_oltw_loop(
    global_cost_matrix: np.ndarray,
    window_cost: np.ndarray,
    window_start: int,
    window_end: int,
    input_index: int,
    min_costs: float,
    min_index: int,
    gamma_arr: np.ndarray,
    max_score_step: int = 1,
    max_time_step: int = 1,
    w_vertical: float = 1.0,
    w_horizontal: float = 1.0,
    w_diagonal: float = 1.0,
):
    """Generalized multi-path Soft-OLTW loop considering paths within a window.

    Inspired by Carabias-Orti et al. (ISMIR 2015) Eq. (2), which considers
    arbitrary step sizes (c_i, c_j) across score and time dimensions.
    """
    N = global_cost_matrix.shape[0]
    num_cols = global_cost_matrix.shape[1]
    curr_col = num_cols - 1

    idx = 0
    score_index = window_start

    if score_index == input_index == 0:
        global_cost_matrix[1, curr_col] = np.sum(window_cost)
        min_costs = global_cost_matrix[1, curr_col]
        min_index = 0

    max_candidates = max_score_step + max_time_step + (max_score_step * max_time_step) + 8
    candidates = np.empty(max_candidates, dtype=np.float64)

    while score_index < window_end:
        if not (score_index == input_index == 0):
            local_dist = window_cost[idx]
            g = gamma_arr[score_index]
            n_cands = 0
            curr_s = score_index + 1

            # 1. Vertical steps: same audio frame (curr_col), previous score frames
            for cs in range(1, max_score_step + 1):
                prev_s = curr_s - cs
                if prev_s >= 0:
                    cost_val = global_cost_matrix[prev_s, curr_col]
                    if cost_val < np.inf:
                        candidates[n_cands] = cost_val + w_vertical * local_dist
                        n_cands += 1

            # 2. Horizontal steps: previous audio frames, same score frame
            for ct in range(1, max_time_step + 1):
                prev_col = curr_col - ct
                if prev_col >= 0 and input_index >= ct:
                    cost_val = global_cost_matrix[curr_s, prev_col]
                    if cost_val < np.inf:
                        candidates[n_cands] = cost_val + w_horizontal * local_dist
                        n_cands += 1

            # 3. Diagonal / multi-step transitions: previous audio frames, previous score frames
            for ct in range(1, max_time_step + 1):
                prev_col = curr_col - ct
                if prev_col >= 0 and input_index >= ct:
                    for cs in range(1, max_score_step + 1):
                        prev_s = curr_s - cs
                        if prev_s >= 0:
                            cost_val = global_cost_matrix[prev_s, prev_col]
                            if cost_val < np.inf:
                                candidates[n_cands] = cost_val + w_diagonal * local_dist
                                n_cands += 1

            if n_cands == 0:
                soft_min_dist = np.inf
            else:
                min_val = np.inf
                for ci in range(n_cands):
                    if candidates[ci] < min_val:
                        min_val = candidates[ci]
                if min_val == np.inf:
                    soft_min_dist = np.inf
                else:
                    exp_sum = 0.0
                    for ci in range(n_cands):
                        exp_sum += np.exp(-(candidates[ci] - min_val) / g)
                    soft_min_dist = min_val - g * np.log(exp_sum + 1e-10)

            global_cost_matrix[curr_s, curr_col] = soft_min_dist
            norm_cost = soft_min_dist / (input_index + score_index + 1.0)
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        idx += 1
        score_index += 1

    for c in range(curr_col):
        for i in range(N):
            global_cost_matrix[i, c] = global_cost_matrix[i, c + 1]
    for i in range(N):
        global_cost_matrix[i, curr_col] = np.inf

    return global_cost_matrix, min_index, min_costs


class PositionTempoKalman:
    """2D Kalman filter tracking score position and tempo."""

    def __init__(
        self,
        init_position: float = 0.0,
        init_tempo: float = 1.0,
        process_var: float = 0.5,
        obs_var: float = DEFAULT_OBS_VAR,
    ):
        self.state = np.array([init_position, init_tempo], dtype=np.float64)
        self.P = np.diag([10.0, 1.0])
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Q = np.array(
            [[process_var * 0.25, process_var * 0.5], [process_var * 0.5, process_var]]
        )
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[obs_var]])

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()

    def update(self, observed_position: float) -> np.ndarray:
        y = observed_position - float((self.H @ self.state).item())
        S = float((self.H @ self.P @ self.H.T + self.R).item())
        K = (self.P @ self.H.T) / S
        self.state = self.state + K.squeeze() * y
        self.P = (np.eye(2) - np.outer(K.squeeze(), self.H.squeeze())) @ self.P
        self.state[1] = np.clip(self.state[1], 0.1, 10.0)
        return self.state.copy()

    @property
    def position(self) -> float:
        return float(self.state[0])

    @property
    def tempo(self) -> float:
        return float(self.state[1])

    @property
    def position_uncertainty(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 1e-6)))

    def reset(self, position: float = 0.0, tempo: float = 1.0) -> None:
        self.state = np.array([position, tempo], dtype=np.float64)
        self.P = np.diag([10.0, 1.0])


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
        single_model: bool = False,
    ):
        if obs_var <= 0 or tempo <= 0 or frame_rate <= 0:
            raise ValueError("Observation variance, tempo and frame rate must be positive")
        self.single_model = single_model
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
        if self.single_model:
            return np.tile([1.0, 0.0, 0.0], (3, 1))
        n = 3 if allow_pause else 2
        durations = np.array([self.measure_seconds, self.beat_seconds, self.beat_seconds])[:n]
        generator = np.ones((n, n)) / (durations[:, None] * (n - 1))
        np.fill_diagonal(generator, -1 / durations)
        matrix = np.zeros((3, 3))
        matrix[:n, :n] = expm(generator * self.dt)
        if not allow_pause:
            matrix[2, :2] = 1 / 2
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

        self.mu = np.array([1.0, 0.0, 0.0])
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

class SoftOnlineTimeWarping(OnlineAlignment):
    DEFAULT_DISTANCE_FUNC: str = "Manhattan"

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: Optional[NDArray[np.float32]] = None,
        window_size: int = 10,
        step_size: int = 3,
        gamma: float = DEFAULT_GAMMA,
        w_horizontal: float = DEFAULT_W_HORIZONTAL,
        prior_lambda: float = 0.0,
        prior_sigma: float = 50.0,
        obs_var: float = DEFAULT_OBS_VAR,
        use_imm: bool = True,
        tempo_model: str = "imm",
        score_observation: str = "variance",
        score_part: Any = None,
        distance_func: Union[str, Callable, Tuple[str, Dict[str, Any]]] = DEFAULT_DISTANCE_FUNC,
        start_window_size: Union[float, int] = 0.1,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: Optional[NDArray] = None,
        queue: Optional[RECVQueue] = None,
        max_score_step: int = 1,
        max_time_step: int = 1,
        repeat_boundaries: Optional[Any] = None,
        gamma_repeat_factor: float = 1.0,
        gamma_repeat_window_beats: float = 1.5,
        tempo: float = 120.0,
        tempo_lookback_sec: float = DEFAULT_TEMPO_LOOKBACK_SEC,
        velocity_scale: float = DEFAULT_VELOCITY_SCALE,
        min_history_sec: float = DEFAULT_MIN_HISTORY_SEC,
        **kwargs,
    ) -> None:
        if ref_frame_to_beat is None and score_positions is not None:
            ref_frame_to_beat = score_positions
        if score_positions is None and ref_frame_to_beat is not None:
            score_positions = ref_frame_to_beat

        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.N_ref: int = self.reference_features.shape[0]
        self.frame_rate = frame_rate
        self.tempo = float(tempo)
        self.tempo_lookback_sec = float(tempo_lookback_sec)
        self.velocity_scale = float(velocity_scale)
        self.min_history_sec = float(min_history_sec)
        self._lookback_frames = max(1, int(np.round(self.tempo_lookback_sec * self.frame_rate)))
        self._min_history_frames = max(2, int(np.round(self.min_history_sec * self.frame_rate)))
        self._ref_frame_to_beat = ref_frame_to_beat
        self.step_size = step_size
        self._window_size = int(np.round(window_size * self.frame_rate))
        self._start_window_size = int(np.round(start_window_size * frame_rate))
        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        self._init_distance_func(distance_func)

        self.gamma = gamma
        self.w_horizontal = w_horizontal
        self.prior_lambda = prior_lambda
        self.prior_sigma = prior_sigma
        self._obs_var = obs_var
        self.expected_advance_rate = 1.0
        self.use_imm = use_imm
        if tempo_model not in ("imm", "cv"):
            raise ValueError("tempo_model must be imm or cv")
        if score_observation not in ("fixed", "variance"):
            raise ValueError("score_observation must be fixed or variance")
        self.score_observation = score_observation
        self.score_part = score_part

        self.max_score_step = max(1, int(max_score_step))
        self.max_time_step = max(1, int(max_time_step))
        self.gamma_repeat_factor = float(gamma_repeat_factor)
        self.gamma_repeat_window_beats = float(gamma_repeat_window_beats)

        if repeat_boundaries is not None:
            self.repeat_boundaries = list(repeat_boundaries)
        else:
            self.repeat_boundaries = self._extract_repeat_boundaries()

        self._init_gamma_array()
        self._init_score_uncertainty()

        if self.use_imm:
            self.kalman = ScoreInformedIMM(
                score_part=self.score_part,
                ref_frame_to_beat=self._ref_frame_to_beat,
                obs_var=self._obs_var,
                tempo=self.tempo,
                frame_rate=self.frame_rate,
                single_model=tempo_model == "cv",
            )
        else:
            self.kalman = PositionTempoKalman(obs_var=self._obs_var)

        self.reset()

    def _init_score_uncertainty(self) -> None:
        frames = np.arange(self.N_ref, dtype=float)
        self._score_position_variance = np.zeros(self.N_ref)
        if self.score_part is None or self._ref_frame_to_beat is None:
            return
        events = {}
        for note in self.score_part.note_array():
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
        beats = self._ref_frame_to_beat
        if run_start is not None and run_start < beats[-1]:
            spans.append((run_start, float(beats[-1])))
        for start, end in spans:
            left, right = np.searchsorted(beats, [start, end])
            start_frame, end_frame = np.interp([start, end], beats, frames)
            self._score_position_variance[left:right] = (end_frame - start_frame)**2 / 12

    def _extract_repeat_boundaries(self) -> list:
        rep_beats: list = []
        if self.score_part is not None:
            try:
                import partitura as pt
                for r in self.score_part.iter_all(pt.score.Repeat):
                    if hasattr(r, "start") and r.start is not None and hasattr(r.start, "t"):
                        rep_beats.append(float(r.start.t))
                    if hasattr(r, "end") and r.end is not None and hasattr(r.end, "t"):
                        rep_beats.append(float(r.end.t))
                for end in self.score_part.iter_all(pt.score.Ending):
                    if hasattr(end, "start") and end.start is not None and hasattr(end.start, "t"):
                        rep_beats.append(float(end.start.t))
            except Exception:
                pass
        return sorted(list(set(rep_beats)))

    def _init_gamma_array(self) -> None:
        self.gamma_arr = np.full(self.N_ref, self.gamma, dtype=np.float64)
        if self.gamma_repeat_factor == 1.0 or not self.repeat_boundaries or self._ref_frame_to_beat is None:
            return

        sigma = max(0.1, self.gamma_repeat_window_beats / 2.0)
        two_sigma_sq = 2.0 * (sigma ** 2)
        factor_diff = self.gamma_repeat_factor - 1.0

        for frame in range(self.N_ref):
            beat = self._frame_to_beat(frame)
            min_dist = min(abs(beat - r_beat) for r_beat in self.repeat_boundaries)
            if min_dist <= self.gamma_repeat_window_beats * 2.0:
                mult = 1.0 + factor_diff * np.exp(-(min_dist ** 2) / two_sigma_sq)
                self.gamma_arr[frame] = float(np.clip(self.gamma * mult, 1e-4, 1.0))

    def _init_distance_func(self, distance_func: Union[str, Callable, Tuple[str, Dict[str, Any]]]) -> None:
        if not (isinstance(distance_func, (str, tuple)) or callable(distance_func)):
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="distance_func",
                required_parameter_type=(str, tuple, Callable),
                actual_parameter_type=type(distance_func),
            )

        if isinstance(distance_func, str):
            if distance_func not in CYTHONIZED_METRICS_WO_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_WO_ARGUMENTS,
                    value=distance_func,
                )
            self.distance_func = getattr(distances, distance_func)()
        elif isinstance(distance_func, tuple):
            if distance_func[0] not in CYTHONIZED_METRICS_W_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_W_ARGUMENTS,
                    value=distance_func[0],
                )
            self.distance_func = getattr(distances, distance_func[0])(**distance_func[1])
        elif callable(distance_func):
            self.distance_func = distance_func

        if isinstance(self.distance_func, Metric):
            self.vdist = vdist
        else:
            self.vdist = lambda X, y, lcf: np.array([lcf(x, y) for x in X]).astype(np.float32)

    def reset(self) -> None:
        self.current_index = 0
        self._current_frame = 0
        self.input_index = 0
        self.input_features: list = []
        self._alignment_path = []
        self.global_cost_matrix = np.full(
            (self.N_ref + 1, self.max_time_step + 1), np.inf, dtype=np.float64
        )
        if hasattr(self, "kalman"):
            self.kalman.reset()
        self._pos_history = []
        self._music_started = False
        self.last_diagnostics: Optional[SoftOLTWStepDiagnostics] = None
        self.mu_history: list = []

    @property
    def window_index(self) -> int:
        return self._current_frame

    def _frame_to_beat(self, frame: Union[int, float]) -> float:
        if self._ref_frame_to_beat is None:
            return float(frame)
        n = len(self._ref_frame_to_beat)
        if n == 0:
            return float(frame)
        f_clamped = max(0.0, min(float(frame), float(n - 1)))
        i = int(f_clamped)
        if i >= n - 1:
            return float(self._ref_frame_to_beat[-1])
        r = f_clamped - i
        return float(
            (1.0 - r) * self._ref_frame_to_beat[i]
            + r * self._ref_frame_to_beat[i + 1]
        )

    def _frame_to_score_idx(self, frame: int) -> int:
        if self.score_positions is None:
            return frame
        beat = self._frame_to_beat(frame)
        idx = int(np.searchsorted(self.score_positions, beat, side="right") - 1)
        return max(0, min(idx, len(self.score_positions) - 1))

    def get_current_position(self) -> float:
        if isinstance(self.kalman, ScoreInformedIMM):
            return self._frame_to_beat(self.kalman.position)
        return self._frame_to_beat(self._current_frame)

    def get_window(self) -> Tuple[int, int]:
        w = self._window_size
        if self.window_index < self._start_window_size:
            w = self._start_window_size
        start = max(self.window_index - w, 0)
        end = min(self.window_index + w, self.N_ref)
        return start, end

    def __call__(self, observation: Any, perf_time: float) -> float:
        t0 = time.time()
        self.input_features.append(observation)
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return beat

    def is_still_following(self) -> bool:
        if self.score_positions is not None and len(self.score_positions) > 0:
            return self.current_index < len(self.score_positions) - 1
        return self._current_frame < self.N_ref - 2

    def step(self, input_features: NDArray[np.float32]) -> None:
        feat = input_features.squeeze()

        # Lead-in silence gating: wait for first audio onset before advancing
        feat_abs = np.abs(feat)
        peakiness = float(np.max(feat_abs)) / (float(np.mean(feat_abs)) + 1e-10)
        is_silent = peakiness < 2.0
        if not self._music_started:
            if is_silent:
                self._pos_history.append(self._current_frame)
                return
            self._music_started = True

        if isinstance(self.kalman, ScoreInformedIMM):
            frame = int(np.clip(round(self.kalman.position), 0, self.N_ref - 1))
            variance = self._score_position_variance[frame] if self.score_observation == "variance" else 0.0
            self.kalman.set_observation_error(variance, initialize=self.input_index == 0)
        if self.input_index > 0:
            self.kalman.predict()

        # Window local distances
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)
        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            feat,
            self.distance_func,
        )

        is_imm = isinstance(self.kalman, ScoreInformedIMM)
        expected_tempo = float(self.kalman.tempo) if is_imm else 1.0
        recent_tempo = 1.0
        if len(self._pos_history) >= self._min_history_frames:
            lookback = min(self._lookback_frames, len(self._pos_history))
            recent_tempo = (
                self._current_frame - self._pos_history[-lookback]
            ) / lookback
        excess_tempo = (recent_tempo - 1.0) / self.velocity_scale
        effective_wh = 1.0 + (self.w_horizontal - 1.0) * np.clip(
            excess_tempo, 0.0, 1.0
        )
        wv = wd = 1.0
        wh = effective_wh

        # Weighted Soft-OLTW DP loop
        if (
            self.max_score_step == 1
            and self.max_time_step == 1
            and self.gamma_repeat_factor == 1.0
        ):
            self.global_cost_matrix, min_index, min_costs = weighted_soft_oltw_loop(
                global_cost_matrix=self.global_cost_matrix,
                window_cost=window_cost,
                window_start=window_start,
                window_end=window_end,
                input_index=self.input_index,
                min_costs=min_costs,
                min_index=min_index,
                gamma=self.gamma,
                w_vertical=wv,
                w_horizontal=wh,
                w_diagonal=wd,
            )
        else:
            self.global_cost_matrix, min_index, min_costs = multi_path_soft_oltw_loop(
                global_cost_matrix=self.global_cost_matrix,
                window_cost=window_cost,
                window_start=window_start,
                window_end=window_end,
                input_index=self.input_index,
                min_costs=min_costs,
                min_index=min_index,
                gamma_arr=self.gamma_arr,
                max_score_step=self.max_score_step,
                max_time_step=self.max_time_step,
                w_vertical=wv,
                w_horizontal=wh,
                w_diagonal=wd,
            )

        # Update position
        allowed_step = max(self.step_size, self.max_score_step)
        if self.input_index == 0:
            self._current_frame = min(
                max(self._current_frame, min_index),
                self._current_frame,
            )
        else:
            self._current_frame = min(
                max(self._current_frame, min_index),
                self._current_frame + allowed_step,
            )
        self.current_index = self._frame_to_score_idx(self._current_frame)
        self._pos_history.append(self._current_frame)

        # IMM observation update

        self.kalman.update(float(self._current_frame))

        mu_val = (
            tuple(float(x) for x in self.kalman.mu)
            if isinstance(self.kalman, ScoreInformedIMM)
            else (1.0, 0.0, 0.0)
        )
        self.mu_history.append(mu_val)

        self.last_diagnostics = SoftOLTWStepDiagnostics(
            input_index=self.input_index,
            score_position=self._current_frame,
            score_beat=self.get_current_position(),
            window_start=window_start,
            window_end=window_end,
            predicted_position=float(self.kalman.position),
            position_uncertainty=float(self.kalman.position_uncertainty),
            normalized_path_cost=(
                float(min_costs) if np.isfinite(min_costs) else None
            ),
            selected_local_cost=None,
            best_local_cost=None,
            median_local_cost=None,
            local_cost_margin=None,
            informativeness=1.0,
            effective_horizontal_weight=float(effective_wh),
            expected_advance_rate=float(expected_tempo),
            is_informative=True,
            mode_probabilities=mu_val,
        )

        self.input_index += 1
