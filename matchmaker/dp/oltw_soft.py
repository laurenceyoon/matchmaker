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
    """Interacting Multiple Model (IMM) estimator with score-informed pause transitions."""

    def __init__(
        self,
        score_part: Any = None,
        ref_frame_to_beat: Optional[NDArray] = None,
        init_position: float = 0.0,
        init_tempo: float = 1.0,
        obs_var: float = DEFAULT_OBS_VAR,
        tempo: float = 120.0,
        frame_rate: int = FRAME_RATE,
    ):
        self.H = np.array([1.0, 0.0, 0.0])
        self.R = float(obs_var)
        self.score_part = score_part
        self.r2b = ref_frame_to_beat
        self.tempo_bpm = float(tempo)
        self.frame_rate = int(frame_rate)

        # Extract score pause ranges (fermatas and global rests across all voices)
        fermatas = (
            list(score_part.iter_all(pt.score.Fermata)) if score_part is not None else []
        )
        fermata_ranges = []
        for f in fermatas:
            try:
                fermata_ranges.append(
                    (float(score_part.beat_map(f.ref.start.t)), float(score_part.beat_map(f.ref.end.t)))
                )
            except Exception:
                pass

        global_rests = []
        if score_part is not None:
            try:
                na = score_part.note_array()
                events = []
                for n in na:
                    events.append((float(n["onset_beat"]), 1))
                    events.append((float(n["onset_beat"] + n["duration_beat"]), -1))
                events.sort(key=lambda x: (x[0], -x[1]))
                active = 0
                last_t = 0.0
                for t, delta in events:
                    if active == 0 and t > last_t and (t - last_t >= 0.5):
                        global_rests.append((last_t, t))
                    active += delta
                    last_t = t
            except Exception:
                pass
        self.pause_ranges = list(set(fermata_ranges + global_rests))

        # Kinematic regimes: 0: CV (steady), 1: CA (varying tempo), 2: ZV (pause/rest)
        dt = 1.0
        self.F = [
            np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]
        q_cv, q_ca = 0.01, 0.05
        self.Q = [
            np.diag([q_cv * 0.25, q_cv, 1e-6]),
            np.array([
                [q_ca * (dt**4) / 4, q_ca * (dt**3) / 2, q_ca * (dt**2) / 2],
                [q_ca * (dt**3) / 2, q_ca * (dt**2), q_ca * dt],
                [q_ca * (dt**2) / 2, q_ca * dt, q_ca],
            ]) + np.diag([1e-4, 1e-4, 1e-4]),
            np.diag([1e-6, 1e-6, 1e-6]),
        ]
        self.M_play = np.array([[0.95, 0.05, 0.00],
                                [0.20, 0.80, 0.00],
                                [0.50, 0.50, 0.00]])
        self.M_pause = np.array([[0.10, 0.05, 0.85],
                                 [0.05, 0.10, 0.85],
                                 [0.02, 0.01, 0.97]])
        self.reset(position=init_position, tempo=init_tempo)

    @property
    def position(self) -> float:
        return float(self.state[0])

    @property
    def tempo(self) -> float:
        return float(self.state[1])

    @property
    def position_uncertainty(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 1e-6)))

    @property
    def mode_probabilities(self) -> Tuple[float, float, float]:
        return (float(self.mu[0]), float(self.mu[1]), float(self.mu[2]))

    def reset(self, position: float = 0.0, tempo: float = 1.0) -> None:
        self.mu = np.array([0.7, 0.2, 0.1])
        self.mu_history: list = [tuple(float(x) for x in self.mu)]
        self.c_bar = self.mu.copy()
        self.states = np.array([
            [position, tempo, 0.0],
            [position, tempo, 0.0],
            [position, 0.0, 0.0],
        ])
        self.P_matrices = np.array([
            np.diag([10.0, 1.0, 0.01]),
            np.diag([10.0, 3.0, 1.0]),
            np.diag([10.0, 0.01, 0.01]),
        ])
        self.state = np.array([position, tempo, 0.0])
        self.P = np.diag([10.0, 1.0, 0.01])

    def predict(self, is_silent: Optional[bool] = None) -> np.ndarray:
        if self.r2b is not None and len(self.r2b) > 0:
            idx = int(np.clip(round(self.position), 0, len(self.r2b) - 1))
            curr_beat = float(self.r2b[idx])
        else:
            curr_beat = float(self.position)

        is_pause = (is_silent is True) or any(
            s <= curr_beat <= e for s, e in self.pause_ranges
        )
        M = self.M_pause if is_pause else self.M_play

        c_bar = self.mu @ M
        self.c_bar = np.maximum(c_bar, 1e-12)
        omega = (M * self.mu[:, None]) / self.c_bar[None, :]

        x_mixed = omega.T @ self.states
        P_mixed = np.zeros((3, 3, 3))
        for j in range(3):
            for i in range(3):
                y = self.states[i] - x_mixed[j]
                P_mixed[j] += omega[i, j] * (self.P_matrices[i] + np.outer(y, y))

        for j in range(3):
            self.states[j] = self.F[j] @ x_mixed[j]
            self.P_matrices[j] = self.F[j] @ P_mixed[j] @ self.F[j].T + self.Q[j]

        self.states[0, 1] = max(0.2, self.states[0, 1])
        self.states[2, 1] = 0.0
        self.states[2, 2] = 0.0

        self.state = self.c_bar @ self.states
        self.P = np.zeros((3, 3))
        for j in range(3):
            dx = self.states[j] - self.state
            self.P += self.c_bar[j] * (self.P_matrices[j] + np.outer(dx, dx))
        return self.state.copy()

    def update(self, z: float) -> np.ndarray:
        z_val = float(z)
        lik = np.zeros(3)
        for j in range(3):
            y = z_val - np.dot(self.H, self.states[j])
            S = np.dot(self.H, self.P_matrices[j] @ self.H) + self.R
            K = (self.P_matrices[j] @ self.H) / S
            self.states[j] += K * y
            self.P_matrices[j] -= np.outer(K, self.H) @ self.P_matrices[j]
            lik[j] = (1.0 / np.sqrt(2.0 * np.pi * S)) * np.exp(-0.5 * (y**2) / S)

        unnorm_mu = self.c_bar * np.maximum(lik, 1e-12)
        self.mu = unnorm_mu / np.sum(unnorm_mu)

        self.states[0, 1] = max(0.2, self.states[0, 1])
        self.states[2, 1] = 0.0
        self.states[2, 2] = 0.0

        self.state = self.mu @ self.states
        self.P = np.zeros((3, 3))
        for j in range(3):
            dx = self.states[j] - self.state
            self.P += self.mu[j] * (self.P_matrices[j] + np.outer(dx, dx))
        self.mu_history.append(tuple(float(x) for x in self.mu))
        return self.state.copy()


class SoftOnlineTimeWarping(OnlineAlignment):
    """Soft-min Online Time Warping with directional weighting and score-informed IMM."""

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

        if self.use_imm and self.score_part is not None:
            self.kalman = ScoreInformedIMM(
                score_part=self.score_part,
                ref_frame_to_beat=self._ref_frame_to_beat,
                obs_var=self._obs_var,
                tempo=self.tempo,
                frame_rate=self.frame_rate,
            )
        else:
            self.kalman = PositionTempoKalman(obs_var=self._obs_var)

        self.reset()

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
        if isinstance(self.kalman, ScoreInformedIMM) and self.input_index > 5:
            if self.kalman.mu[2] > 0.5:
                return self._frame_to_beat(float(self.kalman.position))
            return self._frame_to_beat(self._current_frame)
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
        if not self._music_started:
            if peakiness < 2.0:
                self._pos_history.append(self._current_frame)
                return
            self._music_started = True

        # IMM / Kalman prediction
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

        # Directional weighting: penalize horizontal stalling when tracker is active
        expected_tempo = 1.0
        recent_tempo = 1.0
        if len(self._pos_history) >= self._min_history_frames:
            lookback = min(self._lookback_frames, len(self._pos_history))
            recent_tempo = (
                self._current_frame - self._pos_history[-lookback]
            ) / lookback

        racing_scale = self.velocity_scale * max(expected_tempo, 0.1)
        racing_amount = np.clip(
            (recent_tempo - expected_tempo) / racing_scale, 0.0, 1.0
        )
        effective_wh = 1.0 + (self.w_horizontal - 1.0) * racing_amount
        wv = 1.0
        wd = 1.0
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
            effective_horizontal_weight=float(wh),
            expected_advance_rate=float(self.kalman.tempo),
            is_informative=True,
            mode_probabilities=mu_val,
        )

        self.input_index += 1
