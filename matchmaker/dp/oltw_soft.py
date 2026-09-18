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
    """Interacting Multiple Model (IMM) filter with score-informed transitions and second-order kinematic state [p, v, a]."""

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
        self.H = np.array([[1.0, 0.0, 0.0]])
        self.R = np.array([[obs_var]])
        self.score_part = score_part
        self.r2b = ref_frame_to_beat
        self.tempo_bpm = float(tempo)
        self.frame_rate = int(frame_rate)
        self.frames_per_beat = (self.frame_rate * 60.0) / max(self.tempo_bpm, 1.0)

        # Extract fermata ranges with tie resolution
        fermatas = (
            list(score_part.iter_all(pt.score.Fermata)) if score_part is not None else []
        )
        self.fermata_ranges = []
        for f in fermatas:
            curr = f.ref
            while hasattr(curr, "tie_prev") and curr.tie_prev is not None:
                curr = curr.tie_prev
            try:
                start_b = float(score_part.beat_map(curr.start.t))
                end_b = float(score_part.beat_map(f.ref.end.t))
                self.fermata_ranges.append((start_b, end_b))
            except Exception:
                pass
        self.fermata_ranges = list(set(self.fermata_ranges))

        # Extract true global rests across ALL voices (complete score silence >= 0.5 beats)
        # Single-voice rests must NOT be treated as pauses because other voices are playing notes!
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
                    if active == 0 and t > last_t:
                        if t - last_t >= 0.5:
                            global_rests.append((last_t, t))
                    active += delta
                    last_t = t
            except Exception:
                pass
        self.pause_ranges = list(set(self.fermata_ranges + global_rests))

        # Extract onsets and IOIs
        if score_part is not None:
            try:
                na = score_part.note_array()
                unique_onsets = np.unique(na["onset_beat"])
                self.onsets = unique_onsets
                self.iois = (
                    np.diff(unique_onsets) if len(unique_onsets) > 1 else np.array([1.0])
                )
            except Exception:
                self.onsets = np.array([])
                self.iois = np.array([1.0])
        else:
            self.onsets = np.array([])
            self.iois = np.array([1.0])

        # Kinematic regimes with state vector x = [position, velocity, acceleration]^T:
        # 0: Constant Velocity (CV) - a = 0, v >= v_min
        # 1: Constant Acceleration (CA) - second-order dynamics for tempo variations
        # 2: Zero Velocity (ZV) - stationary regime (v = 0, a = 0) for pauses/rests
        dt = 1.0  # Normalized frame time step
        self.F = [
            np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]
        q_cv = 0.01
        q_ca = 0.05
        self.Q = [
            np.diag([q_cv * 0.25, q_cv, 1e-6]),
            np.array([
                [q_ca * (dt**4) / 4, q_ca * (dt**3) / 2, q_ca * (dt**2) / 2],
                [q_ca * (dt**3) / 2, q_ca * (dt**2), q_ca * dt],
                [q_ca * (dt**2) / 2, q_ca * dt, q_ca],
            ]) + np.diag([1e-4, 1e-4, 1e-4]),
            np.diag([1e-5, 1e-6, 1e-6]),
        ]
        self.reset(position=init_position, tempo=init_tempo)

    @property
    def position(self) -> float:
        return float(self.state[0])

    @property
    def tempo(self) -> float:
        return float(self.state[1])

    @property
    def acceleration(self) -> float:
        return float(self.state[2])

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
        self.states = np.array(
            [
                [position, tempo, 0.0],
                [position, tempo, 0.0],
                [position, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.P_matrices = np.array(
            [
                np.diag([10.0, 1.0, 0.01]),
                np.diag([10.0, 3.0, 1.0]),
                np.diag([10.0, 0.01, 0.01]),
            ],
            dtype=np.float64,
        )
        self.state = np.array([position, tempo, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 1.0, 0.01])
        self.tempo_memory = max(float(tempo), 0.4)

    def get_dynamic_PI(
        self, current_pos_frames: float, is_silent: Optional[bool] = None
    ) -> np.ndarray:
        if self.r2b is not None and len(self.r2b) > 0:
            idx = int(np.clip(round(current_pos_frames), 0, len(self.r2b) - 1))
            curr_beat = float(self.r2b[idx])
        else:
            curr_beat = float(current_pos_frames)

        # Suppress ZV if current score position is close to note onsets
        is_at_onset = False
        if len(self.onsets) > 0:
            dist = np.min(np.abs(self.onsets - curr_beat))
            if dist < 0.15:
                is_at_onset = True

        all_pauses = list(set(self.pause_ranges + self.fermata_ranges))
        matching_pause = None
        is_fermata = False
        for s, e in all_pauses:
            if s <= curr_beat <= e:
                matching_pause = (s, e)
                if (s, e) in self.fermata_ranges:
                    is_fermata = True
                break
        is_score_pause = (not is_at_onset) and (matching_pause is not None)
        if len(all_pauses) == 0 and len(self.onsets) == 0:
            is_pause = bool(is_silent)
        elif is_fermata:
            # Asymmetric fermata handling:
            # Under a score fermata, notes are physically sustained (audio is not silent),
            # but the performer is intentionally holding the chord.
            # ZV must be triggered by score prior regardless of audio level.
            is_pause = is_score_pause
        elif is_silent is False:
            # Active acoustic audio overrides score pause during non-fermata rests
            is_pause = False
        else:
            # is_silent is True or None (unspecified / score-only query): honor score pause
            is_pause = is_score_pause

        if is_pause:
            # Explicit score pause / rest / fermata (or silent in score-free mode): strongly favor Zero-Velocity mode
            pause_dur_beats = (
                max(matching_pause[1] - matching_pause[0], 0.5)
                if matching_pause is not None
                else 1.0
            )
            n_pause_frames = pause_dur_beats * self.frames_per_beat
            p_stay = float(np.clip(1.0 - (1.0 / max(n_pause_frames, 10.0)), 0.90, 0.99))
            p_exit_cv = (1.0 - p_stay) * 0.70
            p_exit_ca = (1.0 - p_stay) * 0.30
            return np.array(
                [
                    [0.10, 0.05, 0.85],
                    [0.05, 0.10, 0.85],
                    [p_exit_cv, p_exit_ca, p_stay],
                ]
            )

        if is_silent:
            # Quiet audio frame during active playing (pianissimo, pedal decay, legato):
            # Maintain Constant Velocity (CV) tempo inertia; do NOT falsely trigger ZV braking.
            return np.array(
                [
                    [0.94, 0.05, 0.01],
                    [0.40, 0.59, 0.01],
                    [0.70, 0.25, 0.05],
                ]
            )

        # Active playing: adapt transition based on CA estimated acceleration
        curr_accel = abs(float(self.states[1, 2]))
        p_maneuver = float(np.clip((curr_accel - 0.012) / 0.025, 0.05, 0.85))

        return np.array(
            [
                [1.0 - p_maneuver - 0.02, p_maneuver, 0.02],
                [0.40, 0.58, 0.02],
                [0.30, 0.10, 0.60],
            ]
        )

    def predict(self, is_silent: bool = False) -> np.ndarray:
        PI = self.get_dynamic_PI(self.position, is_silent=is_silent)
        c_bar = PI.T @ self.mu
        self.c_bar = np.maximum(c_bar, 1e-12)
        omega = (PI * self.mu[:, None]) / self.c_bar[None, :]

        # SKF Insight: Note-length-aware process noise scaling (proportional to local IOI)
        # Fast notes (short IOI) -> high tempo inertia (lower process noise)
        # Long notes (large IOI) -> flexible rubato adaptation (higher process noise)
        ioi_scale = 1.0
        if len(self.onsets) > 0 and len(self.iois) > 0:
            if self.r2b is not None and len(self.r2b) > 0:
                idx = int(np.clip(round(self.position), 0, len(self.r2b) - 1))
                curr_b = float(self.r2b[idx])
            else:
                curr_b = float(self.position)
            k = int(np.clip(np.searchsorted(self.onsets, curr_b, side="right") - 1, 0, len(self.iois) - 1))
            ioi_scale = float(np.clip(np.sqrt(self.iois[k]), 0.5, 2.0))

        Q_scaled = [self.Q[0] * ioi_scale, self.Q[1] * ioi_scale, self.Q[2]]

        x_mixed = np.zeros((3, 3))
        P_mixed = np.zeros((3, 3, 3))
        for j in range(3):
            for i in range(3):
                x_mixed[j] += omega[i, j] * self.states[i]
            for i in range(3):
                dx = self.states[i] - x_mixed[j]
                P_mixed[j] += omega[i, j] * (self.P_matrices[i] + np.outer(dx, dx))

        for j in range(3):
            self.states[j] = self.F[j] @ x_mixed[j]
            self.P_matrices[j] = self.F[j] @ P_mixed[j] @ self.F[j].T + Q_scaled[j]

        # Enforce physical constraints:
        # CV cannot drop to zero velocity (prevents CV from absorbing pauses)
        self.states[0, 1] = max(0.2, self.states[0, 1])
        # ZV strictly maintains zero velocity and acceleration
        self.states[2, 1] = 0.0
        self.states[2, 2] = 0.0

        self.state = np.sum(self.c_bar[:, None] * self.states, axis=0)
        self.P = np.zeros((3, 3))
        for j in range(3):
            dx = self.states[j] - self.state
            self.P += self.c_bar[j] * (self.P_matrices[j] + np.outer(dx, dx))
        return self.state.copy()

    def update(self, z: float) -> np.ndarray:
        z_val = float(z)
        unnorm_mu = np.zeros(3)
        for j in range(3):
            y = z_val - float((self.H @ self.states[j]).item())
            S = float((self.H @ self.P_matrices[j] @ self.H.T + self.R).item())
            K = (self.P_matrices[j] @ self.H.T) / S
            self.states[j] = self.states[j] + K.squeeze() * y
            I_KH = np.eye(3) - np.outer(K.squeeze(), self.H.squeeze())
            self.P_matrices[j] = I_KH @ self.P_matrices[j]
            lik = (1.0 / np.sqrt(2.0 * np.pi * S)) * np.exp(-0.5 * (y**2) / S)
            unnorm_mu[j] = self.c_bar[j] * max(lik, 1e-12)

        sum_unnorm = np.sum(unnorm_mu)
        self.mu = (
            unnorm_mu / sum_unnorm
            if sum_unnorm > 1e-12
            else np.array([0.7, 0.2, 0.1])
        )
        self.mu = np.clip(self.mu, 0.005, 0.99)
        self.mu /= np.sum(self.mu)

        # Enforce physical kinematic constraints post-update:
        self.states[0, 1] = max(0.2, self.states[0, 1])
        self.states[2, 1] = 0.0
        self.states[2, 2] = 0.0

        self.state = np.sum(self.mu[:, None] * self.states, axis=0)
        self.P = np.zeros((3, 3))
        for j in range(3):
            dx = self.states[j] - self.state
            self.P += self.mu[j] * (self.P_matrices[j] + np.outer(dx, dx))
        self.state[1] = np.clip(self.state[1], 0.0, 10.0)
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
        tempo: Optional[float] = None,
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

        if tempo is None:
            if score_part is not None:
                try:
                    from matchmaker.utils.misc import get_tempo_from_score
                    score_tempo = get_tempo_from_score(score_part)
                    tempo = score_tempo if score_tempo is not None else 120.0
                except Exception:
                    tempo = 120.0
            else:
                tempo = 120.0
        self.tempo = float(tempo)

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
            k_pos = float(self.kalman.position)
            if abs(k_pos - self._current_frame) < 3.0:
                return self._frame_to_beat(k_pos)
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

        # Pre-music silence detection
        feat_abs = np.abs(feat)
        peakiness = float(np.max(feat_abs)) / (float(np.mean(feat_abs)) + 1e-10)
        is_uninformative = peakiness < 2.0

        if not self._music_started:
            if is_uninformative:
                self._pos_history.append(self._current_frame)
                return
            self._music_started = True

        # Kalman / IMM prediction
        if isinstance(self.kalman, ScoreInformedIMM):
            predicted_state = self.kalman.predict(is_silent=is_uninformative)
        else:
            predicted_state = self.kalman.predict()
        predicted_pos = predicted_state[0]
        pos_uncertainty = self.kalman.position_uncertainty
        effective_sigma = self.prior_sigma + pos_uncertainty

        # Compute window local distances
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)
        window_start, window_end = self.get_window()
        raw_window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            feat,
            self.distance_func,
        )
        window_cost = raw_window_cost.copy()

        # Cost informativeness
        cost_range = (
            float(np.max(window_cost) - np.min(window_cost))
            if len(window_cost) > 1
            else 0.0
        )
        cost_med = float(np.median(window_cost)) if len(window_cost) > 0 else 1.0
        informativeness = cost_range / (cost_med + 1e-10)

        # Cost-adaptive prior
        if (
            self.prior_lambda > 0
            and self.input_index > 30
            and pos_uncertainty < effective_sigma * 0.5
            and informativeness > 0.5
        ):
            for idx in range(len(window_cost)):
                score_idx = window_start + idx
                deviation = score_idx - predicted_pos
                prior_cost = (deviation**2) / (2.0 * effective_sigma**2)
                window_cost[idx] += self.prior_lambda * prior_cost

        # Tempo-adaptive directional weighting
        in_zv_pause = (
            isinstance(self.kalman, ScoreInformedIMM) and self.kalman.mu[2] > 0.5
        )
        if in_zv_pause:
            wv = 5.0
            wd = 5.0
            wh = 1.0
            effective_wh = 1.0
            expected_tempo = 0.0
        else:
            expected_tempo = 1.0
            recent_tempo = 1.0
            if len(self._pos_history) >= 10:
                lookback = min(30, len(self._pos_history))
                recent_tempo = (
                    self._current_frame - self._pos_history[-lookback]
                ) / lookback

            racing_scale = 0.35 * max(expected_tempo, 0.1)
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

        # Observation update
        adaptive_R = self._obs_var * np.clip(2.0 / (informativeness + 0.5), 0.2, 10.0)
        self.kalman.R = np.array([[adaptive_R]])
        self.kalman.update(float(self._current_frame))

        # Diagnostics
        selected_offset = self._current_frame - window_start
        selected_local_cost = (
            float(raw_window_cost[selected_offset])
            if 0 <= selected_offset < len(raw_window_cost)
            else None
        )
        sorted_local_costs = np.sort(raw_window_cost)
        best_local_cost = (
            float(sorted_local_costs[0]) if len(sorted_local_costs) else None
        )
        local_cost_margin = (
            float(sorted_local_costs[1] - sorted_local_costs[0])
            if len(sorted_local_costs) > 1
            else None
        )
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
            predicted_position=float(predicted_pos),
            position_uncertainty=float(pos_uncertainty),
            normalized_path_cost=(float(min_costs) if np.isfinite(min_costs) else None),
            selected_local_cost=selected_local_cost,
            best_local_cost=best_local_cost,
            median_local_cost=cost_med,
            local_cost_margin=local_cost_margin,
            informativeness=informativeness,
            effective_horizontal_weight=float(effective_wh),
            expected_advance_rate=float(expected_tempo),
            is_informative=not is_uninformative,
            mode_probabilities=mu_val,
        )

        # Dynamic acoustic unfreezing for IMM: release braking upon next note arrival
        if isinstance(self.kalman, ScoreInformedIMM):
            if self.kalman.mu[2] > 0.5:
                self.expected_advance_rate = 0.0
            else:
                self.expected_advance_rate = 1.0
        else:
            self.expected_advance_rate = 1.0

        self.input_index += 1
