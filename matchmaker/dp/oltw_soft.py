import time
from typing import Any, Dict, Optional, Tuple, Union

import numba
import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.oltw_imm import DEFAULT_GAMMA, IMMPathFilter
from matchmaker.prob.imm import (
    DEFAULT_OBS_VAR,
    IMMMotionModels,
    score_position_variance,
    score_activity,
)
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats

DEFAULT_W_HORIZONTAL: float = 10.0
DEFAULT_TEMPO_LOOKBACK_SEC: float = 1.0
DEFAULT_VELOCITY_SCALE: float = 0.35
DEFAULT_MIN_HISTORY_SEC: float = 0.33


@numba.extending.register_jitable
def softmin(a: float, b: float, c: float, gamma: float) -> float:
    """Soft minimum via log-sum-exp."""
    min_val = min(a, b, c)
    if min_val == np.inf or gamma == 0:
        return min_val
    exp_sum = (
        np.exp(-(a - min_val) / gamma)
        + np.exp(-(b - min_val) / gamma)
        + np.exp(-(c - min_val) / gamma)
    )
    return min_val - gamma * np.log(exp_sum + 1e-10)


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


_weighted_soft_oltw_loop_numba = numba.njit(cache=True)(weighted_soft_oltw_loop)


def _manhattan(x, y):
    return np.abs(x - y).sum()


def _python_vdist(x, y, metric):
    # Sequential float32 accumulation matches the accelerated Manhattan metric.
    return np.add.accumulate(np.abs(x - y), axis=1)[:, -1]


class SoftWarpingPath:
    """Acoustic DP state; both backends execute the same Python recurrence."""

    def __init__(self, size, backend="numba"):
        self.size = size
        self._loop = (
            _weighted_soft_oltw_loop_numba
            if backend == "numba"
            else weighted_soft_oltw_loop
        )

    def reset(self, position=0):
        self.costs = np.full((self.size + 1, 2), np.inf)
        self.index = position

    def observe_silence(self, rest_frames):
        return False

    def step(self, distances, start, gamma, input_index, horizontal_weight):
        self.costs, self.index, cost = self._loop(
            self.costs,
            distances,
            start,
            start + len(distances),
            input_index,
            np.inf,
            self.index,
            gamma,
            w_horizontal=horizontal_weight,
        )
        return self.index, cost


class SoftOnlineTimeWarping(OnlineAlignment):
    """Soft-OLTW score follower with candidate-wise IMM path inference.

    ``backend="python"`` uses Python/NumPy alignment calculations.
    ``use_imm=False`` selects acoustic DP for the paper ablation.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: Optional[NDArray[np.float32]] = None,
        window_size: int = 10,
        step_size: int = 3,
        gamma: float = DEFAULT_GAMMA,
        w_horizontal: float = DEFAULT_W_HORIZONTAL,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: Optional[NDArray] = None,
        queue: Optional[RECVQueue] = None,
        backend: str = "numba",
        *,
        score_part: Any = None,
        tempo: float = 120.0,
        use_imm: bool = True,
        use_silence: bool = False,
        imm_modes: Tuple[str, ...] = ("cv", "ca", "zv"),
        score_pause_gating: bool = True,
        correlated_observation: bool = True,
        obs_var: float = DEFAULT_OBS_VAR,
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
        self._lookback_frames = max(
            1, int(np.round(DEFAULT_TEMPO_LOOKBACK_SEC * self.frame_rate))
        )
        self._min_history_frames = max(
            2, int(np.round(DEFAULT_MIN_HISTORY_SEC * self.frame_rate))
        )
        self._ref_frame_to_beat = ref_frame_to_beat
        self.step_size = step_size
        self._window_size = int(np.round(window_size * self.frame_rate))
        self._start_window_size = int(np.round(0.1 * frame_rate))
        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        if backend not in ("numba", "python"):
            raise ValueError("backend must be 'numba' or 'python'")
        self.backend = backend
        if backend == "python":
            self.distance_func, self.vdist = _manhattan, _python_vdist
        else:
            from matchmaker.utils.distances import Manhattan, vdist

            self.distance_func, self.vdist = Manhattan(), vdist

        self.gamma = gamma
        self.w_horizontal = w_horizontal
        self.use_imm = use_imm
        self.use_silence = use_silence
        sounding = score_activity(score_part, ref_frame_to_beat, self.N_ref)
        self.rest_frames = np.flatnonzero(np.diff(np.r_[True, sounding, True])).reshape(-1, 2)
        self.score_part = score_part
        self.tempo = float(tempo)
        self.imm_modes = imm_modes
        self.score_pause_gating = score_pause_gating
        self.correlated_observation = correlated_observation
        self.obs_var = obs_var
        self.path = self._create_path()

        self.reset()

    def _create_path(self):
        if not self.use_imm:
            return SoftWarpingPath(self.N_ref, self.backend)
        model = IMMMotionModels(
            score_part=self.score_part,
            ref_frame_to_beat=self._ref_frame_to_beat,
            obs_var=self.obs_var,
            tempo=self.tempo,
            frame_rate=self.frame_rate,
            modes=self.imm_modes,
            score_pause_gating=self.score_pause_gating,
            retain_tempo=self.use_silence,
        )
        variance = score_position_variance(
            self.score_part if self.correlated_observation else None,
            self._ref_frame_to_beat,
            self.N_ref,
        )
        return IMMPathFilter(model, variance, self.step_size)

    def reset(self, position=0) -> None:
        self.current_index = self._frame_to_score_idx(position)
        self._current_frame = position
        self.input_index = 0
        self.input_features = []
        self._alignment_path = []
        self.path.reset(position)
        self._pos_history = []
        self._music_started = False

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
            (1.0 - r) * self._ref_frame_to_beat[i] + r * self._ref_frame_to_beat[i + 1]
        )

    def _frame_to_score_idx(self, frame: int) -> int:
        if self.score_positions is None:
            return frame
        beat = self._frame_to_beat(frame)
        idx = int(np.searchsorted(self.score_positions, beat, side="right") - 1)
        return max(0, min(idx, len(self.score_positions) - 1))

    def get_current_position(self) -> float:
        frame = self.path.position if self.use_imm else self._current_frame
        return self._frame_to_beat(frame)

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

        feat_abs = np.abs(feat)
        peakiness = float(np.max(feat_abs)) / (float(np.mean(feat_abs)) + 1e-10)
        is_silent = peakiness < 2.0
        if not self._music_started:
            if is_silent:
                self._pos_history.append(self._current_frame)
                return
            self._music_started = True

        if self.use_silence and not feat.any():
            if self.path.observe_silence(self.rest_frames):
                return

        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            feat,
            self.distance_func,
        )

        recent_tempo = 1.0
        if len(self._pos_history) >= self._min_history_frames:
            lookback = min(self._lookback_frames, len(self._pos_history))
            recent_tempo = (
                self._current_frame - self._pos_history[-lookback]
            ) / lookback
        excess_tempo = (recent_tempo - 1.0) / DEFAULT_VELOCITY_SCALE
        horizontal_weight = 1.0 + (self.w_horizontal - 1.0) * np.clip(
            excess_tempo, 0.0, 1.0
        )

        min_index, _ = self.path.step(
            window_cost, window_start, self.gamma, self.input_index, horizontal_weight
        )
        if self.input_index > 0:
            self._current_frame = min(
                max(self._current_frame, min_index),
                self._current_frame + self.step_size,
            )
        self.current_index = self._frame_to_score_idx(self._current_frame)
        self._pos_history.append(self._current_frame)

        self.input_index += 1
