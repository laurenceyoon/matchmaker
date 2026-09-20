import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numba
import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.kalman_path import KalmanPathLattice
from matchmaker.prob.imm import DEFAULT_OBS_VAR, ScoreInformedIMM, score_position_variance
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
DEFAULT_TEMPO_LOOKBACK_SEC: float = 1.0
DEFAULT_VELOCITY_SCALE: float = 0.35
DEFAULT_MIN_HISTORY_SEC: float = 0.33


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
        obs_var: float = DEFAULT_OBS_VAR,
        use_imm: bool = True,
        path_tempo: bool = False,
        score_part: Any = None,
        distance_func: Union[str, Callable, Tuple[str, Dict[str, Any]]] = DEFAULT_DISTANCE_FUNC,
        start_window_size: Union[float, int] = 0.1,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: Optional[NDArray] = None,
        queue: Optional[RECVQueue] = None,
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
        self._obs_var = obs_var
        self.use_imm = use_imm
        self.path_tempo = use_imm and path_tempo
        self.score_part = score_part

        if self.use_imm:
            self._score_position_variance = score_position_variance(
                self.score_part, self._ref_frame_to_beat, self.N_ref,
            )
            self.kalman = ScoreInformedIMM(
                score_part=self.score_part,
                ref_frame_to_beat=self._ref_frame_to_beat,
                obs_var=self._obs_var,
                tempo=self.tempo,
                frame_rate=self.frame_rate,
            )
        else:
            self.kalman = None

        self.reset()

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
        self.global_cost_matrix = (
            None if self.path_tempo else np.full((self.N_ref + 1, 2), np.inf)
        )
        if self.kalman is not None:
            self.kalman.reset()
        self.path_lattice = (
            KalmanPathLattice(self.N_ref, self._obs_var,
                             self.kalman.beat_seconds * self.frame_rate, self.step_size)
            if self.path_tempo else None
        )
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
        if self.kalman is not None:
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

        feat_abs = np.abs(feat)
        peakiness = float(np.max(feat_abs)) / (float(np.mean(feat_abs)) + 1e-10)
        is_silent = peakiness < 2.0
        if not self._music_started:
            if is_silent:
                self._pos_history.append(self._current_frame)
                return
            self._music_started = True

        if self.kalman is not None:
            frame = int(np.clip(round(self.kalman.position), 0, self.N_ref - 1))
            variance = self._score_position_variance[frame]
            self.kalman.set_observation_error(variance, initialize=self.input_index == 0)
            if self.input_index > 0:
                self.kalman.predict()

        min_index = max(self.window_index - self.step_size, 0)
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
        excess_tempo = (recent_tempo - 1.0) / self.velocity_scale
        horizontal_weight = 1.0 + (self.w_horizontal - 1.0) * np.clip(
            excess_tempo, 0.0, 1.0
        )

        if self.path_lattice is not None:
            min_index, _ = self.path_lattice.step(
                window_cost, window_start, self.gamma, self.input_index, horizontal_weight,
            )
        else:
            self.global_cost_matrix, min_index, _ = weighted_soft_oltw_loop(
                global_cost_matrix=self.global_cost_matrix,
                window_cost=window_cost,
                window_start=window_start,
                window_end=window_end,
                input_index=self.input_index,
                min_costs=np.inf,
                min_index=min_index,
                gamma=self.gamma,
                w_horizontal=horizontal_weight,
            )
        if self.input_index > 0:
            self._current_frame = min(
                max(self._current_frame, min_index),
                self._current_frame + self.step_size,
            )
        self.current_index = self._frame_to_score_idx(self._current_frame)
        self._pos_history.append(self._current_frame)

        if self.kalman is not None:
            self.kalman.update(float(self._current_frame))

        self.input_index += 1
