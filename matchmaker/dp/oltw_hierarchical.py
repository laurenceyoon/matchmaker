from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.oltw_soft import DEFAULT_GAMMA, DEFAULT_OBS_VAR, DEFAULT_W_HORIZONTAL, SoftOnlineTimeWarping
from matchmaker.features.audio import FRAME_RATE
from matchmaker.graph.score_graph import ScoreGraph
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats


def logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values)) if m > -math.inf else -math.inf


@dataclass
class GraphHypothesis:
    node_id: str
    log_weight: float
    follower: SoftOnlineTimeWarping
    jump_counts: Dict[str, int] = field(default_factory=dict)
    last_score_beat: float = 0.0


class HierarchicalSoftOnlineTimeWarping(OnlineAlignment):
    """Hierarchical score follower with Measure Graph Beam Search (K=2) and 3-Regime IMM."""

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        score_graph: ScoreGraph,
        beam_size: int = 2,
        gamma: float = DEFAULT_GAMMA,
        w_horizontal: float = DEFAULT_W_HORIZONTAL,
        obs_var: float = DEFAULT_OBS_VAR,
        window_size: int = 10,
        step_size: int = 3,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: Optional[NDArray] = None,
        score_part: Any = None,
        queue: Optional[RECVQueue] = None,
        initial_node: Optional[str] = None,
        boundary_margin_beats: float = 1.0,
        max_score_step: int = 1,
        max_time_step: int = 1,
        gamma_repeat_factor: float = 1.0,
        gamma_repeat_window_beats: float = 1.5,
        tempo: Optional[float] = None,
        **kwargs,
    ) -> None:
        if ref_frame_to_beat is None and score_positions is not None:
            ref_frame_to_beat = score_positions

        super().__init__(reference_features=reference_features, score_positions=score_positions, queue=queue)
        self.score_graph = score_graph
        self.beam_size = max(1, beam_size)
        self.gamma = gamma
        self.w_horizontal = w_horizontal
        self.obs_var = obs_var
        self.window_size = window_size
        self.step_size = step_size
        self.frame_rate = frame_rate
        self.ref_frame_to_beat = np.asarray(ref_frame_to_beat, dtype=float)
        self.score_part = score_part
        self.tempo = tempo
        self.boundary_margin_beats = boundary_margin_beats
        self.max_score_step = max_score_step
        self.max_time_step = max_time_step
        self.gamma_repeat_factor = gamma_repeat_factor
        self.gamma_repeat_window_beats = gamma_repeat_window_beats
        self.kwargs = kwargs

        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

        ordered_nodes = sorted(score_graph.nodes.values(), key=lambda n: n.score_beat)
        self.ordered_nodes = ordered_nodes
        self.node_end_beats: Dict[str, float] = {}
        for idx, node in enumerate(ordered_nodes):
            dur = float(node.metadata.get("duration_beats", 0.0))
            if dur > 0:
                self.node_end_beats[node.node_id] = node.score_beat + dur
            elif idx + 1 < len(ordered_nodes):
                self.node_end_beats[node.node_id] = ordered_nodes[idx + 1].score_beat
            else:
                self.node_end_beats[node.node_id] = float(self.ref_frame_to_beat[-1])

        repeat_boundaries = []
        for src_id, edges in self.score_graph.outgoing_edges.items():
            for edge in edges:
                if edge.kind.value != "linear":
                    if src_id in self.node_end_beats:
                        repeat_boundaries.append(self.node_end_beats[src_id])
                    if edge.target in self.score_graph.nodes:
                        repeat_boundaries.append(self.score_graph.nodes[edge.target].score_beat)
        self.repeat_boundaries = sorted(list(set(repeat_boundaries)))

        self.node_start_frames: Dict[str, int] = {
            node.node_id: min(
                int(np.searchsorted(self.ref_frame_to_beat, node.score_beat, side="left")),
                len(self.ref_frame_to_beat) - 1,
            )
            for node in ordered_nodes
        }

        init_id = initial_node or ordered_nodes[0].node_id
        root_follower = self._make_local_follower(self.node_start_frames[init_id])
        self.hypotheses: List[GraphHypothesis] = [
            GraphHypothesis(
                node_id=init_id,
                log_weight=0.0,
                follower=root_follower,
                last_score_beat=score_graph.nodes[init_id].score_beat,
            )
        ]

        self.current_position = score_graph.nodes[init_id].score_beat
        self.current_index = self.node_start_frames[init_id]
        self.input_index = 0

    def _make_local_follower(self, start_frame: int) -> SoftOnlineTimeWarping:
        local_kwargs = dict(self.kwargs)
        for k in (
            "max_score_step",
            "max_time_step",
            "repeat_boundaries",
            "gamma_repeat_factor",
            "gamma_repeat_window_beats",
            "window_size",
            "step_size",
            "gamma",
            "w_horizontal",
            "obs_var",
            "use_imm",
            "score_part",
            "frame_rate",
            "ref_frame_to_beat",
            "queue",
            "tempo",
        ):
            local_kwargs.pop(k, None)

        follower = SoftOnlineTimeWarping(
            reference_features=self.reference_features,
            score_positions=self.score_positions,
            window_size=self.window_size,
            step_size=self.step_size,
            gamma=self.gamma,
            w_horizontal=self.w_horizontal,
            obs_var=self.obs_var,
            use_imm=True,
            score_part=self.score_part,
            frame_rate=self.frame_rate,
            ref_frame_to_beat=self.ref_frame_to_beat,
            queue=None,
            max_score_step=self.max_score_step,
            max_time_step=self.max_time_step,
            repeat_boundaries=self.repeat_boundaries,
            gamma_repeat_factor=self.gamma_repeat_factor,
            gamma_repeat_window_beats=self.gamma_repeat_window_beats,
            tempo=self.tempo,
            **local_kwargs,
        )
        follower._current_frame = start_frame
        follower.current_index = follower._frame_to_score_idx(start_frame)
        follower._music_started = True
        return follower

    def _fork_hypothesis(self, parent: GraphHypothesis, target_node_id: str, prior_prob: float) -> GraphHypothesis:
        target_frame = self.node_start_frames[target_node_id]
        child_follower = self._make_local_follower(target_frame)

        if hasattr(parent.follower, "kalman") and hasattr(child_follower, "kalman"):
            p_imm, c_imm = parent.follower.kalman, child_follower.kalman
            if hasattr(p_imm, "states") and hasattr(c_imm, "states"):
                c_imm.states[:, 1] = p_imm.states[:, 1]
                c_imm.mu = p_imm.mu.copy()

        new_jumps = dict(parent.jump_counts)
        jump_key = f"{parent.node_id}->{target_node_id}"
        new_jumps[jump_key] = new_jumps.get(jump_key, 0) + 1

        target_node = self.score_graph.nodes[target_node_id]
        return GraphHypothesis(
            node_id=target_node_id,
            log_weight=parent.log_weight + math.log(max(prior_prob, 1e-6)),
            follower=child_follower,
            jump_counts=new_jumps,
            last_score_beat=target_node.score_beat,
        )

    def step(self, features: NDArray[np.float32]) -> None:
        new_hypotheses: List[GraphHypothesis] = []

        for hyp in self.hypotheses:
            hyp.follower.step(features)
            current_beat = hyp.follower.get_current_position()
            hyp.last_score_beat = current_beat

            diag = hyp.follower.last_diagnostics
            if diag is not None and diag.selected_local_cost is not None:
                hyp.log_weight += -float(diag.selected_local_cost) / max(self.gamma * 2.0, 0.05)

            new_hypotheses.append(hyp)

            node_end = self.node_end_beats.get(hyp.node_id, float("inf"))
            if current_beat >= node_end - self.boundary_margin_beats:
                for edge, trans_prob in self.score_graph.transition_distribution(hyp.node_id):
                    if edge.target != hyp.node_id:
                        times = int(edge.metadata.get("times", 2))
                        jump_key = f"{hyp.node_id}->{edge.target}"
                        if hyp.jump_counts.get(jump_key, 0) < times:
                            new_hypotheses.append(self._fork_hypothesis(hyp, edge.target, trans_prob))

        if len(new_hypotheses) > self.beam_size:
            new_hypotheses.sort(key=lambda h: h.log_weight, reverse=True)
            new_hypotheses = new_hypotheses[: self.beam_size]

        total_log_weight = logsumexp([h.log_weight for h in new_hypotheses])
        for h in new_hypotheses:
            h.log_weight -= total_log_weight

        self.hypotheses = new_hypotheses

        best_hyp = max(self.hypotheses, key=lambda h: h.log_weight)
        self.current_position = best_hyp.follower.get_current_position()
        self.current_index = best_hyp.follower.current_index
        self.input_index += 1

    def get_current_position(self) -> float:
        return self.current_position

    def reset(self) -> None:
        self.input_index = 0
        self._alignment_path = []
        init_id = self.ordered_nodes[0].node_id
        root_follower = self._make_local_follower(self.node_start_frames[init_id])
        self.hypotheses = [
            GraphHypothesis(
                node_id=init_id,
                log_weight=0.0,
                follower=root_follower,
                last_score_beat=self.score_graph.nodes[init_id].score_beat,
            )
        ]
        self.current_position = self.score_graph.nodes[init_id].score_beat
        self.current_index = self.node_start_frames[init_id]

    def __call__(self, observation: Any, perf_time: float) -> float:
        t0 = time.time()
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(time.time() - t0, self.latency_stats, self.input_index)
        return beat
