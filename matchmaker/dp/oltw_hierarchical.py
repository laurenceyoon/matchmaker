from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.oltw_soft import (
    DEFAULT_GAMMA,
    DEFAULT_W_HORIZONTAL,
    SoftOnlineTimeWarping,
)
from matchmaker.prob.imm import DEFAULT_OBS_VAR
from matchmaker.features.audio import FRAME_RATE
from matchmaker.graph.score_graph import EdgeKind, ScoreGraph
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
    route: tuple[str, ...] = ()
    last_score_beat: float = 0.0
    branched: bool = False

    @property
    def jump_counts(self) -> dict[str, int]:
        return dict(Counter(self.route))


class HierarchicalSoftOnlineTimeWarping(OnlineAlignment):

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
        tempo: float = 120.0,
        **kwargs,
    ) -> None:
        if ref_frame_to_beat is None and score_positions is not None:
            ref_frame_to_beat = score_positions

        super().__init__(reference_features=reference_features, score_positions=score_positions, queue=queue)
        self.score_graph = score_graph
        self.beam_size = max(1, beam_size)
        self.ref_frame_to_beat = np.asarray(ref_frame_to_beat, dtype=float)
        self.boundary_margin_beats = boundary_margin_beats
        self.local_options = dict(
            kwargs, window_size=window_size, step_size=step_size,
            gamma=gamma, w_horizontal=w_horizontal, obs_var=obs_var,
            use_imm=True, score_part=score_part, frame_rate=frame_rate,
            ref_frame_to_beat=self.ref_frame_to_beat, tempo=tempo,
        )

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
        follower = SoftOnlineTimeWarping(
            reference_features=self.reference_features,
            score_positions=self.score_positions,
            **self.local_options,
        )
        follower.reset(start_frame)
        return follower

    def _fork_hypothesis(self, parent: GraphHypothesis, target_node_id: str, relative_weight: float) -> GraphHypothesis:
        target_frame = self.node_start_frames[target_node_id]
        child_follower = self._make_local_follower(target_frame)

        child_follower.path.restart_from(parent.follower.path, target_frame)
        child_follower._music_started = parent.follower._music_started
        child_follower.input_index = parent.follower.input_index

        jump_key = f"{parent.node_id}->{target_node_id}"

        target_node = self.score_graph.nodes[target_node_id]
        return GraphHypothesis(
            node_id=target_node_id,
            log_weight=parent.log_weight + math.log(relative_weight),
            follower=child_follower,
            route=(*parent.route, jump_key),
            last_score_beat=target_node.score_beat,
        )

    def _advance_node(self, hypothesis: GraphHypothesis) -> None:
        node_id = hypothesis.node_id
        current_start = self.score_graph.nodes[node_id].score_beat
        for node in self.ordered_nodes:
            if current_start < node.score_beat <= hypothesis.last_score_beat:
                node_id = node.node_id
        if node_id != hypothesis.node_id:
            hypothesis.node_id = node_id
            hypothesis.branched = False

    def _expand(self, hypothesis: GraphHypothesis) -> list[GraphHypothesis]:
        node_end = self.node_end_beats[hypothesis.node_id]
        if hypothesis.last_score_beat < node_end - self.boundary_margin_beats:
            return [hypothesis]
        transitions = self.score_graph.transition_distribution(hypothesis.node_id)
        jumps = [(edge, probability) for edge, probability in transitions
                 if edge.kind not in (EdgeKind.LINEAR, EdgeKind.STAY)]
        if not jumps:
            return [hypothesis]
        continuation = sum(probability for edge, probability in transitions
                           if edge.kind in (EdgeKind.LINEAR, EdgeKind.STAY))
        if not continuation:
            return [self._fork_hypothesis(hypothesis, edge.target, probability)
                    for edge, probability in jumps]
        if not hypothesis.branched:
            hypothesis.log_weight += math.log(continuation)
            hypothesis.branched = True
        children = [self._fork_hypothesis(hypothesis, edge.target, probability / continuation)
                    for edge, probability in jumps]
        return [hypothesis, *children]

    def step(self, features: NDArray[np.float32]) -> None:
        audible = not self.local_options.get("use_silence", False) or bool(features.any())
        candidates = []
        for hypothesis in self.hypotheses:
            candidates.extend(self._expand(hypothesis) if audible else [hypothesis])
        for hypothesis in candidates:
            follower = hypothesis.follower
            follower.path.costs[:self.node_start_frames[hypothesis.node_id]] = np.inf
            follower.step(features)
            hypothesis.last_score_beat = follower.get_current_position()
            if audible and follower._music_started:
                frame = follower.path.index
                distance = follower.vdist(
                    self.reference_features[frame:frame + 1],
                    features.squeeze(), follower.distance_func,
                )[0]
                hypothesis.log_weight -= float(distance) / (follower.gamma or DEFAULT_GAMMA)
            self._advance_node(hypothesis)

        routes = {}
        for hypothesis in candidates:
            key = hypothesis.route
            if key not in routes or hypothesis.log_weight > routes[key].log_weight:
                routes[key] = hypothesis
        candidates = list(routes.values())
        candidates.sort(key=lambda h: h.log_weight, reverse=True)
        self.hypotheses = candidates[:self.beam_size]
        total = logsumexp([h.log_weight for h in self.hypotheses])
        for hypothesis in self.hypotheses:
            hypothesis.log_weight -= total
        self.hypotheses = [h for h in self.hypotheses
                           if h.log_weight >= math.log(np.finfo(float).eps)]
        best = self.hypotheses[0]
        self.current_position = best.follower.get_current_position()
        self.current_index = best.follower.current_index
        self.input_index += 1

    def get_current_position(self) -> float:
        return self.current_position

    def is_still_following(self) -> bool:
        return True

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
