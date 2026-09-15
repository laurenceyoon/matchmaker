from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class EdgeKind(str, Enum):
    LINEAR = "linear"
    STAY = "stay"
    REPEAT = "repeat"
    VOLTA = "volta"
    CODA = "coda"
    DAL_SEGNO = "dal-segno"
    DA_CAPO = "da-capo"
    SKIP = "skip"
    MANUAL = "manual"


@dataclass(frozen=True)
class ScoreNode:
    node_id: str
    score_beat: float
    measure: int | None = None
    event_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreEdge:
    source: str
    target: str
    kind: EdgeKind = EdgeKind.LINEAR
    prior: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ScoreGraph:
    def __init__(self, nodes: Iterable[ScoreNode] = ()) -> None:
        self.node_map: dict[str, ScoreNode] = {}
        self.outgoing_edges: dict[str, list[ScoreEdge]] = {}
        for node in nodes:
            self.add_node(node)

    @property
    def nodes(self) -> Mapping[str, ScoreNode]:
        return self.node_map

    def add_node(self, node: ScoreNode) -> None:
        self.node_map[node.node_id] = node
        self.outgoing_edges[node.node_id] = []

    def add_edge(self, edge: ScoreEdge) -> None:
        self.outgoing_edges[edge.source].append(edge)

    def outgoing(self, node_id: str) -> tuple[ScoreEdge, ...]:
        return tuple(self.outgoing_edges.get(node_id, ()))

    def transition_distribution(self, node_id: str) -> tuple[tuple[ScoreEdge, float], ...]:
        edges = self.outgoing(node_id)
        if not edges:
            return ((ScoreEdge(node_id, node_id, EdgeKind.STAY, 1.0), 1.0),)
        total = sum(e.prior for e in edges)
        return tuple((e, e.prior / total) for e in edges)

    @classmethod
    def linear(cls, beats: Iterable[float]) -> ScoreGraph:
        nodes = [ScoreNode(f"beat:{i}", b) for i, b in enumerate(beats)]
        graph = cls(nodes)
        for left, right in zip(nodes, nodes[1:]):
            graph.add_edge(ScoreEdge(left.node_id, right.node_id))
        return graph
