from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import partitura as pt
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

    def transition_distribution(
        self, node_id: str
    ) -> tuple[tuple[ScoreEdge, float], ...]:
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


class ScoreDirectedGraph(ScoreGraph):
    def __init__(
        self,
        nodes: Iterable[ScoreNode] = (),
        measure_nodes: Iterable[str] = (),
    ) -> None:
        super().__init__(nodes)
        self.measure_nodes = tuple(measure_nodes)

    @property
    def graph(self) -> ScoreDirectedGraph:
        return self


class ScoreDirectedGraphBuilder:
    def __init__(self, linear_prior: float = 0.5, jump_prior: float = 0.5) -> None:
        self.linear_prior = linear_prior
        self.jump_prior = jump_prior

    def build(
        self, score: str | Path | pt.score.Score | pt.score.Part
    ) -> ScoreDirectedGraph:
        if isinstance(score, pt.score.Part):
            part = score
        elif isinstance(score, pt.score.Score):
            part = pt.score.merge_parts(score.parts)
        else:
            part = pt.load_score_as_part(str(score))

        measures = list(part.measures)
        if not measures:
            raise ValueError("Score contains no measures")

        graph = ScoreDirectedGraph()
        measure_nodes: list[str] = []

        for idx, m in enumerate(measures):
            start_b = float(part.beat_map(m.start.t))
            end_b = float(part.beat_map(m.end.t))
            node_id = f"measure:{idx}:{m.number}"
            meta = {
                "measure_index": idx,
                "measure_number": m.number,
                "duration_beats": end_b - start_b,
            }
            graph.add_node(ScoreNode(node_id, start_b, idx, metadata=meta))
            measure_nodes.append(node_id)

        graph.measure_nodes = tuple(measure_nodes)

        for src, tgt in zip(measure_nodes, measure_nodes[1:]):
            graph.add_edge(
                ScoreEdge(src, tgt, EdgeKind.LINEAR, prior=self.linear_prior)
            )

        starts = [m.start.t for m in measures]
        ends = [m.end.t for m in measures]

        def at_time(t: float) -> int:
            for i, (s, e) in enumerate(zip(starts, ends)):
                if s <= t < e:
                    return i
            return len(measures) - 1

        def ending_at(t: float) -> int:
            for i, (s, e) in enumerate(zip(starts, ends)):
                if s < t <= e:
                    return i
            return len(measures) - 1

        for r in part.iter_all(pt.score.Repeat):
            if r.end is not None:
                src = ending_at(r.end.t)
                tgt = at_time(r.start.t) if r.start is not None else 0
                graph.add_edge(
                    ScoreEdge(
                        measure_nodes[src],
                        measure_nodes[tgt],
                        EdgeKind.REPEAT,
                        prior=self.jump_prior,
                    )
                )

        endings = list(part.iter_all(pt.score.Ending))
        if len(endings) >= 2:
            src = at_time(endings[0].start.t) - 1
            tgt = at_time(endings[1].start.t)
            if src >= 0 and tgt < len(measure_nodes):
                graph.add_edge(
                    ScoreEdge(
                        measure_nodes[src],
                        measure_nodes[tgt],
                        EdgeKind.VOLTA,
                        prior=self.jump_prior,
                    )
                )

        for dc in part.iter_all(pt.score.DaCapo):
            src = ending_at(dc.start.t)
            graph.add_edge(
                ScoreEdge(
                    measure_nodes[src],
                    measure_nodes[0],
                    EdgeKind.DA_CAPO,
                    prior=self.jump_prior,
                )
            )

        segnos = list(part.iter_all(pt.score.Segno))
        if segnos:
            for ds in part.iter_all(pt.score.DalSegno):
                src = ending_at(ds.start.t)
                tgt = at_time(segnos[0].start.t)
                graph.add_edge(
                    ScoreEdge(
                        measure_nodes[src],
                        measure_nodes[tgt],
                        EdgeKind.DAL_SEGNO,
                        prior=self.jump_prior,
                    )
                )

        codas = list(part.iter_all(pt.score.Coda))
        if codas:
            for tc in part.iter_all(pt.score.ToCoda):
                src = ending_at(tc.start.t)
                tgt = at_time(codas[0].start.t)
                graph.add_edge(
                    ScoreEdge(
                        measure_nodes[src],
                        measure_nodes[tgt],
                        EdgeKind.CODA,
                        prior=self.jump_prior,
                    )
                )

        last_node = measure_nodes[-1]
        if graph.outgoing(last_node):
            graph.add_edge(ScoreEdge(last_node, last_node, EdgeKind.STAY, self.linear_prior))

        return graph

    parse = build


def rebase_measure_beats(
    graph: ScoreDirectedGraph,
    measure_start_beats: list[float] | tuple[float, ...],
    *,
    final_beat: float | None = None,
) -> ScoreDirectedGraph:
    starts = tuple(float(b) for b in measure_start_beats)
    new_graph = ScoreDirectedGraph(measure_nodes=graph.measure_nodes)
    for i, nid in enumerate(graph.measure_nodes):
        old = graph.nodes[nid]
        dur = (
            (starts[i + 1] - starts[i])
            if i + 1 < len(starts)
            else (
                (final_beat - starts[i])
                if final_beat is not None
                else float(old.metadata.get("duration_beats", 0.0))
            )
        )
        meta = dict(old.metadata, duration_beats=dur)
        new_graph.add_node(
            ScoreNode(old.node_id, starts[i], old.measure, old.event_ids, metadata=meta)
        )
    for nid in graph.measure_nodes:
        for edge in graph.outgoing(nid):
            new_graph.add_edge(edge)
    return new_graph
