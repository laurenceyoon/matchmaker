from pathlib import Path
import numpy as np
import partitura as pt
import pytest

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp.oltw_hierarchical import HierarchicalSoftOnlineTimeWarping
from matchmaker.graph.directed_graph import ScoreDirectedGraphBuilder
from matchmaker.graph.score_graph import EdgeKind, ScoreGraph, ScoreNode


def test_score_graph_basic():
    graph = ScoreGraph()
    graph.add_node(ScoreNode("m1", 0.0))
    graph.add_node(ScoreNode("m2", 4.0))
    assert "m1" in graph.nodes
    assert len(graph.outgoing("m1")) == 0

    transitions = graph.transition_distribution("m1")
    assert len(transitions) == 1
    assert transitions[0][0].kind == EdgeKind.STAY


def test_score_directed_graph_builder_simple_mozart():
    piece = EXAMPLE_PIECES["simple_mozart"]
    builder = ScoreDirectedGraphBuilder()
    graph = builder.build(piece["score"])
    assert len(graph.measure_nodes) > 0
    assert "measure:0:1" in graph.nodes
    assert "measure:0:1" in graph.graph.nodes


def test_score_directed_graph_repeat_and_volta():
    part = pt.score.Part("P1")
    for i in range(3):
        part.add(pt.score.Measure(number=i + 1), i * 8, (i + 1) * 8)
    part.add(pt.score.Repeat(), 0, 16)
    part.add(pt.score.Ending(1), 8, 16)
    part.add(pt.score.Ending(2), 16, 24)

    graph = ScoreDirectedGraphBuilder().build(part)
    assert len(graph.measure_nodes) == 3

    m0_targets = [e.target for e in graph.outgoing(graph.measure_nodes[0])]
    assert graph.measure_nodes[1] in m0_targets
    assert graph.measure_nodes[2] in m0_targets

    m1_kinds = [e.kind for e in graph.outgoing(graph.measure_nodes[1])]
    assert EdgeKind.REPEAT in m1_kinds


def test_matchmaker_hierarchical_soft_oltw():
    piece = EXAMPLE_PIECES["simple_mozart"]
    mm = Matchmaker(
        score_file=piece["score"],
        performance_file=piece["audio"],
        method="hierarchical_soft_oltw",
        input_type="audio",
    )
    assert isinstance(mm.score_follower, HierarchicalSoftOnlineTimeWarping)

    # Run full alignment
    for pos in mm.run():
        assert pos >= 0.0

    wp = mm.score_follower.alignment_path
    assert wp.shape[0] == 2
    assert wp.shape[1] > 0
    assert wp[1, -1] > 0.0
