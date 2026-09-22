from pathlib import Path
import numpy as np
import partitura as pt
import pytest

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp.oltw_hierarchical import HierarchicalSoftOnlineTimeWarping
from matchmaker.graph.score_graph import ScoreDirectedGraphBuilder
from matchmaker.graph.score_graph import EdgeKind, ScoreEdge, ScoreGraph, ScoreNode
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END


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

    for node_id in graph.measure_nodes[:2]:
        transitions = graph.transition_distribution(node_id)
        assert len(transitions) == 2
        assert [probability for _, probability in transitions] == [0.5, 0.5]


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


def test_startup_waits_for_music_and_forks_preserve_started_state():
    graph = ScoreGraph()
    graph.add_node(ScoreNode("m1", 0.0))
    graph.add_node(ScoreNode("m2", 4.0))
    reference = np.zeros((240, 12), dtype=np.float32)
    reference[:, 0] = 1.0
    follower = HierarchicalSoftOnlineTimeWarping(
        reference, np.arange(240) / 15, graph,
    )

    for _ in range(2):
        for _ in range(300):
            follower.step(np.zeros(12, dtype=np.float32))
        parent = follower.hypotheses[0]
        assert follower.get_current_position() == 0.0
        assert parent.follower.input_index == 0
        assert not parent.follower._music_started

        follower.step(reference[0])
        assert parent.follower._music_started
        child = follower._fork_hypothesis(parent, "m2", 1.0)
        assert child.follower._music_started
        follower.reset()


@pytest.mark.parametrize("repeat", [False, True])
def test_score_end_keeps_listening_until_stream_end(repeat):
    reference = np.zeros((120, 12), dtype=np.float32)
    reference[:, 0] = 1.0
    positions = np.arange(120) / 15
    graph = ScoreGraph([ScoreNode("start", 0.0), ScoreNode("end", positions[-1])])
    if repeat:
        graph.add_edge(ScoreEdge("end", "start", EdgeKind.REPEAT))
    queue = RECVQueue()
    follower = HierarchicalSoftOnlineTimeWarping(
        reference, positions, graph, queue=queue, initial_node="end",
    )
    assert follower.current_index == len(positions) - 1
    times = np.arange(20) / follower.local_options["frame_rate"]
    for time in times:
        queue.put((reference[-1], time))
    queue.put(STREAM_END)
    queue.put((reference[-1], times[-1] + 1))

    beats = list(follower.run(verbose=False))

    assert len(beats) == len(times)
    assert np.isfinite(beats).all()
    np.testing.assert_array_equal(follower.alignment_path[0], times)
    assert queue.qsize() == 1


@pytest.fixture
def repeated_sections():
    pitches = np.repeat([0, 2, 4, 5, 7, 9, 11, 7, 8, 10, 6, 3], 5)
    reference = np.eye(12, dtype=np.float32)[pitches]
    positions = np.arange(len(reference)) / 15
    graph = ScoreGraph([
        ScoreNode(str(i), i, metadata={"duration_beats": 1}) for i in range(4)
    ])
    for i in range(3):
        graph.add_edge(ScoreEdge(str(i), str(i + 1), prior=0.5))
    graph.add_edge(ScoreEdge("1", "0", EdgeKind.REPEAT, 0.5))
    graph.add_edge(ScoreEdge("3", "2", EdgeKind.REPEAT, 0.5))
    graph.add_edge(ScoreEdge("3", "3", EdgeKind.STAY, 0.5))
    return HierarchicalSoftOnlineTimeWarping(reference, positions, graph)


@pytest.mark.parametrize("repeat_first", [False, True])
@pytest.mark.parametrize("trailing_rest", [0, 6])
def test_acoustic_evidence_selects_repeat_or_continue(
    repeated_sections, repeat_first, trailing_rest
):
    follower = repeated_sections
    follower.node_end_beats["3"] += trailing_rest
    first = np.arange(30)
    second = np.arange(30, 60)
    route = np.concatenate([first] * (1 + repeat_first) + [second, second])
    path, sizes = [], []
    for time, frame in enumerate(route):
        path.append(follower(follower.reference_features[frame], time / 30))
        sizes.append(len(follower.hypotheses))

    truth = follower.score_positions[route]
    assert np.max(np.abs(path - truth)) < 0.5
    np.testing.assert_array_equal(
        np.flatnonzero(np.diff(path) < -1), np.flatnonzero(np.diff(truth) < -1)
    )
    assert max(sizes) <= 2
    assert min(sizes[35:45]) == 1


def test_branch_priors_are_applied_once_per_visit(repeated_sections):
    follower = repeated_sections
    parent = follower.hypotheses[0]
    parent.node_id = "1"
    parent.last_score_beat = 1.5

    for _ in range(10):
        candidates = follower._expand(parent)
        np.testing.assert_allclose([np.exp(h.log_weight) for h in candidates], [0.5, 0.5])
        assert candidates[1].route == ("1->0",)
        assert parent.route == ()


def test_filtered_position_cannot_close_repeat_before_alignment(repeated_sections):
    follower = repeated_sections
    parent = follower.hypotheses[0]
    parent.node_id = "1"
    parent.follower.reset(29)
    parent.follower.path.state[0] = 31
    parent.last_score_beat = parent.follower.get_current_position()

    follower._advance_node(parent)

    assert parent.node_id == "1"
    assert len(follower._expand(parent)) == 2

    parent.follower.path.index = 30
    follower._advance_node(parent)

    assert parent.node_id == "1"

    parent.follower._current_frame = 30
    follower._advance_node(parent)

    assert parent.node_id == "2"


def test_ambiguous_observation_preserves_both_routes(repeated_sections):
    follower = repeated_sections
    follower.reference_features[:] = np.eye(12, dtype=np.float32)[0]
    parent = follower.hypotheses[0]
    parent.follower.reset(22)
    parent.node_id = "1"
    parent.last_score_beat = follower.score_positions[22]

    follower.step(follower.reference_features[0])

    assert len(follower.hypotheses) == 2
    np.testing.assert_allclose([np.exp(h.log_weight) for h in follower.hypotheses], [0.5, 0.5])
