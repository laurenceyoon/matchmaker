import numpy as np
import partitura as pt
import pytest

from matchmaker.dp.oltw_imm import IMMPathFilter
from matchmaker.dp.oltw_hierarchical import HierarchicalSoftOnlineTimeWarping
from matchmaker.dp.oltw_soft import SoftOnlineTimeWarping
from matchmaker.graph.score_graph import EdgeKind, ScoreEdge, ScoreGraph, ScoreNode
from matchmaker.prob.imm import IMMMotionModels, score_activity


def test_silence_retains_position_tempo_and_path_history():
    model = IMMMotionModels(retain_tempo=True)
    path = IMMPathFilter(model, np.zeros(120), 3)
    path.reset(10)
    costs = path.costs.copy()
    for _ in range(150):
        assert path.observe_silence(np.empty((0, 2)))
        np.testing.assert_allclose(path.position, 10)
        np.testing.assert_allclose(path.state[1], model.frame_rate)
        np.testing.assert_array_equal(path.probabilities[path.index], [0, 0, 1])
    np.testing.assert_array_equal(path.costs, costs)
    priors, states, _ = path.predict(np.array([path.index]))
    assert priors[0, 2] == 0
    np.testing.assert_allclose(states[0, :2, 0], 11)


def test_no_zv_ablation_has_no_silence_update():
    path = IMMPathFilter(IMMMotionModels(modes=("cv", "ca")), np.zeros(12), 3)
    original = path.states.copy()
    assert not path.observe_silence(np.empty((0, 2)))
    np.testing.assert_array_equal(path.states, original)


def test_score_activity_includes_overlapping_notes_and_rests():
    part = pt.score.Part("P1", quarter_duration=1)
    part.add(pt.score.Note("C", 4), 0, 2)
    part.add(pt.score.Note("E", 4), 1, 3)
    part.add(pt.score.Note("G", 4), 4, 5)
    np.testing.assert_array_equal(
        score_activity(part, np.arange(6), 6), [True, True, True, False, True, False]
    )


def test_middle_silence_preserves_the_path_clock():
    reference = np.zeros((120, 12), dtype=np.float32)
    reference[:, 0] = 1
    follower = SoftOnlineTimeWarping(reference, use_silence=True)
    for frame in reference[:10]:
        follower.step(frame)
    count = follower.input_index
    position = follower.get_current_position()
    costs = follower.path.costs.copy()
    for _ in range(150):
        follower.step(np.zeros(12, dtype=np.float32))
    assert follower.input_index == count
    assert follower.get_current_position() == pytest.approx(position)
    np.testing.assert_array_equal(follower.path.costs, costs)
    follower.step(reference[10])
    assert follower.input_index == count + 1
    assert follower.get_current_position() > position


@pytest.mark.parametrize("frame", [27, 28, 29])
def test_position_uncertainty_allows_entry_into_a_rest(frame):
    part = pt.score.Part("P1", quarter_duration=1)
    part.add(pt.score.Note("C", 4), 0, 2)
    part.add(pt.score.Note("G", 4), 4, 6)
    reference = np.zeros((90, 12), dtype=np.float32)
    reference[:30, 0] = 1
    reference[60:, 7] = 1
    follower = SoftOnlineTimeWarping(
        reference, np.arange(90) / 15, score_part=part, use_silence=True,
    )
    follower.reset(frame)
    follower._music_started = True
    follower.input_index = 1
    for _ in range(30):
        follower.step(np.zeros(12, dtype=np.float32))
    assert 3 < follower.get_current_position() < 4
    for feature in reference[60:]:
        follower.step(feature)
    assert follower.get_current_position() > 5.5


def test_silence_preserves_repeat_hypotheses_without_new_evidence():
    graph = ScoreGraph([ScoreNode("start", 0.0), ScoreNode("next", 2.0)])
    graph.add_edge(ScoreEdge("start", "next", EdgeKind.LINEAR, 0.5))
    graph.add_edge(ScoreEdge("start", "start", EdgeKind.REPEAT, 0.5))
    reference = np.zeros((120, 12), dtype=np.float32)
    reference[:, 0] = 1
    follower = HierarchicalSoftOnlineTimeWarping(
        reference, np.arange(120) / 15, graph, use_silence=True,
    )
    for feature in reference[:20]:
        follower.step(feature)
    assert len(follower.hypotheses) == 2
    hypotheses = [(h.route, h.log_weight) for h in follower.hypotheses]
    position = follower.get_current_position()
    for _ in range(150):
        follower.step(np.zeros(12, dtype=np.float32))
    assert [(h.route, h.log_weight) for h in follower.hypotheses] == hypotheses
    assert follower.get_current_position() == pytest.approx(position)
