import numpy as np
import partitura as pt
from scipy.linalg import expm

from matchmaker.prob.imm import IMMMotionModels, score_activity


def test_continuous_time_model_is_invariant_to_frame_subdivision():
    full = IMMMotionModels(frame_rate=30)
    half = IMMMotionModels(frame_rate=60)
    np.testing.assert_allclose(full.M_pause, half.M_pause @ half.M_pause, atol=1e-12)
    np.testing.assert_allclose(full.M_play, half.M_play @ half.M_play, atol=1e-12)


def test_mode_transition_is_exact_continuous_time_chain():
    imm = IMMMotionModels()
    generator = np.full((3, 3), 1 / (2 * imm.beat_seconds))
    np.fill_diagonal(generator, -1 / imm.beat_seconds)
    np.testing.assert_allclose(imm.M_pause, expm(generator * imm.dt), atol=1e-12)
    np.testing.assert_allclose(imm.M_pause.sum(axis=1), 1)
    assert np.all(imm.M_play[:, 2] == 0)


def test_removed_modes_never_receive_probability():
    for removed in ("cv", "ca", "zv"):
        modes = tuple(mode for mode in ("cv", "ca", "zv") if mode != removed)
        model = IMMMotionModels(modes=modes)
        disabled = ("cv", "ca", "zv").index(removed)
        for matrix in (model.M_play, model.M_pause):
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
            assert np.all(matrix[:, disabled] == 0.0)
        assert model.initial_probabilities[disabled] == 0.0


def test_pauses_are_notated_fermatas_and_rests():
    part = pt.score.Part("P1", quarter_duration=1)
    part.add(pt.score.Note("C", 4), 0, 1)
    part.add(pt.score.Note("E", 4), 2, 3)
    assert IMMMotionModels(score_part=part).pause_ranges == [(1.0, 2.0)]


def test_score_activity_includes_overlapping_notes_and_rests():
    part = pt.score.Part("P1", quarter_duration=1)
    part.add(pt.score.Note("C", 4), 0, 2)
    part.add(pt.score.Note("E", 4), 1, 3)
    part.add(pt.score.Note("G", 4), 4, 5)
    np.testing.assert_array_equal(
        score_activity(part, np.arange(6), 6), [True, True, True, False, True, False]
    )
