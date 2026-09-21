import numpy as np
from scipy.linalg import expm
from matchmaker.prob.imm import IMMMotionModels


def test_continuous_time_model_is_invariant_to_frame_subdivision():
    full = IMMMotionModels(frame_rate=30)
    half = IMMMotionModels(frame_rate=60)
    np.testing.assert_allclose(full.F, half.F @ half.F)
    np.testing.assert_allclose(
        full.Q, half.F @ half.Q @ half.F.transpose(0, 2, 1) + half.Q, atol=1e-12
    )
    np.testing.assert_allclose(full.M_pause, half.M_pause @ half.M_pause, atol=1e-12)
    np.testing.assert_allclose(full.M_play, half.M_play @ half.M_play, atol=1e-12)


def test_mode_transition_is_exact_continuous_time_chain():
    imm = IMMMotionModels()
    generator = np.full((3, 3), 1 / (2 * imm.beat_seconds))
    np.fill_diagonal(generator, -1 / imm.beat_seconds)
    np.testing.assert_allclose(imm.M_pause, expm(generator * imm.dt), atol=1e-12)
    np.testing.assert_allclose(imm.M_pause.sum(axis=1), 1)
    assert np.all(imm.M_play[:, 2] == 0)


def test_score_repeated_chord_duration_defines_uncertainty():
    import partitura as pt
    from matchmaker.prob.imm import score_position_variance

    part = pt.score.Part("P1")
    part.set_quarter_duration(0, 1)
    part.add(pt.score.TimeSignature(4, 4), 0)
    part.add(pt.score.Note("C", 4), 0, 4)
    part.add(pt.score.Note("C", 4), 4, 8)
    part.add(pt.score.Note("G", 4), 8, 9)
    beats = np.arange(100, dtype=float) / 10
    variance = score_position_variance(part, beats, 100)
    np.testing.assert_allclose(variance[20], 80**2 / 12)
    np.testing.assert_allclose(variance[85], 10**2 / 12)


def test_removed_modes_never_receive_probability():
    from matchmaker.dp.oltw_imm import IMMPathFilter

    for removed in ("cv", "ca", "zv"):
        modes = tuple(mode for mode in ("cv", "ca", "zv") if mode != removed)
        model = IMMMotionModels(modes=modes, score_pause_gating=False)
        path = IMMPathFilter(model, np.ones(80), 3)
        disabled = ("cv", "ca", "zv").index(removed)
        for matrix in (model.M_play, model.M_pause):
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
            assert np.all(matrix[:, disabled] == 0.0)
        for frame in range(40):
            path.step(abs(np.arange(80) - frame), 0, 0.05, frame, 1.0)
            assert np.all(path.probabilities[:, disabled] == 0.0)
            assert np.isfinite(path.state).all()


def test_colored_error_preserves_stationary_variance_without_measurements():
    model = IMMMotionModels(score_pause_gating=False)
    states, covariance, probabilities = model.initial_state()
    covariance[:, 3, 3] = 100.0
    for _ in range(100):
        priors, prediction, predicted_covariance = model.predict(
            probabilities[None], states[None], covariance[None], np.full(100, 100.0)
        )
        probabilities, states, covariance = (
            priors[0],
            prediction[0],
            predicted_covariance[0],
        )
        np.testing.assert_allclose(covariance[:, 3, 3], 100.0)


def test_score_gate_controls_zv_prior():
    for gated in (True, False):
        model = IMMMotionModels(score_pause_gating=gated)
        states, covariance, probabilities = model.initial_state()
        priors, _, _ = model.predict(
            probabilities[None], states[None], covariance[None], np.zeros(10)
        )
        assert (priors[0, 2] == 0) == gated
