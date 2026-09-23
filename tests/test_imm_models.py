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


def test_imm_primitives_match_filterpy_imm_estimator():
    from filterpy.kalman import IMMEstimator, KalmanFilter
    from matchmaker.prob.imm import interact, kalman_update, stationary_distribution

    rng = np.random.default_rng(0)
    transition = np.array([[0.97, 0.03], [0.2, 0.8]])
    Q, H, R = np.array([1e-4, 0.3]), 1.7, 0.05
    filters = []
    for q in Q:
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x, kf.P, kf.F, kf.Q, kf.H, kf.R = np.array([[1.0]]), np.array([[0.5]]), np.eye(1), np.array([[q]]), np.array([[H]]), np.array([[R]])
        filters.append(kf)
    mu0 = stationary_distribution(transition)
    reference = IMMEstimator(filters, mu0.copy(), transition)
    mu, x, P = mu0[None], np.ones((1, 2)), np.full((1, 2), 0.5)
    for z in rng.normal(1.7, 0.4, 50):
        reference.predict()
        reference.update(np.array([[z]]))
        c, x, P = interact(mu, transition, x, P)
        P = P + Q
        S = H ** 2 * P + R
        likelihood = np.exp(-0.5 * (z - H * x) ** 2 / S) / np.sqrt(2 * np.pi * S)
        mu = c * likelihood / (c * likelihood).sum()
        x, P = kalman_update(x, P, z - H * x, H, R)
        np.testing.assert_allclose(mu[0], reference.mu, rtol=1e-9)
        np.testing.assert_allclose(x[0], [f.x[0, 0] for f in reference.filters], rtol=1e-9)
        np.testing.assert_allclose(P[0], [f.P[0, 0] for f in reference.filters], rtol=1e-9)
