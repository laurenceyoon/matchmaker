import numpy as np
import partitura as pt

from matchmaker.prob.imm import score_activity


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
    F = np.array([np.diag([1.0, 0.9]), np.diag([1.0, 0.5])])
    Q = np.array([np.diag([1e-4, 0.01]), np.diag([1e-4, 0.3])])
    H, R = np.array([1.7, 1.7]), 0.05
    filters = []
    for f, q in zip(F, Q):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x, kf.P, kf.F, kf.Q, kf.H, kf.R = np.array([[1.0], [0.0]]), np.diag([0.5, 0.1]), f, q, H[None], np.array([[R]])
        filters.append(kf)
    mu0 = stationary_distribution(transition)
    reference = IMMEstimator(filters, mu0.copy(), transition)
    mu, x, P = mu0[None], np.tile([1.0, 0.0], (1, 2, 1)), np.tile(np.diag([0.5, 0.1]), (1, 2, 1, 1))
    for z in rng.normal(1.7, 0.4, 50):
        reference.predict()
        reference.update(np.array([[z]]))
        c, x, P = interact(mu, transition, x, P)
        x, P = np.einsum("jab,hjb->hja", F, x), np.einsum("jab,hjbc,jdc->hjad", F, P, F) + Q
        S = np.einsum("a,hjab,b->hj", H, P, H) + R
        residual = z - x @ H
        likelihood = np.exp(-0.5 * residual ** 2 / S) / np.sqrt(2 * np.pi * S)
        mu = c * likelihood / (c * likelihood).sum()
        x, P = kalman_update(x, P, residual, np.broadcast_to(H, x.shape), np.full(residual.shape, R))
        np.testing.assert_allclose(mu[0], reference.mu, rtol=1e-9)
        np.testing.assert_allclose(x[0], [f.x[:, 0] for f in reference.filters], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(P[0], [f.P for f in reference.filters], rtol=1e-9, atol=1e-12)
