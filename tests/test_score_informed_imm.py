import numpy as np
from scipy.linalg import expm
from matchmaker.prob.imm import ScoreInformedIMM


def test_continuous_time_model_is_invariant_to_frame_subdivision():
    full = ScoreInformedIMM(frame_rate=30)
    half = ScoreInformedIMM(frame_rate=60)
    np.testing.assert_allclose(full.F, half.F @ half.F)
    np.testing.assert_allclose(full.Q, half.F @ half.Q @ half.F.transpose(0, 2, 1) + half.Q,
                               atol=1e-12)
    np.testing.assert_allclose(full.M_pause, half.M_pause @ half.M_pause, atol=1e-12)
    np.testing.assert_allclose(full.M_play, half.M_play @ half.M_play, atol=1e-12)


def test_mode_transition_is_exact_continuous_time_chain():
    imm = ScoreInformedIMM()
    generator = np.full((3, 3), 1 / (2 * imm.beat_seconds))
    np.fill_diagonal(generator, -1 / imm.beat_seconds)
    np.testing.assert_allclose(imm.M_pause, expm(generator * imm.dt), atol=1e-12)
    np.testing.assert_allclose(imm.M_pause.sum(axis=1), 1)
    assert np.all(imm.M_play[:, 2] == 0)


def test_slow_tempo_has_no_artificial_velocity_floor():
    imm = ScoreInformedIMM(init_tempo=0.05)
    for i in range(1, 121):
        imm.predict()
        imm.update(i * 0.05)
    np.testing.assert_allclose(imm.tempo, 0.05, atol=1e-8)


def test_covariance_and_probabilities_remain_valid_at_pause_and_resume():
    imm = ScoreInformedIMM()
    position = 1000.0
    imm.reset(position)
    for velocity in [1.] * 100 + [0.] * 100 + [1.] * 100:
        position += velocity
        imm.predict(is_silent=velocity == 0)
        imm.update(position)
        assert np.isfinite(imm.state).all()
        np.testing.assert_allclose(imm.mu.sum(), 1)
        np.testing.assert_allclose(imm.P, imm.P.T, atol=1e-10)
        assert np.linalg.eigvalsh(imm.P).min() >= -1e-10
    assert abs(imm.position-position) < 1


def test_scalar_reference_imm_equations_match_vectorized_implementation():
    imm = ScoreInformedIMM()
    imm.mu = np.array([0.3, 0.6, 0.1])
    imm.states[:] = np.array([[40.,30.,0.,2.],[41.,35.,4.,-1.],[39.,0.,0.,3.]])
    imm.set_observation_error(4.0, initialize=True)
    states, covs = imm.states.copy(), imm.P_matrices.copy()
    c = imm.mu @ imm.M_pause
    omega = imm.mu[:,None] * imm.M_pause / c
    expected_x, expected_p = [], []
    for j in range(3):
        x = sum(omega[i,j] * states[i] for i in range(3))
        p = sum(omega[i,j] * (covs[i] + np.outer(states[i]-x, states[i]-x)) for i in range(3))
        expected_x.append(imm.F[j] @ x)
        expected_p.append(imm.F[j] @ p @ imm.F[j].T + imm.Q[j])
    imm.predict(is_silent=True)
    np.testing.assert_allclose(imm.states, expected_x)
    np.testing.assert_allclose(imm.P_matrices, expected_p, atol=1e-10)
    z = 42.
    likelihoods = []
    for j in range(3):
        y = z - imm.H @ expected_x[j]
        cross = expected_p[j] @ imm.H
        S = imm.H @ cross + imm.R
        K = cross / S
        expected_x[j] += K * y
        expected_p[j] -= np.outer(K, cross)
        likelihoods.append(np.exp(-y*y/(2*S))/np.sqrt(2*np.pi*S))
    mu = c * likelihoods
    mu /= mu.sum()
    imm.update(z)
    np.testing.assert_allclose(imm.mu, mu)
    np.testing.assert_allclose(imm.states, expected_x)
    np.testing.assert_allclose(imm.P_matrices, expected_p, atol=1e-10)
    mean = sum(mu[j] * expected_x[j] for j in range(3))
    covariance = sum(mu[j] * (expected_p[j] + np.outer(expected_x[j] - mean, expected_x[j] - mean))
                     for j in range(3))
    np.testing.assert_allclose(imm.state, mean)
    np.testing.assert_allclose(imm.P, covariance, atol=1e-10)


def test_score_repeated_chord_duration_defines_uncertainty():
    import partitura as pt
    from matchmaker.dp.oltw_soft import SoftOnlineTimeWarping
    part = pt.score.Part('P1')
    part.set_quarter_duration(0, 1)
    part.add(pt.score.TimeSignature(4, 4), 0)
    part.add(pt.score.Note('C', 4), 0, 4)
    part.add(pt.score.Note('C', 4), 4, 8)
    part.add(pt.score.Note('G', 4), 8, 9)
    beats = np.arange(100, dtype=float) / 10
    follower = SoftOnlineTimeWarping(reference_features=np.zeros((100, 12)),
                                    score_part=part, ref_frame_to_beat=beats,
                                    frame_rate=10)
    np.testing.assert_allclose(follower._score_position_variance[20], 80**2 / 12)
    np.testing.assert_allclose(follower._score_position_variance[85], 10**2 / 12)


def test_correlated_observation_update_matches_scalar_kalman_reference():
    imm = ScoreInformedIMM()
    imm.set_observation_error(4.0, initialize=True)
    imm.predict()
    state, covariance = imm.states[0].copy(), imm.P_matrices[0].copy()
    observation = 4.0
    cross_covariance = covariance @ imm.H
    innovation_variance = imm.H @ cross_covariance + imm.R
    gain = cross_covariance / innovation_variance
    expected_state = state + gain * (observation - imm.H @ state)
    residual = np.eye(4) - np.outer(gain, imm.H)
    expected_covariance = residual @ covariance @ residual.T + imm.R * np.outer(gain, gain)
    imm.update(observation)
    np.testing.assert_allclose(imm.states[0], expected_state)
    np.testing.assert_allclose(imm.P_matrices[0], expected_covariance)


def test_observation_error_preserves_stationary_variance_without_measurements():
    imm = ScoreInformedIMM()
    imm.set_observation_error(100.0, initialize=True)
    for _ in range(100):
        imm.predict()
        imm.update(None)
        np.testing.assert_allclose(imm.P_matrices[:, 3, 3], 100.0)


def test_persistent_observation_error_keeps_covariance_and_modes_valid():
    imm = ScoreInformedIMM()
    imm.set_observation_error(4.0, initialize=True)
    for frame in range(1, 501):
        imm.predict()
        imm.update(frame + 3 * np.sin(frame / 30))
        assert np.isfinite(imm.states).all()
        np.testing.assert_allclose(imm.P_matrices, imm.P_matrices.transpose(0, 2, 1), atol=1e-9)
        assert np.linalg.eigvalsh(imm.P_matrices).min() > -1e-9
        np.testing.assert_allclose(imm.mu.sum(), 1.0)


def test_uncertain_observation_weights_prediction_more():
    certain = ScoreInformedIMM()
    uncertain = ScoreInformedIMM()
    certain.predict()
    uncertain.predict()
    prediction = certain.position
    certain.update(10.0)
    uncertain.update(10.0, obs_var=1000.0)
    assert abs(uncertain.position - prediction) < abs(certain.position - prediction)


def test_removed_modes_never_receive_probability():
    for removed in ("cv", "ca", "zv"):
        modes = tuple(mode for mode in ("cv", "ca", "zv") if mode != removed)
        imm = ScoreInformedIMM(modes=modes)
        disabled = ("cv", "ca", "zv").index(removed)
        for matrix in (imm.M_play, imm.M_pause):
            np.testing.assert_allclose(matrix.sum(axis=1), 1.)
            assert np.all(matrix[:, disabled] == 0.)
        for frame in range(50):
            imm.predict(is_silent=20 <= frame < 30)
            imm.update(float(frame))
            assert imm.mu[disabled] == 0.
            assert np.isfinite(imm.state).all()
            assert np.linalg.eigvalsh(imm.P).min() > -1e-10


def test_single_mode_reduces_to_ordinary_kalman_recursion():
    for index, mode in enumerate(('cv', 'ca')):
        imm = ScoreInformedIMM(modes=(mode,))
        imm.set_observation_error(4., initialize=True)
        state = imm.states[index].copy()
        covariance = imm.P_matrices[index].copy()
        for frame in range(1, 31):
            state = imm.F[index] @ state
            covariance = imm.F[index] @ covariance @ imm.F[index].T + imm.Q[index]
            observation = frame + np.sin(frame)
            cross = covariance @ imm.H
            gain = cross / (imm.H @ cross + imm.R)
            state += gain * (observation - imm.H @ state)
            covariance -= np.outer(gain, cross)
            imm.predict(is_silent=frame > 15)
            imm.update(observation)
            assert imm.mu[index] == 1.
            np.testing.assert_allclose(imm.state, state, atol=1e-10)
            np.testing.assert_allclose(imm.P, covariance, atol=1e-10)


def test_ungated_zv_is_available_without_forcing_a_pause():
    gated = ScoreInformedIMM()
    ungated = ScoreInformedIMM(score_pause_gating=False)
    gated.predict()
    ungated.predict()
    assert gated.c_bar[2] == 0.
    assert 0. < ungated.c_bar[2] < ungated.c_bar[0]
    np.testing.assert_allclose(ungated.c_bar, ungated.M_pause[0])
