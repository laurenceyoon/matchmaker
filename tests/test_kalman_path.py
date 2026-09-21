import numpy as np
from matchmaker.dp.kalman_path import KalmanPathLattice


def test_path_mixture_matches_scalar_branch_enumeration():
    model = KalmanPathLattice(12, 5., 15., 3)
    distances = np.linspace(.1, .6, 6)
    model.step(distances, 0, .05, 0, 1.)
    model.step(distances, 0, .05, 1, 1.)
    position = 2
    branch_costs, branch_states, branch_covariances = [], [], []
    for step in range(3):
        ancestor = position - step
        mean = model.F @ model.states[ancestor]
        covariance = model.F @ model.covariances[ancestor] @ model.F.T + model.Q
        residual = position - mean[0]
        variance = covariance[0, 0] + model.R
        acoustic = distances[position] if step == 0 else distances[ancestor + 1:position + 1].sum()
        branch_costs.append(model.costs[ancestor] + .5 * (residual**2 / variance + np.log(2 * np.pi * variance)) + acoustic / .05)
        cross = covariance[:, 0].copy()
        branch_states.append(mean + cross * residual / variance)
        branch_covariances.append(covariance - np.outer(cross, cross) / variance)
    branch_costs = np.array(branch_costs)
    weights = np.exp(branch_costs.min() - branch_costs)
    total = weights.sum()
    weights /= total
    expected_mean = sum(w * x for w, x in zip(weights, branch_states))
    expected_covariance = sum(w * (p + np.outer(x - expected_mean, x - expected_mean))
                              for w, x, p in zip(weights, branch_states, branch_covariances))
    model.step(distances, 0, .05, 2, 1.)
    np.testing.assert_allclose(model.states[position], expected_mean)
    np.testing.assert_allclose(model.covariances[position], expected_covariance)
    np.testing.assert_allclose(model.costs[position], branch_costs.min() - np.log(total))


def test_path_lattice_tracks_ramp_with_valid_covariances():
    model = KalmanPathLattice(200, 5., 15., 3)
    model.step(np.array([0., 1., 2.]), 0, .05, 0, 1.)
    for frame in range(1, 80):
        start, end = max(frame - 20, 0), min(frame + 20, 200)
        position, cost = model.step(abs(np.arange(start, end) - frame).astype(float), start, .05, frame, 1.)
        active = np.isfinite(model.costs)
        assert abs(position - frame) <= 1
        assert np.isfinite(cost)
        assert np.linalg.eigvalsh(model.covariances[active]).min() >= -1e-10
        np.testing.assert_allclose(model.covariances[active], model.covariances[active].transpose(0, 2, 1), atol=1e-10)


def test_motion_covariance_matches_observation_scale_over_one_beat():
    model = KalmanPathLattice(20, 5., 15., 3)
    covariance = np.zeros((2, 2))
    for _ in range(15):
        covariance = model.F @ covariance @ model.F.T + model.Q
    np.testing.assert_allclose(covariance[0, 0], model.R)


def test_hard_min_keeps_best_branch_and_its_posterior():
    model = KalmanPathLattice(12, 5., 15., 3)
    distances = np.linspace(.1, .6, 6)
    model.step(distances, 0, .05, 0, 1., hard_min=True)
    model.step(distances, 0, .05, 1, 1., hard_min=True)
    position = 2
    branches = []
    for ancestor in range(position + 1):
        state = model.F @ model.states[ancestor]
        covariance = model.F @ model.covariances[ancestor] @ model.F.T + model.Q
        residual = position - state[0]
        variance = covariance[0, 0] + model.R
        acoustic = distances[position] if ancestor == position else distances[ancestor + 1:position + 1].sum()
        cost = model.costs[ancestor] + .5 * (residual**2 / variance + np.log(2 * np.pi * variance)) + acoustic / .05
        cross = covariance[:, 0].copy()
        branches.append((cost, state + cross * residual / variance,
                         covariance - np.outer(cross, cross) / variance))
    cost, state, covariance = min(branches, key=lambda branch: branch[0])
    model.step(distances, 0, .05, 2, 1., hard_min=True)
    np.testing.assert_allclose(model.costs[position], cost)
    np.testing.assert_allclose(model.states[position], state)
    np.testing.assert_allclose(model.covariances[position], covariance)
