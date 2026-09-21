import numpy as np
from matchmaker.prob.imm import ScoreInformedIMM
from matchmaker.dp.oltw_imm import IMMPathFilter


def test_one_path_reduces_to_standard_imm():
    model = ScoreInformedIMM(
        ref_frame_to_beat=np.arange(4) / 15, score_pause_gating=False
    )
    path = IMMPathFilter(model, np.full(4, 3.0), 0)
    reference = ScoreInformedIMM(
        ref_frame_to_beat=np.arange(4) / 15, score_pause_gating=False
    )
    reference.set_observation_error(3.0, initialize=True)
    path.step(np.array([0.2]), 0, 0.05, 0, 1.0)
    for i in range(1, 8):
        reference.set_observation_error(3.0)
        reference.predict()
        reference.update(0.0)
        path.step(np.array([0.2]), 0, 0.05, i, 1.0)
        np.testing.assert_allclose(path.probabilities[0], reference.mu, atol=1e-12)
        np.testing.assert_allclose(path.states[0], reference.states, atol=1e-12)
        np.testing.assert_allclose(
            path.covariances[0], reference.P_matrices, atol=1e-10
        )
        np.testing.assert_allclose(path.state, reference.state, atol=1e-12)


def test_branches_match_scalar_enumeration():
    model = ScoreInformedIMM(ref_frame_to_beat=np.arange(12) / 15)
    path = IMMPathFilter(model, np.zeros(12), 3)
    distances = np.linspace(0.1, 0.6, 6)
    path.step(distances, 0, 0.05, 0, 1.0)
    path.step(distances, 0, 0.05, 1, 1.0)
    priors, states, covariances = path.predict(np.arange(6))
    position = 2
    branches = []
    for ancestor in range(position + 1):
        for mode in np.flatnonzero(priors[ancestor] > 0):
            state, covariance = states[ancestor, mode], covariances[ancestor, mode]
            residual = position - model.H @ state
            cross = covariance @ model.H
            variance = model.H @ cross + model.R
            acoustic = (
                distances[position]
                if ancestor == position
                else distances[ancestor + 1 : position + 1].sum()
            )
            cost = (
                path.costs[ancestor]
                - np.log(priors[ancestor, mode])
                + 0.5 * (residual**2 / variance + np.log(2 * np.pi * variance))
                + acoustic / 0.05
            )
            branches.append(
                (
                    cost,
                    mode,
                    state + cross * residual / variance,
                    covariance - np.outer(cross, cross) / variance,
                )
            )
    costs = np.array([b[0] for b in branches])
    weights = np.exp(costs.min() - costs)
    weights /= weights.sum()
    path.step(distances, 0, 0.05, 2, 1.0)
    for mode in (0, 1):
        subset = [(w, x, p) for w, (_, m, x, p) in zip(weights, branches) if m == mode]
        total = sum(w for w, _, _ in subset)
        mean = sum(w * x for w, x, _ in subset) / total
        covariance = (
            sum(w * (p + np.outer(x - mean, x - mean)) for w, x, p in subset) / total
        )
        np.testing.assert_allclose(path.probabilities[position, mode], total)
        np.testing.assert_allclose(path.states[position, mode], mean)
        np.testing.assert_allclose(
            path.covariances[position, mode], covariance, atol=1e-12
        )


def test_ramp_keeps_distinct_paths_and_valid_covariances():
    model = ScoreInformedIMM(ref_frame_to_beat=np.arange(160) / 15)
    path = IMMPathFilter(model, np.full(160, 3.0), 3)
    path.step(np.array([0.0, 1.0, 2.0]), 0, 0.05, 0, 1.0)
    for frame in range(1, 80):
        start, end = max(frame - 20, 0), min(frame + 20, 160)
        winner, _ = path.step(
            abs(np.arange(start, end) - frame).astype(float), start, 0.05, frame, 1.0
        )
        active = np.isfinite(path.costs)
        assert abs(winner - frame) <= 1
        assert active.sum() > 1
        assert np.linalg.eigvalsh(path.covariances[active]).min() >= -1e-10
        np.testing.assert_allclose(path.probabilities[active].sum(axis=1), 1.0)


def test_hard_min_selects_one_path_and_retains_imm_mode_probabilities():
    model = ScoreInformedIMM(ref_frame_to_beat=np.arange(12) / 15)
    path = IMMPathFilter(model, np.zeros(12), 3)
    distances = np.linspace(0.1, 0.6, 6)
    path.step(distances, 0, 0.05, 0, 1.0)
    path.step(distances, 0, 0.05, 1, 1.0)
    priors, states, covariances = path.predict(np.arange(6))
    position, branches = 2, []
    for ancestor in range(position + 1):
        reference = ScoreInformedIMM(ref_frame_to_beat=np.arange(12) / 15)
        reference.states = states[ancestor].copy()
        reference.P_matrices = covariances[ancestor].copy()
        reference.c_bar = priors[ancestor].copy()
        residuals = position - reference.states @ reference.H
        variances = reference.P_matrices @ reference.H @ reference.H + reference.R
        evidence = np.sum(
            priors[ancestor]
            * np.exp(-0.5 * residuals**2 / variances)
            / np.sqrt(2 * np.pi * variances)
        )
        acoustic = (
            distances[position]
            if ancestor == position
            else distances[ancestor + 1 : position + 1].sum()
        )
        cost = path.costs[ancestor] - np.log(evidence) + acoustic / 0.05
        reference.update(position)
        branches.append((cost, reference))
    expected_cost, expected = min(branches, key=lambda item: item[0])
    path.step(distances, 0, 0.05, 2, 1.0, hard_min=True)
    np.testing.assert_allclose(path.costs[position], expected_cost)
    np.testing.assert_allclose(path.probabilities[position], expected.mu)
    active = expected.mu > 0
    np.testing.assert_allclose(path.states[position, active], expected.states[active])
    np.testing.assert_allclose(
        path.covariances[position, active], expected.P_matrices[active], atol=1e-12
    )


def test_reference_frame_coordinates_do_not_require_a_beat_map():
    path = IMMPathFilter(ScoreInformedIMM(), np.zeros(12), 3)
    distances = np.arange(6, dtype=float)
    path.step(np.zeros(6), 0, 0.05, 0, 1.0)
    position, cost = path.step(abs(distances - 1), 0, 0.05, 1, 1.0)
    assert position == 1
    assert np.isfinite(cost)
    assert np.isfinite(path.position)
