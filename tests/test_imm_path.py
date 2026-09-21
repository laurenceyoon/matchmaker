import numpy as np
import pytest
from matchmaker.prob.imm import IMMMotionModels
from matchmaker.dp.oltw_imm import IMMOnlineTimeWarping, IMMPathFilter


def scalar_imm_step(model, states, covariances, probabilities, observation, variance):
    priors = probabilities @ model.M_pause
    dynamics, noise = model.F.copy(), model.Q.copy()
    rho = np.exp(-1 / np.sqrt(12 * variance))
    dynamics[:, 3, 3] = rho
    noise[:, 3, 3] = variance * (1 - rho**2)
    updated, covs, likelihoods = [], [], []
    for mode in range(3):
        weights = probabilities * model.M_pause[:, mode] / priors[mode]
        mean = sum(w * x for w, x in zip(weights, states))
        covariance = sum(
            w * (p + np.outer(x - mean, x - mean))
            for w, x, p in zip(weights, states, covariances)
        )
        mean = dynamics[mode] @ mean
        covariance = dynamics[mode] @ covariance @ dynamics[mode].T + noise[mode]
        innovation = observation - model.H @ mean
        cross = covariance @ model.H
        residual_variance = model.H @ cross + model.R
        gain = cross / residual_variance
        updated.append(mean + gain * innovation)
        residual = np.eye(4) - np.outer(gain, model.H)
        covs.append(residual @ covariance @ residual.T + model.R * np.outer(gain, gain))
        likelihoods.append(
            np.exp(-(innovation**2) / (2 * residual_variance))
            / np.sqrt(2 * np.pi * residual_variance)
        )
    probabilities = priors * likelihoods
    return np.array(updated), np.array(covs), probabilities / probabilities.sum()


def test_one_path_reduces_to_standard_imm():
    model = IMMMotionModels(
        ref_frame_to_beat=np.arange(4) / 15, score_pause_gating=False
    )
    path = IMMPathFilter(model, np.full(4, 3.0), 0)
    states, covariances, probabilities = model.initial_state()
    covariances[:, 3, 3] = 3.0
    path.step(np.array([0.2]), 0, 0.05, 0, 1.0)
    for i in range(1, 8):
        states, covariances, probabilities = scalar_imm_step(
            model, states, covariances, probabilities, 0.0, 3.0
        )
        path.step(np.array([0.2]), 0, 0.05, i, 1.0)
        np.testing.assert_allclose(path.probabilities[0], probabilities, atol=1e-12)
        np.testing.assert_allclose(path.states[0], states, atol=1e-12)
        np.testing.assert_allclose(path.covariances[0], covariances, atol=1e-10)
        np.testing.assert_allclose(path.state, probabilities @ states, atol=1e-12)


def test_branches_match_scalar_enumeration():
    model = IMMMotionModels(ref_frame_to_beat=np.arange(12) / 15)
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
    model = IMMMotionModels(ref_frame_to_beat=np.arange(160) / 15)
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
    model = IMMMotionModels(ref_frame_to_beat=np.arange(12) / 15)
    path = IMMPathFilter(model, np.zeros(12), 3)
    distances = np.linspace(0.1, 0.6, 6)
    path.step(distances, 0, 0.05, 0, 1.0)
    path.step(distances, 0, 0.05, 1, 1.0)
    priors, states, covariances = path.predict(np.arange(6))
    position, branches = 2, []
    for ancestor in range(position + 1):
        residual = position - states[ancestor] @ model.H
        cross = covariances[ancestor] @ model.H
        variances = cross @ model.H + model.R
        evidence = (
            priors[ancestor]
            * np.exp(-0.5 * residual**2 / variances)
            / np.sqrt(2 * np.pi * variances)
        )
        acoustic = (
            distances[position]
            if ancestor == position
            else distances[ancestor + 1 : position + 1].sum()
        )
        cost = path.costs[ancestor] - np.log(evidence.sum()) + acoustic / 0.05
        branches.append(
            (
                cost,
                states[ancestor] + cross * (residual / variances)[:, None],
                evidence / evidence.sum(),
            )
        )
    cost, state, probabilities = min(branches, key=lambda item: item[0])
    path.step(distances, 0, 0.0, 2, 1.0)
    np.testing.assert_allclose(path.costs[position], cost)
    np.testing.assert_allclose(path.probabilities[position], probabilities)
    np.testing.assert_allclose(
        path.states[position, probabilities > 0], state[probabilities > 0]
    )


def test_reference_frame_coordinates_do_not_require_a_beat_map():
    path = IMMPathFilter(IMMMotionModels(), np.zeros(12), 3)
    distances = np.arange(6, dtype=float)
    path.step(np.zeros(6), 0, 0.05, 0, 1.0)
    position, cost = path.step(abs(distances - 1), 0, 0.05, 1, 1.0)
    assert position == 1
    assert np.isfinite(cost)
    assert np.isfinite(path.position)


def test_alignment_reset_reproduces_the_same_path_and_state():
    features = np.eye(12, dtype=np.float32)[np.arange(90) // 6 % 12]
    follower = IMMOnlineTimeWarping(features, ref_frame_to_beat=np.arange(90) / 15)
    outputs = []
    for _ in range(2):
        positions = []
        for frame in features[:30]:
            follower.step(frame)
            positions.append(follower.get_current_position())
        outputs.append(
            (
                np.array(positions),
                follower.path.states.copy(),
                follower.path.covariances.copy(),
            )
        )
        follower.reset()
        assert follower.input_index == 0
        assert follower.path.position == 0
    for first, second in zip(*outputs):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("option", ["path_tempo", "filter_output"])
def test_legacy_filter_switches_are_not_silently_overridden(option):
    with pytest.raises(TypeError, match=option):
        IMMOnlineTimeWarping(np.eye(12, dtype=np.float32), **{option: False})


def test_without_imm_matches_acoustic_baseline_and_constructs_no_models(monkeypatch):
    import matchmaker.dp.oltw_imm as module
    from matchmaker.dp.oltw_soft import SoftOnlineTimeWarping, SoftWarpingPath

    def forbidden(*args, **kwargs):
        raise AssertionError("IMM disabled")

    monkeypatch.setattr(module, "IMMMotionModels", forbidden)
    monkeypatch.setattr(module, "score_position_variance", forbidden)
    features = np.eye(12, dtype=np.float32)[np.arange(90) // 6 % 12]
    baseline = SoftOnlineTimeWarping(features)
    ablation = IMMOnlineTimeWarping(features, use_imm=False)
    assert isinstance(ablation.path, SoftWarpingPath)
    for _ in range(2):
        for frame in features:
            baseline.step(frame)
            ablation.step(frame)
            assert baseline.get_current_position() == ablation.get_current_position()
            np.testing.assert_array_equal(baseline.path.costs, ablation.path.costs)
        baseline.reset()
        ablation.reset()


def test_path_restart_preserves_motion_and_reanchors_position():
    features = np.eye(12, dtype=np.float32)[np.arange(90) // 6 % 12]
    follower = IMMOnlineTimeWarping(features)
    for frame in features[:30]:
        follower.step(frame)
    child = IMMOnlineTimeWarping(features)
    child.reset(60)
    child.path.restart_from(follower.path, 60)
    np.testing.assert_array_equal(
        child.path.states[60, :, 1:], follower.path.states[follower.path.index, :, 1:]
    )
    assert child.path.position == pytest.approx(60)
    child._music_started = True
    child.step(features[60])
    assert child.path.index == 60
    assert child.get_current_position() == pytest.approx(60)
