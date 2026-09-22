import numpy as np
import pytest
from scipy.special import ndtr

from matchmaker.prob.duration import gaussian_duration_stay, score_time_duration_variance


def test_timing_variance_is_invariant_to_score_subdivision():
    whole = score_time_duration_variance(0.25, 2.0, 0.05)
    subdivided = 4 * score_time_duration_variance(0.0625, 2.0, 0.05)
    assert whole == pytest.approx(subdivided)


def test_first_frame_includes_short_positive_durations():
    mean, std, dt = 0.01, 0.005, 1 / 30
    expected = ndtr((mean - dt) / std) / ndtr(mean / std)
    assert gaussian_duration_stay(mean, std**2, 1, dt) == pytest.approx(expected)
    assert expected < 1e-5


def test_discrete_duration_probabilities_telescope_to_the_cdf():
    mean, std, dt = 0.4, 0.1, 1 / 30
    age = np.arange(1, 61)
    stay = gaussian_duration_stay(mean, std**2, age, dt)
    alive = np.r_[1, np.cumprod(stay)[:-1]]
    probability = alive * (1 - stay)
    expected_cdf = (ndtr((age * dt - mean) / std) - ndtr(-mean / std)) / ndtr(mean / std)
    np.testing.assert_allclose(np.cumsum(probability), expected_cdf, atol=1e-14)


def test_delayed_hypotheses_keep_a_finite_nonzero_survival_probability():
    stay = gaussian_duration_stay(0.0, 1.0, np.array([10, 30, 100]), 1.0)
    assert np.all(np.isfinite(stay))
    assert np.all((stay > 0) & (stay < 1))


def test_uniform_time_rescaling_does_not_change_stay_probability():
    a = gaussian_duration_stay(0.4, 0.02, np.arange(1, 12), 0.03)
    b = gaussian_duration_stay(0.8, 0.08, np.arange(1, 12), 0.06)
    np.testing.assert_allclose(a, b)
