import numpy as np
import pytest

from matchmaker.prob.chord_position import estimate_chord_position


def estimate(chords, ages, routes, probabilities, tempos=None, modes=None):
    n = len(chords)
    return estimate_chord_position(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.full(4, 0.25),
        np.array(chords),
        np.array(ages),
        np.array(routes),
        np.array(probabilities),
        np.ones((n, 1)) if modes is None else np.array(modes),
        np.full((n, 1), 2.0) if tempos is None else np.array(tempos),
        0.1,
    )


def test_adjacent_chords_are_averaged_without_a_map_discontinuity():
    before, _ = estimate([0, 1], [5, 1], [0, 0], [0.51, 0.49])
    after, _ = estimate([0, 1], [5, 1], [0, 0], [0.49, 0.51])
    assert before == pytest.approx(0.898)
    assert after - before == pytest.approx(0.004)


def test_route_is_selected_by_total_mass_not_best_individual_hypothesis():
    position, route = estimate([0, 1, 3], [1, 1, 1], [7, 7, 2], [0.3, 0.3, 0.4])
    assert route == 7
    assert position == pytest.approx(0.5)


def test_repeat_positions_are_never_averaged_across_routes():
    position, route = estimate([0, 3], [1, 1], [1, 0], [0.6, 0.4])
    assert (position, route) == (0.0, 1)


def test_average_position_per_mode_instead_of_inverting_average_tempo():
    position, _ = estimate([0], [3], [0], [1], [[1.0, 4.0]], [[0.5, 0.5]])
    assert position == pytest.approx(0.5)


def test_last_chord_is_absorbing_and_age_one_is_onset():
    assert estimate([2], [1], [0], [1])[0] == 2.0
    assert estimate([3], [10000], [0], [1])[0] == 3.0


def test_splitting_an_equivalent_hypothesis_does_not_change_estimate():
    a = estimate([0, 1], [3, 2], [0, 0], [0.6, 0.4])
    b = estimate([0, 0, 1], [3, 3, 2], [0, 0, 0], [0.2, 0.4, 0.4])
    assert a == pytest.approx(b)
