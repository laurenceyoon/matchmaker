"""Read out a continuous position from a chord/age/route posterior."""

import numpy as np


def estimate_chord_position(
    onset_beats,
    lengths,
    chords,
    ages,
    routes,
    probabilities,
    mode_probabilities,
    tempos,
    frame_seconds,
):
    """Return ``(position_in_beats, route_id)`` without changing the filter.

    First choose the route with greatest marginal probability. Then minimize
    squared position error by averaging the continuous positions of *all*
    hypotheses and modes on that route. Positions across different repeat
    histories must not be averaged: their score coordinates are incompatible.

    Ages count emitted frames, so age one is the chord onset. Tempo is seconds
    per whole note and ``lengths`` are whole notes; ``onset_beats`` can use any
    score beat unit. This readout is causal and carries no smoothing state.
    """
    route_ids, inverse = np.unique(routes, return_inverse=True)
    route_mass = np.bincount(inverse, weights=probabilities)
    route = route_ids[np.argmax(route_mass)]
    selected = routes == route
    k = chords[selected]
    next_k = np.minimum(k + 1, len(onset_beats) - 1)
    duration = lengths[k, None] * tempos[selected]
    elapsed = (ages[selected] - 1)[:, None] * frame_seconds
    fraction = np.clip(elapsed / np.maximum(duration, np.finfo(float).tiny), 0, 1)
    positions = onset_beats[k, None] + fraction * (
        onset_beats[next_k] - onset_beats[k]
    )[:, None]
    weights = probabilities[selected, None] * mode_probabilities[selected]
    position = np.sum(weights * positions) / np.sum(weights)
    return float(position), int(route)
