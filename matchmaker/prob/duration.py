"""Duration uncertainty and discrete survival probabilities for chord filters."""

import numpy as np
from scipy.special import log_ndtr


def score_time_duration_variance(length, tempo, jitter_scale):
    """Timing-noise variance accumulated over ``length`` whole notes.

    ``tempo`` is seconds per whole note. Independent timing increments have
    additive variance in score time, so splitting a chord preserves its total
    timing variance. Tempo-state uncertainty is added separately by the caller.
    """
    return length * (jitter_scale * tempo) ** 2


def gaussian_duration_stay(mean, variance, age, frame_seconds):
    """Conditional chance of staying for the next observed frame.

    Age one is the first frame of the chord. Its outgoing transition covers
    durations in (0, dt], age two covers (dt, 2dt], etc. Starting the denominator
    at dt would silently exclude every chord shorter than one frame. The
    positive-duration truncation cancels in this ratio of survival functions.

    Log survival avoids cancellation in ``1 - ndtr(z)`` for delayed hypotheses.
    Arrays broadcast; variance must be strictly positive and age at least one.
    """
    std = np.sqrt(variance)
    lower = (age - 1) * frame_seconds
    upper = age * frame_seconds
    log_stay = log_ndtr((mean - upper) / std) - log_ndtr((mean - lower) / std)
    return np.exp(np.minimum(log_stay, 0.0))
