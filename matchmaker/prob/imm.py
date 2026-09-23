from typing import Any, Tuple

import numpy as np
import partitura as pt
from scipy.linalg import expm

from matchmaker.features.audio import FRAME_RATE


class IMMMotionModels:
    """Score-informed CV/CA/ZV mode bank: per-frame mode transitions and pause gating.

    Residence times come from the score (beat and measure durations at the marked
    tempo); ZV (a pause) is only reachable inside notated fermatas and rests.
    """

    def __init__(
        self,
        score_part: Any = None,
        tempo: float = 120.0,
        frame_rate: int = FRAME_RATE,
        modes: Tuple[str, ...] = ("cv", "ca", "zv"),
    ):
        if tempo <= 0 or frame_rate <= 0:
            raise ValueError("Tempo and frame rate must be positive")
        self.enabled = np.array([mode in modes for mode in ("cv", "ca", "zv")])
        if not self.enabled[:2].any():
            raise ValueError("At least one moving mode is required")
        self.dt = 1.0 / frame_rate
        self.beat_seconds = 60.0 / tempo
        self.pause_ranges = self._score_pause_ranges(score_part)
        measure_beats = []
        if score_part is not None:
            for measure in score_part.iter_all(pt.score.Measure):
                if measure.start is not None and measure.end is not None:
                    duration = float(
                        score_part.beat_map(measure.end.t)
                        - score_part.beat_map(measure.start.t)
                    )
                    if duration > 0:
                        measure_beats.append(duration)
        self.measure_seconds = self.beat_seconds * (
            float(np.median(measure_beats)) if measure_beats else 1.0
        )
        self.M_play = self._transition_matrix(allow_pause=False)
        self.M_pause = self._transition_matrix(allow_pause=True)
        self.initial_probabilities = np.zeros(3)
        self.initial_probabilities[np.flatnonzero(self.enabled[:2])[0]] = 1.0

    @staticmethod
    def _score_pause_ranges(score_part: Any) -> list:
        if score_part is None:
            return []
        ranges = []
        for fermata in score_part.iter_all(pt.score.Fermata):
            ref = fermata.ref
            if (
                getattr(ref, "start", None) is not None
                and getattr(ref, "end", None) is not None
            ):
                ranges.append(
                    (
                        float(score_part.beat_map(ref.start.t)),
                        float(score_part.beat_map(ref.end.t)),
                    )
                )

        notes = score_part.note_array()
        if len(notes):
            intervals = sorted(
                (float(n["onset_beat"]), float(n["onset_beat"] + n["duration_beat"]))
                for n in notes
            )
            end = intervals[0][1]
            for start, stop in intervals[1:]:
                if start > end:
                    ranges.append((end, start))
                end = max(end, stop)
        return sorted(set(ranges))

    def _transition_matrix(self, allow_pause: bool) -> np.ndarray:
        active = np.flatnonzero(self.enabled & np.array([True, True, allow_pause]))
        n = len(active)
        # A tempo drift (CA) is a phrase-scale gesture, like the steady pulse it
        # perturbs (CV), not an instantaneous event like a pause (ZV): both share
        # the measure-scale residence time, so a rubato gesture has room to
        # accumulate before mixing reverts it to the steady-pulse mode.
        durations = np.array(
            [self.measure_seconds, self.measure_seconds, self.beat_seconds]
        )[active]
        matrix = np.zeros((3, 3))
        matrix[:, active] = 1 / n
        if n > 1:
            generator = np.ones((n, n)) / (durations[:, None] * (n - 1))
            np.fill_diagonal(generator, -1 / durations)
            matrix[np.ix_(active, active)] = expm(generator * self.dt)
        return matrix


def score_activity(score_part, beats, size):
    if score_part is None or beats is None:
        return np.ones(size, dtype=bool)
    notes = score_part.note_array()
    starts = np.sort(notes["onset_beat"])
    ends = np.sort(notes["onset_beat"] + notes["duration_beat"])
    return np.searchsorted(starts, beats, side="right") > np.searchsorted(
        ends, beats, side="right"
    )


# ---------------------------------------------------------------------------
# Vectorised IMM primitives (Blom & Bar-Shalom): a batch of hypotheses, each with an
# n-dimensional state per mode. Checked against filterpy.kalman.IMMEstimator in the tests.
# ---------------------------------------------------------------------------

def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Stationary distribution of a Markov transition matrix (rows sum to one)."""
    values, vectors = np.linalg.eig(transition.T)
    pi = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    return pi / pi.sum()


def interact(mu: np.ndarray, transition: np.ndarray, x: np.ndarray, P: np.ndarray):
    """IMM interaction: predicted mode probabilities and each mode's mixed initial condition.

    mu: (hypotheses, modes); transition: (modes, modes) or one per hypothesis;
    x: (hypotheses, modes, n); P: (hypotheses, modes, n, n). Returns (c, x0, P0).
    """
    transition = np.broadcast_to(transition, (len(mu),) + transition.shape[-2:])
    c = np.einsum("hi,hij->hj", mu, transition)
    mix = mu[:, :, None] * transition / np.maximum(c[:, None, :], np.finfo(float).tiny)
    x0 = np.einsum("hij,hia->hja", mix, x)
    spread = x[:, :, None, :] - x0[:, None, :, :]
    P0 = np.einsum("hij,hiab->hjab", mix, P) + np.einsum("hij,hija,hijb->hjab", mix, spread, spread)
    return c, x0, P0


def kalman_update(x: np.ndarray, P: np.ndarray, residual: np.ndarray, H: np.ndarray, R: np.ndarray):
    """Kalman update by one scalar measurement, Joseph-form covariance.

    x, H: (..., n); P: (..., n, n); residual, R: (...).
    """
    PH = np.einsum("...ab,...b->...a", P, H)
    S = np.einsum("...a,...a->...", H, PH) + R
    K = PH / S[..., None]
    IKH = np.eye(x.shape[-1]) - K[..., :, None] * H[..., None, :]
    P = np.einsum("...ab,...bc,...dc->...ad", IKH, P, IKH) + R[..., None, None] * K[..., :, None] * K[..., None, :]
    return x + K * residual[..., None], P
