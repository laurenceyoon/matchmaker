from typing import Any, Optional, Tuple

import numpy as np
import partitura as pt
from numpy.typing import NDArray
from scipy.linalg import expm

from matchmaker.features.audio import FRAME_RATE

DEFAULT_OBS_VAR = 5.0


class IMMMotionModels:
    """Score-informed dynamics and IMM prediction, owned by IMMPathFilter.

    States contain position, velocity, acceleration and correlated observation
    error. This component neither consumes audio nor emits an alignment path.
    """

    def __init__(
        self,
        score_part: Any = None,
        ref_frame_to_beat: Optional[NDArray] = None,
        obs_var: float = DEFAULT_OBS_VAR,
        tempo: float = 120.0,
        frame_rate: int = FRAME_RATE,
        modes: Tuple[str, ...] = ("cv", "ca", "zv"),
        score_pause_gating: bool = True,
        retain_tempo: bool = False,
    ):
        if obs_var <= 0 or tempo <= 0 or frame_rate <= 0:
            raise ValueError(
                "Observation variance, tempo and frame rate must be positive"
            )
        self.enabled = np.array([mode in modes for mode in ("cv", "ca", "zv")])
        if not self.enabled[:2].any():
            raise ValueError("At least one moving mode is required")
        self.R = float(obs_var)
        self.H = np.array([1.0, 0.0, 0.0, 1.0])
        self.score_part = score_part
        self.score_pause_gating = score_pause_gating
        self.r2b = ref_frame_to_beat
        self.tempo_bpm = float(tempo)
        self.frame_rate = float(frame_rate)
        self.dt = 1.0 / self.frame_rate
        self.beat_seconds = 60.0 / self.tempo_bpm
        if self.r2b is not None and len(self.r2b) > 1:
            increments = np.diff(self.r2b)
            positive = increments[increments > 0]
            if len(positive):
                self.beat_seconds = self.dt / float(np.median(positive))
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

        dt = self.dt
        self.F = np.array(
            [
                [[1, dt, 0], [0, 1, 0], [0, 0, 0]],
                [[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]],
                [[1, 0, 0], [0, int(retain_tempo), 0], [0, 0, 0]],
            ],
            dtype=float,
        )

        q_ca = 20 * self.R / self.beat_seconds**5
        self.Q = np.array(
            [
                np.zeros((3, 3)),
                q_ca
                * np.array(
                    [
                        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                        [dt**3 / 6, dt**2 / 2, dt],
                    ]
                ),
                np.zeros((3, 3)),
            ]
        )
        self.F = np.pad(self.F, ((0, 0), (0, 1), (0, 1)))
        self.Q = np.pad(self.Q, ((0, 0), (0, 1), (0, 1)))
        self.M_play = self._transition_matrix(allow_pause=False)
        self.M_pause = self._transition_matrix(allow_pause=True)

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
        durations = np.array(
            [self.measure_seconds, self.beat_seconds, self.beat_seconds]
        )[active]
        matrix = np.zeros((3, 3))
        matrix[:, active] = 1 / n
        if n > 1:
            generator = np.ones((n, n)) / (durations[:, None] * (n - 1))
            np.fill_diagonal(generator, -1 / durations)
            matrix[np.ix_(active, active)] = expm(generator * self.dt)
        return matrix

    def initial_state(self, position=0.0):
        velocity = 1.0 / self.dt
        probabilities = np.zeros(3)
        probabilities[np.flatnonzero(self.enabled[:2])[0]] = 1.0
        states = np.array(
            [
                [position, velocity, 0.0, 0.0],
                [position, velocity, 0.0, 0.0],
                [position, 0.0, 0.0, 0.0],
            ]
        )
        covariances = np.array(
            [
                np.diag([self.R, velocity**2, 0, 0]),
                np.diag([self.R, velocity**2, (velocity / self.beat_seconds) ** 2, 0]),
                np.diag([self.R, 0, 0, 0]),
            ]
        )
        return states, covariances, probabilities

    def predict(self, probabilities, states, covariance, score_variance, allow_pause=False):
        means = np.einsum("am,amd->ad", probabilities, states)
        beats = means[:, 0]
        if self.r2b is not None:
            beats = np.interp(beats, np.arange(len(self.r2b)), self.r2b)
        allowed = np.full(len(states), allow_pause or not self.score_pause_gating)
        for start, end in self.pause_ranges:
            allowed |= (beats >= start) & (beats < end)
        transition = np.where(allowed[:, None, None], self.M_pause, self.M_play)
        joint = probabilities[:, :, None] * transition
        priors = joint.sum(axis=1)
        mixing = np.divide(
            joint,
            priors[:, None, :],
            out=np.zeros_like(joint),
            where=priors[:, None, :] > 0,
        )
        mixed = np.einsum("aij,aid->ajd", mixing, states)
        residuals = states[:, :, None, :] - mixed[:, None, :, :]
        covariance = np.einsum("aij,aikl->ajkl", mixing, covariance)
        covariance += np.einsum("aij,aijk,aijl->ajkl", mixing, residuals, residuals)
        frames = np.clip(np.rint(means[:, 0]).astype(int), 0, len(score_variance) - 1)
        variance = score_variance[frames]
        duration = np.sqrt(12 * variance)
        correlation = np.exp(
            np.divide(
                -1.0, duration, out=np.full_like(duration, -np.inf), where=duration > 0
            )
        )
        dynamics = np.tile(self.F, (len(states), 1, 1, 1))
        noise = np.tile(self.Q, (len(states), 1, 1, 1))
        dynamics[:, :, 3, 3] = correlation[:, None]
        noise[:, :, 3, 3] = (variance * (1 - correlation**2))[:, None]
        predicted = np.einsum("amij,amj->ami", dynamics, mixed)
        covariance = dynamics @ covariance @ dynamics.swapaxes(-1, -2) + noise
        return priors, predicted, covariance


def score_activity(score_part, beats, size):
    if score_part is None or beats is None:
        return np.ones(size, dtype=bool)
    notes = score_part.note_array()
    starts = np.sort(notes["onset_beat"])
    ends = np.sort(notes["onset_beat"] + notes["duration_beat"])
    return np.searchsorted(starts, beats, side="right") > np.searchsorted(
        ends, beats, side="right"
    )


def score_position_variance(score_part, ref_frame_to_beat, n_frames):
    frames = np.arange(n_frames, dtype=float)
    variance = np.zeros(n_frames)
    if score_part is None or ref_frame_to_beat is None:
        return variance
    events = {}
    for note in score_part.note_array():
        start, duration = float(note["onset_beat"]), float(note["duration_beat"])
        if duration <= 0:
            continue
        pc = int(note["pitch"]) % 12
        for beat, delta in ((start, 1), (start + duration, -1)):
            events.setdefault(beat, []).append((pc, delta))
    counts = np.zeros(12, dtype=int)
    signature, run_start, spans = None, None, []
    for beat, changes in sorted(events.items()):
        for pc, delta in changes:
            counts[pc] += delta
        updated = tuple(np.flatnonzero(counts > 0))
        if updated != signature:
            if run_start is not None:
                spans.append((run_start, beat))
            run_start, signature = beat, updated
    beats = ref_frame_to_beat
    if run_start is not None and run_start < beats[-1]:
        spans.append((run_start, float(beats[-1])))
    for start, end in spans:
        left, right = np.searchsorted(beats, [start, end])
        start_frame, end_frame = np.interp([start, end], beats, frames)
        variance[left:right] = (end_frame - start_frame) ** 2 / 12
    return variance
