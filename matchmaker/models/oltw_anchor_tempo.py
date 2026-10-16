#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Anchor-Point Tempo-Driven Online Dynamic Time Warping (OLTW).

Based on:
    F. J. Rodriguez-Serrano, J. J. Carabias-Orti, P. Vera-Candeas, D. Martinez-Munoz (2016),
    "Tempo Driven Audio-to-Score Alignment Using Spectral Decomposition and
    Online Dynamic Time Warping", ACM Transactions on Intelligent Systems and
    Technology (TIST), 8(2), Article 22.

Key Innovations:
1. Anchor Points: Score locations where consecutive spectral/chroma representations
   have normalized cross-correlation / cosine similarity below a threshold (cc < Th),
   identifying unambiguous harmonic / chord / note transitions.
2. Interpretation Speed Tracking: When the online tracker crosses each anchor point i,
   the ratio of elapsed score frames (Δτ) to performance audio frames (Δt) provides
   an observation of interpretation speed v_interp = Δτ / Δt, smoothed via EMA:
       v_estim = alpha * v_interp + (1 - alpha) * v_estim
3. Zero Score Alteration: Unlike Arzt & Widmer (2010), score feature arrays and cost
   matrices are never mutated at runtime. Instead, estimated tempo guides step
   progression and smooths intra-anchor trajectories, eliminating stochastic jumps,
   reducing latency, and improving SPARC smoothness.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame
from matchmaker.utils.tempo_models import TempoModel


class AnchorTempoModel(TempoModel):
    """
    Anchor-Point Tempo Model of Rodriguez-Serrano et al. (ACM TIST 2016).

    Parameters
    ----------
    corr_threshold : float
        Correlation threshold Th below which an onset is selected as an anchor
        point (default: 0.70).
    alpha : float
        Exponential moving average smoothing factor alpha_v for tempo updates
        (default: 0.10, as recommended in Sec. 4.4).
    min_anchor_distance : int
        Minimum score frames between successive anchor points to prevent
        micro-clustering (default: 8).
    tempo_min : float
        Lower bound on relative tempo factor (default: 0.50).
    tempo_max : float
        Upper bound on relative tempo factor (default: 2.00).
    position_blend : float
        Weight for intra-anchor tempo-projected position blending in [0, 1]
        (default: 0.35). 0.0 uses raw DTW position; higher values produce
        smoother trajectories (better SPARC) by mitigating staircase quantization.
    adaptive_step : bool
        If True, dynamically scales tracker's step_size proportional to the
        current estimated tempo (default: True).
    init_beat_period : float
        Initial tempo in seconds per beat (default: 0.50).
    init_score_onset : float
        Initial score onset time in beats (default: 0.0).
    score_onsets : Optional[np.ndarray]
        Score onsets array in beats.
    """

    def __init__(
        self,
        corr_threshold: float = 0.70,
        alpha: float = 0.10,
        min_anchor_distance: int = 8,
        tempo_min: float = 0.50,
        tempo_max: float = 2.00,
        position_blend: float = 0.35,
        adaptive_step: bool = True,
        init_beat_period: float = 0.50,
        init_score_onset: float = 0.0,
        score_onsets: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
            score_onsets=score_onsets,
        )
        self.corr_threshold = float(corr_threshold)
        self.alpha = float(alpha)
        self.min_anchor_distance = int(min_anchor_distance)
        self.tempo_min = float(tempo_min)
        self.tempo_max = float(tempo_max)
        self.position_blend = float(position_blend)
        self.adaptive_step = bool(adaptive_step)

        self._current_tempo: float = 1.0
        self.anchor_frames: NDArray[np.int64] = np.array([0], dtype=np.int64)
        self._anchor_idx: int = 0
        self._last_anchor_score_frame: int = 0
        self._last_anchor_perf_frame: int = 0
        self._base_step_size: int = 3
        self._last_guided_beat: float = 0.0

        # Zero-alteration statistics (always 0, for interface parity with ArztTempoModel)
        self.n_inserted: int = 0
        self.n_deleted: int = 0
        self.n_postponed: int = 0

        # Standalone onset history
        self._history_perf: List[float] = []
        self._history_score: List[float] = []

    @property
    def current_tempo(self) -> float:
        """Current relative tempo (performance speed / score nominal speed)."""
        return self._current_tempo

    def reset(self, tracker=None) -> None:
        """Reset the model state and precompute anchor points from score features."""
        self.counter = 0
        self.prev_perf_onset = None
        self.est_onset = None
        self.asynchrony = 0.0
        self.beat_period = getattr(self, "init_beat_period", 0.50)
        self._current_tempo = 1.0
        self._anchor_idx = 0
        self._last_anchor_score_frame = 0
        self._last_anchor_perf_frame = 0
        self._last_guided_beat = 0.0
        self._history_perf = []
        self._history_score = []

        if tracker is not None:
            self._base_step_size = getattr(tracker, "step_size", 3)
            self._precompute_anchors(tracker)

    def _precompute_anchors(self, tracker) -> None:
        """
        Identify anchor points from reference features via cosine similarity drops.

        Eq. 6-7:
            cc(m) = <bs_m, bs_{m-1}> / (||bs_m|| * ||bs_{m-1}||)
        Select state/frame onsets where cc(m) < Th.
        """
        features = getattr(tracker, "reference_features", None)
        if features is None or len(features) < 2:
            self.anchor_frames = np.array([0], dtype=np.int64)
            return

        N = len(features)
        V = np.asarray(features, dtype=np.float32)

        # Norms
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        V_norm = V / norms

        # Cosine similarity between consecutive frames
        cos_sim = np.sum(V_norm[1:] * V_norm[:-1], axis=1)

        # Candidate anchor frames where similarity drops below threshold
        drops = np.where(cos_sim < self.corr_threshold)[0] + 1

        # Also consider score onsets if available
        is_onset = getattr(tracker, "is_onset", None)
        if is_onset is not None and np.any(is_onset):
            onset_indices = np.where(is_onset)[0]
            # Keep drops that align close to onset frames or strong drops
            combined = np.unique(np.concatenate(([0], drops, onset_indices)))
        else:
            combined = np.unique(np.concatenate(([0], drops)))

        # Enforce minimum anchor distance to avoid micro-clustering
        filtered_anchors = [0]
        for idx in combined:
            if idx - filtered_anchors[-1] >= self.min_anchor_distance and idx < N:
                filtered_anchors.append(int(idx))

        if filtered_anchors[-1] != N - 1:
            filtered_anchors.append(N - 1)

        self.anchor_frames = np.array(filtered_anchors, dtype=np.int64)
        self._anchor_idx = 0

    def before_step(self, tracker) -> None:
        """
        Called before each tracker step: check anchor crossings and update tempo.
        """
        current_score_frame = getattr(tracker, "_current_frame", 0)
        perf_frame = getattr(tracker, "input_index", 0)

        # Check if the tracker has crossed one or more anchor points
        while (
            self._anchor_idx < len(self.anchor_frames)
            and current_score_frame >= self.anchor_frames[self._anchor_idx]
        ):
            anchor_score_frame = int(self.anchor_frames[self._anchor_idx])

            if self._anchor_idx > 0:
                delta_tau = anchor_score_frame - self._last_anchor_score_frame
                delta_t = perf_frame - self._last_anchor_perf_frame

                if delta_t > 0 and delta_tau > 0:
                    # Eq. 8: v_interp = Δτ / Δt
                    v_interp = float(delta_tau) / float(delta_t)

                    # Outlier rejection / confidence interval clipping
                    v_interp = max(self.tempo_min, min(self.tempo_max, v_interp))

                    # Eq. 9: EMA update
                    self._current_tempo = (
                        self.alpha * v_interp + (1.0 - self.alpha) * self._current_tempo
                    )
                    base_bp = getattr(self, "init_beat_period", 0.50)
                    self.beat_period = base_bp / max(1e-4, self._current_tempo)

                self._last_anchor_score_frame = anchor_score_frame
                self._last_anchor_perf_frame = perf_frame

            self._anchor_idx += 1

        # Adaptive step size scaling
        if self.adaptive_step and hasattr(tracker, "step_size"):
            new_step = int(np.round(self._base_step_size * self._current_tempo))
            tracker.step_size = max(1, min(new_step, self._base_step_size * 2))

    def guide_position(self, tracker, raw_beat: float) -> float:
        """
        Guide and smooth tracker position between anchor points.

        Mitigates DTW staircase quantization artifacts and path jitter by blending
        raw DTW beat with the intra-anchor linear tempo projection.

        Parameters
        ----------
        tracker : OnlineTimeWarpingArztFrame
            The active tracker instance.
        raw_beat : float
            The discrete beat corresponding to tracker._current_frame.

        Returns
        -------
        float
            Smoothed, monotonic guided beat.
        """
        if self.position_blend <= 0.0 or tracker.input_index < 2:
            self._last_guided_beat = max(self._last_guided_beat, raw_beat)
            return self._last_guided_beat

        perf_frame = tracker.input_index - 1
        delta_t = perf_frame - self._last_anchor_perf_frame

        if delta_t >= 0 and self._current_tempo > 0:
            pred_score_frame = (
                self._last_anchor_score_frame + delta_t * self._current_tempo
            )
            pred_score_frame = min(
                pred_score_frame, len(tracker.reference_features) - 1
            )
            pred_beat = tracker._frame_to_beat(int(round(pred_score_frame)))

            # If prediction deviates excessively from DTW (> 2.0 beats), trust DTW to avoid desync
            if abs(pred_beat - raw_beat) <= 2.0:
                blended = (
                    1.0 - self.position_blend
                ) * raw_beat + self.position_blend * pred_beat
            else:
                blended = raw_beat
        else:
            blended = raw_beat

        # Enforce weak monotonicity to prevent backward jitter
        guided = max(self._last_guided_beat, blended)
        self._last_guided_beat = guided
        return float(guided)

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Standalone onset update: update tempo using recent onset IOI ratio with EMA.
        """
        self.est_onset = performed_onset
        self._history_perf.append(performed_onset)
        self._history_score.append(score_onset)
        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset

        if len(self._history_perf) < 2:
            return

        dp = self._history_perf[-1] - self._history_perf[-2]
        ds = self._history_score[-1] - self._history_score[-2]

        if dp > 0 and ds > 0:
            raw_ratio = ds / dp  # score beats per perf second
            base_bp = getattr(self, "init_beat_period", 0.50)
            nominal_ratio = 1.0 / base_bp
            v_interp = raw_ratio / nominal_ratio

            v_interp = max(self.tempo_min, min(self.tempo_max, v_interp))
            self._current_tempo = (
                self.alpha * v_interp + (1.0 - self.alpha) * self._current_tempo
            )
            self.beat_period = base_bp / max(1e-4, self._current_tempo)


class OnlineTimeWarpingAnchorTempoFrame(OnlineTimeWarpingArztFrame):
    """
    Frame-level Online Time Warping follower equipped with AnchorTempoModel.

    Convenience wrapper around OnlineTimeWarpingArztFrame with default
    tempo_model=AnchorTempoModel.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        ref_frame_to_beat: NDArray,
        corr_threshold: float = 0.70,
        alpha: float = 0.10,
        position_blend: float = 0.35,
        adaptive_step: bool = True,
        tempo_min: float = 0.50,
        tempo_max: float = 2.00,
        **kwargs,
    ) -> None:
        tempo_model = AnchorTempoModel(
            corr_threshold=corr_threshold,
            alpha=alpha,
            position_blend=position_blend,
            adaptive_step=adaptive_step,
            tempo_min=tempo_min,
            tempo_max=tempo_max,
        )
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            ref_frame_to_beat=ref_frame_to_beat,
            tempo_model=tempo_model,
            **kwargs,
        )
