#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Anchor-Point Tempo-Driven OLTW model (Rodriguez-Serrano et al., 2016).
"""

import unittest
from types import SimpleNamespace

import numpy as np

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame
from matchmaker.models.oltw_anchor_tempo import (
    AnchorTempoModel,
    OnlineTimeWarpingAnchorTempoFrame,
)
from matchmaker.registry import REGISTRY
from matchmaker.utils.tempo_models import TempoModel, make_tempo_model


class TestAnchorTempoModel(unittest.TestCase):
    def setUp(self):
        # Create synthetic 12-dimensional chroma-like features with 4 distinct blocks
        # Block 0: frames 0..19 (Chord 1: strong pitch 0, 4, 7)
        # Block 1: frames 20..39 (Chord 2: strong pitch 2, 5, 9)
        # Block 2: frames 40..59 (Chord 3: strong pitch 5, 9, 0)
        # Block 3: frames 60..79 (Chord 4: strong pitch 7, 11, 2)
        feat = np.zeros((80, 12), dtype=np.float32)
        feat[0:20, [0, 4, 7]] = 1.0
        feat[20:40, [2, 5, 9]] = 1.0
        feat[40:60, [5, 9, 0]] = 1.0
        feat[60:80, [7, 11, 2]] = 1.0
        self.features = feat
        self.ref_frame_to_beat = np.linspace(0.0, 16.0, 80)
        self.score_positions = np.arange(0.0, 16.0, 1.0)

    def test_anchor_detection(self):
        """Verify that transitions between orthogonal blocks are detected as anchors."""
        model = AnchorTempoModel(
            corr_threshold=0.70,
            min_anchor_distance=8,
        )
        fake_tracker = SimpleNamespace(
            reference_features=self.features,
            is_onset=None,
            step_size=3,
        )
        model.reset(fake_tracker)

        anchors = model.anchor_frames
        self.assertGreaterEqual(len(anchors), 4)
        self.assertEqual(anchors[0], 0)
        self.assertEqual(anchors[-1], 79)

        # Frame 20, 40, 60 are sharp transition boundaries
        for transition in [20, 40, 60]:
            # Should have an anchor at or immediately near transition
            self.assertTrue(
                any(abs(a - transition) <= 1 for a in anchors),
                f"Expected anchor near frame {transition}, got {anchors}",
            )

        # Check min_anchor_distance constraint
        diffs = np.diff(anchors)
        # Last element might be N-1 which could be < min_anchor_distance from previous, but interior ones respect it
        for d in diffs[:-1]:
            self.assertGreaterEqual(d, 8)

    def test_online_tempo_update(self):
        """Simulate fast and slow performance and verify EMA tempo convergence."""
        model = AnchorTempoModel(
            corr_threshold=0.70,
            alpha=0.30,
            min_anchor_distance=5,
            tempo_min=0.5,
            tempo_max=2.0,
        )
        fake_tracker = SimpleNamespace(
            reference_features=self.features,
            is_onset=None,
            step_size=3,
            _current_frame=0,
            input_index=0,
        )
        model.reset(fake_tracker)

        # Simulate fast performance: performer reaches score frame 40 in 20 audio frames (v = 2.0)
        anchors = model.anchor_frames
        self.assertGreaterEqual(len(anchors), 3)

        # Step tracker through frames
        for a in anchors[1:4]:
            fake_tracker._current_frame = int(a)
            # Fast: perf frames advance half as many as score frames
            fake_tracker.input_index = int(round(a * 0.5))
            model.before_step(fake_tracker)

        # Tempo should adapt to > 1.0 (faster)
        self.assertGreater(model.current_tempo, 1.0)
        self.assertLess(model.beat_period, 0.50)

        # Now simulate slow performance
        for a in anchors[3:]:
            fake_tracker._current_frame = int(a)
            # Slow: perf frames advance twice as many
            fake_tracker.input_index = int(round(a * 2.0))
            model.before_step(fake_tracker)

        # Tempo should adapt downwards
        self.assertLess(model.current_tempo, 1.5)

    def test_position_guidance_monotonicity(self):
        """Verify guide_position enforces monotonicity and blends smoothly."""
        model = AnchorTempoModel(position_blend=0.40)
        tracker = SimpleNamespace(
            reference_features=self.features,
            _current_frame=0,
            input_index=1,
            _frame_to_beat=lambda f: float(f) * 0.2,
            step_size=3,
        )
        model.reset(tracker)

        guided_positions = []
        raw_positions = [0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.8, 1.0]

        for i, raw in enumerate(raw_positions):
            tracker.input_index = i + 1
            tracker._current_frame = int(round(raw / 0.2))
            guided = model.guide_position(tracker, raw)
            guided_positions.append(guided)

        # Monotonicity check
        for j in range(1, len(guided_positions)):
            self.assertGreaterEqual(
                guided_positions[j],
                guided_positions[j - 1],
                "guide_position must be weakly monotonic",
            )

    def test_zero_score_alterations(self):
        """Verify that AnchorTempoModel never alters the score."""
        model = AnchorTempoModel()
        fake_tracker = SimpleNamespace(
            reference_features=self.features,
            step_size=3,
        )
        model.reset(fake_tracker)
        self.assertEqual(model.n_inserted, 0)
        self.assertEqual(model.n_deleted, 0)
        self.assertEqual(model.n_postponed, 0)

    def test_standalone_onset_tracking(self):
        """Verify update_beat_period with discrete onsets."""
        model = AnchorTempoModel(alpha=0.5, init_beat_period=0.50)
        model.reset()

        # Normal tempo: 1 beat every 0.5s
        bp, est = model(performed_onset=0.0, score_onset=0.0)
        self.assertEqual(bp, 0.50)

        # Faster tempo: 1 beat every 0.3s
        for i in range(1, 10):
            bp, est = model(performed_onset=i * 0.3, score_onset=float(i))

        # Relative tempo should be > 1.0 (faster)
        self.assertGreater(model.current_tempo, 1.0)
        self.assertLess(model.beat_period, 0.50)

    def test_make_tempo_model_alias(self):
        """Verify make_tempo_model accepts 'anchor' alias and AnchorTempoModel."""
        tm1 = make_tempo_model("anchor")
        self.assertIsInstance(tm1, AnchorTempoModel)

        tm2 = make_tempo_model(AnchorTempoModel, corr_threshold=0.65)
        self.assertIsInstance(tm2, AnchorTempoModel)
        self.assertEqual(tm2.corr_threshold, 0.65)

    def test_tracker_integration(self):
        """End-to-end integration test with OnlineTimeWarpingArztFrame."""
        oltw = OnlineTimeWarpingArztFrame(
            reference_features=self.features,
            score_positions=self.score_positions,
            ref_frame_to_beat=self.ref_frame_to_beat,
            window_size=5,
            step_size=3,
            frame_rate=50,
            tempo_model="anchor",
        )
        self.assertIsInstance(oltw.tempo_model, AnchorTempoModel)
        self.assertEqual(oltw.n_deleted, 0)
        self.assertEqual(oltw.n_inserted, 0)

        # Feed some observation frames
        positions = []
        for i in range(25):
            obs = self.features[min(i, len(self.features) - 1)]
            beat = oltw(obs, perf_time=i * 0.02)
            self.assertIsInstance(beat, float)
            self.assertGreaterEqual(beat, 0.0)
            positions.append(beat)

        self.assertEqual(len(positions), 25)
        self.assertGreaterEqual(oltw.current_relative_tempo, 0.5)

    def test_convenience_subclass(self):
        """Verify OnlineTimeWarpingAnchorTempoFrame initializes cleanly."""
        tracker = OnlineTimeWarpingAnchorTempoFrame(
            reference_features=self.features,
            score_positions=self.score_positions,
            ref_frame_to_beat=self.ref_frame_to_beat,
            window_size=5,
            step_size=3,
            frame_rate=50,
        )
        self.assertIsInstance(tracker.tempo_model, AnchorTempoModel)

        obs = self.features[0]
        beat = tracker(obs, perf_time=0.0)
        self.assertIsInstance(beat, float)

    def test_registry_lookup(self):
        """Verify arzt_anchor_tempo is properly registered and accessible."""
        self.assertTrue(issubclass(AnchorTempoModel, TempoModel))

        spec = REGISTRY.method("audio", "arzt_anchor_tempo")
        self.assertEqual(spec.cls_path, "matchmaker.dp:OnlineTimeWarpingArztFrame")
        self.assertIs(spec.default_kwargs.get("tempo_model"), AnchorTempoModel)
        self.assertEqual(spec.default_kwargs.get("processor"), "chroma")


if __name__ == "__main__":
    unittest.main()
