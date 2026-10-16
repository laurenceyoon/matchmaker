#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended unit tests for AnchorTempoModel integration into:
- OnlineTimeWarpingDixonFrame (audio)
- OnlineTimeWarpingArztEvent (midi)
- OnlineTimeWarpingDixonEvent (midi)
- PitchIOIHMM (midi)
- Methods registry lookups
"""

import unittest

import numpy as np

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztEvent
from matchmaker.dp.oltw_dixon import (
    OnlineTimeWarpingDixonEvent,
    OnlineTimeWarpingDixonFrame,
)
from matchmaker.models.oltw_anchor_tempo import AnchorTempoModel
from matchmaker.prob.hmm import PitchIOIHMM
from matchmaker.registry import REGISTRY


class TestAnchorTempoExtended(unittest.TestCase):
    def setUp(self):
        # 80 frames of 12-dim features
        feat = np.zeros((80, 12), dtype=np.float32)
        feat[0:20, [0, 4, 7]] = 1.0
        feat[20:40, [2, 5, 9]] = 1.0
        feat[40:60, [5, 9, 0]] = 1.0
        feat[60:80, [7, 11, 2]] = 1.0
        self.features = feat
        self.ref_frame_to_beat = np.linspace(0.0, 16.0, 80)
        self.score_positions = np.arange(0.0, 16.0, 1.0)

    def test_dixon_frame_with_anchor_tempo(self):
        """Test OnlineTimeWarpingDixonFrame with AnchorTempoModel."""
        dixon = OnlineTimeWarpingDixonFrame(
            reference_features=self.features,
            score_positions=self.score_positions,
            ref_frame_to_beat=self.ref_frame_to_beat,
            window_size=5,
            frame_rate=50,
            tempo_model="anchor",
        )
        self.assertIsInstance(dixon.tempo_model, AnchorTempoModel)

        # Feed some frames
        for i in range(20):
            obs = self.features[i]
            beat = dixon(obs, perf_time=i * 0.02)
            self.assertIsInstance(beat, float)
            self.assertGreaterEqual(beat, 0.0)

    def test_arzt_event_with_anchor_tempo(self):
        """Test OnlineTimeWarpingArztEvent with AnchorTempoModel."""
        event_features = np.zeros((16, 88), dtype=np.float32)
        for i in range(16):
            event_features[i, 40 + (i % 12)] = 1.0
        score_positions = np.arange(0.0, 16.0, 1.0)

        arzt_ev = OnlineTimeWarpingArztEvent(
            reference_features=event_features,
            score_positions=score_positions,
            window_size=4,
            step_size=3,
            tempo_model="anchor",
        )
        self.assertIsInstance(arzt_ev.tempo_model, AnchorTempoModel)

        for i in range(10):
            obs = event_features[i]
            beat = arzt_ev(obs, perf_time=i * 0.5)
            self.assertIsInstance(beat, float)
            self.assertGreaterEqual(beat, 0.0)

    def test_dixon_event_with_anchor_tempo(self):
        """Test OnlineTimeWarpingDixonEvent with AnchorTempoModel."""
        event_features = np.zeros((16, 88), dtype=np.float32)
        for i in range(16):
            event_features[i, 40 + (i % 12)] = 1.0
        score_positions = np.arange(0.0, 16.0, 1.0)

        dixon_ev = OnlineTimeWarpingDixonEvent(
            reference_features=event_features,
            score_positions=score_positions,
            window_size=10,
            tempo_model="anchor",
        )
        self.assertIsInstance(dixon_ev.tempo_model, AnchorTempoModel)

        for i in range(10):
            obs = event_features[i]
            beat = dixon_ev(obs, perf_time=i * 0.5)
            self.assertIsInstance(beat, float)
            self.assertGreaterEqual(beat, 0.0)

    def test_pitch_ioi_hmm_with_anchor_tempo(self):
        """Test PitchIOIHMM with AnchorTempoModel."""
        dtype = [
            ("pitch", "i4"),
            ("onset_beat", "f4"),
            ("duration_beat", "f4"),
        ]
        notes = np.zeros(8, dtype=dtype)
        notes["pitch"] = [60, 62, 64, 65, 67, 69, 71, 72]
        notes["onset_beat"] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        notes["duration_beat"] = [1.0] * 8

        hmm = PitchIOIHMM(
            reference_features=notes,
            tempo_model=AnchorTempoModel,
            piano_range=True,
        )
        self.assertIsInstance(hmm.tempo_model, AnchorTempoModel)

        for i, pitch in enumerate([60, 62, 64, 65]):
            obs = np.array([pitch - 21], dtype=int)
            pos = hmm(obs, perf_time=i * 0.5)
            self.assertIsInstance(pos, float)

    def test_registry_methods(self):
        """Verify that all new methods are registered in methods.yaml."""
        # Audio
        dixon_spec = REGISTRY.method("audio", "dixon_anchor_tempo")
        self.assertEqual(
            dixon_spec.cls_path, "matchmaker.dp:OnlineTimeWarpingDixonFrame"
        )
        self.assertIs(dixon_spec.default_kwargs.get("tempo_model"), AnchorTempoModel)

        # MIDI
        hmm_spec = REGISTRY.method("midi", "hmm_anchor_tempo")
        self.assertEqual(hmm_spec.cls_path, "matchmaker.prob:PitchIOIHMM")
        self.assertIs(hmm_spec.default_kwargs.get("tempo_model"), AnchorTempoModel)

        arzt_midi_spec = REGISTRY.method("midi", "arzt_anchor_tempo")
        self.assertEqual(
            arzt_midi_spec.cls_path, "matchmaker.dp:OnlineTimeWarpingArztEvent"
        )
        self.assertIs(
            arzt_midi_spec.default_kwargs.get("tempo_model"), AnchorTempoModel
        )

        dixon_midi_spec = REGISTRY.method("midi", "dixon_anchor_tempo")
        self.assertEqual(
            dixon_midi_spec.cls_path, "matchmaker.dp:OnlineTimeWarpingDixonEvent"
        )
        self.assertIs(
            dixon_midi_spec.default_kwargs.get("tempo_model"), AnchorTempoModel
        )


if __name__ == "__main__":
    unittest.main()
