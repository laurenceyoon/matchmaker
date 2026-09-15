#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Soft-OLTW score follower and Score-Informed IMM.
"""

import unittest
import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp.oltw_soft import (
    PositionTempoKalman,
    ScoreInformedIMM,
    SoftOnlineTimeWarping,
)
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(42)


class TestSoftOnlineTimeWarping(unittest.TestCase):
    def test_synthetic_sequence_alignment(self):
        X, Y, _ = generate_example_sequences(
            lenX=10,
            centers=3,
            n_features=3,
            maxreps=3,
            minreps=1,
            noise_scale=0.0,
            random_state=RNG,
            dtype=np.float32,
        )
        score_positions = np.arange(X.shape[0], dtype=np.float32)

        tracker = SoftOnlineTimeWarping(
            reference_features=X,
            score_positions=score_positions,
            ref_frame_to_beat=score_positions,
            window_size=3,
            step_size=1,
            frame_rate=1,
            use_imm=False,
        )

        positions = []
        for i, obs in enumerate(Y):
            beat = tracker(obs, float(i))
            positions.append(beat)
            self.assertIsInstance(beat, float)

        self.assertGreaterEqual(positions[-1], positions[0])
        self.assertEqual(len(tracker.alignment_path[0]), len(Y))

    def test_score_informed_imm_transitions(self):
        imm = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None)
        imm.onsets = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 3.0, 5.0])
        imm.iois = np.diff(imm.onsets)
        imm.fermata_ranges = [(4.0, 4.5)]

        # Fermata region -> high zero-velocity transition probability
        pi_fermata = imm.get_dynamic_PI(current_pos_frames=4.2)
        self.assertGreater(pi_fermata[0, 2], 0.8)

        # Rapid notes (IOI = 0.25 <= 0.5) -> high constant-velocity persistence
        pi_fast = imm.get_dynamic_PI(current_pos_frames=0.3)
        self.assertGreaterEqual(pi_fast[0, 0], 0.9)

        # IMM predict and update
        pred = imm.predict()
        self.assertEqual(pred.shape, (3,))
        updated = imm.update(1.0)
        self.assertEqual(updated.shape, (3,))
        self.assertAlmostEqual(np.sum(imm.mu), 1.0, places=5)

    def test_matchmaker_audio_integration(self):
        score_file = EXAMPLE_PIECES["simple_mozart"]["score"]
        audio_file = EXAMPLE_PIECES["simple_mozart"]["audio"]

        mm = Matchmaker(
            score_file=score_file,
            performance_file=audio_file,
            input_type="audio",
            method="oltw_soft",
            wait=False,
        )
        self.assertIsInstance(mm.score_follower, SoftOnlineTimeWarping)
        self.assertIsInstance(mm.score_follower.kalman, ScoreInformedIMM)

        gen = mm.run(verbose=False)
        first_beat = next(gen)
        second_beat = next(gen)
        gen.close()

        self.assertIsInstance(first_beat, float)
        self.assertIsInstance(second_beat, float)

    def test_imm_kinematic_mode_switching(self):
        """Verify that CV dominates in steady tempo, CA during accelerando, and ZV during pauses."""
        imm = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None, obs_var=1.0)
        
        # 1. Steady tempo: constant velocity v = 1.0
        p = 0.0
        for _ in range(30):
            p += 1.0
            imm.predict(is_silent=False)
            imm.update(p)
        self.assertGreater(imm.mu[0], 0.5, "CV should dominate in steady tempo")

        # 2. Strong acceleration: velocity increasing by 0.15 each step
        v = 1.0
        for _ in range(25):
            v += 0.15
            p += v
            imm.predict(is_silent=False)
            imm.update(p)
        self.assertGreater(imm.mu[1], 0.45, "CA should activate during accelerando")

        # 3. Musical pause: position fixed, silent
        for _ in range(20):
            imm.predict(is_silent=True)
            imm.update(p)
        self.assertGreater(imm.mu[2], 0.75, "ZV should dominate during pause")


if __name__ == "__main__":
    unittest.main()
