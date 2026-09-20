#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Soft-OLTW score follower and Score-Informed IMM.
"""

import unittest
import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp.oltw_soft import (
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
        imm.pause_ranges = [(4.0, 4.5)]

        # Pause regions permit ZV without forcing the score clock to stop.
        imm.reset(position=4.2, tempo=1.0)
        imm.predict()
        self.assertGreater(imm.M_pause[0, 2], 0.0)
        self.assertGreater(imm.M_pause[0, 0], imm.M_pause[0, 2])

        # IMM predict and update
        pred = imm.predict()
        self.assertEqual(pred.shape, (4,))
        updated = imm.update(4.2)
        self.assertEqual(updated.shape, (4,))
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
        self.assertGreater(imm.mu[1], 0.15, "CA should activate during accelerando")

        # 3. Musical pause: position fixed, silent
        for _ in range(20):
            imm.predict(is_silent=True)
            imm.update(p)
        self.assertEqual(int(np.argmax(imm.mu)), 2, "ZV should be the most likely pause model")


    def test_imm_large_innovation_preserves_likelihood_ranking(self):
        """A far-away observation must not collapse all likelihoods to a floor."""
        imm = ScoreInformedIMM(obs_var=1.0)
        imm.predict(is_silent=False)
        imm.states[0, 0] = 0.0
        imm.states[1, 0] = 20.0
        imm.P_matrices[:] = np.eye(len(imm.H))
        imm.update(100.0)
        self.assertGreater(imm.mu[1], 0.99)
        self.assertEqual(imm.mu[2], 0.0)
        for covariance in imm.P_matrices:
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), 0.0)

    def test_imm_pause_resume_far_from_origin(self):
        """An inactive pause mode must not pull a late pause toward frame zero."""
        imm = ScoreInformedIMM(init_position=1000.0)
        for p in range(1001, 1031):
            imm.predict(is_silent=False)
            imm.update(float(p))
        for _ in range(20):
            imm.predict(is_silent=True)
            imm.update(1030.0)
        self.assertLess(abs(imm.position - 1030.0), 1.0)
        self.assertEqual(int(np.argmax(imm.mu)), 2)
        for p in range(1031, 1051):
            imm.predict(is_silent=False)
            imm.update(float(p))
        self.assertLess(abs(imm.position - 1050.0), 1.0)
        self.assertEqual(imm.mu[2], 0.0)

    def test_filter_ablation_preserves_acoustic_path(self):
        """Turning feedback off isolates filtering from acoustic alignment."""
        reference = np.eye(12, dtype=np.float32)[np.arange(48) % 12]
        common = dict(reference_features=reference, frame_rate=10,
                      window_size=2, start_window_size=0.3)
        baseline = SoftOnlineTimeWarping(**common, use_imm=False)
        filtered = SoftOnlineTimeWarping(**common)
        positions = []
        for feature in np.repeat(reference, 2, axis=0):
            baseline.step(feature)
            filtered.step(feature)
            self.assertEqual(filtered._current_frame, baseline._current_frame)
            positions.append(filtered.get_current_position())
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(any(abs(p - round(p)) > 1e-6 for p in positions))

    def test_continuous_subframe_interpolation(self):
        """Verify that continuous float frames interpolate smoothly between discrete beat points."""
        r2b = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tracker = SoftOnlineTimeWarping(
            reference_features=np.zeros((5, 3), dtype=np.float32),
            score_positions=r2b,
            ref_frame_to_beat=r2b,
            window_size=3,
            step_size=1,
            frame_rate=10,
            use_imm=False,
        )
        b_half = tracker._frame_to_beat(1.5)
        self.assertAlmostEqual(b_half, 1.5, places=4)
        b_quarter = tracker._frame_to_beat(2.25)
        self.assertAlmostEqual(b_quarter, 2.25, places=4)


def test_baseline_constructs_no_kalman_models(monkeypatch):
    import matchmaker.dp.oltw_soft as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not construct Kalman models")

    monkeypatch.setattr(module, "ScoreInformedIMM", forbidden)
    monkeypatch.setattr(module, "KalmanPathLattice", forbidden)
    monkeypatch.setattr(module, "score_position_variance", forbidden)
    reference = np.eye(12, dtype=np.float32)
    follower = SoftOnlineTimeWarping(reference, use_imm=False, path_tempo=True)
    for feature in reference:
        follower.step(feature)
    follower.reset()
    assert follower.kalman is None
    assert follower.path_lattice is None


if __name__ == "__main__":
    unittest.main()
