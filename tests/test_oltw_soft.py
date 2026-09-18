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

    def test_beat_normalized_dynamics(self):
        """Verify that acceleration thresholds and pause stay probabilities scale with score tempo."""
        # 60 BPM -> 50 frames/beat, 120 BPM -> 25 frames/beat, 180 BPM -> 16.67 frames/beat
        imm60 = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None, tempo=60.0, frame_rate=50)
        imm120 = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None, tempo=120.0, frame_rate=50)
        imm180 = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None, tempo=180.0, frame_rate=50)

        self.assertAlmostEqual(imm60.frames_per_beat, 50.0, places=2)
        self.assertAlmostEqual(imm120.frames_per_beat, 25.0, places=2)
        self.assertAlmostEqual(imm180.frames_per_beat, 50.0 / 3.0, places=2)

        # Apply same physical acceleration: 30% tempo change per beat
        # For 60 BPM: a = 0.30 / 50 = 0.006
        # For 120 BPM: a = 0.30 / 25 = 0.012
        # For 180 BPM: a = 0.30 / 16.67 = 0.018
        imm60.states[1, 2] = 0.006
        imm120.states[1, 2] = 0.012
        imm180.states[1, 2] = 0.018

        pi60 = imm60.get_dynamic_PI(current_pos_frames=0.0)
        pi120 = imm120.get_dynamic_PI(current_pos_frames=0.0)
        pi180 = imm180.get_dynamic_PI(current_pos_frames=0.0)

        # Maneuver probability should be identical at baseline (0.05) across all tempos
        self.assertAlmostEqual(pi60[0, 1], pi120[0, 1], places=2)
        self.assertAlmostEqual(pi120[0, 1], pi180[0, 1], places=2)
        self.assertAlmostEqual(pi60[0, 1], 0.05, places=2)

        # Apply 60% tempo change per beat (delta_v_beat = 0.60): (0.60 - 0.30) / 0.625 = 0.48
        imm60.states[1, 2] = 0.60 / 50.0
        imm120.states[1, 2] = 0.60 / 25.0
        imm180.states[1, 2] = 0.60 / (50.0 / 3.0)

        pi60_acc = imm60.get_dynamic_PI(current_pos_frames=0.0)
        pi120_acc = imm120.get_dynamic_PI(current_pos_frames=0.0)
        pi180_acc = imm180.get_dynamic_PI(current_pos_frames=0.0)

        self.assertAlmostEqual(pi60_acc[0, 1], 0.48, places=2)
        self.assertAlmostEqual(pi120_acc[0, 1], 0.48, places=2)
        self.assertAlmostEqual(pi180_acc[0, 1], 0.48, places=2)

    def test_active_tempo_coupling_and_advance_rate(self):
        """Verify that SoftOnlineTimeWarping.step couples expected_tempo to kalman.tempo."""
        X, Y, _ = generate_example_sequences(
            lenX=20,
            centers=3,
            n_features=3,
            maxreps=2,
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
            window_size=5,
            step_size=2,
            frame_rate=10,
            use_imm=False,  # Uses PositionTempoKalman
        )

        # Artificially set kalman velocity to 1.4 (accelerando)
        tracker.kalman.state[1] = 1.4
        # Populate position history to simulate steady velocity of 1.4
        tracker._music_started = True
        tracker._current_frame = 14
        tracker._pos_history = [int(i * 1.4) for i in range(11)]

        tracker.step(Y[0])
        # Expected advance rate should reflect the kalman tempo (1.4)
        self.assertAlmostEqual(tracker.expected_advance_rate, 1.4, places=1)
        # Racing amount should not heavily penalize horizontal transitions
        self.assertLess(tracker.last_diagnostics.effective_horizontal_weight, 1.5)

    def test_tempo_injection_from_matchmaker(self):
        """Verify that Matchmaker passes score tempo to SoftOnlineTimeWarping and ScoreInformedIMM."""
        score_file = EXAMPLE_PIECES["simple_mozart"]["score"]
        audio_file = EXAMPLE_PIECES["simple_mozart"]["audio"]

        mm = Matchmaker(
            score_file=score_file,
            performance_file=audio_file,
            input_type="audio",
            method="oltw_soft",
            tempo=144.0,
            wait=False,
        )
        self.assertEqual(mm.tempo, 144.0)
        self.assertEqual(mm.score_follower.tempo, 144.0)
        self.assertEqual(mm.score_follower.kalman.tempo_bpm, 144.0)
        self.assertAlmostEqual(mm.score_follower.kalman.frames_per_beat, (mm.frame_rate * 60) / 144.0, places=2)

    def test_softmin_gibbs_observation(self):
        """Verify that step computes continuous Gibbs posterior mean and variance."""
        X, Y, _ = generate_example_sequences(
            lenX=25,
            centers=3,
            n_features=3,
            maxreps=2,
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
            window_size=5,
            step_size=2,
            frame_rate=10,
            use_imm=True,
        )
        tracker._music_started = True

        for obs in Y[:5]:
            tracker.step(obs)
            # Kalman position should be a continuous float
            self.assertIsInstance(tracker.kalman.position, float)
            # Kalman observation variance R should be finite and positive
            self.assertGreaterEqual(float(tracker.kalman.R[0, 0]), tracker._obs_var)

    def test_asymmetric_fermata_and_tempo_memory(self):
        """Verify that fermata triggers ZV even with active audio, and preserves tempo memory."""
        imm = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None, tempo=120.0, frame_rate=50)
        imm.fermata_ranges = [(4.0, 6.0)]
        imm.pause_ranges = [(4.0, 6.0)]
        imm.onsets = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
        imm.iois = np.diff(imm.onsets)

        # Steady playing at v = 1.3
        for i in range(20):
            imm.states[0, 1] = 1.3
            imm.predict(is_silent=False)
            imm.update(float(i * 1.3))

        self.assertGreater(imm.mu[0], 0.4)

        # Enter fermata at beat 4.5: even if is_silent is False (loud sustained chord), ZV must activate
        pi_fermata = imm.get_dynamic_PI(current_pos_frames=4.5, is_silent=False)
        self.assertGreater(pi_fermata[0, 2], 0.8, "Fermata with sounding audio must still trigger ZV")

        # Step into pause
        imm.predict(is_silent=False)
        self.assertAlmostEqual(imm.tempo_memory, 1.3, places=1, msg="Tempo memory must capture pre-pause velocity")

    def test_innovation_gating(self):
        """Verify that 4-sigma innovation gating prevents state explosion on extreme outlier observations."""
        imm = ScoreInformedIMM(score_part=None, ref_frame_to_beat=None, obs_var=1.0)
        p_init = 10.0
        imm.reset(position=p_init, tempo=1.0)
        imm.predict(is_silent=False)

        # Extreme outlier observation: z = 1000.0 (error of 990 frames)
        state_after = imm.update(1000.0)

        # Without gating, position would jump by hundreds of frames
        # With 4-sigma gating, position update is bounded
        self.assertLess(state_after[0], p_init + 50.0, "Innovation gating must prevent massive state jump")
        self.assertTrue(np.all(np.isfinite(state_after)))


if __name__ == "__main__":
    unittest.main()
