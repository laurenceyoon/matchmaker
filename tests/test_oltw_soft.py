import unittest
import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp.oltw_soft import (
    SoftOnlineTimeWarping,
    softmin,
)
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(42)


def test_zero_gamma_selects_hard_minimum():
    assert softmin(3.0, 1.0, 2.0, 0.0) == 1.0
    assert softmin(np.inf, np.inf, np.inf, 0.0) == np.inf


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
        )

        positions = []
        for i, obs in enumerate(Y):
            beat = tracker(obs, float(i))
            positions.append(beat)
            self.assertIsInstance(beat, float)

        self.assertGreaterEqual(positions[-1], positions[0])
        self.assertEqual(len(tracker.alignment_path[0]), len(Y))

    def test_matchmaker_audio_integration(self):
        score_file = EXAMPLE_PIECES["simple_mozart"]["score"]
        audio_file = EXAMPLE_PIECES["simple_mozart"]["audio"]

        mm = Matchmaker(
            score_file=score_file,
            performance_file=audio_file,
            input_type="audio",
            method="softoltw",
            wait=False,
        )
        self.assertIsInstance(mm.score_follower, SoftOnlineTimeWarping)
        from matchmaker.dp.oltw_imm import IMMPathFilter

        self.assertIsInstance(mm.score_follower.path, IMMPathFilter)

        gen = mm.run(verbose=False)
        first_beat = next(gen)
        second_beat = next(gen)
        gen.close()

        self.assertIsInstance(first_beat, float)
        self.assertIsInstance(second_beat, float)

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
        )
        b_half = tracker._frame_to_beat(1.5)
        self.assertAlmostEqual(b_half, 1.5, places=4)
        b_quarter = tracker._frame_to_beat(2.25)
        self.assertAlmostEqual(b_quarter, 2.25, places=4)
