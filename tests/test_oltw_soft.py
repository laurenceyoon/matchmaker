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
            method="soft_oltw",
            wait=False,
        )
        self.assertIs(type(mm.score_follower), SoftOnlineTimeWarping)
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


def test_only_two_soft_oltw_methods_are_registered():
    from matchmaker import AVAILABLE_METHODS

    assert {name for name in AVAILABLE_METHODS["audio"] if "soft" in name} == {
        "soft_oltw",
        "hierarchical_soft_oltw",
    }


def test_python_backend_matches_accelerated_acoustic_and_imm_paths(monkeypatch):
    import matchmaker.dp.oltw_soft as module
    from matchmaker.utils import distances

    rng = np.random.default_rng(17)
    features = rng.random((90, 12), dtype=np.float32) * 0.05
    features[np.arange(90), np.arange(90) // 6 % 12] += 1.0
    observations = features[np.repeat(np.arange(45), 2)]
    expected = {}
    for use_imm in (False, True):
        for gamma in (0.0, 0.05, 0.5):
            follower = SoftOnlineTimeWarping(features, gamma=gamma, use_imm=use_imm)
            positions = [follower(frame, i) for i, frame in enumerate(observations)]
            expected[use_imm, gamma] = positions, follower.path

    def forbidden(*args, **kwargs):
        raise AssertionError("Python backend must not call accelerated kernels")

    monkeypatch.setattr(module, "_weighted_soft_oltw_loop_numba", forbidden)
    monkeypatch.setattr(distances, "vdist", forbidden)
    monkeypatch.setattr(distances, "Manhattan", forbidden)
    for (use_imm, gamma), (positions, path) in expected.items():
        follower = SoftOnlineTimeWarping(
            features, gamma=gamma, backend="python", use_imm=use_imm
        )
        for _ in range(2):
            actual = [follower(frame, i) for i, frame in enumerate(observations)]
            np.testing.assert_allclose(actual, positions, atol=1e-10)
            np.testing.assert_allclose(follower.path.costs, path.costs, atol=1e-6)
            if use_imm:
                for attr in ("states", "covariances", "probabilities"):
                    np.testing.assert_array_equal(
                        getattr(follower.path, attr), getattr(path, attr)
                    )
            follower.reset()


def test_soft_follower_directly_inherits_online_alignment():
    from matchmaker.base import OnlineAlignment

    assert SoftOnlineTimeWarping.__bases__ == (OnlineAlignment,)
