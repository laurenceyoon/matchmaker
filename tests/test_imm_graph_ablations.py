import numpy as np
import pytest

from matchmaker.prob.imm_graph import IMMGraphFollower


def follower(**kwargs):
    onsets = np.arange(12, dtype=float) / 5
    notes = np.zeros(12, dtype=[("onset_beat", "f8"), ("onset_quarter", "f8"),
                                ("duration_beat", "f8"), ("pitch", "i4")])
    notes["onset_beat"] = notes["onset_quarter"] = onsets
    notes["duration_beat"] = 0.2
    notes["pitch"] = np.arange(60, 72)
    reference = np.full((36, 12), 0.05)
    reference[np.arange(36), np.arange(36) // 3] = 1
    return IMMGraphFollower(reference_features=reference, score_positions=onsets,
        ref_frame_to_beat=np.arange(36) / 15, note_array=notes, frame_rate=30, **kwargs)


@pytest.mark.parametrize("modes", [("cv", "ca", "zv"), ("cv", "zv"), ("cv", "ca"), ("cv",)])
def test_every_ablation_keeps_a_finite_normalized_posterior(modes):
    sf = follower(modes=modes)
    for feature in sf.reference_features:
        sf.step(feature)
        assert np.isfinite(sf.get_current_position())
        assert sf.p.sum() == pytest.approx(1)
        np.testing.assert_allclose(sf.w.sum(axis=1), 1)
        assert np.all(np.isfinite(sf.x))
        assert np.min(np.linalg.eigvalsh(sf.P)) >= -1e-10
    assert sf.get_current_position() > 1.5
