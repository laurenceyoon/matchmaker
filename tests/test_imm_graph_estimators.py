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


def test_readout_does_not_feed_back_into_inference():
    a = follower(position_estimator="map")
    b = follower(position_estimator="mean")
    for feature in np.repeat(a.reference_features, 2, axis=0):
        a.step(feature)
        b.step(feature)
        for field in ("k", "a", "r", "p", "x", "P", "w"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("estimator", ["map", "mean"])
def test_score_time_model_keeps_a_finite_normalized_posterior(estimator):
    sf = follower(position_estimator=estimator, duration_model="score_time")
    for feature in sf.reference_features:
        sf.step(feature)
        assert np.isfinite(sf.get_current_position())
        assert sf.p.sum() + sf.waiting == pytest.approx(1)
        np.testing.assert_allclose(sf.w.sum(axis=1), 1)
        assert np.all(np.isfinite(sf.x))
        assert np.min(np.linalg.eigvalsh(sf.P)) >= -1e-10
    assert sf.get_current_position() > 1.5


@pytest.mark.parametrize("kwargs", [{"position_estimator": "median"}, {"duration_model": "typo"}])
def test_invalid_model_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        follower(**kwargs)
