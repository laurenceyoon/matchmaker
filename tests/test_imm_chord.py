import numpy as np
import pytest

from matchmaker.graph.score_graph import ScoreDirectedGraphBuilder
from matchmaker.prob.imm_chord import IMMChordFollower
from tests.test_imm_graph import FRAMES_PER_CHORD, _chord_pitches, _chroma, _repeat_part


def _follower(part, onset=False, **kwargs):
    notes = part.note_array()
    onsets = np.unique(notes["onset_beat"])
    reference = np.zeros((len(onsets) * FRAMES_PER_CHORD, 12 + onset), dtype=np.float32)
    for k, pitch in enumerate(_chord_pitches(notes)):
        reference[k * FRAMES_PER_CHORD:(k + 1) * FRAMES_PER_CHORD, pitch % 12] = 1.0
    beats = np.repeat(onsets, FRAMES_PER_CHORD) + np.tile(np.arange(FRAMES_PER_CHORD) / FRAMES_PER_CHORD, len(onsets))
    return IMMChordFollower(reference_features=reference, score_positions=onsets, frame_rate=30, ref_frame_to_beat=beats,
                            note_array=notes, score_graph=ScoreDirectedGraphBuilder().build(part), score_part=part, **kwargs)


@pytest.mark.parametrize("modes", [("steady",), ("steady", "hold"), ("steady", "maneuver", "hold")])
@pytest.mark.parametrize("onset", [False, True])
def test_follows_a_performance_through_the_repeat(modes, onset):
    part = _repeat_part()
    follower = _follower(part, onset=onset, modes=modes)
    pitches = [p % 12 for p in _chord_pitches(part.note_array())]
    played = list(range(8)) + list(range(12))
    positions = []
    for chord in played:
        for frame in range(FRAMES_PER_CHORD):
            chroma = _chroma(pitches[chord])
            # an attack is heard in the frame after each onset
            features = np.r_[chroma, 0.5 if frame == 1 else 0.02] if onset else chroma
            follower.step(features)
            positions.append(follower.current_index)
    positions = np.array(positions).reshape(len(played), FRAMES_PER_CHORD)
    assert positions[-1, -1] == 11
    assert np.mean(np.abs(positions[:, -1] - np.array(played)) <= 1) > 0.8
    assert follower.current_route == ("measure:1:2->measure:0:1",)
    top = np.argmax(follower.p)
    assert np.exp(follower.u[top]) == pytest.approx(2.0, rel=0.05)   # one-beat chords at 120 bpm
    assert np.all(np.isfinite(follower.P)) and np.all(follower.P >= 0)


def test_rejects_unknown_modes():
    with pytest.raises(ValueError):
        _follower(_repeat_part(), modes=("cv",))
