import numpy as np
import partitura as pt
import pytest

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.graph.score_graph import ScoreDirectedGraphBuilder
from matchmaker.prob.imm_graph import ZV, IMMGraphFollower

FRAME_RATE = 30
FRAMES_PER_CHORD = 15  # one-beat chords at 120 bpm


def _repeat_part():
    part = pt.score.Part("P1", quarter_duration=1)
    for i in range(3):
        part.add(pt.score.Measure(number=i + 1), i * 4, (i + 1) * 4)
    part.add(pt.score.Repeat(), 0, 8)
    for i in range(12):
        part.add(pt.score.Note(id=f"n{i}", step="CDEFGAB"[i % 7], octave=4), i, i + 1)
    return part


def _chord_pitches(note_array):
    order = np.argsort(note_array["onset_beat"], kind="stable")
    return [int(p) for p in note_array["pitch"][order]]


def _follower(part, **kwargs):
    note_array = part.note_array()
    onsets = np.unique(note_array["onset_beat"])
    chroma = np.zeros((len(onsets) * FRAMES_PER_CHORD, 12), dtype=np.float32)
    for k, pitch in enumerate(_chord_pitches(note_array)):
        chroma[k * FRAMES_PER_CHORD:(k + 1) * FRAMES_PER_CHORD, pitch % 12] = 1.0
    beats = np.repeat(onsets, FRAMES_PER_CHORD) + np.tile(np.arange(FRAMES_PER_CHORD) / FRAMES_PER_CHORD, len(onsets))
    return IMMGraphFollower(
        reference_features=chroma, score_positions=onsets, frame_rate=FRAME_RATE,
        ref_frame_to_beat=beats, note_array=note_array,
        score_graph=ScoreDirectedGraphBuilder().build(part), score_part=part, **kwargs,
    )


def _chroma(pitch_class):
    frame = np.zeros(12, dtype=np.float32)
    frame[pitch_class] = 1.0
    return frame


def test_chord_jumps_map_repeat_to_first_chord_of_target():
    follower = _follower(_repeat_part())
    assert follower.K == 12
    linear, targets = follower.jumps[7]
    assert set(follower.jumps) == {7}
    assert linear == pytest.approx(0.5)
    assert [(t, s) for t, s, _ in targets] == [(0, pytest.approx(0.5))]


def test_repeat_jump_splits_advance_by_graph_prior(monkeypatch):
    follower = _follower(_repeat_part(), modes=("cv",))
    monkeypatch.setattr(follower, "_log_frame_likelihoods", lambda features: np.zeros(len(follower.log_reference) + 1))
    follower.step(_chroma(0))  # enter the first chord, then place its hypothesis at the measure end
    follower.waiting = 0.0
    follower.k, follower.a, follower.r = follower.k[:1] * 0 + 7, follower.a[:1] * 0 + 10_000, follower.r[:1]
    follower.p, follower.x, follower.P, follower.w = np.ones(1), follower.x[:1], follower.P[:1], follower.w[:1]
    follower.step(_chroma(0))
    chords = dict(zip(follower.k.tolist(), follower.p.tolist()))
    assert chords[0] == pytest.approx(chords[8])
    jumped = follower.r[follower.k == 0][0]
    assert follower.routes[jumped] == ("measure:1:2->measure:0:1",)


def test_follows_synthetic_performance_through_the_repeat():
    part = _repeat_part()
    follower = _follower(part)
    pitches = [p % 12 for p in _chord_pitches(part.note_array())]
    played = list(range(8)) + list(range(12))  # first section repeated, then to the end
    positions = []
    for chord in played:
        for _ in range(FRAMES_PER_CHORD):
            follower.step(_chroma(pitches[chord]))
            positions.append(follower.current_index)
    positions = np.array(positions).reshape(len(played), FRAMES_PER_CHORD)
    assert positions[-1, -1] == 11
    assert np.mean(np.abs(positions[:, -1] - np.array(played)) <= 1) > 0.8
    assert follower.current_route == ("measure:1:2->measure:0:1",)


def test_silence_before_the_music_starts_is_ignored():
    follower = _follower(_repeat_part())
    for _ in range(400):  # 13 s of flat (silent) chroma against one-beat chords
        follower.step(np.full(12, 1 / 12, dtype=np.float32))
    assert follower.current_index == 0
    assert follower.waiting > 0.5
    assert follower.get_current_position() == pytest.approx(follower.onset_beats[0], abs=0.5)


def test_held_chord_stays_while_its_sound_continues():
    part = _repeat_part()
    part.remove(next(n for n in part.iter_all(pt.score.Note) if n.id == "n0"))
    part.add(pt.score.Note(id="n0", step="C", octave=4), 0, 0.5)  # a rest follows the first chord
    with_hold = _follower(part)
    without_hold = _follower(part, modes=("cv", "ca"))
    for follower in (with_hold, without_hold):
        follower.log_rest = np.full((1, 12), -np.log(12.0))  # a rendered rest is silence, not the held pitch
        for _ in range(400):
            follower.step(_chroma(0))
    assert with_hold.current_index == 0
    assert with_hold.w[np.argmax(with_hold.p)][ZV] > 0.5
    assert without_hold.current_index >= with_hold.current_index


def test_example_piece_runs_to_the_end_without_backward_jumps():
    piece = EXAMPLE_PIECES["simple_mozart"]
    mm = Matchmaker(score_file=piece["score"], performance_file=piece["audio"],
                    input_type="audio", method="imm_graph", wait=False)
    for _ in mm.run(verbose=False):
        pass
    path = mm.score_follower.alignment_path
    assert path[1, -1] == pytest.approx(mm.score_follower.onset_beats[-1])
    assert np.all(np.diff(path[1]) > -1.0)
