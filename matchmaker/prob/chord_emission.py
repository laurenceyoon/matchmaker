"""Pool rendered-frame acoustic evidence into chord-state likelihoods."""

import numpy as np


class ChordFramePool:
    """Marginalize an unknown acoustic phase within each score chord.

    Each frame belongs to its sounding chord. A chord shorter than the rendering
    frame interval additionally owns the frame at its onset, so it cannot have
    zero likelihood simply because the rendering did not sample its interior.
    """

    def __init__(self, frame_beats, onset_beats):
        self.size = len(onset_beats)
        chord = np.clip(np.searchsorted(onset_beats, frame_beats, side="right") - 1,
                        0, self.size - 1)
        onset_frame = np.clip(np.searchsorted(frame_beats, onset_beats), 0, len(frame_beats) - 1)
        pairs = np.unique(np.concatenate([
            np.column_stack([np.arange(len(frame_beats)), chord]),
            np.column_stack([onset_frame, np.arange(self.size)]),
        ]), axis=0)
        self.frames, self.chords = pairs.T
        self.counts = np.bincount(self.chords, minlength=self.size)

    def __call__(self, log_frame_likelihoods):
        values = log_frame_likelihoods[self.frames]
        maximum = np.full(self.size, -np.inf)
        np.maximum.at(maximum, self.chords, values)
        total = np.bincount(self.chords, np.exp(values - maximum[self.chords]),
                            minlength=self.size)
        return maximum + np.log(total / self.counts)
