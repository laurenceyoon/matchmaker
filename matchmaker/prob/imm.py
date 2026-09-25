import numpy as np


def score_activity(score_part, beats, size):
    if score_part is None or beats is None:
        return np.ones(size, dtype=bool)
    notes = score_part.note_array()
    starts = np.sort(notes["onset_beat"])
    ends = np.sort(notes["onset_beat"] + notes["duration_beat"])
    return np.searchsorted(starts, beats, side="right") > np.searchsorted(
        ends, beats, side="right"
    )


# ---------------------------------------------------------------------------
# Vectorised IMM primitives (Blom & Bar-Shalom): a batch of hypotheses, each with an
# n-dimensional state per mode. Checked against filterpy.kalman.IMMEstimator in the tests.
# ---------------------------------------------------------------------------

def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Stationary distribution of a Markov transition matrix (rows sum to one)."""
    values, vectors = np.linalg.eig(transition.T)
    pi = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    return pi / pi.sum()


def interact(mu: np.ndarray, transition: np.ndarray, x: np.ndarray, P: np.ndarray):
    """IMM interaction: predicted mode probabilities and each mode's mixed initial condition.

    mu: (hypotheses, modes); transition: (modes, modes) or one per hypothesis;
    x: (hypotheses, modes, n); P: (hypotheses, modes, n, n). Returns (c, x0, P0).
    """
    transition = np.broadcast_to(transition, (len(mu),) + transition.shape[-2:])
    c = np.einsum("hi,hij->hj", mu, transition)
    mix = mu[:, :, None] * transition / np.maximum(c[:, None, :], np.finfo(float).tiny)
    x0 = np.einsum("hij,hia->hja", mix, x)
    spread = x[:, :, None, :] - x0[:, None, :, :]
    P0 = np.einsum("hij,hiab->hjab", mix, P) + np.einsum("hij,hija,hijb->hjab", mix, spread, spread)
    return c, x0, P0


def kalman_update(x: np.ndarray, P: np.ndarray, residual: np.ndarray, H: np.ndarray, R: np.ndarray):
    """Kalman update by one scalar measurement, Joseph-form covariance.

    x, H: (..., n); P: (..., n, n); residual, R: (...).
    """
    PH = np.einsum("...ab,...b->...a", P, H)
    S = np.einsum("...a,...a->...", H, PH) + R
    K = PH / S[..., None]
    IKH = np.eye(x.shape[-1]) - K[..., :, None] * H[..., None, :]
    P = np.einsum("...ab,...bc,...dc->...ad", IKH, P, IKH) + R[..., None, None] * K[..., :, None] * K[..., None, :]
    return x + K * residual[..., None], P

