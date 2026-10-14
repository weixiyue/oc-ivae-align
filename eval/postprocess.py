"""Health indicator postprocessing utilities."""

import numpy as np

_EPS = 1e-8


def normalize_hi(
    hi_raw: np.ndarray,
) -> np.ndarray:
    """Per-unit min-max normalization to ``[0, 1]``."""
    hi = np.asarray(hi_raw, dtype=np.float32)
    return (hi - hi.min()) / (hi.max() - hi.min() + _EPS)


def smooth_hi(
    hi: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply causal EWMA smoothing to an HI sequence."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1")

    hi_array = np.asarray(hi, dtype=np.float32)
    if hi_array.size == 0:
        return hi_array.copy()

    smoothed = np.empty_like(hi_array, dtype=np.float32)
    smoothed[0] = hi_array[0]
    for index in range(1, hi_array.shape[0]):
        smoothed[index] = alpha * hi_array[index] + (1 - alpha) * smoothed[index - 1]

    return smoothed


def postprocess_unit(
    hi_raw: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """Apply normalize then smooth to one unit's HI sequence."""
    return smooth_hi(normalize_hi(hi_raw), alpha=alpha)
