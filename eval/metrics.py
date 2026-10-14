"""Evaluation metrics for health indicator sequences."""

import numpy as np

from eval.postprocess import postprocess_unit

_EPS = 1e-8


def monotonicity(hi: np.ndarray) -> float:
    """Return decrease fraction minus increase fraction for consecutive pairs."""
    hi_array = np.asarray(hi, dtype=np.float32)
    if hi_array.shape[0] < 2:
        return 0.0

    diff = np.diff(hi_array)
    n_pairs = hi_array.shape[0] - 1
    decreases = np.sum(diff < 0)
    increases = np.sum(diff > 0)
    return float(decreases / n_pairs - increases / n_pairs)


def trendability(hi: np.ndarray) -> float:
    """Return absolute Pearson correlation between HI and time index."""
    hi_array = np.asarray(hi, dtype=np.float32)
    n_cycles = hi_array.shape[0]
    if n_cycles < 2:
        return 0.0

    time_index = np.arange(n_cycles, dtype=np.float32)
    hi_centered = hi_array - hi_array.mean()
    time_centered = time_index - time_index.mean()
    denominator = np.linalg.norm(hi_centered) * np.linalg.norm(time_centered)
    if denominator <= _EPS:
        return 0.0

    corr = np.sum(hi_centered * time_centered) / (denominator + _EPS)
    return float(abs(corr))


def prognosability(hi: np.ndarray) -> float:
    """Return the final HI value for one unit."""
    hi_array = np.asarray(hi, dtype=np.float32)
    if hi_array.shape[0] < 1:
        return 0.0

    return float(hi_array[-1])


def mae(
    hi_pred: np.ndarray,
    hi_gt: np.ndarray,
) -> float:
    """Mean Absolute Error between predicted and ground truth HI."""
    pred_array = np.asarray(hi_pred, dtype=np.float32)
    gt_array = np.asarray(hi_gt, dtype=np.float32)
    if pred_array.shape != gt_array.shape:
        raise ValueError("hi_pred and hi_gt must have the same shape")
    if pred_array.shape[0] < 1:
        return 0.0

    return float(np.abs(pred_array - gt_array).mean())


def evaluate_all(
    results: dict[int, dict],
    alpha: float = 0.3,
) -> dict:
    """Compute all HI metrics across all units.

    Args:
        results: Output of ``extract_hi``.
        alpha: EWMA smoothing parameter.

    Returns:
        Aggregate metrics and per-unit processed HI details.
    """
    per_unit: dict[int, dict] = {}
    monotonicity_values: list[float] = []
    trendability_values: list[float] = []
    mae_values: list[float] = []
    eol_values: list[float] = []

    for unit_id, unit_results in sorted(results.items()):
        hi_proc = postprocess_unit(unit_results["hi_raw"], alpha=alpha)
        hi_gt_raw = unit_results["hi_gt"]
        hi_gt = (
            None
            if hi_gt_raw is None
            else np.asarray(hi_gt_raw, dtype=np.float32)
        )
        mono = monotonicity(hi_proc)
        trend = trendability(hi_proc)
        if hi_gt is not None:
            mae_value = mae(hi_proc, hi_gt)
        else:
            mae_value = float("nan")
        eol_value = prognosability(hi_proc)

        monotonicity_values.append(mono)
        trendability_values.append(trend)
        if hi_gt is not None:
            mae_values.append(mae_value)
        if hi_proc.shape[0] >= 1:
            eol_values.append(eol_value)

        per_unit[int(unit_id)] = {
            "hi_proc": hi_proc,
            "hi_gt": hi_gt,
            "monotonicity": mono,
            "trendability": trend,
            "mae": mae_value,
            "eol_value": eol_value,
        }

    eol_array = np.asarray(eol_values, dtype=np.float32)
    if eol_array.size == 0:
        prog = 0.0
    else:
        prog = float(np.exp(-eol_array.std() / (eol_array.mean() + _EPS)))

    return {
        "monotonicity": _mean_or_zero(monotonicity_values),
        "trendability": _mean_or_zero(trendability_values),
        "prognosability": prog,
        "mae": _mean_or_nan(mae_values),
        "per_unit": per_unit,
    }


def _mean_or_zero(values: list[float]) -> float:
    """Return the mean of values, or zero for an empty list."""
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _mean_or_nan(values: list[float]) -> float:
    """Return the mean of values, or NaN for an empty list."""
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))
