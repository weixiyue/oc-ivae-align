"""UnsupervisedHI six-metric protocol (no postprocessing inside metrics)."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


def monotonicity(hi: np.ndarray) -> float:
    """Return absolute nonpositive/positive difference imbalance.

    Args:
        hi: One unit's cycle-sorted HI values.
    Returns:
        Reference monotonicity, or NaN if no consecutive pair exists.
    """
    diff = np.diff(np.asarray(hi, dtype=float))
    if not len(diff):
        return float('nan')
    return float(abs(np.count_nonzero(diff <= 0) - np.count_nonzero(diff > 0)) / len(diff))


def trendability(hi: np.ndarray, cycles: np.ndarray) -> float:
    """Return signed Spearman correlation with actual cycles; NaN becomes zero.

    Args:
        hi: One unit's HI values.
        cycles: Corresponding actual cycle ids.
    Returns:
        Signed unit correlation, before cross-unit averaging/absolute value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr = float(spearmanr(hi, cycles).statistic)
    return 0.0 if np.isnan(corr) else corr


def mae(hi_pred: np.ndarray, hi_gt: np.ndarray) -> float:
    """Return equal-cycle mean absolute error.

    Args:
        hi_pred: Predicted HI for each cycle.
        hi_gt: Corresponding reference HI.
    Returns:
        Mean absolute cycle error.
    """
    return float(np.mean(np.abs(np.asarray(hi_pred) - np.asarray(hi_gt))))


def mape(hi_pred: np.ndarray, hi_gt: np.ndarray) -> tuple[float, int]:
    """Return percent relative error and number of nonzero reference cycles.

    Args:
        hi_pred: Predicted HI for each cycle.
        hi_gt: Corresponding reference HI; zeros are excluded.
    Returns:
        MAPE percentage (NaN when no eligible labels) and valid count.
    """
    valid = np.asarray(hi_gt) != 0
    n = int(valid.sum())
    return (float(np.mean(np.abs((hi_gt[valid] - hi_pred[valid]) / hi_gt[valid])) * 100)
            if n else float('nan'), n)


def evaluate_all(frame: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    """Evaluate keyed, globally scaled cycles with equal-unit structural metrics.

    Args:
        frame: One split, columns unit/cycle/hi/hi_gt/rul, one row per flight.
        random_state: Reproducibility addition to reference MI defaults.
    Returns:
        Overall and per-unit metrics, counts, and diagnostic reasons; no FC per unit.
    """
    if frame.empty or frame.duplicated(['unit', 'cycle']).any():
        raise ValueError('metrics require nonempty unique flight keys')
    frame = frame.sort_values(['unit', 'cycle'])
    if not np.isfinite(frame[['unit', 'cycle', 'hi', 'rul']].to_numpy()).all():
        raise ValueError('nonfinite evaluation input')
    has_gt = not frame.hi_gt.isna().all()
    if has_gt and not np.isfinite(frame.hi_gt).all():
        raise ValueError('partial or nonfinite reference HI')
    per_unit: dict[str, Any] = {}
    starts, ends = [], []
    for unit, group in frame.groupby('unit', sort=True):
        hi, gt, rul = (group[k].to_numpy(dtype=float) for k in ['hi', 'hi_gt', 'rul'])
        issues: dict[str, str] = {}
        mono = monotonicity(hi)
        if len(hi) < 2:
            issues['monotonicity'] = 'fewer than two cycles'
        if np.ptp(hi) == 0:
            issues['constant_output'] = 'constant HI: monotonicity can equal 1 without degradation information'
        trend = trendability(hi, group.cycle.to_numpy())
        if np.ptp(hi) == 0 or len(hi) < 2:
            issues['trendability'] = 'NaN Spearman mapped to zero per reference'
        try:
            raw_mi = float(mutual_info_regression(hi[:, None], rul, random_state=random_state)[0])
            mi = float(1 - np.exp(-raw_mi))
        except ValueError as exc:
            raw_mi = mi = float('nan')
            issues['mi'] = str(exc)
        map_value, n_valid = mape(hi, gt) if has_gt else (float('nan'), 0)
        if not has_gt:
            issues.update(mae='reference HI unavailable', mape='reference HI unavailable')
        elif not n_valid:
            issues['mape'] = 'no nonzero reference HI'
        values = dict(mae=mae(hi, gt) if has_gt else float('nan'), mape=map_value,
                      monotonicity=mono, trendability=trend, mi=mi)
        for name, value in values.items():
            if not np.isfinite(value):
                issues.setdefault(name, 'reference metric formula is nonfinite')
        per_unit[str(int(unit))] = dict(values, mi_raw=raw_mi, n_cycles=len(hi),
                                        mape_valid_count=n_valid, issues=issues)
        starts.append(hi[0]); ends.append(hi[-1])
    denominator = float(np.mean(np.asarray(starts) - ends))
    end_std = float(np.std(ends, ddof=0))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        fc = float(np.exp(-np.divide(end_std, denominator)))
    issues = {}
    if denominator <= 0:
        issues['fc_denominator'] = 'mean(first HI - last HI) is nonpositive; FC is not evidence of effective degradation learning'
    if not np.isfinite(fc):
        issues['fc'] = 'reference FC formula is nonfinite'
    if np.ptp(frame.hi.to_numpy()) == 0:
        issues['constant_output'] = 'constant output is degenerate, regardless of high metric values'
    def unit_mean(name: str) -> float:
        values = [v[name] for v in per_unit.values()]
        if not np.isfinite(values).all():
            issues[name] = 'one or more unit metrics unavailable; no units silently excluded'
        return float(np.mean(values))
    map_value, n_valid = mape(frame.hi.to_numpy(), frame.hi_gt.to_numpy()) if has_gt else (float('nan'), 0)
    if not has_gt:
        issues.update(mae='reference HI unavailable', mape='reference HI unavailable')
    elif not n_valid:
        issues['mape'] = 'no nonzero reference HI'
    result = dict(mae=mae(frame.hi.to_numpy(), frame.hi_gt.to_numpy()) if has_gt else float('nan'),
                  mape=map_value, monotonicity=unit_mean('monotonicity'),
                  trendability=abs(unit_mean('trendability')), fc=fc, mi=unit_mean('mi'),
                  n_cycles=len(frame), n_units=len(per_unit), mape_valid_count=n_valid,
                  fc_denominator=denominator, fc_end_std=end_std, per_unit=per_unit, issues=issues)
    for name in ['mae', 'mape', 'monotonicity', 'trendability', 'fc', 'mi']:
        if not np.isfinite(result[name]):
            issues.setdefault(name, 'reference metric formula is nonfinite')
    return _finite_json(result)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
