"""Key validation and dev-only global scaling for N-CMAPSS evaluation."""
from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

PROTOCOL_VERSION = 'unsupervisedhi-ncmapss-v1'
REFERENCE_COMMIT = '157ebfae7baeee1e0cd52180f8e58ead3dcb448e'


def validate_keys(frame: pd.DataFrame, name: str) -> None:
    """Reject duplicate, missing, nonfinite, or fractional flight keys.

    Args:
        frame: Table containing unit and cycle columns.
        name: Diagnostic source name.
    Returns:
        None; raises ValueError for invalid keys.
    """
    keys = frame[['unit', 'cycle']].to_numpy(dtype=float)
    if not np.isfinite(keys).all() or not np.equal(keys, np.floor(keys)).all():
        raise ValueError(f'{name}: invalid unit/cycle keys')
    if frame.duplicated(['unit', 'cycle']).any():
        raise ValueError(f'{name}: duplicate unit/cycle keys')


def read_rul(h5_path: str, split: str) -> pd.DataFrame:
    """Read original Y and reject RUL changes within a flight before downsampling.

    Args:
        h5_path: Original N-CMAPSS file.
        split: Original dev or test split.
    Returns:
        Sorted unique unit/cycle/RUL rows.
    """
    with h5py.File(h5_path, 'r') as source:
        def array(key: str) -> np.ndarray:
            for candidate in [f'{key}_{split}', f'{split}/{key}', key]:
                if candidate in source:
                    return np.asarray(source[candidate])
            raise KeyError(f'missing {key} for {split}')
        a, y = array('A'), array('Y')
    if y.shape not in [(len(a),), (len(a), 1)]:
        raise ValueError('Y must contain one original RUL value per row')
    rows = pd.DataFrame({'unit': a[:, 0], 'cycle': a[:, 1], 'rul': y.ravel()})
    if not np.isfinite(rows.to_numpy()).all():
        raise ValueError('nonfinite original A keys or Y RUL')
    if not np.equal(rows[['unit','cycle']], np.floor(rows[['unit','cycle']])).all().all():
        raise ValueError('fractional original flight keys')
    grouped = rows.groupby(['unit', 'cycle'], sort=True).rul.agg(['min', 'max'])
    if (grouped['min'] != grouped['max']).any():
        raise ValueError('original Y RUL is not constant within flight')
    return grouped['min'].rename('rul').reset_index().astype({'unit': int, 'cycle': int})


def read_reference(path: str | None, keys: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate reference flight keys, documenting legacy order-derived cycle ids.

    Args:
        path: Reference HDF file; None explicitly disables HI labels.
        keys: Unique original flight keys.
    Returns:
        Keyed reference HI table and cycle-key provenance.
    """
    if path is None:
        return keys[['unit', 'cycle']].assign(hi_gt=np.nan), 'unavailable (explicit null or no file)'
    keys = keys.sort_values(['unit', 'cycle'])
    validate_keys(keys, 'original flights')
    ref = pd.read_hdf(path, key='df')
    if not {'unit', 'HI'} <= set(ref.columns):
        raise ValueError('reference requires unit and HI')
    if 'cycle' in ref:
        provenance = 'explicit unit/cycle columns'
    else:
        # Published cycle HI files have no cycle column. Refuse ambiguous order.
        if not ref.index.is_unique or not ref.index.is_monotonic_increasing:
            raise ValueError('legacy reference index must be unique and increasing')
        for unit, group in keys.groupby('unit'):
            if not np.array_equal(group.cycle.to_numpy(), np.arange(1, len(group) + 1)):
                raise ValueError('legacy reference requires contiguous cycles starting at 1')
        blocks = ref.unit[ref.unit.ne(ref.unit.shift())]
        if blocks.duplicated().any():
            raise ValueError('legacy reference has repeated unit blocks')
        ref = ref.copy()
        ref['cycle'] = ref.groupby('unit', sort=False).cumcount() + 1
        provenance = 'legacy per-unit row ordinal; index/order/count/contiguous raw cycles validated; semantic row order cannot be independently verified'
    ref = ref[['unit', 'cycle', 'HI']].rename(columns={'HI': 'hi_gt'})
    validate_keys(ref, 'reference')
    if not np.isfinite(ref.hi_gt).all():
        raise ValueError('nonfinite reference HI')
    return ref, provenance


def associate(results: dict[int, dict], rul: pd.DataFrame, reference: pd.DataFrame,
              allow_windows: bool = False) -> pd.DataFrame:
    """Join predictions, reference HI and original RUL one-to-one by flight keys.

    Args:
        results: Raw inference output, before any scaling.
        rul: Unique original Y flight labels.
        reference: Unique reference HI flight labels.
        allow_windows: Explicitly allow within-flight score means for window predictions.
    Returns:
        One sorted row per unit/cycle, with score/hi_gt/rul.
    """
    rows = [(unit, cycle, float(score)) for unit, result in results.items()
            for cycle, score in zip(result['cycle_ids'], result['hi_raw'], strict=True)]
    pred = pd.DataFrame(rows, columns=['unit', 'cycle', 'score'])
    if not np.isfinite(pred.score).all():
        raise ValueError('nonfinite prediction score')
    if allow_windows:
        pred = pred.groupby(['unit','cycle'], as_index=False).score.mean()
    for name, frame in [('prediction', pred), ('RUL', rul), ('reference', reference)]:
        validate_keys(frame, name)
    if not np.isfinite(rul.rul).all():
        raise ValueError('nonfinite RUL')
    if not reference.hi_gt.isna().all() and not np.isfinite(reference.hi_gt).all():
        raise ValueError('partial or nonfinite reference HI')
    joined = pred
    for labels in [rul, reference]:
        joined = joined.merge(labels, on=['unit','cycle'], how='outer', validate='one_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError('missing or extra flight keys in predictions/reference/RUL')
        joined = joined.drop(columns='_merge')
    return joined.sort_values(['unit','cycle']).reset_index(drop=True)


def scale_dev_test(dev: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit sklearn MinMaxScaler on final dev scores only and transform both splits.

    Args:
        dev: Unique final development scores.
        test: Unique final test scores, excluded from fitting.
    Returns:
        Scaled dev, scaled test, and fitted sklearn parameters/provenance.
    """
    for name, frame in [('dev',dev),('test',test)]:
        validate_keys(frame, name)
        if frame.empty or not np.isfinite(frame.score).all():
            raise ValueError(f'{name}: empty or nonfinite scores')
    scaler = MinMaxScaler(clip=False).fit(dev[['score']].to_numpy(dtype=float))
    scaled = []
    for split, frame in [('dev', dev), ('test', test)]:
        frame = frame.copy()
        frame['hi'] = scaler.transform(frame[['score']].to_numpy(dtype=float)).ravel()
        frame['split'] = split
        scaled.append(frame)
    meta = {k: getattr(scaler, k).tolist() for k in ['data_min_', 'data_max_', 'data_range_', 'scale_', 'min_']}
    meta.update(fit_split='dev', n_samples_seen_=int(scaler.n_samples_seen_),
                fit_keys=dev[['unit','cycle']].astype(int).values.tolist(),
                feature_range=[0,1], clip=False, sklearn_version=sklearn.__version__,
                constant_dev_output=bool(scaler.data_range_[0] == 0),
                constant_behavior='sklearn substitutes unit scale for zero range: equal dev scores map to 0; test transforms as score - dev_constant, without epsilon or clipping',
                labels_used=False, window_aggregation='none: one prediction per flight')
    return *scaled, meta
