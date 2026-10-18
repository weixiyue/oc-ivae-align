"""Frozen, keyed cycle representations and train-device-only linear HI probes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from dataset import NCMAPSSDataset
from evaluate import validate_keys
from models.oc_ivae import OC_iVAE
from models.ocald import OCALD, to_valid_prefix
from models.proposal_cnn import ProposalCNN


def extract_representations(model: torch.nn.Module, dataset: NCMAPSSDataset,
                            device: str, batch_size: int) -> pd.DataFrame:
    """Return complete health means/Z with unit/cycle keys; never sample or apply PCA.

    The table interface is cycle-level. A future temporal model must return its
    aggregated cycle vectors through this same keyed boundary, not extra labels.
    Current models already emit one vector per flight; duplicates are errors.
    """
    model.requires_grad_(False).to(device).eval()
    keys, vectors = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for x, mask, _, w, units, cycles, _ in loader:
            x, mask = x.to(device), mask.to(device)
            if isinstance(model, OCALD):
                x, _, mask = to_valid_prefix(x, w.to(device), mask)
                vector = model.encode(x, mask)[0]  # mu_I only, no condition block
            elif isinstance(model, ProposalCNN):
                vector = model.encode(x)[:, None]
            elif isinstance(model, OC_iVAE):
                vector = model.zd_branch(model.backbone(x, mask), mask)[0]  # mu_d
            else:
                raise ValueError('linear probes support OC-iVAE, OCALD and Proposal-C only')
            if vector.ndim != 2 or not torch.isfinite(vector).all():
                raise ValueError('expected one finite representation vector per flight')
            keys.extend(zip(units.tolist(), cycles.tolist(), strict=True))
            vectors.append(vector.cpu().numpy())
    values = np.concatenate(vectors).astype(np.float64)
    frame = pd.DataFrame(keys, columns=['unit', 'cycle'])
    validate_keys(frame, 'probe representations')
    frame[[f'representation_{i}' for i in range(values.shape[1])]] = values
    return frame.sort_values(['unit', 'cycle']).reset_index(drop=True)


def resolve_probe_split(config: dict, checkpoint: dict, dev_units: list[int],
                        test_units: list[int], explicit: dict | None = None) -> dict:
    """Use saved train/val units or a recorded split seed plus training val_split.

    Explicit YAML can supply a missing partition, but cannot contradict recorded
    provenance. No default seed or validation fraction is invented for probing.
    """
    candidates, partial = [], []
    for name, metadata in [('checkpoint.run_metadata', checkpoint.get('run_metadata', {})),
                           ('config.run_metadata', config.get('run_metadata', {}))]:
        if 'train_units' in metadata and 'val_units' in metadata:
            candidates.append((name, dict(fit_units=metadata['train_units'], val_units=metadata['val_units'])))
        else:
            for saved_key, probe_key in [('train_units', 'fit_units'), ('val_units', 'val_units')]:
                if saved_key in metadata:
                    partial.append((name, probe_key, metadata[saved_key]))
    seeds = [value for value in (
        checkpoint.get('run_metadata', {}).get('split_seed'),
        config.get('run_metadata', {}).get('split_seed'), config.get('training', {}).get('split_seed'))
        if value is not None]
    if seeds and 'val_split' in config['data']:
        if len(set(seeds)) != 1 or not isinstance(seeds[0], int) or isinstance(seeds[0], bool):
            raise ValueError('conflicting or invalid recorded split seeds')
        fraction = config['data']['val_split']
        if not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
            raise ValueError('invalid recorded data.val_split')
        split = NCMAPSSDataset._split_units(sorted(dev_units), fraction, seeds[0])
        candidates.append(('recorded split_seed + data.val_split',
                           dict(fit_units=sorted(split['train']), val_units=sorted(split['val']))))
    if explicit is not None:
        candidates.append(('explicit split YAML', explicit))
    if not candidates:
        raise ValueError('training split provenance missing; provide --split-config YAML with fit_units and val_units')
    resolved = None
    for source, values in candidates:
        groups = {}
        for name in ('fit_units', 'val_units'):
            units = values[name]
            if (not isinstance(units, list) or any(not isinstance(u, int) or isinstance(u, bool) for u in units)
                    or len(set(units)) != len(units)):
                raise ValueError(f'{source}: {name} must be a list of unique integer unit ids')
            groups[name] = sorted(units)
        if resolved is not None and groups != resolved:
            raise ValueError('probe split configuration contradicts recorded training split')
        resolved = groups
    for source, name, units in partial:
        if not isinstance(units, list) or sorted(units) != resolved[name]:
            raise ValueError(f'{source}: partial unit metadata contradicts the resolved training split')
    fit, val, test = (set(resolved['fit_units']), set(resolved['val_units']), set(test_units))
    if not fit or not test or fit & val or (fit | val) != set(dev_units) or (fit | val) & test:
        raise ValueError('fit/val must partition dev, with nonempty fit and disjoint original test units')
    return dict(**resolved, test_units=sorted(test), provenance=[name for name, _ in candidates])


def join_probe_features(representations: pd.DataFrame, labelled_hi: pd.DataFrame) -> pd.DataFrame:
    """One-to-one key join, rejecting missing coverage and nonfinite labels/features."""
    validate_keys(representations, 'probe representations')
    validate_keys(labelled_hi, 'labelled public HI')
    frame = labelled_hi.merge(representations, on=['unit', 'cycle'], how='outer',
                              validate='one_to_one', indicator=True)
    if not frame['_merge'].eq('both').all():
        raise ValueError('representation/reference flight-key coverage mismatch')
    frame = frame.drop(columns='_merge')
    columns = ['hi_gt', 'hi'] + [c for c in frame if c.startswith('representation_')]
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('linear probe requires finite reference HI and features for every cycle')
    return frame.sort_values(['unit', 'cycle']).reset_index(drop=True)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    """Pooled per-cycle R², MAE and RMSE; do not coerce undefined R² to 0 or 1."""
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 1 or not np.isfinite([target, prediction]).all():
        raise ValueError('regression metrics require aligned finite one-dimensional arrays')
    reason = ('no_samples' if not len(target) else 'fewer_than_two_samples' if len(target) < 2
              else 'constant_target' if np.all(target == target[0]) else None)
    return dict(n_samples=len(target), r2=None if reason else float(r2_score(target, prediction, force_finite=False)),
                r2_undefined_reason=reason,
                mae=float(mean_absolute_error(target, prediction)) if len(target) else None,
                rmse=float(np.sqrt(np.mean((target - prediction) ** 2))) if len(target) else None)


def fit_linear_probes(dev: pd.DataFrame, test: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Fit independent scaler+OLS pipelines on fit_units; expose numeric fitted state."""
    features = [c for c in dev if c.startswith('representation_')]
    if not features:
        raise ValueError('no cycle representation columns supplied')
    validate_keys(dev, 'probe dev')
    validate_keys(test, 'probe test')
    dev = dev.assign(probe_split=np.where(dev.unit.isin(split['fit_units']), 'fit', 'val'))
    frame = pd.concat([dev, test.assign(probe_split='test')], ignore_index=True)
    if set(dev.unit) != set(split['fit_units'] + split['val_units']) or set(test.unit) != set(split['test_units']):
        raise ValueError('probe table devices do not match the resolved partition')
    fit = frame.probe_split.eq('fit')
    metrics, parameters = {}, {}
    for name, columns in [('representation', features), ('hi', ['hi'])]:
        x, y = frame[columns].to_numpy(dtype=float), frame.hi_gt.to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('probe fitting requires finite features and reference HI')
        pipeline = Pipeline([('scaler', StandardScaler()), ('regression', LinearRegression(fit_intercept=True))])
        pipeline.fit(x[fit], y[fit])
        prediction = pipeline.predict(x)  # no output transform, clipping or unit calibration
        frame[f'predicted_hi_{name}'] = prediction
        scaler, regression = pipeline.named_steps['scaler'], pipeline.named_steps['regression']
        parameters[name] = dict(feature_columns=columns, n_fit=int(fit.sum()), n_features=len(columns),
            rank=int(regression.rank_), rank_deficient=bool(regression.rank_ < len(columns)),
            underdetermined=bool(fit.sum() < len(columns) + 1),
            singular_values=regression.singular_.tolist(), scaler_mean=scaler.mean_.tolist(),
            scaler_var=scaler.var_.tolist(), scaler_scale=scaler.scale_.tolist(),
            coefficient=regression.coef_.tolist(), intercept=float(regression.intercept_))
        metrics[name] = {}
        for group in ('fit', 'val', 'test'):
            selected = frame.probe_split.eq(group)
            metrics[name][group] = regression_metrics(y[selected], prediction[selected])
            metrics[name][group]['per_unit'] = {
                str(unit): regression_metrics(part.hi_gt.to_numpy(), part[f'predicted_hi_{name}'].to_numpy())
                for unit, part in frame[selected].groupby('unit')}
    metrics['unprobed_hi'] = {group: regression_metrics(
        part.hi_gt.to_numpy(), part.hi.to_numpy()) for group in ('fit', 'val', 'test')
        for part in [frame[frame.probe_split.eq(group)]]}
    return frame, metrics, parameters
