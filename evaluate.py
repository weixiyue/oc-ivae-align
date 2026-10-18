"""Evaluate complete N-CMAPSS dev/test flights with dev-only HI calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from dataset import NCMAPSSDataset, ShortWindowDataset
from models import build_model
from models.ocald import OCALD
from models.proposal_cnn import ProposalCNN
from models.residual import ResidualModel, residual_scores, predict_window_flights
from models.tcn import W_SUMMARY_DIM
from utils.metrics import evaluate_all

PROTOCOL_VERSION = 'unsupervisedhi-ncmapss-v1'
REFERENCE_COMMIT = '157ebfae7baeee1e0cd52180f8e58ead3dcb448e'


def read_evaluation_checkpoint(config_path: str, checkpoint_path: str | None = None,
                               device: str | None = None) -> tuple[dict, dict, dict, Path, str]:
    """Read saved settings with the same model/data guards for all evaluation tasks."""
    with open(config_path, encoding='utf-8') as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError('config file must contain a YAML mapping')
    data_config = raw.get('data')
    if not isinstance(data_config, dict):
        raise ValueError("config must include section 'data'")
    requested = raw
    exp_name = Path(config_path).stem
    run_dir = Path(raw.get('run_dir', Path('runs') / exp_name))
    checkpoint_path = checkpoint_path or str(run_dir / 'best.pt')
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    saved = checkpoint.get('config')
    if saved is not None:
        # New checkpoints carry the complete actual settings. External YAML must
        # agree at model/data boundaries; all other settings come from the save.
        if requested.get('model_type', 'oc_ivae') != checkpoint['model_type']:
            raise ValueError('external model_type differs from saved checkpoint')
        if requested['model'] != saved['model']:
            raise ValueError('external model structure differs; use the saved run/config.yaml')
        def canonical_data(data: dict) -> dict:
            values = dict(data)
            for key in ('h5_path', 'cycle_hi_path', 'test_cycle_hi_path', 'processed_dir'):
                if values.get(key) is not None:
                    values[key] = str(Path(values[key]).resolve())
            return values
        if canonical_data(requested['data']) != canonical_data(saved['data']):
            raise ValueError('external preprocessing/data settings differ from saved checkpoint')
        raw = saved
        if checkpoint['model_type'] == 'ocald' and checkpoint['stage'] == 'representation':
            raise ValueError('representation checkpoint has no trained HIHead; evaluate best.pt')
    flat = {
        key: value
        for section in raw.values() if isinstance(section, dict)
        for key, value in section.items()
    }
    flat.setdefault('w_summary_dim', W_SUMMARY_DIM)
    return raw, flat, checkpoint, checkpoint_file, device


def load_evaluation_model(raw: dict, flat: dict, checkpoint: dict,
                          dataset: NCMAPSSDataset | ShortWindowDataset) -> torch.nn.Module:
    """Validate checkpoint/input compatibility and strictly restore network weights."""
    model_type = raw.get('model_type', 'oc_ivae')
    in_features = int(dataset[0][0].shape[-1])
    if model_type != 'oc_ivae':
        if in_features != raw['model']['in_features']:
            raise ValueError('checkpoint feature dimension differs from prepared data')
        if checkpoint.get('config') is not None and str(dataset.processed_path) != checkpoint['processed_data']:
            raise ValueError('checkpoint preprocessing source differs from current prepared data')
        if model_type == 'proposal_cnn' and dataset.L_max != raw['model']['window_size']:
            raise ValueError('checkpoint window_size differs from prepared L_max')
    flat['in_features'] = in_features
    model = build_model(raw, flat)
    saved_switch = checkpoint.get('decoder_use_W_summary', True)
    configured_switch = flat.get('decoder_use_W_summary', True)
    if model_type == 'oc_ivae' and saved_switch != configured_switch:
        raise ValueError('checkpoint decoder_use_W_summary does not match evaluation config')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    return model


def run_evaluation(
    config_path: str,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    *,
    save_plots: bool = True,
) -> Path:
    """Evaluate all dev/test flights with one dev-only scaler and save an isolated run.

    Args:
        config_path: Original experiment YAML, including original train/val fraction.
        checkpoint_path: Old model-state checkpoint or bare state dictionary.
        output_dir: New directory, or the matching training run awaiting its first evaluation.
        device: Optional inference device; otherwise select available CUDA or CPU.
        save_plots: Save per-split HI curves; paired experiments draw combined curves.
    Returns:
        Evaluation directory with scores, metrics (including scaler/provenance), and optional curves.
    """
    raw, flat, checkpoint, checkpoint_file, device = read_evaluation_checkpoint(
        config_path, checkpoint_path, device)
    data_config = raw['data']
    exp_name = Path(config_path).stem
    model_type = raw.get('model_type', 'oc_ivae')
    existing_run = output_dir is not None and Path(output_dir).exists()
    if existing_run:
        destination = Path(output_dir)
        # Only a matching fresh training run may receive its first evaluation.
        artifacts = ['scores.csv', 'metrics.json', 'hi_curves_dev.png', 'hi_curves_test.png']
        if (Path(config_path).resolve() != (destination / 'config.yaml').resolve()
                or checkpoint_file.resolve() != (destination / 'best.pt').resolve()
                or not (destination / 'history.csv').is_file()
                or any((destination / name).exists() for name in artifacts)):
            raise FileExistsError(f'evaluation directory already exists: {output_dir}')
    h5_path = _resolve_main_h5_path(data_config)
    # No held-out dev units during evaluation; labels are associated separately.
    window_mode = model_type == 'residual' and data_config.get('input_mode') == 'short_window'
    dataset_class = ShortWindowDataset if window_mode else NCMAPSSDataset
    dev_dataset = dataset_class(h5_path, None, flat, split='train', val_fraction=0.0)
    model = load_evaluation_model(raw, flat, checkpoint, dev_dataset)
    test_dataset = dataset_class(h5_path, None, flat, split='test')
    configured_switch = flat.get('decoder_use_W_summary', True)
    residual_state = checkpoint['residual_state'] if model_type == 'residual' else None
    if residual_state is not None:
        fit_keys = dev_dataset.flight_keys if window_mode else dev_dataset.samples
        if residual_state['fit_keys'] != [list(k) for k in fit_keys]:
            raise ValueError('checkpoint PCA fit keys differ from complete dev flights')
        if window_mode and (residual_state['n_samples_seen'] != len(dev_dataset) or
                residual_state['fit_window_key_sha256'] != hashlib.sha256(dev_dataset.samples).hexdigest()):
            raise ValueError('checkpoint PCA window fit set differs from complete dev windows')
    frames: dict[str, pd.DataFrame] = {}
    reference_meta: dict[str, dict[str, Any]] = {}
    for split, dataset in [('dev', dev_dataset), ('test', test_dataset)]:
        rul = dataset.rul
        path = _resolve_cycle_hi_path(flat, h5_path, split)
        ref, provenance = read_reference(path, rul)
        frames[split] = associate(predict_cycles(
            model, dataset, device=device, batch_size=raw.get('evaluation', {}).get('batch_size', 64),
            residual_state=residual_state), rul, ref)
        reference_meta[split] = dict(
            path=path, key_provenance=provenance,
            original_Y_constant_within_flight=True,
        )
    dev, test, scaler = scale_dev_test(frames['dev'], frames['test'])
    metrics = {
        split: evaluate_all(frame, random_state=42)
        for split, frame in [('dev', dev), ('test', test)]
    }
    original_units = NCMAPSSDataset._split_units(
        sorted({int(key[0]) for key in dev_dataset.samples}), float(raw['data']['val_split']), 42)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = (
        Path(output_dir) if output_dir
        else Path('runs') / exp_name / f'evaluation_{PROTOCOL_VERSION}_{timestamp}'
    )
    destination.mkdir(parents=True, exist_ok=existing_run)
    source_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    metadata = dict(
        protocol_version=PROTOCOL_VERSION,
        decoder_use_W_summary=configured_switch,
        training_run_metadata=checkpoint.get('run_metadata', {}),
        reference_repository='https://github.com/KBaja/UnsupervisedHI',
        reference_commit=REFERENCE_COMMIT,
        reference_metric_file='CMAPSS/eval_utils.py',
        reference_path=(
            'CMAPSS/03_train_evaluate_proposals.ipynb -> '
            'train_and_evaluate_proposed_model -> latent layer Z'
        ),
        checkpoint=dict(
            path=str(checkpoint_file.resolve()), sha256=source_hash,
            epoch=checkpoint.get('epoch'),
            source_config=str(Path(config_path).resolve()),
        ),
        device=device, mi_random_state=42,
        mi_note=(
            'random_state=42 is a reproducibility addition; '
            'all other estimator defaults match reference'
        ),
        mi_estimator_defaults=dict(n_neighbors=3, discrete_features='auto', copy=True),
        score=(
            'deterministic posterior mean, existing decreasing-health linear HI head, '
            'no sigmoid or sign selection'
        ),
        input_preprocessing=(
            'unchanged per-unit sensor standardization; '
            'NOT aligned with reference input preprocessing'
        ),
        original_unit_split={k: sorted(v) for k, v in original_units.items()},
        original_split_seed=42, evaluated_splits=['dev', 'test'],
        dev_definition='all original development flights (train union validation), not train',
        reference_HI=reference_meta, scaling=scaler,
        n_cycles={s: len(f) for s, f in [('dev', dev), ('test', test)]},
        L_max=dev_dataset.L_max, in_features=flat['in_features'],
        processed_data=str(dev_dataset.processed_path),
        postprocessing=dict(
            per_unit_scaling=False, ewma=False, smoothing=False, clipping=False,
        ),
        ignored_legacy_config={'postprocess': raw.get('postprocess')},
        config=raw, resolved_model_data_config=flat, resolved_h5_path=h5_path,
    )
    if model_type != 'oc_ivae':
        metadata.pop('decoder_use_W_summary')
        metadata.update(model_type=model_type, source=raw['source'],
                        score='-score(mu_I), deterministic posterior mean' if model_type == 'ocald'
                        else 'Proposal-C scalar Z, no sign conversion',
                        selection=checkpoint['selection'], stage=checkpoint['stage'])
    if model_type == 'residual':
        metadata.update(
            score='oriented PCA of time-first absolute residuals including padding',
            reference_path='CMAPSS/02_train_evaluate.ipynb -> train_and_evaluate_residual_model',
            healthy_network_prior='train units only, cycle<=20; no HI/RUL filtering',
            residual_pca={key: value for key, value in residual_state.items()
                          if key not in ('mean', 'components', 'explained_variance')},
        )
    if window_mode:
        metadata.update(
            input_preprocessing='short windows with OC-iVAE normalization; not full source-paper protocol',
            score='window absolute residual -> dev PCA -> dev direction -> mean per flight',
            window_length=dev_dataset.window_length, stride=dev_dataset.stride,
            n_windows=dict(dev=len(dev_dataset), test=len(test_dataset)),
            no_L_max_truncation=True, no_input_padding=True,
        )
        metadata['scaling']['window_aggregation'] = 'mean per unit/cycle AFTER PCA and direction'
    predictions = pd.concat([dev, test], ignore_index=True)
    predictions[['split', 'unit', 'cycle', 'score', 'hi', 'hi_gt', 'rul']].to_csv(
        destination / 'scores.csv', index=False,
    )
    (destination / 'metrics.json').write_text(
        json.dumps(dict(metadata=metadata, **metrics), indent=2, allow_nan=False),
        encoding='utf-8',
    )
    if save_plots:
        _plot_curves(dev, destination / 'hi_curves_dev.png')
        _plot_curves(test, destination / 'hi_curves_test.png')
    for split, values in metrics.items():
        print(split, {k: values[k] for k in ['mae','mape','monotonicity','trendability','fc','mi']}, flush=True)
        if values['issues']:
            print('Diagnostics:', values['issues'], flush=True)
    if scaler['constant_dev_output']:
        print('Degenerate dev scores: constant MinMaxScaler input; see metrics.json metadata.scaling', flush=True)
    print(f'Evaluation saved to: {destination}', flush=True)
    return destination


def predict_cycles(
    model: torch.nn.Module,
    dataset: NCMAPSSDataset,
    device: str = 'cpu',
    batch_size: int = 64,
    *, residual_state: dict | None = None,
) -> pd.DataFrame:
    """Collect deterministic model scores keyed by unit and cycle.

    Args:
        model: Trained OC-iVAE or an existing baseline.
        dataset: Dataset providing the seven-field cycle samples.
        device: Inference device.
        batch_size: Inference batch size; never changes the evaluated flight set.
    Returns:
        Sorted unit/cycle/score rows; reference labels are read separately.
    """
    if isinstance(model, ResidualModel):
        if residual_state is None:
            raise ValueError('residual scoring requires saved dev PCA and direction state')
        if isinstance(dataset, ShortWindowDataset):
            return predict_window_flights(model, dataset, residual_state, device, batch_size)
        if residual_state.get('input_mode', 'flight') != 'flight':
            raise ValueError('short-window checkpoint cannot use legacy flight inference')
        residuals = predict_residuals(model, dataset, device, batch_size)
        scores = residual_scores(residuals, residual_state)
        return pd.DataFrame(dataset.samples, columns=['unit', 'cycle']).assign(score=scores).sort_values(
            ['unit', 'cycle']).reset_index(drop=True)
    model.to(torch.device(device))
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for x, mask, u, W_seq, unit_id, cycle_id, _ in loader:
            if isinstance(model, OCALD):
                scores = model.health_score(x.to(device), W_seq.to(device), mask.to(device))
            elif isinstance(model, ProposalCNN):
                scores = model.encode(x.to(device))
            else:
                scores = model(x.to(device), mask.to(device), u.to(device), W_seq.to(device))['hi_raw']
            if not torch.isfinite(scores).all():
                raise FloatingPointError('nonfinite model scores')
            rows.extend(zip(unit_id.tolist(), cycle_id.tolist(), scores.cpu().tolist(), strict=True))
    return pd.DataFrame(rows, columns=['unit', 'cycle', 'score']).sort_values(
        ['unit', 'cycle']).reset_index(drop=True)


def predict_residuals(model: ResidualModel, dataset: NCMAPSSDataset,
                      device: str, batch_size: int) -> np.ndarray:
    """Infer absolute residual vectors per flight in dataset order, without masking."""
    model.to(device).eval()
    residuals = []
    with torch.no_grad():
        for x, _, _, w, _, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x, w = x.to(device), w.to(device)
            residual = (model(x, w) - x).abs().reshape(len(x), -1)
            if not torch.isfinite(residual).all():
                raise FloatingPointError('nonfinite residual predictions')
            residuals.append(residual.cpu().numpy())
    return np.concatenate(residuals, axis=0)


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


def associate(pred: pd.DataFrame, rul: pd.DataFrame,
              reference: pd.DataFrame) -> pd.DataFrame:
    """Join predictions, reference HI and original RUL one-to-one by flight keys.

    Args:
        pred: Unit/cycle/score rows, before any scaling.
        rul: Unique original Y flight labels.
        reference: Unique reference HI flight labels.
    Returns:
        One sorted row per unit/cycle, with score/hi_gt/rul.
    """
    pred = pred.assign(score=pred.score.astype(float))
    if not np.isfinite(pred.score).all():
        raise ValueError('nonfinite prediction score')
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


def _plot_curves(frame: pd.DataFrame, path: Path) -> None:
    """Plot each unit against actual cycle ids without smoothing or clipping."""
    # Keep matplotlib cache and all plot artifacts inside the independent evaluation run.
    os.environ.setdefault('MPLCONFIGDIR', str(path.parent.resolve() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_units = frame.unit.nunique()
    fig, axes = plt.subplots(math.ceil(n_units / 3), 3, figsize=(15, 3.5 * math.ceil(n_units / 3)), squeeze=False)
    for ax, (unit, group) in zip(axes.ravel(), frame.groupby('unit')):
        ax.plot(group.cycle, group.hi, label='HI prediction')
        if not group.hi_gt.isna().all():
            ax.plot(group.cycle, group.hi_gt, '--', label='Reference HI')
        ax.set(title=f'{frame.split.iloc[0]} unit {unit}', xlabel='Actual cycle', ylabel='HI')
        ax.legend()
    for ax in axes.ravel()[n_units:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _resolve_main_h5_path(data_config: dict[str, Any]) -> str:
    """Resolve configured main h5 path, accepting one local filename variant."""
    configured_path = str(data_config["h5_path"])
    if os.path.exists(configured_path):
        return configured_path

    parent = os.path.dirname(configured_path)
    stem = os.path.splitext(os.path.basename(configured_path))[0]
    candidates = sorted(
        os.path.join(parent, filename)
        for filename in os.listdir(parent)
        if filename.startswith(stem)
        and filename.endswith(".h5")
        and "_cycle_" not in filename
    )
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"main h5 path {configured_path} does not exist and no unique "
        f"matching h5 file was found in {parent}"
    )


def _resolve_cycle_hi_path(
    config: dict[str, Any],
    h5_path: str,
    split: str,
) -> str | None:
    """Resolve cycle-level HI path, returning None when GT is unavailable."""
    explicit_key = "cycle_hi_path" if split == "dev" else f"{split}_cycle_hi_path"
    if explicit_key in config:
        configured_value = config[explicit_key]
        if configured_value is None:
            return None
        configured_path = str(configured_value)
        if os.path.exists(configured_path):
            return configured_path
        raise FileNotFoundError(f"{explicit_key} does not exist: {configured_path}")

    if split != "dev" and config.get("cycle_hi_path", "__missing__") is None:
        return None

    parent = os.path.dirname(h5_path)
    candidates = sorted(
        os.path.join(parent, filename)
        for filename in os.listdir(parent)
        if filename.endswith(f"cycle_{split}.h5")
    )
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"expected one cycle_{split}.h5 reference file, found {len(candidates)}; "
        "configure the reference path or explicitly set cycle_hi_path: null"
    )


def main() -> None:
    """Parse CLI arguments and evaluate complete dev/test flights."""
    parser = argparse.ArgumentParser(description='Evaluate complete dev and test with dev-only calibration.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', choices=['cpu','cuda'])
    parser.add_argument('--threads', type=int, default=2, help='CPU inference threads')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    run_evaluation(args.config, args.checkpoint, args.output_dir, args.device)


if __name__ == '__main__':
    main()
