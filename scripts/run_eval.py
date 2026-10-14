"""Unified, dev-calibrated UnsupervisedHI evaluation entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import NCMAPSSDataset
from eval.inference import extract_hi
from eval.metrics import evaluate_all
from eval.protocol import (PROTOCOL_VERSION, REFERENCE_COMMIT, associate,
                           read_reference, read_rul, scale_dev_test)
from model.constants import W_SUMMARY_DIM
from model.oc_ivae import OC_iVAE

def _build_model(raw_config: dict[str, Any], model_config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the correct model class based on experiment.baseline_type.

    Returns an nn.Module whose forward() matches the OC_iVAE interface.
    If experiment.baseline_type is absent or None, returns a plain OC_iVAE.

    Args:
        raw_config: The full nested YAML config dict (with sections).
        model_config: The flat model config dict passed to the model constructor.

    Returns:
        An nn.Module ready for training.
    """
    baseline_type = raw_config.get("experiment", {}).get("baseline_type") or None
    if baseline_type is None:
        return OC_iVAE(model_config)
    if baseline_type == "cvae":
        from baselines.cvae import build_cvae_model

        return build_cvae_model(model_config)
    elif baseline_type == "vae":
        from baselines.vae import build_vae_model

        return build_vae_model(model_config)
    raise ValueError(
        f"Unknown baseline_type {baseline_type!r}. "
        "Supported values: None, 'cvae', 'vae'."
    )


def run_evaluation(config_path: str, checkpoint_path: str | None = None,
                   output_dir: str | None = None, device: str | None = None) -> Path:
    """Evaluate all dev/test flights with one dev-only scaler and save an isolated run.

    Args:
        config_path: Original experiment YAML, including original train/val fraction.
        checkpoint_path: Old model-state checkpoint or bare state dictionary.
        output_dir: New directory; existing directories are rejected.
        device: Optional inference device; otherwise select available CUDA or CPU.
    Returns:
        New evaluation directory with predictions, metrics, scaler and provenance.
    """
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'evaluation directory already exists: {output_dir}')
    raw = _load_config(config_path)
    flat = _merge_config_sections(raw)
    flat.setdefault('w_summary_dim', W_SUMMARY_DIM)
    exp_name = Path(config_path).stem
    checkpoint_path = checkpoint_path or f'checkpoints/{exp_name}/best_model.pt'
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_path)
    h5_path = _resolve_main_h5_path(_require_section(raw, 'data'))
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Preserve the original dataset processing exactly; labels are read separately by key.
    # Dataset with val_fraction=0 exposes the union of train and validation (all dev).
    dev_dataset = NCMAPSSDataset(h5_path, None, flat, split='train', val_fraction=0.0)
    flat['in_features'] = int(dev_dataset[0][0].shape[-1])
    test_dataset = NCMAPSSDataset(h5_path, None, flat, split='test',
                                  train_stats=dev_dataset.get_train_stats(), L_max=dev_dataset.L_max)
    model = _build_model(raw, flat)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    saved_switch = checkpoint.get('decoder_use_W_summary', True)
    configured_switch = flat.get('decoder_use_W_summary', True)
    if saved_switch != configured_switch:
        raise ValueError('checkpoint decoder_use_W_summary does not match evaluation config')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    frames, reference_meta = {}, {}
    for split, dataset in [('dev', dev_dataset), ('test', test_dataset)]:
        rul = read_rul(h5_path, split)
        path = _resolve_cycle_hi_path(flat, h5_path, split)
        ref, provenance = read_reference(path, rul)
        frames[split] = associate(extract_hi(model, dataset, device=device), rul, ref)
        reference_meta[split] = dict(path=path, key_provenance=provenance,
                                     original_Y_constant_within_flight=True)
    dev, test, scaler = scale_dev_test(frames['dev'], frames['test'])
    metrics = {split: evaluate_all(frame, random_state=42) for split, frame in [('dev', dev), ('test', test)]}
    original_units = NCMAPSSDataset._split_units(
        sorted({u for u, _ in dev_dataset.samples}), float(raw['data']['val_split']), 42)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = Path(output_dir) if output_dir else Path('runs') / exp_name / f'evaluation_{PROTOCOL_VERSION}_{timestamp}'
    destination.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    metadata = dict(protocol_version=PROTOCOL_VERSION,
                    decoder_use_W_summary=configured_switch,
                    training_run_metadata=checkpoint.get('run_metadata', {}),
                    reference_repository='https://github.com/KBaja/UnsupervisedHI',
                    reference_commit=REFERENCE_COMMIT,
                    reference_metric_file='CMAPSS/eval_utils.py',
                    reference_path='CMAPSS/03_train_evaluate_proposals.ipynb -> train_and_evaluate_proposed_model -> latent layer Z',
                    checkpoint=dict(path=str(checkpoint_file.resolve()), sha256=source_hash,
                                    epoch=checkpoint.get('epoch'), source_config=str(Path(config_path).resolve())),
                    device=device, mi_random_state=42,
                    mi_note='random_state=42 is a reproducibility addition; all other estimator defaults match reference',
                    mi_estimator_defaults=dict(n_neighbors=3, discrete_features='auto', copy=True),
                    score='deterministic posterior mean, existing decreasing-health linear HI head, no sigmoid or sign selection',
                    input_preprocessing='unchanged per-unit sensor standardization; NOT aligned with reference input preprocessing',
                    original_unit_split={k: sorted(v) for k,v in original_units.items()},
                    original_split_seed=42, evaluated_splits=['dev','test'],
                    dev_definition='all original development flights (train union validation), not train',
                    reference_HI=reference_meta, scaling=scaler,
                    n_cycles={s: len(f) for s,f in [('dev',dev),('test',test)]},
                    L_max=dev_dataset.L_max, in_features=flat['in_features'],
                    postprocessing=dict(per_unit_scaling=False, ewma=False, smoothing=False, clipping=False),
                    ignored_legacy_config={'postprocess':raw.get('postprocess')},
                    config=raw, resolved_model_data_config=flat, resolved_h5_path=h5_path)
    predictions = pd.concat([dev, test], ignore_index=True)
    predictions[['split','unit','cycle','score','hi','hi_gt','rul']].to_csv(destination / 'predictions.csv', index=False)
    for name, content in [('metrics.json', dict(metadata=metadata, **metrics)), ('scaler.json', scaler)]:
        (destination / name).write_text(json.dumps(content, indent=2, allow_nan=False), encoding='utf-8')
    (destination / 'evaluation_config.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    _plot_curves(dev, destination / 'hi_curves_dev.png')
    _plot_curves(test, destination / 'hi_curves_test.png')
    for split, values in metrics.items():
        print(split, {k: values[k] for k in ['mae','mape','monotonicity','trendability','fc','mi']}, flush=True)
        if values['issues']:
            print('Diagnostics:', values['issues'], flush=True)
    if scaler['constant_dev_output']:
        print('Degenerate dev scores: constant MinMaxScaler input; see scaler.json', flush=True)
    print(f'Evaluation saved to: {destination}', flush=True)
    return destination


def _plot_curves(frame: pd.DataFrame, path: Path) -> None:
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


def main() -> None:
    """Run dev/test evaluation; legacy split/alpha flags explicitly fail."""
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


def _load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config file must contain a YAML mapping")
    return config


def _merge_config_sections(config: dict[str, Any]) -> dict[str, Any]:
    """Merge all mapping sections into one flat config dictionary."""
    merged: dict[str, Any] = {}
    for value in config.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a required mapping section from config."""
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config must include section {section!r}")
    return value


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


if __name__ == '__main__':
    main()
