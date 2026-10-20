"""Sequential knowledge ablations using the current kgld.yaml and shared train loop."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from dataset import NCMAPSSDataset, UnitStratifiedSampler
import evaluate as evaluation
from models.vae import build_knowledge_ablation
from models.tcn import W_SUMMARY_DIM
import train as training
from utils.metrics import evaluate_all

BASE_CONFIG = Path(__file__).resolve().parents[1] / 'config/kgld.yaml'
VARIANTS = {
    'no_prior': 'w/o conditional prior',
    'no_film': 'w/o FiLM',
    'no_prior_no_film': 'w/o conditional prior and FiLM',
}
CHECKPOINT_TYPE = 'kgld_knowledge_ablation'
METRICS = (('MAE↓', 'mae'), ('RMSE↓', 'rmse'), ('Mon↑', 'monotonicity'),
           ('Tren↑', 'trendability'), ('FC↑', 'fc'), ('MI↑', 'mi'))


def check_experiment_parameters(baseline: dict, config: dict) -> None:
    """Allow only variant/seed metadata, output paths and derived input dimensions."""
    expected, actual = copy.deepcopy(baseline), copy.deepcopy(config)
    for value in (expected, actual):
        value.pop('run_metadata', None)
        value.pop('run_dir', None)
        for key in ('log_dir', 'checkpoint_dir'):
            value.get('logging', {}).pop(key, None)
        for key in ('seed', 'knowledge_ablation'):
            value.get('experiment', {}).pop(key, None)
        if not value.get('experiment'):
            value.pop('experiment', None)
    for key in ('in_features', 'w_summary_dim'):
        if key not in baseline['model']:
            actual['model'].pop(key, None)
    if actual != expected:
        raise ValueError('ablation experiment parameters differ from current kgld.yaml baseline')


def variant_configs(baseline: dict, output: Path, seed: int) -> dict[str, dict]:
    """Deepcopy the one current baseline; no historical/full-run settings are read."""
    if baseline.get('model_type') != 'kgld' or baseline.get('experiment', {}).get('baseline_type'):
        raise ValueError('config/kgld.yaml must describe the complete KGLD model')
    configs = {}
    for variant in VARIANTS:
        config = copy.deepcopy(baseline)
        config.setdefault('experiment', {}).update(seed=seed, knowledge_ablation=variant)
        config['run_dir'] = str((output / variant).resolve())
        config.setdefault('logging', {}).update(log_dir=config['run_dir'], checkpoint_dir=config['run_dir'])
        config.setdefault('run_metadata', {}).update(
            seed=seed, variant=variant, split_seed=42, sampler_seed=42,
            source_config=str(BASE_CONFIG), checkpoint_loader='experiments.run_knowledge_ablation.load_ablation_checkpoint',
        )
        check_experiment_parameters(baseline, config)
        configs[variant] = config
    return configs


def seed_model(seed: int) -> None:
    """Reset model/training randomness without changing the baseline data split seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mark_ablation_checkpoint(path: Path, baseline: dict) -> None:
    """Finalize this new run's selected checkpoint with its actual variant/config."""
    config = yaml.safe_load((path.parent / 'config.yaml').read_text(encoding='utf-8'))
    check_experiment_parameters(baseline, config)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    variant = config['experiment']['knowledge_ablation']
    if variant not in VARIANTS or checkpoint['run_metadata']['variant'] != variant:
        raise ValueError('checkpoint variant does not match run config')
    checkpoint.update(model_type=CHECKPOINT_TYPE, variant=variant, config=config,
                      seed=config['run_metadata']['seed'], selection='minimum_validation_reconstruction')
    torch.save(checkpoint, path)


def load_ablation_checkpoint(path: str | Path, device: str = 'cpu') -> tuple[torch.nn.Module, dict, dict]:
    """Rebuild the recorded ablation, strictly loading its selected state_dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('model_type') != CHECKPOINT_TYPE:
        raise ValueError('expected a finalized knowledge-ablation checkpoint')
    config = checkpoint['config']
    variant = checkpoint['variant']
    if (variant not in VARIANTS or config['experiment']['knowledge_ablation'] != variant
            or checkpoint['run_metadata']['variant'] != variant):
        raise ValueError('inconsistent checkpoint ablation identifiers')
    model = build_knowledge_ablation(config['model'], variant)
    if checkpoint['decoder_use_W_summary'] != model.decoder.use_W_summary:
        raise ValueError('checkpoint decoder_use_W_summary differs from saved config')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval(), config, checkpoint


def evaluate_ablation(path: Path, device: str) -> dict:
    """Adapt only model loading; reuse public inference, labels, scaling and metrics."""
    model, config, checkpoint = load_ablation_checkpoint(path, device)
    output = path.parent
    for name in ('metrics.json', 'scores.csv', 'hi_curves_dev.png', 'hi_curves_test.png'):
        if (output / name).exists():
            raise FileExistsError(output / name)
    data = config['data']
    h5_path = evaluation._resolve_main_h5_path(data)
    frames, references, processed = {}, {}, {}
    for split in ('dev', 'test'):
        dataset = NCMAPSSDataset(h5_path, None, data, split='train' if split == 'dev' else 'test',
                                val_fraction=0.0)
        if str(dataset.processed_path) != config['run_metadata']['processed_data']:
            raise ValueError('evaluation preprocessing source differs from training')
        reference_path = evaluation._resolve_cycle_hi_path(data, h5_path, split)
        reference, provenance = evaluation.read_reference(reference_path, dataset.rul)
        predicted = evaluation.predict_cycles(model, dataset, device=device,
            batch_size=config.get('evaluation', {}).get('batch_size', 64))
        frames[split] = evaluation.associate(predicted, dataset.rul, reference)
        references[split] = dict(path=reference_path, key_provenance=provenance)
        processed[split] = str(dataset.processed_path)
    dev, test, scaler = evaluation.scale_dev_test(frames['dev'], frames['test'])
    metrics = {split: evaluate_all(frame, random_state=42) for split, frame in [('dev', dev), ('test', test)]}
    metadata = dict(
        protocol_version=evaluation.PROTOCOL_VERSION, model_type=CHECKPOINT_TYPE,
        variant=checkpoint['variant'], config=config, training_run_metadata=checkpoint['run_metadata'],
        checkpoint=dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        epoch=checkpoint['epoch'], selection=checkpoint['selection']),
        scaling=scaler, reference_HI=references, processed_data=processed, device=device,
        mi_random_state=42, score='deterministic posterior mean through unchanged KGLD HIHead',
        dev_definition='all original development flights (train union validation)',
    )
    pd.concat([dev, test], ignore_index=True)[
        ['split', 'unit', 'cycle', 'score', 'hi', 'hi_gt', 'rul']].to_csv(output / 'scores.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(dict(metadata=metadata, **metrics),
        indent=2, allow_nan=False), encoding='utf-8')
    for split, frame in [('dev', dev), ('test', test)]:
        evaluation._plot_curves(frame, output / f'hi_curves_{split}.png')
    return metrics


def write_comparison(output: Path, results: dict) -> None:
    """Export test overall metrics separately for each ablation; do not rerun full KGLD."""
    columns = ['变体', *(label for label, _ in METRICS)]
    rows = [dict({'变体': VARIANTS[variant]}, **{label: values['test'][key] for label, key in METRICS})
            for variant, values in results.items()]
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * len(METRICS)) + ' |']
    for row in rows:
        cells = [row['变体']] + ['—' if row[label] is None else f'{row[label]:.6f}' for label, _ in METRICS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(output: Path, seed: int, device: str, dry_run: bool = False) -> None:
    """Train exactly three variants through train_model, then reevaluate selected weights."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'output root already exists: {output}')
    source = BASE_CONFIG.read_bytes()
    baseline = yaml.safe_load(source)
    configs = variant_configs(baseline, output, seed)
    for variant, config in configs.items():
        differences = {key: value for key, value in config.items() if baseline.get(key) != value}
        print(json.dumps(dict(variant=variant, description=VARIANTS[variant], output=config['run_dir'],
                              config_changes=differences), ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        print('Dry run: experiment parameters match kgld.yaml; no model/data loaded or output created.')
        return
    # Resolve paths exactly as train.main does, preserving the shared preprocessing protocol.
    data = baseline['data']
    h5_path = training._resolve_main_h5_path(data)
    cycle_hi_path = training._resolve_cycle_hi_path(data, h5_path)
    data['h5_path'] = str(h5_path.absolute())
    data['cycle_hi_path'] = None if cycle_hi_path is None else str(cycle_hi_path.absolute())
    if data.get('test_cycle_hi_path') is not None:
        data['test_cycle_hi_path'] = str(Path(data['test_cycle_hi_path']).absolute())
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for variant, config in configs.items():
        config['data'] = copy.deepcopy(data)
        train = NCMAPSSDataset(str(h5_path), None if cycle_hi_path is None else str(cycle_hi_path),
            data, split='train', val_fraction=float(data['val_split']))
        val = NCMAPSSDataset(str(h5_path), None if cycle_hi_path is None else str(cycle_hi_path),
            data, split='val', val_fraction=float(data['val_split']))
        config['model'].setdefault('in_features', int(train[0][0].shape[-1]))
        config['model'].setdefault('w_summary_dim', W_SUMMARY_DIM)
        config['run_metadata'].update(
            source_config_sha256=hashlib.sha256(source).hexdigest(), processed_data=str(train.processed_path),
            train_units=sorted({key[0] for key in train.samples}), val_units=sorted({key[0] for key in val.samples}),
        )
        check_experiment_parameters(baseline, config)
        settings = config['training']
        k, m = int(settings['k_units_per_batch']), int(settings['m_cycles_per_unit'])
        train_loader = DataLoader(train, batch_sampler=UnitStratifiedSampler(train, k=k, m=m))
        val_loader = DataLoader(val, batch_size=k * m, shuffle=False)
        seed_model(seed)
        model = build_knowledge_ablation(config['model'], variant)
        result = training.train_model(model, train_loader, val_loader, config, device,
                                      run_dir=output / variant)
        checkpoint_path = Path(result['best_checkpoint'])
        mark_ablation_checkpoint(checkpoint_path, baseline)
        results[variant] = evaluate_ablation(checkpoint_path, device)
        del model, train_loader, val_loader, train, val
    write_comparison(output, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run(args.output_dir, args.seed, args.device, args.dry_run)


if __name__ == '__main__':
    main()
