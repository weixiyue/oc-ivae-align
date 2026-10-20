"""Train three isolated KGLD loss ablations using the ordinary public interfaces."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset import NCMAPSSDataset, UnitStratifiedSampler
from evaluate import run_evaluation
from models.kgld import KGLD
from models.tcn import W_SUMMARY_DIM
import train as training

BASE_CONFIG = Path(__file__).resolve().parents[1] / 'config/kgld.yaml'
EXPECTED_LOSS_WEIGHTS = dict(
    beta_d=0.1, beta_c=0.25, lambda_hsic=1.0, lambda_mono=10, lambda_spearman=0.5,
    ranking_margin=0.01, ranking_delta_t=3, spearman_tau=0.1,
)
VARIANTS = dict(no_hsic='lambda_hsic', no_pairwise='lambda_mono', no_spearman='lambda_spearman')
METRICS = (('MAE↓', 'mae'), ('RMSE↓', 'rmse'), ('Mon↑', 'monotonicity'),
           ('Tren↑', 'trendability'), ('FC↑', 'fc'), ('MI↑', 'mi'))


def check_variant_parameters(baseline: dict, config: dict, variant: str) -> None:
    """Require exactly the designated loss change, allowing only run bookkeeping."""
    expected, actual = copy.deepcopy(baseline), copy.deepcopy(config)
    expected['loss_weights'][VARIANTS[variant]] = 0
    for value in (expected, actual):
        for key in ('run_dir', 'run_metadata', 'log_dir', 'checkpoint_dir'):
            value.pop(key, None)
        for key in ('log_dir', 'checkpoint_dir'):
            value.get('logging', {}).pop(key, None)
        for key in ('name', 'description', 'seed', 'loss_ablation'):
            value.get('experiment', {}).pop(key, None)
        if not value.get('experiment'):
            value.pop('experiment', None)
    for key in ('in_features', 'w_summary_dim'):
        if key not in baseline['model']:
            actual['model'].pop(key, None)
    if actual != expected:
        raise ValueError(f'{variant}: parameters differ beyond the designated loss weight')


def variant_configs(baseline: dict, output: Path, seed: int) -> dict[str, dict]:
    """Validate current loss values before independently copying each variant."""
    if (baseline.get('loss_weights') != EXPECTED_LOSS_WEIGHTS
            or any(isinstance(value, bool) for value in baseline['loss_weights'].values())):
        raise ValueError(f'config/kgld.yaml loss_weights mismatch: expected {EXPECTED_LOSS_WEIGHTS!r}; '
                         f"found {baseline.get('loss_weights')!r}")
    if baseline.get('model_type') != 'kgld' or baseline.get('experiment', {}).get('baseline_type'):
        raise ValueError('config/kgld.yaml must describe full KGLD')
    configs = {}
    for variant, weight in VARIANTS.items():
        config = copy.deepcopy(baseline)
        config['loss_weights'][weight] = 0
        for key in ('run_metadata', 'run_dir', 'log_dir', 'checkpoint_dir'):
            config.pop(key, None)
        config.setdefault('experiment', {}).pop('description', None)
        config['experiment'].update(name=variant, loss_ablation=variant, seed=seed)
        config['run_dir'] = str((output / variant).resolve())
        config.setdefault('logging', {}).update(log_dir=config['run_dir'], checkpoint_dir=config['run_dir'])
        config['run_metadata'] = dict(
            variant=variant, seed=seed, source_config=str(BASE_CONFIG), split_seed=42, sampler_seed=42,
        )
        check_variant_parameters(baseline, config, variant)
        configs[variant] = config
    return configs


def checkpoint_stage(epoch: int, settings: dict, epochs_completed: int) -> dict:
    """Map saved completed-epoch numbering to compute_total_loss's zero-based phase."""
    epoch_index = epoch - 1
    transition = int(settings['transition_end_epoch'])
    phase = ('reconstruction' if epoch_index < int(settings['warmup_end_epoch'])
             else 'kl_hsic_transition' if epoch_index < transition else 'temporal_constraints')
    return dict(
        best_epoch=epoch, best_training_epoch_zero_based=epoch_index, phase=phase,
        first_temporal_checkpoint_epoch=transition + 1,
        best_checkpoint_in_temporal_stage=epoch_index >= transition,
        epochs_completed=epochs_completed, training_reached_temporal_stage=epochs_completed > transition,
        selection='minimum_validation_reconstruction',
    )


def write_results(output: Path, results: dict) -> None:
    """Export one test row per variant, keeping missing values blank or as a dash."""
    columns = ['Variant', *(label for label, _ in METRICS)]
    rows = [dict({'Variant': variant}, **{label: values['test'].get(key) for label, key in METRICS})
            for variant, values in results.items()]
    with (output / 'loss_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * len(METRICS)) + ' |']
    for row in rows:
        cells = [row['Variant']] + ['—' if row[label] is None else f'{row[label]:.6f}' for label, _ in METRICS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'loss_ablation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(output: Path, seed: int, device: str, dry_run: bool = False) -> None:
    """Start fresh models sequentially; never train full or alter checkpoint selection."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'output root already exists: {output}')
    source = BASE_CONFIG.read_bytes()
    baseline = yaml.safe_load(source)
    configs = variant_configs(baseline, output, seed)
    for variant, config in configs.items():
        weight = VARIANTS[variant]
        print(json.dumps(dict(variant=variant, seed=seed, output=config['run_dir'],
            loss_change={weight: dict(before=baseline['loss_weights'][weight], after=0)},
            bookkeeping={key: config[key] for key in ('experiment', 'run_dir', 'logging', 'run_metadata')}),
            ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        print('Dry run: only the designated loss weight differs; no data/model loaded or output created.')
        return
    # Match train.main's path resolution, unit split, derived dimensions and sampling.
    data = baseline['data']
    h5_path = training._resolve_main_h5_path(data)
    cycle_hi_path = training._resolve_cycle_hi_path(data, h5_path)
    data['h5_path'] = str(h5_path.absolute())
    data['cycle_hi_path'] = None if cycle_hi_path is None else str(cycle_hi_path.absolute())
    if data.get('test_cycle_hi_path') is not None:
        data['test_cycle_hi_path'] = str(Path(data['test_cycle_hi_path']).absolute())
    output.mkdir(parents=True, exist_ok=False)
    results, reports = {}, {}
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
        check_variant_parameters(baseline, config, variant)
        settings = config['training']
        k, m = int(settings['k_units_per_batch']), int(settings['m_cycles_per_unit'])
        train_loader = DataLoader(train, batch_sampler=UnitStratifiedSampler(train, k=k, m=m))
        val_loader = DataLoader(val, batch_size=k * m, shuffle=False)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = KGLD(config['model'])
        run_dir = output / variant
        result = training.train_model(model, train_loader, val_loader, config, device, run_dir=run_dir)
        actual = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))
        check_variant_parameters(baseline, actual, variant)
        checkpoint_path = Path(result['best_checkpoint'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['run_metadata']['variant'] != variant:
            raise ValueError('checkpoint variant differs from requested loss ablation')
        report = checkpoint_stage(checkpoint['epoch'], actual['training'], result['epochs_completed'])
        # Enrich only this newly trained checkpoint; weights/optimizer/selected epoch stay unchanged.
        checkpoint.update(config=actual, model_type='kgld', variant=variant, seed=seed,
                          selection=report['selection'], checkpoint_phase=report)
        torch.save(checkpoint, checkpoint_path)
        del checkpoint
        reports[variant] = report
        (run_dir / 'checkpoint_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'{variant} selected checkpoint:', json.dumps(report), flush=True)
        evaluated = run_evaluation(str(run_dir / 'config.yaml'), str(checkpoint_path), str(run_dir), device)
        results[variant] = json.loads((evaluated / 'metrics.json').read_text(encoding='utf-8'))
        del model, train_loader, val_loader, train, val
    write_results(output, results)
    (output / 'checkpoint_reports.json').write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8')


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
