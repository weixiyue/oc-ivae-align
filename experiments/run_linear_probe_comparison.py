"""Compare frozen linear probes of explicit saved runs; exclude OCALD and Residual."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import traceback
from types import FunctionType

import h5py
import torch
import yaml

from evaluate import read_evaluation_checkpoint
from experiments import evaluate_linear_probe as probe
from experiments.run_knowledge_ablation import CHECKPOINT_TYPE, load_ablation_checkpoint
from models import get_model_type
from models.tcn import W_SUMMARY_DIM
from utils.linear_probe import resolve_probe_split

COLUMNS = ['Method', 'Variant', 'Run', 'Feature dim',
           'Representation R²↑', 'Representation MAE↓', 'Representation RMSE↓',
           'HI-probe R²↑', 'HI-probe MAE↓', 'HI-probe RMSE↓']
STRUCTURAL_VARIANTS = ('no_prior', 'no_film', 'no_prior_no_film')


def inspect_run(source: Path) -> dict:
    """Check saved configuration and split against existing cache keys, without inference."""
    snapshot = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
    if not isinstance(snapshot, dict):
        raise ValueError('run config must be a YAML mapping')
    kind = get_model_type(snapshot)
    if kind in ('ocald', 'residual'):
        return dict(status='skipped', reason=f'{kind} excluded from this probe comparison')
    path = source / 'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    structural = checkpoint.get('model_type') == CHECKPOINT_TYPE
    if structural:
        config = checkpoint['config']
        variant = checkpoint['variant']
        if (variant not in STRUCTURAL_VARIANTS or config['experiment']['knowledge_ablation'] != variant
                or checkpoint['run_metadata']['variant'] != variant):
            raise ValueError('inconsistent structural-ablation checkpoint identifiers')
        if snapshot['model'] != config['model'] or snapshot['data'] != config['data']:
            raise ValueError('run snapshot model/data differs from saved structural checkpoint')
    else:
        config, _, checkpoint, _, _ = read_evaluation_checkpoint(str(source / 'config.yaml'), str(path), 'cpu')
        if config.get('experiment', {}).get('knowledge_ablation'):
            raise ValueError('structural ablation requires its dedicated finalized checkpoint')
        identities = [value for value in (config.get('experiment', {}).get('loss_ablation'),
            checkpoint.get('variant'), checkpoint.get('run_metadata', {}).get('variant'),
            config.get('run_metadata', {}).get('variant')) if value is not None]
        if len(set(identities)) > 1:
            raise ValueError('conflicting saved loss-ablation identifiers')
        variant = identities[0] if identities else 'full'
    kind = get_model_type(config)
    if kind not in ('kgld', 'proposal_cnn') or config.get('experiment', {}).get('baseline_type'):
        return dict(status='skipped', reason=f'unsupported model configuration: {kind}')
    if config['data'].get('input_mode', 'flight') != 'flight':
        raise ValueError('linear probes require one complete input per flight')
    if kind == 'proposal_cnn':
        variant = 'proposal_cnn_c'
    elif not structural and variant not in ('full', 'no_hsic', 'no_pairwise', 'no_spearman'):
        raise ValueError(f'unknown loss ablation: {variant}')
    if kind == 'kgld' and not structural and variant != 'full':
        weight = dict(no_hsic='lambda_hsic', no_pairwise='lambda_mono', no_spearman='lambda_spearman')[variant]
        if config['loss_weights'][weight] != 0:
            raise ValueError(f'{variant} checkpoint does not have zero {weight}')
    metadata = checkpoint.get('run_metadata', {})
    config_metadata = config.get('run_metadata', {})
    seeds = [value for value in (checkpoint.get('seed'), metadata.get('seed'), config_metadata.get('seed'))
             if value is not None]
    if len(set(seeds)) > 1:
        raise ValueError('conflicting recorded actual training seeds')
    cache_paths = [Path(value).resolve() for value in (checkpoint.get('processed_data'),
        metadata.get('processed_data'), config_metadata.get('processed_data')) if value is not None]
    if not cache_paths or len(set(cache_paths)) != 1:
        raise ValueError('missing or conflicting checkpoint preprocessing cache provenance')
    cache = cache_paths[0]
    with h5py.File(cache, 'r') as file:
        dev_units = sorted(set(map(int, file['dev/keys'][:, 0])))
        test_units = sorted(set(map(int, file['test/keys'][:, 0])))
    split = resolve_probe_split(config, checkpoint, dev_units, test_units)
    return dict(status='pending', config=config, checkpoint=checkpoint, structural=structural,
        info=dict(method='KGLD' if kind == 'kgld' else 'Proposal-C', variant=variant,
            model_type=kind, checkpoint_type=checkpoint.get('model_type', 'kgld'),
            checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actual_seed=seeds[0] if seeds else None,
            seed_note='recorded actual seed' if seeds else 'actual seed unavailable; configured seed not substituted',
            feature_dim=int(config['model'].get('d_zd', 2)) if kind == 'kgld' else 1,
            processed_data=str(cache), **split))


def run_structural_probe(source: Path, destination: Path, context: dict, device: str) -> Path:
    """Use the existing probe body with private loader bindings, without mutating its module.

    The single-run function has no loader argument. Binding only these two names
    preserves its complete fitting/export workflow and leaves concurrent callers untouched.
    """
    def read_checkpoint(config_path, checkpoint_path, requested_device):
        config = copy.deepcopy(context['config'])
        flat = {key: value for section in config.values() if isinstance(section, dict)
                for key, value in section.items()}
        flat.setdefault('w_summary_dim', W_SUMMARY_DIM)
        return config, flat, context['checkpoint'], source / 'best.pt', requested_device

    def load_model(config, flat, checkpoint, dataset):
        model, saved, _ = load_ablation_checkpoint(source / 'best.pt', device)
        if saved != config or int(dataset[0][0].shape[-1]) != config['model']['in_features']:
            raise ValueError('structural checkpoint configuration/input dimensions changed')
        return model

    namespace = dict(probe.run.__globals__, read_evaluation_checkpoint=read_checkpoint,
                     load_evaluation_model=load_model)
    adapted = FunctionType(probe.run.__code__, namespace, 'run_structural_probe', probe.run.__defaults__)
    return adapted(str(source / 'config.yaml'), str(source / 'best.pt'), str(destination), device)


def write_comparison(output: Path, rows: list[dict]) -> None:
    """Write per-run pooled test metrics, retaining missing values as blank/dash."""
    with (output / 'linear_probe_comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(COLUMNS) + ' |',
             '| ' + ' | '.join(['---'] * 3 + ['---:'] * 7) + ' |']
    for row in rows:
        cells = []
        for column in COLUMNS:
            value = row[column]
            if value is None:
                cell = '—'
            elif column == 'Feature dim':
                cell = str(value)
            elif isinstance(value, (float, int)):
                cell = f'{value:.6f}'
            else:
                cell = str(value).replace('|', '\\|').replace('\n', '<br>')
            cells.append(cell)
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', 'Proposal-C: Z and final HI differ only by dev-global affine scaling; '
                  'the two probes may be equivalent and do not represent independent gains.'])
    (output / 'linear_probe_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(run_dirs: list[str], output_dir: str, device: str = 'cpu', *, dry_run: bool = False) -> dict:
    """Process explicit source runs independently; never train or modify source artifacts."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    sources = [Path(name).resolve() for name in run_dirs]
    if any(output.is_relative_to(source) for source in sources):
        raise ValueError('output must not be inside a source run')
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
    records, rows = [], []
    for index, source in enumerate(sources, 1):
        destination = output / 'probes' / f'{index:03d}_{source.name}'
        record = dict(source_run=str(source), output=str(destination))
        try:
            context = inspect_run(source)
            record['status'] = context['status']
            if context['status'] == 'skipped':
                record['reason'] = context['reason']
            else:
                record.update(context['info'])
                if not dry_run:
                    with torch.no_grad():
                        if context['structural']:
                            evaluated = run_structural_probe(source, destination, context, device)
                        else:
                            evaluated = probe.run(str(source / 'config.yaml'), str(source / 'best.pt'),
                                                  str(destination), device)
                    metrics_path = evaluated / 'probe_metrics.json'
                    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                    parameters = json.loads((evaluated / 'probe_parameters.json').read_text(encoding='utf-8'))
                    if hashlib.sha256((source / 'best.pt').read_bytes()).hexdigest() != record['checkpoint_sha256']:
                        raise ValueError('source checkpoint changed during batch probe')
                    actual_split = {key: metrics['metadata'][key] for key in ('fit_units', 'val_units', 'test_units')}
                    if actual_split != {key: record[key] for key in actual_split}:
                        raise ValueError('actual probe split differs from preflight validation')
                    record['feature_dim'] = parameters['representation']['n_features']
                    record['status'] = 'success'
                    metrics['metadata'].update(source_run=str(source), comparison_run=record.copy())
                    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
                    row = dict(zip(COLUMNS[:4], [record['method'], record['variant'], str(source), record['feature_dim']]))
                    for prefix, name in [('Representation', 'representation'), ('HI-probe', 'hi')]:
                        for label, key in [('R²↑', 'r2'), ('MAE↓', 'mae'), ('RMSE↓', 'rmse')]:
                            row[f'{prefix} {label}'] = metrics[name]['test'][key]
                    rows.append(row)
        except Exception as exc:
            record.update(status='failed', reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    summary = dict(dry_run=dry_run, device=device, output_dir=str(output), runs=records,
                   counts={status: sum(record['status'] == status for record in records)
                           for status in ('pending', 'success', 'skipped', 'failed')})
    if not dry_run:
        write_comparison(output, rows)
        (output / 'comparison_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Summary:', summary['counts'], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run(args.run_dirs, args.output_dir, args.device, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
