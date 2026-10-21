"""Frozen KGLD knowledge-ablation probes with dedicated CSV and Markdown reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback

import h5py
import numpy as np
import pandas as pd
import torch

from evaluate import _resolve_cycle_hi_path, _resolve_main_h5_path, read_reference
from experiments import evaluate_linear_probe as probe
from experiments.run_linear_probe_comparison import inspect_run, run_structural_probe

LABELS = {
    'full': 'KGLD (full)',
    'no_prior': 'w/o conditional prior',
    'no_film': 'w/o FiLM',
    'no_prior_no_film': 'w/o conditional prior and FiLM',
}
REPORT_NAME = 'knowledge_ablation_linear_prob.md'
CSV_NAME = 'knowledge_ablation_linear_prob.csv'


def reference_manifest(context: dict) -> dict:
    """Read public reference labels and verify complete cached flight-key coverage."""
    data = context['config']['data']
    h5_path = _resolve_main_h5_path(data)
    references = {}
    with h5py.File(context['info']['processed_data'], 'r') as cache:
        for split in ('dev', 'test'):
            keys = pd.DataFrame(cache[f'{split}/keys'][()], columns=['unit', 'cycle'])
            keys = keys.sort_values(['unit', 'cycle']).reset_index(drop=True)
            path = _resolve_cycle_hi_path(data, h5_path, split)
            if path is None:
                raise ValueError(f'{split}: linear probes require reference HI')
            reference, provenance = read_reference(path, keys)
            reference = reference.sort_values(['unit', 'cycle']).reset_index(drop=True)
            if not np.array_equal(reference[['unit', 'cycle']].to_numpy(), keys.to_numpy()):
                raise ValueError(f'{split}: reference HI does not cover exactly the cached flight keys')
            # Compare semantic keys/values, not HDF storage bytes or filenames.
            content = reference.to_csv(index=False, float_format='%.17g').encode('utf-8')
            references[split] = dict(path=str(Path(path).resolve()), n_cycles=len(reference),
                sha256=hashlib.sha256(content).hexdigest(), key_provenance=provenance)
    return references


def compare_protocols(records: list[dict]) -> str | None:
    """Record differences without pooling or silently forcing the same partition/labels."""
    checked = [record for record in records if 'reference_HI' in record]
    if not checked:
        return None
    baseline = next((record for record in checked if record['variant'] == 'full'), checked[0])
    for record in checked:
        matches = {name: record[name] == baseline[name] for name in ('fit_units', 'val_units', 'test_units')}
        matches.update({f'{split}_reference_HI': record['reference_HI'][split]['sha256']
                        == baseline['reference_HI'][split]['sha256'] for split in ('dev', 'test')})
        record['protocol_comparison'] = dict(reference_run=baseline['source_run'], matches=matches,
                                             differences=[name for name, same in matches.items() if not same])
    return baseline['source_run']


def write_report(output: Path, summary: dict) -> None:
    """Keep each run/seed separate, with pooled test metrics and explicit protocol checks."""
    columns = ['Method', 'Run', 'Seed', 'Feature dim',
               'Representation R²↑', 'Representation MAE↓', 'Representation RMSE↓',
               'HI-probe R²↑', 'HI-probe MAE↓', 'HI-probe RMSE↓']
    rows = []
    for record in summary['runs']:
        if record['status'] != 'success':
            continue
        row = [record['method'], record['source_run'], record['actual_seed'], record['feature_dim']]
        row.extend(record['test_metrics'][name][key] for name in ('representation', 'hi')
                   for key in ('r2', 'mae', 'rmse'))
        rows.append(row)
    with (output / CSV_NAME).open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    lines = ['# KGLD knowledge-ablation linear probes', '',
             'Frozen complete mu_d and public scalar HI; independent StandardScaler → LinearRegression probes. '
             'Probe fitting uses the original training units only; test metrics pool cycles equally. '
             'No network training, prediction rescaling, clipping or smoothing.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * 2 + ['---:'] * 8) + ' |']
    for row in rows:
        cells = ['—' if value is None else str(value) if index < 4 else f'{value:.6f}'
                 for index, value in enumerate(row)]
        lines.append('| ' + ' | '.join(cell.replace('|', '\\|').replace('\n', '<br>') for cell in cells) + ' |')
    lines.extend(['', '## Protocol checks and provenance', '',
                  f"Comparison reference: `{summary['protocol_reference_run']}`. "
                  'If full is unavailable, the first run with valid label checks is used explicitly.',
                  'Unknown actual training seeds remain unavailable; configured seeds are not substituted.', ''])
    for record in summary['runs']:
        lines.extend([f"### {record.get('method', record['source_run'])}", '',
                      f"- Run: `{record['source_run']}`; status: **{record['status']}**."])
        if 'reason' in record:
            lines.append('- Reason: ' + record['reason'].replace('\n', ' '))
        if 'reference_HI' in record:
            lines.extend([f"- Checkpoint: `{record['checkpoint']}`; SHA256: `{record['checkpoint_sha256']}`.",
                          f"- Variant: `{record['variant']}`; actual seed: `{record['actual_seed']}`; "
                          f"feature dimension: {record['feature_dim']}.",
                          f"- fit units: `{record['fit_units']}`; val units: `{record['val_units']}`; "
                          f"test units: `{record['test_units']}`."])
            differences = record['protocol_comparison']['differences']
            lines.append('- Compared with reference run: ' + (', '.join(differences) + ' DIFFER.' if differences
                                                               else 'all device partitions and reference HI agree.'))
            for split, reference in record['reference_HI'].items():
                lines.append(f"- {split} reference HI: `{reference['path']}`; {reference['n_cycles']} cycles; "
                             f"key/value SHA256 `{reference['sha256']}`; {reference['key_provenance']}.")
        lines.append('')
    lines.extend(['## Summary', '', json.dumps(summary['counts']), ''])
    (output / REPORT_NAME).write_text('\n'.join(lines), encoding='utf-8')


def run(run_dirs: list[str], output_dir: str, device: str = 'cpu', *, dry_run: bool = False) -> dict:
    """Reuse existing per-run probing; only orchestration and the report are specialized."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    sources = [Path(name).resolve() for name in run_dirs]
    if any(output.is_relative_to(source) for source in sources):
        raise ValueError('output must not be inside a source run')
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, source in enumerate(sources, 1):
        destination = output / 'probes' / f'{index:03d}_{source.name}'
        record = dict(source_run=str(source), output=str(destination))
        try:
            context = inspect_run(source)
            record['status'] = context['status']
            if context['status'] == 'skipped':
                record['reason'] = context['reason']
            elif context['info']['model_type'] != 'kgld' or context['info']['variant'] not in LABELS:
                record.update(status='skipped', reason='this entry accepts full KGLD and knowledge ablations only')
            else:
                record.update(context['info'])
                record['method'] = LABELS[record['variant']]
                record['reference_HI'] = reference_manifest(context)
                if not dry_run:
                    with torch.no_grad():
                        if context['structural']:
                            evaluated = run_structural_probe(source, destination, context, device)
                        else:
                            evaluated = probe.run(str(source / 'config.yaml'), str(source / 'best.pt'),
                                                  str(destination), device)
                    if hashlib.sha256((source / 'best.pt').read_bytes()).hexdigest() != record['checkpoint_sha256']:
                        raise ValueError('source checkpoint changed during probing')
                    metrics = json.loads((evaluated / 'probe_metrics.json').read_text(encoding='utf-8'))
                    parameters = json.loads((evaluated / 'probe_parameters.json').read_text(encoding='utf-8'))
                    if any(metrics['metadata'][name] != record[name] for name in ('fit_units', 'val_units', 'test_units')):
                        raise ValueError('actual probe partitions differ from preflight validation')
                    record.update(status='success', feature_dim=parameters['representation']['n_features'],
                        test_metrics={name: {key: metrics[name]['test'][key] for key in ('r2', 'mae', 'rmse')}
                                      for name in ('representation', 'hi')})
        except Exception as exc:
            record.update(status='failed', reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        records.append(record)
    reference_run = compare_protocols(records)
    summary = dict(dry_run=dry_run, device=device, output_dir=str(output), protocol_reference_run=reference_run,
                   runs=records, counts={status: sum(record['status'] == status for record in records)
                                        for status in ('pending', 'success', 'skipped', 'failed')})
    for record in records:
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        if record['status'] == 'success':
            path = Path(record['output']) / 'probe_metrics.json'
            metrics = json.loads(path.read_text(encoding='utf-8'))
            metrics['metadata']['knowledge_ablation_comparison'] = record
            path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    if not dry_run:
        write_report(output, summary)
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
