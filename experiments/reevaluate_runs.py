"""Reevaluate saved runs through the public evaluator and export a test-only table."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import traceback

from evaluate import run_evaluation
from models import get_model_type


TABLE_METRICS = (
    ('MAE↓', 'mae'), ('RMSE↓', 'rmse'), ('Mon↑', 'monotonicity'),
    ('Tren↑', 'trendability'), ('FC↑', 'fc'), ('MI↑', 'mi'),
)
TABLE_COLUMNS = ['方法', '运行名称', *(label for label, _ in TABLE_METRICS)]


def discover_runs(root: Path) -> list[dict]:
    """Find run artifacts or empty leaf directories, pruning every history subtree.

    Intermediate grouping directories without run artifacts are not runs.
    Directory symlinks are not followed.
    """
    if not root.is_dir():
        raise NotADirectoryError(root)
    if root.name == 'history':
        return []
    records = []
    markers = {'config.yaml', 'best.pt', 'history.csv', 'metrics.json', 'scores.csv'}
    for directory, children, files in os.walk(root, followlinks=False):
        path = Path(directory)
        has_children = bool(children)
        children[:] = sorted(name for name in children if name != 'history')
        if not markers.intersection(files) and (has_children or files or path == root):
            continue
        missing = [name for name in ('config.yaml', 'best.pt') if not (path / name).is_file()]
        records.append(dict(
            run=path.relative_to(root).as_posix(), source=str(path),
            status='skipped' if missing else 'pending',
            reason='missing ' + ', '.join(missing) if missing else None,
        ))
    return sorted(records, key=lambda record: record['run'])


def method_name(config: dict) -> str:
    """Name the actual saved model variant, never infer it from the directory name."""
    model_type = get_model_type(config)
    if model_type == 'kgld':
        baseline = config.get('experiment', {}).get('baseline_type')
        if not baseline:
            return 'KGLD'
        return {'vae': 'VAE', 'cvae': 'cVAE'}[baseline]
    if model_type == 'ocald':
        pipeline = config['training']['pipeline']
        decoder_w = config['model'].get('decoder_use_w', False)
        return f'OCALD ({pipeline}, decoder W={decoder_w})'
    if model_type == 'proposal_cnn':
        return 'Proposal-C'
    if model_type == 'residual':
        return f"Residual-{config['model']['variant']} ({config['data'].get('input_mode', 'flight')})"
    raise ValueError(f'unknown model_type: {model_type}')


def write_main_results(output: Path, rows: list[dict]) -> None:
    """Write one row per successful run; blanks/dashes denote unavailable metrics."""
    with (output / 'main_results.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(TABLE_COLUMNS) + ' |',
             '| ' + ' | '.join(['---', '---'] + ['---:'] * len(TABLE_METRICS)) + ' |']
    for row in rows:
        cells = []
        for column in TABLE_COLUMNS:
            value = row[column]
            if value is None:
                cell = '—'
            elif isinstance(value, (float, int)):
                cell = f'{value:.6f}'
            else:
                cell = str(value).replace('|', '\\|').replace('\n', '<br>')
            cells.append(cell)
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'main_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def reevaluate_runs(runs_dir: str, output_dir: str, device: str = 'cpu',
                    *, dry_run: bool = False) -> dict:
    """Evaluate each complete run in isolation; only per-run failures are caught."""
    root, output = Path(runs_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    records = discover_runs(root)
    # New batch directories may be under runs/, but never inside a source run.
    for record in records:
        if output.is_relative_to(Path(record['source'])):
            raise ValueError(f"output must not be inside a source run: {record['source']}")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
    rows = []
    for record in records:
        if record['status'] == 'skipped' or dry_run:
            print(f"{record['status']}: {record['run']}" +
                  (f" ({record['reason']})" if record['reason'] else ''), flush=True)
            continue
        source = Path(record['source'])
        destination = output / 'evaluations' / record['run']
        try:
            # The evaluator reads authoritative checkpoint settings and guards
            # model/data agreement; its metadata carries that resolved config.
            evaluated = run_evaluation(
                config_path=str(source / 'config.yaml'), checkpoint_path=str(source / 'best.pt'),
                output_dir=str(destination), device=device,
            )
            metrics = json.loads((evaluated / 'metrics.json').read_text(encoding='utf-8'))
            method = method_name(metrics['metadata']['config'])
            row = {'方法': method, '运行名称': record['run']}
            row.update({label: metrics['test'].get(key) for label, key in TABLE_METRICS})
            rows.append(row)
            record.update(status='success', method=method, output=str(evaluated))
        except Exception as exc:
            # A batch must continue, while preserving the original failure traceback.
            record.update(status='failed', reason=f'{type(exc).__name__}: {exc}',
                          traceback=traceback.format_exc(), output=str(destination))
        print(f"{record['status']}: {record['run']}" +
              (f" ({record['reason']})" if record['reason'] else ''), flush=True)
    summary = dict(
        runs_dir=str(root), output_dir=str(output), device=device, dry_run=dry_run,
        counts={status: sum(record['status'] == status for record in records)
                for status in ('pending', 'success', 'skipped', 'failed')},
        runs=records,
    )
    if not dry_run:
        write_main_results(output, rows)
        (output / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Summary:', summary['counts'], flush=True)
    return summary


def main() -> None:
    """List or reevaluate saved models without training or overwriting outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dry-run', action='store_true', help='list runs without loading checkpoints or writing files')
    args = parser.parse_args()
    reevaluate_runs(args.runs_dir, args.output_dir, args.device, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
