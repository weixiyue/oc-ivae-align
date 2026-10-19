"""Run the seed-42 paired ablation of only the decoder W-summary input."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import NCMAPSSDataset, UnitStratifiedSampler
from evaluate import run_evaluation, _resolve_main_h5_path
from models.kgld import KGLD
from train import state_digest, train_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = ('mae', 'mape', 'monotonicity', 'trendability', 'fc', 'mi')


def seed_training(seed: int) -> None:
    """Reset Python, NumPy and Torch RNGs and enable deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RecordedSampler(UnitStratifiedSampler):
    """Keep actual batches in memory for the training history's epoch hash."""

    def __init__(self, dataset: NCMAPSSDataset, config: dict) -> None:
        super().__init__(
            dataset, k=config['k_units_per_batch'],
            m=config['m_cycles_per_unit'], random_seed=42,
        )
        self.last_batches: list[list[int]] = []

    def __iter__(self) -> Iterator[list[int]]:
        self.last_batches = []
        for batch in super().__iter__():
            self.last_batches.append(batch)
            yield batch


def run_pair(output: Path, device: str) -> None:
    """Train and evaluate both variants with shared initialization and RNGs.

    Args:
        output: New experiment directory; existing results are never overwritten.
        device: CPU or CUDA training and inference device.
    """
    names = {
        'true': 'exp_tuning_mono_weight_w_summary_seed42',
        'false': 'exp_tuning_mono_weight_no_w_summary_seed42',
    }
    configs = {
        role: yaml.safe_load((PROJECT_ROOT / f'config/history/{name}.yaml').read_text())
        for role, name in names.items()
    }
    off_copy = copy.deepcopy(configs['false'])
    off_copy['model']['decoder_use_W_summary'] = True
    if configs['true'] != off_copy:
        raise ValueError('paired configs differ beyond W-summary switch')
    baseline = yaml.safe_load(
        (PROJECT_ROOT / 'config/history/exp_tuning_mono_weight.yaml').read_text(),
    )
    baseline['experiment'] = dict(seed=42, paired_w_summary_ablation=True)
    if configs['true'] != baseline:
        raise ValueError('control settings differ from specified baseline')

    output.mkdir(parents=True, exist_ok=False)
    data = configs['true']['data']
    h5 = _resolve_main_h5_path(data)
    candidates = list(Path(h5).parent.glob('*cycle_dev.h5'))
    path = data.get('cycle_hi_path') or (
        str(candidates[0]) if len(candidates) == 1 else None
    )
    seed_training(42)
    train = NCMAPSSDataset(
        h5, path, data, split='train', val_fraction=data['val_split'], random_seed=42,
    )
    val = NCMAPSSDataset(
        h5, path, data, split='val', val_fraction=data['val_split'], random_seed=42,
    )
    model_cfg = dict(configs['true']['model'], in_features=int(train[0][0].shape[-1]))
    seed_training(42)
    initial = KGLD(model_cfg)
    shared_state = {key: value.detach().clone().cpu()
                    for key, value in initial.state_dict().items()}
    initial_digest = state_digest(shared_state)
    training_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    del initial

    histories = {}
    metrics_by_role = {}
    for role in ('true', 'false'):
        run_dir = output / role
        config = copy.deepcopy(configs[role])
        config['model']['in_features'] = model_cfg['in_features']
        config['data']['h5_path'] = str(Path(h5).absolute())
        config['data']['cycle_hi_path'] = None if path is None else str(Path(path).absolute())
        config['run_metadata'] = dict(
            seed=42, role=role, initial_state_sha256=initial_digest,
            deterministic_algorithms=True, torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            training_start_rng='shared post-initialization Torch state',
            sampler_seed=42, num_workers=0, processed_data=str(train.processed_path),
        )
        seed_training(42)
        model = KGLD(config['model'])
        model.load_state_dict(shared_state)
        if state_digest(model.state_dict()) != initial_digest:
            raise RuntimeError('paired initial weights differ')
        sampler = RecordedSampler(train, config['training'])
        train_loader = DataLoader(train, batch_sampler=sampler, num_workers=0)
        val_loader = DataLoader(
            val, batch_size=(config['training']['k_units_per_batch']
                             * config['training']['m_cycles_per_unit']),
            shuffle=False, num_workers=0,
        )
        torch.set_rng_state(training_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        result = train_model(
            model, train_loader, val_loader, config, device,
            run_dir=run_dir, audit=True,
        )
        histories[role] = result['history']
        # Test evaluation happens only after validation-based checkpoint selection.
        eval_dir = run_evaluation(
            str(run_dir / 'config.yaml'), result['best_checkpoint'], str(run_dir), device,
            save_plots=False,
        )
        metrics = json.loads((eval_dir / 'metrics.json').read_text())
        for split in ('dev', 'test'):
            for key in METRIC_KEYS:
                if metrics[split][key] is None or not np.isfinite(metrics[split][key]):
                    raise FloatingPointError(f'nonfinite {role}/{split}/{key}; stopping pair')
        metrics_by_role[role] = metrics
        del result, model, train_loader, val_loader

    common_epochs = min(len(history) for history in histories.values())
    for index in range(common_epochs):
        for key in ('rng_before_training', 'rng_after_training',
                    'rng_after_validation', 'batch_sha256', 'lr'):
            if histories['true'][index][key] != histories['false'][index][key]:
                raise RuntimeError(f'paired audit mismatch epoch {index + 1}: {key}')
    # Pair-level facts are recorded once both histories have passed comparison.
    for role in ('true', 'false'):
        config_path = output / role / 'config.yaml'
        snapshot = yaml.safe_load(config_path.read_text())
        snapshot['run_metadata']['paired_audit'] = dict(
            common_epochs_compared=common_epochs,
            batch_rng_and_lr_sequences_identical=True,
            completed_epochs={role: len(history) for role, history in histories.items()},
        )
        config_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding='utf-8')

    rows = []
    for split in ('dev', 'test'):
        for metric in METRIC_KEYS:
            on = metrics_by_role['true'][split][metric]
            off = metrics_by_role['false'][split][metric]
            rows.append(dict(split=split, metric=metric, true=on, false=off,
                             false_minus_true=off - on))
    with (output / 'comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_comparison(output)
    print('Pair complete:', output, flush=True)


def _plot_comparison(output: Path) -> None:
    """Plot both variants and reference HI together for each engine."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    on = pd.read_csv(output / 'true/scores.csv')
    off = pd.read_csv(output / 'false/scores.csv')
    joined = on.merge(
        off, on=['split', 'unit', 'cycle'], suffixes=('_true', '_false'),
        validate='one_to_one',
    )
    for split, group in joined.groupby('split'):
        units = list(group.unit.unique())
        nrows = (len(units) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.5 * nrows), squeeze=False)
        for ax, unit in zip(axes.ravel(), units):
            cycle_scores = group[group.unit == unit].sort_values('cycle')
            ax.plot(cycle_scores.cycle, cycle_scores.hi_true, label='W summary: true')
            ax.plot(cycle_scores.cycle, cycle_scores.hi_false, label='W summary: false')
            ax.plot(cycle_scores.cycle, cycle_scores.hi_gt_true, '--', label='Reference HI')
            ax.set(title=f'{split} unit {unit}', xlabel='Actual cycle', ylabel='HI')
            ax.legend(fontsize=8)
        for ax in axes.ravel()[len(units):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / f'comparison_curves_{split}.png', dpi=150)
        plt.close(fig)


def main() -> None:
    """Run one pair and let failures surface with their original traceback."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    if output.exists():
        parser.error(f'experiment directory already exists: {output}')
    run_pair(output, args.device)


if __name__ == '__main__':
    main()
