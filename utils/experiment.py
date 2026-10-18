"""Run snapshots and state-dict outputs shared by migrated experiments."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from dataset import NCMAPSSDataset
from evaluate import _resolve_main_h5_path


def seed_run(seed: int) -> None:
    """Recordable Python/NumPy/PyTorch initialization; no cross-framework RNG claim."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def start_run(config_path: str, run_dir: str | None, model_type: str,
              device: str, smoke: bool) -> tuple[dict, Path, NCMAPSSDataset, NCMAPSSDataset]:
    """Resolve common data and record the actual train/val units, without labels."""
    with open(config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or config.get('model_type') != model_type:
        raise ValueError(f'configuration must specify model_type: {model_type}')
    training = config['training']
    if training['checkpoint_selection'] != 'last_epoch' or training['split_seed'] != 42:
        raise ValueError('migrated experiments require last_epoch and the shared split_seed=42')
    seed_run(training['seed'])
    training['device'] = device
    output = Path(run_dir or f'runs/{Path(config_path).stem}')
    if output.exists():
        raise FileExistsError(f'run directory already exists: {output}')
    data = config['data']
    data['h5_path'] = str(Path(_resolve_main_h5_path(data)).resolve())
    for key in ('cycle_hi_path', 'test_cycle_hi_path', 'processed_dir'):
        if data.get(key) is not None:
            data[key] = str(Path(data[key]).resolve())
    train = NCMAPSSDataset(data['h5_path'], None, data, split='train',
                           val_fraction=data['val_split'], random_seed=training['split_seed'])
    val = NCMAPSSDataset(data['h5_path'], None, data, split='val',
                         val_fraction=data['val_split'], random_seed=training['split_seed'])
    config['model']['in_features'] = int(train.x.shape[-1])
    config['run_dir'] = str(output.resolve())
    config['run_metadata'] = dict(
        source_config=str(Path(config_path).resolve()), seed=training['seed'],
        split_seed=training['split_seed'], smoke=smoke, device=device,
        processed_data=str(train.processed_path), torch_version=str(torch.__version__),
        numpy_version=np.__version__, torch_num_threads=torch.get_num_threads(),
        train_units=sorted({u for u, _ in train.samples}),
        val_units=sorted({u for u, _ in val.samples}),
    )
    output.mkdir(parents=True, exist_ok=False)
    return config, output, train, val


def save_config(output: Path, config: dict) -> None:
    """Save the resolved actual configuration, including smoke limits."""
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')


def append_history(output: Path, row: dict, *, first: bool) -> None:
    """Persist only the model's real loss terms, weights and work counters."""
    with (output / 'history.csv').open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


def save_experiment_checkpoint(path: Path, model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer, config: dict,
                               stage: str, epoch: int, *, residual_state: dict | None = None) -> None:
    """Store complete actual settings; 'best.pt' deliberately selects the last epoch."""
    checkpoint = dict(
        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        config=copy.deepcopy(config), model_config=copy.deepcopy(config['model']),
        model_type=config['model_type'], stage=stage, epoch=epoch,
        selection='last_epoch', seed=config['training']['seed'],
        processed_data=config['run_metadata']['processed_data'],
        run_metadata=copy.deepcopy(config['run_metadata']),
    )
    if residual_state is not None:
        checkpoint['residual_state'] = residual_state
    torch.save(checkpoint, path)
