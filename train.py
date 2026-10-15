"""Direct training workflow shared by OC-iVAE and the existing VAE baselines."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import random
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import NCMAPSSDataset, UnitStratifiedSampler
from evaluate import run_evaluation
from models import build_model
from models.oc_ivae import OC_iVAE
from models.tcn import W_SUMMARY_DIM
from utils.losses import compute_total_loss, masked_recon_loss


def train_model(
    model: OC_iVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: str = 'cpu',
    *,
    run_dir: Path,
    audit: bool = False,
) -> dict[str, Any]:
    """Train, validate and select the strict minimum reconstruction checkpoint.

    Args:
        model: OC-iVAE or an existing baseline with the common forward interface.
        train_loader: Unit-stratified cycle batches.
        val_loader: Validation batches, excluded from parameter updates.
        config: Nested experiment configuration or flat training/loss settings.
        device: Training device.
        run_dir: New directory for the configuration, checkpoint and history.
        audit: Record paired-experiment RNG/batch details; requires a recorded sampler.
    Returns:
        Completed epochs, best reconstruction loss, checkpoint path and history.
    """
    settings = _training_config(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    config.pop('checkpoint_dir', None)
    config.pop('log_dir', None)
    config['run_dir'] = str(run_dir.resolve())
    model_config = config.setdefault('model', {})
    model_config.setdefault('in_features', model.backbone.input_conv.conv.in_channels)
    model_config.setdefault('w_summary_dim', model.decoder.w_summary_dim)
    model_config['decoder_use_W_summary'] = model.decoder.use_W_summary
    logging = config.setdefault('logging', {})
    logging['log_dir'] = str(run_dir.resolve())
    logging['checkpoint_dir'] = str(run_dir.resolve())
    metadata = config.setdefault('run_metadata', {})
    metadata.setdefault('seed', None)
    metadata['configured_seed'] = config.get('experiment', {}).get('seed')
    metadata['initial_state_sha256'] = state_digest(model.state_dict())
    metadata['rng_at_training_start'] = rng_digest()
    metadata['device'] = str(torch.device(device))
    metadata['torch_num_threads'] = torch.get_num_threads()
    metadata['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    (run_dir / 'config.yaml').write_text(
        yaml.safe_dump(config, sort_keys=False), encoding='utf-8',
    )
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings['lr']),
        weight_decay=float(settings['weight_decay']),
    )
    total_epochs = int(settings['total_epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    checkpoint_path = run_dir / 'best.pt'
    best_val_recon = float('inf')
    epochs_without_improvement = 0
    completed_epochs = 0
    history: list[dict[str, Any]] = []
    use_early_stopping = settings.get('use_early_stopping', False)
    patience = int(settings.get('early_stopping_patience', 30))

    for epoch in range(total_epochs):
        rng_before = rng_digest() if audit else None
        train_losses = train_epoch(model, train_loader, optimizer, epoch, settings, torch_device)
        rng_after = rng_digest() if audit else None
        completed_epochs = epoch + 1
        val_recon = validate(model, val_loader, completed_epochs, torch_device)
        row = dict(epoch=completed_epochs, **train_losses, val_recon=val_recon,
                   lr=optimizer.param_groups[0]['lr'])
        if audit:
            row.update(
                best_val_before_epoch=best_val_recon if np.isfinite(best_val_recon) else None,
                selected_by_val_recon=val_recon < best_val_recon,
                rng_before_training=rng_before, rng_after_training=rng_after,
                rng_after_validation=rng_digest(),
                batch_sha256=hashlib.sha256(
                    json.dumps(train_loader.batch_sampler.last_batches).encode()).hexdigest(),
            )
            print(json.dumps(row), flush=True)
        with (run_dir / 'history.csv').open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if not history:
                writer.writeheader()
            writer.writerow(row)
        history.append(row)
        scheduler.step()

        if val_recon < best_val_recon:
            best_val_recon = val_recon
            epochs_without_improvement = 0
            if audit:
                config['run_metadata'].update(
                    epochs_recorded=completed_epochs, rng_at_checkpoint=rng_digest(),
                )
            save_checkpoint(checkpoint_path, model, optimizer, config,
                            completed_epochs, best_val_recon)
        else:
            epochs_without_improvement += 1

        status = (f'| patience={epochs_without_improvement}/{patience}'
                  if use_early_stopping else '| (no early stop)')
        print(
            f'Epoch {completed_epochs}/{total_epochs} | '
            f"recon={train_losses['recon']:.4f} "
            f"kl_d={train_losses['kl_d']:.4f} "
            f"kl_c={train_losses['kl_c']:.4f} "
            f"hsic={train_losses['hsic']:.4f} "
            f"mono={train_losses['mono']:.4f} "
            f"spearman={train_losses['spearman']:.4f} | "
            f'val_recon={val_recon:.4f} {status}'
        )
        if use_early_stopping and epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {completed_epochs}')
            break

    return dict(epochs_completed=completed_epochs, best_val_recon=best_val_recon,
                best_checkpoint=str(checkpoint_path), history=history)


def train_epoch(
    model: OC_iVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Update parameters for one epoch and average losses equally over batches.

    Args:
        model: Model with the common OC-iVAE forward fields.
        loader: Training cycle batches.
        optimizer: Optimizer owning the model parameters.
        epoch: Zero-indexed loss schedule epoch.
        config: Flat training and loss settings.
        device: Device holding model parameters.
    Returns:
        Per-batch averaged component and total losses.
    """
    model.train()
    loss_sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        x, mask, u, W_seq, unit_id, cycle_id, _ = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, mask, u, W_seq)
        total_loss, losses = compute_total_loss(
            out, x, mask, cycle_id, unit_id, epoch, config,
        )
        if not torch.isfinite(total_loss) or any(
            not torch.isfinite(torch.as_tensor(value)).all() for value in losses.values()
        ):
            raise FloatingPointError(f'nonfinite training loss at epoch {epoch}')
        total_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise FloatingPointError(f'nonfinite gradient at epoch {epoch}')
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(config.get('gradient_clip_norm', 1.0)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError(f'nonfinite parameter at epoch {epoch}')
        for key, value in losses.items():
            scalar = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            loss_sums[key] = loss_sums.get(key, 0.0) + scalar
        n_batches += 1
    if n_batches == 0:
        raise ValueError('train_loader produced no batches')
    return {key: value / n_batches for key, value in loss_sums.items()}


def validate(model: OC_iVAE, loader: DataLoader, epoch: int,
             device: torch.device) -> float:
    """Average masked validation reconstruction loss equally over batches.

    Args:
        model: Model to evaluate deterministically.
        loader: Validation cycle batches.
        epoch: Completed epoch count, used in error messages.
        device: Device holding model parameters.
    Returns:
        Mean reconstruction loss for checkpoint selection.
    """
    model.eval()
    recon_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x, mask, u, W_seq, _, _, _ = (tensor.to(device) for tensor in batch)
            out = model(x, mask, u, W_seq)
            recon = masked_recon_loss(out['x_hat'], x, mask)
            if not torch.isfinite(recon):
                raise FloatingPointError(f'nonfinite validation loss at epoch {epoch}')
            recon_sum += float(recon.item())
            n_batches += 1
    if n_batches == 0:
        raise ValueError('val_loader produced no batches')
    return recon_sum / n_batches


def save_checkpoint(
    path: str | Path,
    model: OC_iVAE,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best_val_recon: float,
) -> None:
    """Save the existing state-dict checkpoint format without changing keys.

    Args:
        path: Destination file in an existing checkpoint directory.
        model: Model whose parameters are saved.
        optimizer: Corresponding optimizer state.
        config: Original nested config, including optional run_metadata.
        epoch: Completed epoch count.
        best_val_recon: Best validation reconstruction loss so far.
    Returns:
        None.
    """
    model_config = dict(config.get('model', {}))
    model_config['decoder_use_W_summary'] = model.decoder.use_W_summary
    torch.save(dict(
        model_state_dict=model.state_dict(), model_config=model_config,
        decoder_use_W_summary=model.decoder.use_W_summary,
        run_metadata=dict(config.get('run_metadata', {})),
        optimizer_state_dict=optimizer.state_dict(), epoch=epoch,
        best_val_recon=best_val_recon,
    ), path)


def load_checkpoint(path: str | Path, model: OC_iVAE,
                    optimizer: torch.optim.Optimizer, device: str = 'cpu') -> dict[str, Any]:
    """Restore model and optimizer with the existing W-summary compatibility check.

    Args:
        path: Existing state-dict checkpoint.
        model: Model built with the original architecture and W-summary switch.
        optimizer: Optimizer owning this model's parameters.
        device: Device used when reading tensors.
    Returns:
        Stored epoch and best reconstruction loss; scheduler/RNG are not restored.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('decoder_use_W_summary', True) != model.decoder.use_W_summary:
        raise ValueError('checkpoint decoder_use_W_summary does not match model')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return dict(epoch=int(checkpoint['epoch']), best_val_recon=float(checkpoint['best_val_recon']))


def rng_digest() -> str:
    """Hash Python, NumPy, CPU and available CUDA RNG states for paired-run audits.

    Returns:
        SHA256 of the same serialized RNG states used by the paired experiment.
    """
    states = (
        random.getstate(), np.random.get_state(), torch.get_rng_state().numpy(),
        [s.cpu().numpy() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )
    return hashlib.sha256(pickle.dumps(states)).hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash named tensor values to record initialization independently of serialization.

    Args:
        state: Model state dictionary.
    Returns:
        SHA256 of sorted tensor names, shapes, dtypes and bytes.
    """
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve training/loss/logging merge order and scalar top-level overrides."""
    flat: dict[str, Any] = {}
    for section in ('training', 'loss_weights', 'logging'):
        value = config.get(section)
        if isinstance(value, Mapping):
            flat.update(value)
    flat.update((key, value) for key, value in config.items() if not isinstance(value, Mapping))
    return flat


def main() -> None:
    """Load an experiment, train its model, evaluate the best checkpoint and log it."""
    parser = argparse.ArgumentParser(description='Train OC-iVAE or a VAE baseline on N-CMAPSS.')
    parser.add_argument('--config', default='config/ds03_default.yaml')
    parser.add_argument('--run-dir', type=Path, help='New run directory; existing paths are rejected')
    args = parser.parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError('config file must contain a YAML mapping')
    if config.get('model_type', 'oc_ivae') != 'oc_ivae':
        raise ValueError('use experiments.train_ocald or experiments.train_proposal_cnn for migrated models')
    for section in ('data', 'training', 'model'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'config must include section {section!r}')
    exp_name = config_path.stem
    run_dir = args.run_dir or Path('runs') / exp_name
    if run_dir.exists():
        raise FileExistsError(f'run directory already exists: {run_dir}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Experiment: {exp_name}')
    print(f'Run directory: {run_dir}')
    data_config = config['data']
    training_config = config['training']
    model_config = dict(config['model'])
    h5_path = _resolve_main_h5_path(data_config)
    cycle_hi_path = _resolve_cycle_hi_path(data_config, h5_path)
    data_config['h5_path'] = str(h5_path.absolute())
    data_config['cycle_hi_path'] = None if cycle_hi_path is None else str(cycle_hi_path.absolute())
    if data_config.get('test_cycle_hi_path') is not None:
        data_config['test_cycle_hi_path'] = str(Path(data_config['test_cycle_hi_path']).absolute())
    cycle_hi_arg = None if cycle_hi_path is None else str(cycle_hi_path)
    train_dataset = NCMAPSSDataset(
        str(h5_path), cycle_hi_arg, data_config, split='train',
        val_fraction=float(data_config['val_split']),
    )
    val_dataset = NCMAPSSDataset(
        str(h5_path), cycle_hi_arg, data_config, split='val',
        val_fraction=float(data_config['val_split']),
    )
    model_config.setdefault('in_features', int(train_dataset[0][0].shape[-1]))
    model_config.setdefault('w_summary_dim', W_SUMMARY_DIM)
    k = int(training_config['k_units_per_batch'])
    m = int(training_config['m_cycles_per_unit'])
    sampler = UnitStratifiedSampler(train_dataset, k=k, m=m)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=k * m, shuffle=False)
    model = build_model(config, model_config)
    config['model'] = model_config
    config['run_metadata'] = dict(
        source_config=str(config_path.resolve()), split_seed=42, sampler_seed=42,
        processed_data=str(train_dataset.processed_path),
    )
    result = train_model(model, train_loader, val_loader, config, device, run_dir=run_dir)
    evaluation_dir = run_evaluation(
        str(run_dir / 'config.yaml'), result['best_checkpoint'], str(run_dir), device,
    )
    with Path(__file__).with_name('EXPERIMENT_LOG.md').open('a', encoding='utf-8') as file:
        file.write(
            f'\n## {datetime.now():%Y-%m-%d %H:%M:%S} | training-run | {config_path}\n'
            f"- Epochs completed: {result['epochs_completed']}\n"
            f"- Checkpoint: {result['best_checkpoint']}\n"
            f'- Unified all-dev/test evaluation: {evaluation_dir}\n'
        )


def _resolve_main_h5_path(data_config: dict[str, Any]) -> Path:
    """Resolve the main N-CMAPSS h5 path, tolerating one matching local variant."""
    configured_path = Path(str(data_config["h5_path"]))
    if configured_path.exists():
        return configured_path

    parent = configured_path.parent
    candidates = sorted(
        path
        for path in parent.glob(f"{configured_path.stem}*.h5")
        if "_cycle_" not in path.name
    )
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"main h5 path {configured_path} does not exist and no unique "
        f"matching h5 file was found in {parent}"
    )


def _resolve_cycle_hi_path(data_config: dict[str, Any], h5_path: Path) -> Path | None:
    """Resolve cycle-HI h5 path from config or the main h5 directory."""
    configured_value = data_config.get("cycle_hi_path")
    if "cycle_hi_path" in data_config and configured_value is None:
        return None
    if configured_value is not None:
        configured_path = Path(str(configured_value))
        if configured_path.exists():
            return configured_path
        raise FileNotFoundError(f"cycle_hi_path does not exist: {configured_path}")

    candidates = sorted(h5_path.parent.glob("*cycle_dev.h5"))
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "config data section must include 'cycle_hi_path' when the data "
        "directory does not contain exactly one *cycle_dev.h5 file"
    )


if __name__ == "__main__":
    main()
