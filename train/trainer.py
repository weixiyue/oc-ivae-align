"""Training loop for OC-iVAE."""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader

from data.dataset import NCMAPSSDataset
from data.sampler import UnitStratifiedSampler
from loss.recon import masked_recon_loss
from loss.total import compute_total_loss
from model.oc_ivae import OC_iVAE

_BEST_MODEL_FILENAME = "best_model.pt"
_DEFAULT_GRADIENT_CLIP_NORM = 1.0


class Trainer:
    """Three-phase trainer for OC-iVAE."""

    def __init__(
        self,
        model: OC_iVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cpu",
    ) -> None:
        """Initialize optimizer, scheduler, checkpointing, and trainer state.

        Args:
            model: OC-iVAE model to train.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            config: Flat trainer/loss config.
            device: Device string passed to ``torch.device``.
        """
        self.config = _flatten_config(config)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(_get_config_value(self.config, "lr")),
            weight_decay=float(_get_config_value(self.config, "weight_decay")),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(_get_config_value(self.config, "total_epochs")),
        )
        self.gradient_clip_norm = float(
            self.config.get("gradient_clip_norm", _DEFAULT_GRADIENT_CLIP_NORM)
        )
        self.use_early_stopping = self.config.get("use_early_stopping", False)
        self.patience = int(self.config.get("early_stopping_patience", 30))
        self.best_val_recon = float("inf")
        self.epochs_without_improvement = 0
        self.epoch = 0

        self.checkpoint_dir = str(_get_config_value(self.config, "checkpoint_dir"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            _BEST_MODEL_FILENAME,
        )

    def train_epoch(self) -> dict:
        """Run one training epoch and return averaged loss metrics."""
        self.model.train()
        loss_sums: dict[str, float] = {}
        n_batches = 0

        for batch in self.train_loader:
            x, mask, u, W_seq, unit_id, cycle_id, hi = _move_batch(
                batch,
                self.device,
            )
            del hi

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x, mask, u, W_seq)
            total_loss, loss_dict = compute_total_loss(
                out,
                x,
                mask,
                cycle_id,
                unit_id,
                self.epoch,
                self.config,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
            )
            self.optimizer.step()

            _accumulate_losses(loss_sums, loss_dict)
            n_batches += 1

        if n_batches == 0:
            raise ValueError("train_loader produced no batches")

        self.epoch += 1
        return _average_losses(loss_sums, n_batches)

    def val_epoch(self) -> dict:
        """Run validation and return averaged reconstruction loss."""
        self.model.eval()
        recon_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, mask, u, W_seq, unit_id, cycle_id, hi = _move_batch(
                    batch,
                    self.device,
                )
                del unit_id, cycle_id, hi

                out = self.model(x, mask, u, W_seq)
                recon = masked_recon_loss(out["x_hat"], x, mask)
                recon_sum += float(recon.item())
                n_batches += 1

        if n_batches == 0:
            raise ValueError("val_loader produced no batches")

        return {"val_recon": recon_sum / n_batches}

    def train(self) -> None:
        """Run the main training loop with validation early stopping."""
        total_epochs = int(_get_config_value(self.config, "total_epochs"))
        while self.epoch < total_epochs:
            train_losses = self.train_epoch()
            val_losses = self.val_epoch()
            self.scheduler.step()

            val_recon = val_losses["val_recon"]
            if val_recon < self.best_val_recon:
                self.best_val_recon = val_recon
                self.epochs_without_improvement = 0
                self.save_checkpoint(self.best_checkpoint_path)
            else:
                self.epochs_without_improvement += 1

            if self.use_early_stopping:
                early_stop_status = (
                    f"| patience={self.epochs_without_improvement}/{self.patience}"
                )
            else:
                early_stop_status = "| (no early stop)"
            print(
                f"Epoch {self.epoch}/{total_epochs} | "
                f"recon={train_losses['recon']:.4f} "
                f"kl_d={train_losses['kl_d']:.4f} "
                f"kl_c={train_losses['kl_c']:.4f} "
                f"hsic={train_losses['hsic']:.4f} "
                f"mono={train_losses['mono']:.4f} "
                f"spearman={train_losses['spearman']:.4f} | "
                f"val_recon={val_recon:.4f} "
                f"{early_stop_status}"
            )

            if (
                self.use_early_stopping
                and self.epochs_without_improvement >= self.patience
            ):
                print(f"Early stopping at epoch {self.epoch}")
                break

    def save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, epoch, and best validation state."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_val_recon": self.best_val_recon,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model, optimizer, epoch, and best validation state."""
        # weights_only=False is required because the checkpoint dict contains
        # optimizer state (not pure model weights); suppress the FutureWarning explicitly.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = int(checkpoint["epoch"])
        self.best_val_recon = float(checkpoint["best_val_recon"])


def _move_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Move all batch tensors to ``device``."""
    return tuple(tensor.to(device) for tensor in batch)


def _accumulate_losses(loss_sums: dict[str, float], loss_dict: dict) -> None:
    """Accumulate one batch of loss values into running sums."""
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            scalar_value = float(value.item())
        else:
            scalar_value = float(value)
        loss_sums[key] = loss_sums.get(key, 0.0) + scalar_value


def _average_losses(loss_sums: dict[str, float], n_batches: int) -> dict:
    """Return average loss values over ``n_batches``."""
    return {key: value / n_batches for key, value in loss_sums.items()}


def _flatten_config(config: dict) -> dict:
    """Accept flat config or merge common nested config sections."""
    flat_config: dict = {}
    for section in ("training", "loss_weights", "logging"):
        section_config = config.get(section)
        if isinstance(section_config, Mapping):
            flat_config.update(section_config)

    for key, value in config.items():
        if not isinstance(value, Mapping):
            flat_config[key] = value

    return flat_config


def _get_config_value(config: dict, key: str) -> object:
    """Read a required trainer config value."""
    if key not in config:
        raise ValueError(f"config must include {key!r}")
    return config[key]
