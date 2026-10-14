"""Training entry point for OC-iVAE."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import NCMAPSSDataset
from data.sampler import UnitStratifiedSampler
from evaluate import run_evaluation
from models import build_model
from models.tcn import W_SUMMARY_DIM
from train.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

_DEFAULT_CONFIG_PATH = "configs/ds03_default.yaml"
_TRAIN_SPLIT = "train"
_VAL_SPLIT = "val"
_EXPERIMENT_LOG_PATH = "EXPERIMENT_LOG.md"


def main() -> None:
    """Load config, train OC-iVAE, evaluate HI metrics, and log results."""
    args = _parse_args()
    config_path = Path(args.config)
    config = _load_config(config_path)
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    config["checkpoint_dir"] = os.path.join("checkpoints", exp_name)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    config["log_dir"] = os.path.join("runs", exp_name)
    os.makedirs(config["log_dir"], exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Checkpoint dir: {config['checkpoint_dir']}")
    print(f"Log dir: {config['log_dir']}")

    data_config = _require_section(config, "data")
    training_config = _require_section(config, "training")
    model_config = dict(_require_section(config, "model"))
    h5_path = _resolve_main_h5_path(data_config)
    cycle_hi_path = _resolve_cycle_hi_path(data_config, h5_path)
    cycle_hi_path_arg = None if cycle_hi_path is None else str(cycle_hi_path)

    train_dataset = NCMAPSSDataset(
        h5_path=str(h5_path),
        cycle_hi_path=cycle_hi_path_arg,
        config=data_config,
        split=_TRAIN_SPLIT,
        val_fraction=float(_get_required(data_config, "val_split")),
    )
    val_dataset = NCMAPSSDataset(
        h5_path=str(h5_path),
        cycle_hi_path=cycle_hi_path_arg,
        config=data_config,
        split=_VAL_SPLIT,
        val_fraction=float(_get_required(data_config, "val_split")),
    )

    train_sample = train_dataset[0]
    model_config.setdefault("in_features", int(train_sample[0].shape[-1]))
    model_config.setdefault("w_summary_dim", W_SUMMARY_DIM)

    k_units = int(_get_required(training_config, "k_units_per_batch"))
    m_cycles = int(_get_required(training_config, "m_cycles_per_unit"))
    train_sampler = UnitStratifiedSampler(
        train_dataset,
        k=k_units,
        m=m_cycles,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(
        val_dataset,
        batch_size=k_units * m_cycles,
        shuffle=False,
    )

    model = build_model(config, model_config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    trainer.train()

    evaluation_dir = run_evaluation(str(config_path), trainer.best_checkpoint_path,
                                    device=str(trainer.device))
    with (PROJECT_ROOT / _EXPERIMENT_LOG_PATH).open('a', encoding='utf-8') as file:
        file.write(f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} | training-run | {config_path}\n"
                   f"- Epochs completed: {trainer.epoch}\n"
                   f"- Checkpoint: {trainer.best_checkpoint_path}\n"
                   f"- Unified all-dev/test evaluation: {evaluation_dir}\n")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train OC-iVAE on N-CMAPSS DS03.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config file must contain a YAML mapping")
    return config


def _require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a required mapping section from the config."""
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config must include section {section!r}")
    return value


def _get_required(config: dict[str, Any], key: str) -> Any:
    """Return a required config value."""
    if key not in config:
        raise ValueError(f"config section must include {key!r}")
    return config[key]


def _resolve_main_h5_path(data_config: dict[str, Any]) -> Path:
    """Resolve the main N-CMAPSS h5 path, tolerating one matching local variant."""
    configured_path = Path(str(_get_required(data_config, "h5_path")))
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
