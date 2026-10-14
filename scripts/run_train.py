"""Training entry point for OC-iVAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import NCMAPSSDataset
from data.sampler import UnitStratifiedSampler
from eval.inference import extract_hi
from eval.metrics import evaluate_all
from model.constants import W_SUMMARY_DIM
from model.oc_ivae import OC_iVAE
from train.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

_DEFAULT_CONFIG_PATH = "configs/ds03_default.yaml"
_TRAIN_SPLIT = "train"
_VAL_SPLIT = "val"
_EXPERIMENT_LOG_PATH = "EXPERIMENT_LOG.md"


def _build_model(raw_config: dict[str, Any], model_config: dict[str, Any]):
    """Instantiate the correct model class based on experiment.baseline_type.

    Returns an nn.Module whose forward() matches the OC_iVAE interface.
    If experiment.baseline_type is absent or None, returns a plain OC_iVAE.

    Args:
        raw_config: The full nested YAML config dict (with sections).
        model_config: The flat model config dict passed to the model constructor.

    Returns:
        An nn.Module ready for training.
    """
    baseline_type = raw_config.get("experiment", {}).get("baseline_type") or None
    if baseline_type is None:
        return OC_iVAE(model_config)
    if baseline_type == "cvae":
        from baselines.cvae import build_cvae_model

        return build_cvae_model(model_config)
    elif baseline_type == "vae":
        from baselines.vae import build_vae_model

        return build_vae_model(model_config)
    raise ValueError(
        f"Unknown baseline_type {baseline_type!r}. "
        "Supported values: None, 'cvae', 'vae'."
    )


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
    postprocess_config = _require_section(config, "postprocess")
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

    model = _build_model(config, model_config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    trainer.train()

    trainer.load_checkpoint(trainer.best_checkpoint_path)
    results = extract_hi(trainer.model, train_dataset, device=str(trainer.device))
    metrics = evaluate_all(
        results,
        alpha=float(_get_required(postprocess_config, "ewma_alpha")),
    )

    _print_final_metrics(metrics)
    metrics_path = os.path.join(config["log_dir"], "metrics_train.json")
    _save_metrics_json(metrics, metrics_path)
    _append_experiment_log(
        config_path=config_path,
        trainer=trainer,
        metrics=metrics,
    )


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


def _print_final_metrics(metrics: dict[str, Any]) -> None:
    """Print final aggregate metrics."""
    print("Final train metrics:")
    print(f"  monotonicity:   {metrics['monotonicity']:.6f}")
    print(f"  trendability:   {metrics['trendability']:.6f}")
    print(f"  prognosability: {metrics['prognosability']:.6f}")


def _save_metrics_json(metrics: dict[str, Any], metrics_path: str) -> None:
    """Save metrics dictionary as JSON."""
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(_to_jsonable(metrics), file, indent=2)


def _to_jsonable(value: Any) -> Any:
    """Convert NumPy values and arrays into JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _append_experiment_log(
    config_path: Path,
    trainer: Trainer,
    metrics: dict[str, Any],
) -> None:
    """Append final run metrics to the experiment log."""
    log_path = PROJECT_ROOT / _EXPERIMENT_LOG_PATH
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n## {timestamp} | training-run | {config_path}\n"
        f"- Epochs completed: {trainer.epoch}\n"
        f"- Best val recon: {trainer.best_val_recon:.6f}\n"
        f"- Checkpoint: {trainer.best_checkpoint_path}\n"
        f"- Train monotonicity: {metrics['monotonicity']:.6f}\n"
        f"- Train trendability: {metrics['trendability']:.6f}\n"
        f"- Train prognosability: {metrics['prognosability']:.6f}\n"
    )
    with log_path.open("a", encoding="utf-8") as file:
        file.write(entry)


if __name__ == "__main__":
    main()
