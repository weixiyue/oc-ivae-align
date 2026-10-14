"""Standalone evaluation entry point for OC-iVAE."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import NCMAPSSDataset
from eval.inference import extract_hi
from eval.metrics import evaluate_all
from model.constants import W_SUMMARY_DIM
from model.oc_ivae import OC_iVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

_DEFAULT_ALPHA = 0.3
_N_PLOT_COLS = 3


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
    """Run model evaluation and save metrics/HI plots."""
    args = _parse_args()
    raw_config = _load_config(args.config)
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    flat_config = _merge_config_sections(raw_config)
    flat_config.setdefault("w_summary_dim", W_SUMMARY_DIM)

    alpha = _resolve_alpha(args.alpha, raw_config)
    checkpoint_path = args.checkpoint or os.path.join(
        "checkpoints",
        exp_name,
        "best_model.pt",
    )
    output_dir = os.path.join("runs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    data_config = _require_section(raw_config, "data")
    h5_path = _resolve_main_h5_path(data_config)
    cycle_hi_path = _resolve_cycle_hi_path(flat_config, h5_path, "dev")
    eval_hi_path = (
        _resolve_cycle_hi_path(flat_config, h5_path, "test")
        if args.split == "test"
        else cycle_hi_path
    )

    train_dataset = NCMAPSSDataset(
        h5_path=h5_path,
        cycle_hi_path=cycle_hi_path,
        config=flat_config,
        split="train",
        val_fraction=0.0,
    )
    # Infer in_features from the first sample, same as run_train.py
    in_features = train_dataset[0][0].shape[-1]
    flat_config["in_features"] = int(in_features)
    train_stats = train_dataset.get_train_stats()

    if args.split == "test":
        eval_dataset = NCMAPSSDataset(
            h5_path=h5_path,
            cycle_hi_path=eval_hi_path,
            config=flat_config,
            split="test",
            train_stats=train_stats,
            L_max=train_dataset.L_max,
        )
    else:
        eval_dataset = train_dataset

    model = _build_model(raw_config, flat_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    results = extract_hi(model, eval_dataset, device=device)
    metrics = evaluate_all(results, alpha=alpha)

    _print_metrics_table(metrics, args.split)
    plot_path = os.path.join(output_dir, f"hi_curves_{args.split}.png")
    metrics_path = os.path.join(output_dir, f"metrics_{args.split}.json")
    _plot_hi_curves(results, metrics, args.split, plot_path)
    _save_metrics_json(metrics, metrics_path)

    print(f"Saved HI curves to: {plot_path}")
    print(f"Saved metrics to: {metrics_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate OC-iVAE HI curves.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args()


def _load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config file must contain a YAML mapping")
    return config


def _merge_config_sections(config: dict[str, Any]) -> dict[str, Any]:
    """Merge all mapping sections into one flat config dictionary."""
    merged: dict[str, Any] = {}
    for value in config.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a required mapping section from config."""
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config must include section {section!r}")
    return value


def _resolve_alpha(arg_alpha: float | None, config: dict[str, Any]) -> float:
    """Resolve EWMA alpha from CLI, config, or default."""
    if arg_alpha is not None:
        return arg_alpha

    postprocess_config = config.get("postprocess")
    if isinstance(postprocess_config, dict) and "ewma_alpha" in postprocess_config:
        return float(postprocess_config["ewma_alpha"])

    return _DEFAULT_ALPHA


def _resolve_main_h5_path(data_config: dict[str, Any]) -> str:
    """Resolve configured main h5 path, accepting one local filename variant."""
    configured_path = str(data_config["h5_path"])
    if os.path.exists(configured_path):
        return configured_path

    parent = os.path.dirname(configured_path)
    stem = os.path.splitext(os.path.basename(configured_path))[0]
    candidates = sorted(
        os.path.join(parent, filename)
        for filename in os.listdir(parent)
        if filename.startswith(stem)
        and filename.endswith(".h5")
        and "_cycle_" not in filename
    )
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"main h5 path {configured_path} does not exist and no unique "
        f"matching h5 file was found in {parent}"
    )


def _resolve_cycle_hi_path(
    config: dict[str, Any],
    h5_path: str,
    split: str,
) -> str | None:
    """Resolve cycle-level HI path, returning None when GT is unavailable."""
    explicit_key = "cycle_hi_path" if split == "dev" else f"{split}_cycle_hi_path"
    if explicit_key in config:
        configured_value = config[explicit_key]
        if configured_value is None:
            return None
        configured_path = str(configured_value)
        if os.path.exists(configured_path):
            return configured_path
        raise FileNotFoundError(f"{explicit_key} does not exist: {configured_path}")

    if split != "dev" and config.get("cycle_hi_path", "__missing__") is None:
        return None

    parent = os.path.dirname(h5_path)
    candidates = sorted(
        os.path.join(parent, filename)
        for filename in os.listdir(parent)
        if filename.endswith(f"cycle_{split}.h5")
    )
    if len(candidates) == 1:
        return candidates[0]

    return None


def _print_metrics_table(metrics: dict[str, Any], split: str) -> None:
    """Print aggregate and per-unit metrics using simple box text."""
    metric_name_width = 16
    unit_rows = [
        (
            f"Unit {int(unit_id):02d}",
            f"MAE={_format_metric(unit_metrics['mae'])} "
            f"Mono={unit_metrics['monotonicity']:.4f}",
        )
        for unit_id, unit_metrics in sorted(metrics["per_unit"].items())
    ]
    title = f"OC-iVAE Evaluation - {split} split"
    metric_value_width = max(18, *(len(row[1]) for row in unit_rows))
    width = max(len(title) + 4, metric_name_width + metric_value_width + 1)
    metric_value_width = width - metric_name_width - 1

    print("┌" + "─" * width + "┐")
    print("│" + title.center(width) + "│")
    print("├" + "─" * metric_name_width + "┬" + "─" * metric_value_width + "┤")
    _print_metric_row("Monotonicity", metrics["monotonicity"], metric_value_width)
    _print_metric_row("Trendability", metrics["trendability"], metric_value_width)
    _print_metric_row("Prognosability", metrics["prognosability"], metric_value_width)
    _print_metric_row("MAE", metrics["mae"], metric_value_width)
    print("├" + "─" * metric_name_width + "┼" + "─" * metric_value_width + "┤")
    for left, right in unit_rows:
        print(
            "│"
            + left.ljust(metric_name_width)
            + "│"
            + right.ljust(metric_value_width)
            + "│"
        )
    print("└" + "─" * metric_name_width + "┴" + "─" * metric_value_width + "┘")


def _print_metric_row(name: str, value: float, value_width: int) -> None:
    """Print one aggregate metric table row."""
    print("│" + name.ljust(16) + "│" + _format_metric(value).ljust(value_width) + "│")


def _format_metric(value: float, precision: int = 4) -> str:
    """Format a metric value, using N/A for NaN."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.{precision}f}"


def _plot_hi_curves(
    results: dict[int, dict],
    metrics: dict[str, Any],
    split: str,
    plot_path: str,
) -> None:
    """Plot postprocessed predicted HI against ground truth HI for all units."""
    n_units = len(results)
    n_rows = max(1, math.ceil(n_units / _N_PLOT_COLS))
    fig, axes = plt.subplots(
        n_rows,
        _N_PLOT_COLS,
        figsize=(5 * _N_PLOT_COLS, 3.5 * n_rows),
    )
    axes_array = np.asarray(axes).reshape(-1)

    for index, (unit_id, unit_results) in enumerate(sorted(results.items())):
        ax = axes_array[index]
        unit_metrics = metrics["per_unit"][unit_id]
        hi_proc = unit_metrics["hi_proc"]
        hi_gt = unit_metrics.get("hi_gt")
        cycles = unit_results["cycle_ids"]
        if hi_gt is not None and not np.all(np.isnan(hi_gt)):
            ax.plot(cycles, hi_gt, "b--", alpha=0.6, label="HI gt")
        ax.plot(cycles, hi_proc, "r-", alpha=0.8, label="HI pred")
        mae_str = (
            f" MAE={unit_metrics['mae']:.3f}"
            if not np.isnan(unit_metrics["mae"])
            else ""
        )
        ax.set_title(
            f"Unit {unit_id} | "
            f"Mono={unit_metrics['monotonicity']:.2f} "
            f"Trend={unit_metrics['trendability']:.2f}"
            f"{mae_str}"
        )
        ax.set_xlabel("Cycle index")
        ax.set_ylabel("HI")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    for ax in axes_array[n_units:]:
        ax.axis("off")

    fig.suptitle(
        f"OC-iVAE | {split} split | "
        f"Mono={metrics['monotonicity']:.3f} "
        f"Trend={metrics['trendability']:.3f} "
        f"Prog={metrics['prognosability']:.3f} "
        f"MAE={_format_metric(metrics['mae'], precision=3)}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


if __name__ == "__main__":
    main()
