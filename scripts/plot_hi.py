import sys
from pathlib import Path
import os
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import NCMAPSSDataset
from model.oc_ivae import OC_iVAE
from eval.inference import extract_hi
from eval.metrics import evaluate_all
from model.constants import W_SUMMARY_DIM
import yaml


def main() -> None:
    """Plot train-set HI curves from the best checkpoint."""
    with open("configs/ds03_default.yaml", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    data_config = raw_config["data"]
    model_config = dict(raw_config["model"])
    config = {
        **raw_config["data"],
        **raw_config["model"],
        **raw_config["training"],
        **raw_config["loss_weights"],
        **raw_config["postprocess"],
        **raw_config["logging"],
    }

    h5_path = _resolve_main_h5_path(data_config)
    cycle_hi_path = _resolve_cycle_hi_path(data_config, h5_path)
    ds = NCMAPSSDataset(
        str(h5_path),
        str(cycle_hi_path),
        config=config,
        split="train",
    )

    sample = ds[0]
    model_config.setdefault("in_features", int(sample[0].shape[-1]))
    model_config.setdefault("w_summary_dim", W_SUMMARY_DIM)
    model = OC_iVAE(model_config)
    ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    results = extract_hi(model, ds)
    metrics = evaluate_all(results, alpha=config.get("ewma_alpha", 0.3))

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (uid, res) in enumerate(sorted(results.items())):
        ax = axes[i]
        hi_proc = metrics["per_unit"][uid]["hi_proc"]
        hi_gt = metrics["per_unit"][uid]["hi_gt"]
        cycles = res["cycle_ids"]

        ax.plot(cycles, hi_gt, "b--", alpha=0.6, label="HI gt")
        ax.plot(cycles, hi_proc, "r-", alpha=0.8, label="HI pred")
        mono = metrics["per_unit"][uid]["monotonicity"]
        trend = metrics["per_unit"][uid]["trendability"]
        ax.set_title(f"Unit {uid} | Mono={mono:.2f} Trend={trend:.2f}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("HI")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 1.1)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle(
        f"OC-iVAE Baseline | "
        f"Mono={metrics['monotonicity']:.3f} "
        f"Trend={metrics['trendability']:.3f} "
        f"Prog={metrics['prognosability']:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    output_path = Path(config["log_dir"]) / "hi_curves_baseline.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"保存到 {output_path}")


def _resolve_main_h5_path(data_config: dict[str, Any]) -> Path:
    """Resolve configured main h5 path, accepting one local filename variant."""
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


def _resolve_cycle_hi_path(data_config: dict[str, Any], h5_path: Path) -> Path:
    """Resolve cycle-level HI h5 path from config or the main h5 directory."""
    configured_value = data_config.get("cycle_hi_path")
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
