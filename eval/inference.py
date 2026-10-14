"""Health indicator inference utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import NCMAPSSDataset
from model.oc_ivae import OC_iVAE

_INFERENCE_BATCH_SIZE = 64


def extract_hi(
    model: OC_iVAE,
    dataset: NCMAPSSDataset,
    device: str = "cpu",
) -> dict[int, dict]:
    """Extract HI for all cycles in dataset using posterior mean.

    Args:
        model: Trained OC-iVAE model.
        dataset: Dataset to evaluate.
        device: Device used for model inference.

    Returns:
        Dictionary keyed by unit id with sorted cycle ids, raw HI, and optional
        GT HI. ``hi_gt`` is ``None`` when ground truth is unavailable.
    """
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=_INFERENCE_BATCH_SIZE, shuffle=False)
    collected: dict[int, list[tuple[int, float, float]]] = {}

    with torch.no_grad():
        for batch in loader:
            x, mask, u, W_seq, unit_id, cycle_id, hi = (
                tensor.to(torch_device) for tensor in batch
            )
            out = model(x, mask, u, W_seq)
            hi_raw = out["hi_raw"].detach().cpu().numpy()
            unit_np = unit_id.detach().cpu().numpy()
            cycle_np = cycle_id.detach().cpu().numpy()
            hi_batch = hi.detach().cpu().numpy()

            for unit, cycle, hi_value, gt_value in zip(
                unit_np,
                cycle_np,
                hi_raw,
                hi_batch,
            ):
                unit_key = int(unit)
                collected.setdefault(unit_key, []).append(
                    (int(cycle), float(hi_value), float(gt_value))
                )

    results: dict[int, dict] = {}
    for unit_id, rows in collected.items():
        rows_sorted = sorted(rows, key=lambda item: item[0])
        hi_gt = np.asarray(
            [gt_value for _, _, gt_value in rows_sorted],
            dtype=np.float32,
        )
        if np.all(np.isnan(hi_gt)):
            hi_gt = None
        results[unit_id] = {
            "cycle_ids": [cycle for cycle, _, _ in rows_sorted],
            "hi_raw": np.asarray(
                [hi_value for _, hi_value, _ in rows_sorted],
                dtype=np.float32,
            ),
            "hi_gt": hi_gt,
        }

    return dict(sorted(results.items()))
