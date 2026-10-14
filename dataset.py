"""Dataset and unit-stratified sampler for N-CMAPSS cycle samples."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from utils.preprocess import (
    compute_Lmax,
    compute_W_stats,
    compute_unit_stats,
    construct_u,
    group_by_unit_cycle,
    load_h5,
    process_cycle,
)


class NCMAPSSDataset(Dataset):
    """PyTorch dataset for processed N-CMAPSS cycle samples."""

    def __init__(
        self,
        h5_path: str,
        cycle_hi_path: str | None = None,
        config: dict | None = None,
        split: str = "train",
        val_fraction: float = 0.2,
        random_seed: int = 42,
        train_stats: dict | None = None,
        L_max: int | None = None,
    ) -> None:
        """Load raw N-CMAPSS data, split units, and cache preprocessing metadata.

        Args:
            h5_path: Path to the main flat-row N-CMAPSS h5 file.
            cycle_hi_path: Optional path to the cycle-level HI h5 file.
                ``None`` disables ground-truth HI loading.
            config: Dataset config with ``downsample_factor`` and ``n_fc``.
            split: Unit split to expose, ``"train"``, ``"val"``, or ``"test"``.
            val_fraction: Fraction of units assigned to validation.
            random_seed: Seed used for deterministic unit splitting.
            train_stats: Optional normalization stats exported from train data.
            L_max: Optional shared sequence length from train data.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if val_fraction < 0.0 or val_fraction > 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        config = {} if config is None else config
        self.split = split
        self.val_fraction = val_fraction
        self.config = dict(config)
        self.downsample_factor = _get_config_int(config, "downsample_factor")
        self.n_fc = _get_config_int(config, "n_fc")
        self.config["downsample_factor"] = self.downsample_factor
        self.config["n_fc"] = self.n_fc

        h5_split = "test" if self.split == "test" else "dev"
        raw_arrays = load_h5(h5_path, split=h5_split)
        self.cycles = group_by_unit_cycle(
            raw_arrays["A"],
            raw_arrays["W"],
            raw_arrays["X_s"],
            factor=self.downsample_factor,
        )
        self.L_max = int(L_max) if L_max is not None else compute_Lmax(self.cycles)

        if cycle_hi_path is None:
            self.has_hi = False
            self.hi_lookup = {}
        else:
            self.has_hi = True
            cycle_hi_df = pd.read_hdf(cycle_hi_path, key="df")
            self.hi_lookup = self._build_hi_lookup(cycle_hi_df)

        unit_ids = sorted({unit_id for unit_id, _ in self.cycles})
        if self.split == "test":
            self.samples = sorted(self.cycles)
        else:
            split_units = self._split_units(unit_ids, self.val_fraction, random_seed)
            selected_units = split_units[self.split]
            self.samples = sorted(
                (unit_id, cycle_id)
                for unit_id, cycle_id in self.cycles
                if unit_id in selected_units
            )
        if self.has_hi:
            self._validate_hi_lookup()

        if train_stats is None:
            self.W_mean_global, self.W_std_global = compute_W_stats(self.cycles)
            self.unit_stats = {
                unit_id: compute_unit_stats(self.cycles, unit_id)
                for unit_id in unit_ids
            }
        else:
            self.W_mean_global = np.asarray(train_stats["W_mean_global"])
            self.W_std_global = np.asarray(train_stats["W_std_global"])
            self.unit_stats = {
                int(unit_id): stats
                for unit_id, stats in train_stats.items()
                if unit_id not in {"W_mean_global", "W_std_global"}
            }

    def __len__(self) -> int:
        """Return the number of cycle samples in this split."""
        return len(self.samples)

    def get_train_stats(self) -> dict:
        """Export normalization stats for use with test set."""
        stats = dict(self.unit_stats)
        stats["W_mean_global"] = self.W_mean_global
        stats["W_std_global"] = self.W_std_global
        return stats

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return one processed cycle and its scalar HI target or NaN."""
        unit_id, cycle_id = self.samples[idx]
        cycle_data = self.cycles[(unit_id, cycle_id)]
        if unit_id in self.unit_stats:
            mean, std, valid_mask = self.unit_stats[unit_id]
        else:
            mean, std, valid_mask = compute_unit_stats(self.cycles, unit_id)
            self.unit_stats[unit_id] = (mean, std, valid_mask)
        x, W_seq, mask = process_cycle(
            cycle_data,
            self.L_max,
            mean,
            std,
            valid_mask,
            W_mean_global=self.W_mean_global,
            W_std_global=self.W_std_global,
        )
        u = construct_u(
            cycle_data["W"],
            cycle_data["Fc"],
            n_fc=self.config["n_fc"],
            W_mean_global=self.W_mean_global,
            W_std_global=self.W_std_global,
        )

        if self.has_hi:
            hi = torch.tensor(
                self.hi_lookup[(unit_id, cycle_id)],
                dtype=torch.float32,
            )
        else:
            hi = torch.tensor(float("nan"), dtype=torch.float32)

        return (
            x,
            mask,
            u,
            W_seq,
            torch.tensor(unit_id, dtype=torch.int64),
            torch.tensor(cycle_id, dtype=torch.int64),
            hi,
        )

    @staticmethod
    def _build_hi_lookup(cycle_hi_df: pd.DataFrame) -> dict[tuple[int, int], float]:
        """Build ``(unit_id, cycle_id) -> HI`` using row order within each unit."""
        missing_columns = {
            column
            for column in ("unit", "HI")
            if column not in cycle_hi_df.columns
        }
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"cycle_hi_path is missing required columns: {missing}")

        hi_lookup: dict[tuple[int, int], float] = {}
        cycle_counts: dict[int, int] = {}
        for _, row in cycle_hi_df.iterrows():
            unit_id = int(row["unit"])
            cycle_id = cycle_counts.get(unit_id, 0) + 1
            cycle_counts[unit_id] = cycle_id
            hi_lookup[(unit_id, cycle_id)] = float(row["HI"])

        return hi_lookup

    @staticmethod
    def _split_units(
        unit_ids: list[int],
        val_fraction: float,
        random_seed: int,
    ) -> dict[str, set[int]]:
        """Split sorted unit ids into deterministic train and validation sets."""
        shuffled_units = np.asarray(unit_ids, dtype=np.int64)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(shuffled_units)

        n_val = math.ceil(len(unit_ids) * val_fraction)
        val_units = set(shuffled_units[-n_val:].tolist()) if n_val else set()
        train_units = set(shuffled_units.tolist()) - val_units
        return {
            "train": train_units,
            "val": val_units,
        }

    def _validate_hi_lookup(self) -> None:
        """Raise if any exposed sample is missing a cycle-level HI value."""
        for unit_id, cycle_id in self.samples:
            if (unit_id, cycle_id) not in self.hi_lookup:
                raise ValueError(
                    "cycle_hi_path has no HI for "
                    f"unit {unit_id}, cycle {cycle_id}"
                )


def _get_config_int(config: dict, key: str) -> int:
    """Read an integer dataset config value."""
    if key not in config:
        raise ValueError(f"config must include {key!r}")

    value = config[key]
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"config[{key!r}] must be an integer")

    return int(value)


class UnitStratifiedSampler(Sampler[list[int]]):
    """Yield batches containing ``k`` units and ``m`` cycles per unit."""

    def __init__(
        self,
        dataset: Any,
        k: int = 8,
        m: int = 12,
        random_seed: int = 42,
    ) -> None:
        """Build a unit-to-index map from ``dataset.samples``.

        Args:
            dataset: Dataset exposing ``samples`` as ``(unit_id, cycle_id)`` pairs.
            k: Number of units per batch.
            m: Number of cycles sampled per unit in each batch.
            random_seed: Base seed for deterministic per-epoch shuffling.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if m < 1:
            raise ValueError("m must be at least 1")
        if not hasattr(dataset, "samples"):
            raise ValueError("dataset must expose a samples attribute")

        self.k = k
        self.m = m
        self.random_seed = random_seed
        self.epoch_counter = 0

        unit_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            unit_id, _ = sample
            unit_to_indices[int(unit_id)].append(index)

        self.unit_to_indices = dict(unit_to_indices)
        self.unit_ids = sorted(self.unit_to_indices)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one epoch of flat index batches."""
        rng = np.random.default_rng(self.random_seed + self.epoch_counter)
        self.epoch_counter += 1

        shuffled_units = np.asarray(self.unit_ids, dtype=np.int64)
        rng.shuffle(shuffled_units)
        if len(shuffled_units) == 0:
            return

        effective_k = min(self.k, len(shuffled_units))
        stop = len(shuffled_units) - effective_k + 1
        for start in range(0, stop, effective_k):
            batch_units = shuffled_units[start : start + effective_k]
            batch_indices: list[int] = []
            for unit_id in batch_units:
                indices = self.unit_to_indices[int(unit_id)]
                replace = len(indices) < self.m
                sampled = rng.choice(indices, size=self.m, replace=replace)
                batch_indices.extend(int(index) for index in sampled.tolist())

            yield batch_indices

    def __len__(self) -> int:
        """Return the number of K-unit batches per epoch."""
        if 0 < len(self.unit_ids) < self.k:
            return 1
        return len(self.unit_ids) // self.k
