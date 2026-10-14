"""Batch sampler for unit-stratified N-CMAPSS cycle batches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np
from torch.utils.data import Sampler


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
