"""Dataset and unit-stratified sampler for N-CMAPSS cycle samples."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from utils.preprocess import prepare_data


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
    ) -> None:
        """Load cached tensors, then select units and independently attach HI.

        Args:
            h5_path: Original N-CMAPSS source, used to identify prepared data.
            cycle_hi_path: Optional cycle-HI reference; never stored in input cache.
            config: Data settings, including optional processed_dir.
            split: Train/val unit subset of dev, or the original test split.
            val_fraction: Fraction of dev units reserved for validation.
            random_seed: Seed for the unchanged unit split.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if val_fraction < 0.0 or val_fraction > 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        config = {} if config is None else config
        self.split = split
        self.val_fraction = val_fraction
        self.config = dict(config)
        self.processed_path = prepare_data(h5_path, config)
        h5_split = 'test' if split == 'test' else 'dev'
        with h5py.File(self.processed_path, 'r') as source:
            self.L_max = int(source.attrs['L_max'])
            group = source[h5_split]
            keys = [tuple(map(int, key)) for key in group['keys'][()]]
            unit_ids = sorted({unit for unit, _ in keys})
            if split == 'test':
                indices = list(range(len(keys)))
            else:
                selected_units = self._split_units(unit_ids, val_fraction, random_seed)[split]
                indices = [index for index, (unit, _) in enumerate(keys) if unit in selected_units]
            self.samples = [keys[index] for index in indices]
            self.x = torch.from_numpy(group['x'][indices])
            self.mask = torch.from_numpy(group['mask'][indices])
            self.u = torch.from_numpy(group['u'][indices])
            self.W_seq = torch.from_numpy(group['W_seq'][indices])
            self.rul = pd.DataFrame(group['keys'][()], columns=['unit', 'cycle'])
            self.rul['rul'] = group['rul'][()]
            expected = len(self.samples)
            if (self.x.ndim != 3 or self.x.shape[:2] != (expected, self.L_max)
                    or self.W_seq.shape != (expected, self.L_max, 4)
                    or self.mask.shape != (expected, self.L_max)
                    or self.u.shape != (expected, 8 + int(config['n_fc']))
                    or self.x.dtype != torch.float32 or self.W_seq.dtype != torch.float32
                    or self.u.dtype != torch.float32 or self.mask.dtype != torch.bool):
                raise ValueError(f'invalid tensor shapes/dtypes in {self.processed_path}')

        if cycle_hi_path is None:
            self.has_hi = False
            self.hi_lookup = {}
        else:
            self.has_hi = True
            cycle_hi_df = pd.read_hdf(cycle_hi_path, key="df")
            self.hi_lookup = self._build_hi_lookup(cycle_hi_df)

        if self.has_hi:
            self._validate_hi_lookup()

    def __len__(self) -> int:
        """Return the number of cycle samples in this split."""
        return len(self.samples)

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
        if self.has_hi:
            hi = torch.tensor(
                self.hi_lookup[(unit_id, cycle_id)],
                dtype=torch.float32,
            )
        else:
            hi = torch.tensor(float("nan"), dtype=torch.float32)

        return (
            self.x[idx],
            self.mask[idx],
            self.u[idx],
            self.W_seq[idx],
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


def valid_pair_pools(units: torch.Tensor, cycles: torch.Tensor, min_cycle_gap: int) -> list:
    """OCALD source rule: all eligible ordered flight pairs within each unit."""
    if min_cycle_gap <= 0:
        raise ValueError('min_cycle_gap must be positive')
    pools = []
    for unit in torch.unique(units):
        indices = torch.where(units == unit)[0]
        indices = indices[torch.argsort(cycles[indices])]
        ordered_cycles = cycles[indices]
        if torch.unique(ordered_cycles).numel() != len(indices):
            raise ValueError(f'duplicate cycle keys for unit {unit.item()}')
        offsets = torch.nonzero(ordered_cycles[None, :] - ordered_cycles[:, None] >= min_cycle_gap)
        if len(offsets):
            pools.append((unit, indices[offsets]))
    if not pools:
        raise ValueError(f'no same-unit pairs with cycle gap >= {min_cycle_gap}')
    return pools


def sample_pairs(pools: list, count: int) -> torch.Tensor:
    """OCALD: uniform eligible unit, then uniform whole valid pair."""
    if not pools or count <= 0:
        raise ValueError('pair sampling requires nonempty pools and positive count')
    device = pools[0][1].device
    chosen_units = torch.randint(len(pools), (count,), device=device)
    pairs = torch.empty(count, 2, dtype=torch.int64, device=device)
    for index, (_, candidates) in enumerate(pools):
        slots = torch.where(chosen_units == index)[0]
        draws = torch.randint(len(candidates), (len(slots),), device=device)
        pairs[slots] = candidates[draws]
    return pairs


def ocald_joint_schedule(samples: list[tuple[int, int]], epochs: int,
                          batches: int, seed: int) -> dict[str, torch.Tensor]:
    """Source grouped schedule: 4 units x 4 flights, 16 eligible pairs per batch.

    CPU flight selection and pair draws have independent seed-initialized RNGs,
    isolated from model initialization and reparameterization noise.
    """
    units = torch.tensor([unit for unit, _ in samples])
    cycles = torch.tensor([cycle for _, cycle in samples])
    groups = [torch.where(units == unit)[0] for unit in torch.unique(units)]
    if len(groups) < 4 or any(len(g) < 4 or cycles[g].max() - cycles[g].min() < 5 for g in groups):
        raise ValueError('OCALD grouped sampling requires >=4 units, >=4 flights/unit and valid gaps')
    flight_rng = torch.Generator().manual_seed(seed)
    batch_list, pair_list = [], []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for _ in range(epochs * batches):
            selections = []
            for group_index in torch.randperm(len(groups), generator=flight_rng)[:4]:
                group = groups[int(group_index)]
                while True:
                    selected = group[torch.randperm(len(group), generator=flight_rng)[:4]]
                    if cycles[selected].max() - cycles[selected].min() >= 5:
                        selections.append(selected)
                        break
            indices = torch.cat(selections)
            pairs = sample_pairs(valid_pair_pools(units[indices], cycles[indices], 5), 16)
            batch_list.append(indices)
            pair_list.append(pairs)
    return dict(flights=torch.stack(batch_list).reshape(epochs, batches, 16),
                pairs=torch.stack(pair_list).reshape(epochs, batches, 16, 2))


class ShuffledSequentialBatches(Sampler[list[int]]):
    """Proposal-C: form consecutive batches first, shuffle only batch order."""
    def __init__(self, size: int, batch_size: int, seed: int):
        if size < 1 or batch_size < 1:
            raise ValueError('batch sampling requires positive sizes')
        self.batches = [list(range(start, min(start + batch_size, size)))
                        for start in range(0, size, batch_size)]
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for index in torch.randperm(len(self.batches), generator=generator).tolist():
            yield self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


def healthy_training_indices(dataset: NCMAPSSDataset | ShortWindowDataset, cycle_threshold: int = 20) -> list[int]:
    """Select the reference healthy prefix from train units using cycle keys only."""
    if dataset.split != 'train' or dataset.val_fraction == 0:
        raise ValueError('healthy network fitting requires the held-out-unit training split')
    indices = [i for i, key in enumerate(dataset.samples) if key[1] <= cycle_threshold]
    if not indices:
        raise ValueError('no training flights satisfy the healthy cycle threshold')
    return indices


class ShortWindowDataset(Dataset):
    """On-demand within-flight slices of full normalized pre-truncation sequences.

    samples are (unit, cycle, start) in downsampled coordinates; index contains
    only (flight index, start). No overlapping X/W windows are materialized.
    The seven fields are X, all-valid mask, flight u, W, unit, cycle, start.
    Labels remain one row per original flight in rul, never per window.
    """
    def __init__(self, h5_path: str, cycle_hi_path: None = None, config: dict | None = None,
                 split: str = 'train', val_fraction: float = .2, random_seed: int = 42):
        if cycle_hi_path is not None:
            raise ValueError('window labels must be associated only after flight aggregation')
        if split not in ('train', 'val', 'test') or not 0 <= val_fraction <= 1:
            raise ValueError('invalid window split or validation fraction')
        self.config = dict(config or {})
        if self.config.get('input_mode') != 'short_window':
            raise ValueError('ShortWindowDataset requires explicit input_mode: short_window')
        self.split, self.val_fraction = split, val_fraction
        self.window_length, self.stride = self.config['window_length'], self.config['stride']
        self.processed_path = prepare_data(h5_path, self.config)
        self.flight_keys, self.flights_x, self.flights_w, self.flights_u = [], [], [], []
        indices = []
        with h5py.File(self.processed_path, 'r') as source:
            self.L_max = int(source.attrs['L_max'])  # common provenance, NOT input length
            group = source['test' if split == 'test' else 'dev']
            keys, offsets = group['keys'][()], group['offsets'][()]
            units = sorted(set(map(int, keys[:, 0])))
            selected = units if split == 'test' else NCMAPSSDataset._split_units(
                units, val_fraction, random_seed)[split]
            self.rul = pd.DataFrame(keys, columns=['unit', 'cycle']).assign(rul=group['rul'][()])
            short = [tuple(map(int, key)) for i, key in enumerate(keys)
                     if key[0] in selected and offsets[i + 1] - offsets[i] < self.window_length]
            if short:
                raise ValueError(f'flights shorter than window_length={self.window_length}: {short}')
            for i, key in enumerate(keys):
                if key[0] not in selected:
                    continue
                flight = len(self.flight_keys)
                self.flight_keys.append(tuple(map(int, key)))
                self.flights_x.append(torch.from_numpy(group['x'][offsets[i]:offsets[i + 1]]))
                self.flights_w.append(torch.from_numpy(group['W_seq'][offsets[i]:offsets[i + 1]]))
                self.flights_u.append(torch.from_numpy(group['u'][i]))
                indices.extend((flight, start) for start in range(
                    0, int(offsets[i + 1] - offsets[i]) - self.window_length + 1, self.stride))
        self.index = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
        self.samples = np.empty((len(self.index), 3), dtype=np.int64)
        if len(self.index):
            self.samples[:, :2] = np.asarray(self.flight_keys)[self.index[:, 0]]
            self.samples[:, 2] = self.index[:, 1]
        self.valid_mask = torch.ones(self.window_length, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple:
        flight, start = self.index[index]
        stop = start + self.window_length
        unit, cycle = self.flight_keys[flight]
        return (self.flights_x[flight][start:stop], self.valid_mask, self.flights_u[flight],
                self.flights_w[flight][start:stop], unit, cycle, int(start))
