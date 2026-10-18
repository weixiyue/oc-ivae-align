"""Preprocessing utilities for flat N-CMAPSS h5 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
import torch

_A_UNIT_COL = 0
_A_CYCLE_COL = 1
_A_FC_COL = 2
_A_HS_COL = 3
_STD_EPS = 1e-8
_DEAD_SENSOR_STD = 1e-3


def load_h5(h5_path: str, split: str = "dev") -> dict[str, np.ndarray]:
    """Load raw arrays from an h5 file for one split.

    Args:
        h5_path: Path to the raw N-CMAPSS h5 file.
        split: Dataset split, either ``"dev"`` or ``"test"``.

    Returns:
        Dictionary with ``A``, ``W``, ``X_s``, and ``Y`` arrays.
    """
    split_name = split.lower()
    if split_name not in {"dev", "test"}:
        raise ValueError("split must be either 'dev' or 'test'")

    arrays: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5_file:
        for base_key in ("A", "W", "X_s", "Y"):
            dataset_key = _resolve_h5_key(h5_file, base_key, split_name)
            arrays[base_key] = np.asarray(h5_file[dataset_key][()])

    return arrays


def downsample(
    X_s: np.ndarray,
    W: np.ndarray,
    factor: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample aligned sensor and operating-condition arrays by striding.

    Args:
        X_s: Sensor array with shape ``(T, S)``.
        W: Operating-condition array with shape ``(T, 4)``.
        factor: Keep every ``factor``-th timestep.

    Returns:
        Downsampled ``X_s`` and ``W`` arrays.
    """
    if factor < 1:
        raise ValueError("factor must be at least 1")
    if X_s.ndim != 2 or W.ndim != 2:
        raise ValueError("X_s and W must be 2D arrays")
    if X_s.shape[0] != W.shape[0]:
        raise ValueError("X_s and W must have the same length")

    return X_s[::factor], W[::factor]


def group_by_unit_cycle(
    A: np.ndarray,
    W: np.ndarray,
    X_s: np.ndarray,
    factor: int = 10,
) -> dict[tuple[int, int], dict]:
    """Split flat arrays into valid per-(unit, cycle) segments.

    Args:
        A: Auxiliary array with columns ``unit``, ``cycle``, ``Fc``, ``hs``.
        W: Operating-condition array aligned with ``A`` by timestep.
        X_s: Sensor array aligned with ``A`` by timestep.

    Returns:
        Dictionary keyed by ``(unit_id, cycle_id)`` with cycle arrays and ids.
    """
    _validate_aligned_rows(A, W, X_s)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    if A.shape[1] <= _A_HS_COL:
        raise ValueError("A must include unit, cycle, Fc, and hs columns")

    if A.shape[0] == 0:
        return {}

    A_valid = A
    W_valid = W
    X_valid = X_s

    units = A_valid[:, _A_UNIT_COL].astype(np.int64)
    cycles = A_valid[:, _A_CYCLE_COL].astype(np.int64)
    original_order = np.arange(A_valid.shape[0])
    sorted_indices = np.lexsort((original_order, cycles, units))

    sorted_units = units[sorted_indices]
    sorted_cycles = cycles[sorted_indices]
    boundaries = np.flatnonzero(
        (sorted_units[1:] != sorted_units[:-1])
        | (sorted_cycles[1:] != sorted_cycles[:-1])
    ) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_indices.shape[0]]))

    grouped: dict[tuple[int, int], dict] = {}
    for start, end in zip(starts, ends):
        row_indices = sorted_indices[start:end]
        unit = int(sorted_units[start])
        cycle = int(sorted_cycles[start])
        fc_values = A_valid[row_indices, _A_FC_COL].astype(np.int64)
        if not np.all(fc_values == fc_values[0]):
            raise ValueError(f"Fc is not constant for unit {unit}, cycle {cycle}")

        X_seg, W_seg = downsample(
            X_valid[row_indices],
            W_valid[row_indices],
            factor=factor,
        )
        grouped[(unit, cycle)] = {
            "X_s": np.asarray(X_seg),
            "W": np.asarray(W_seg),
            "Fc": int(fc_values[0]),
            "unit": unit,
            "cycle": cycle,
        }

    return grouped


def compute_Lmax(
    cycles: dict[tuple[int, int], dict],
    percentile: float = 90.0,
) -> int:
    """Return the ceil-rounded cycle-length percentile across all cycles.

    Args:
        cycles: Per-cycle dictionary returned by ``group_by_unit_cycle``.
        percentile: Percentile of valid cycle lengths to use.

    Returns:
        P-percentile length rounded up to the nearest integer.
    """
    if not cycles:
        raise ValueError("cycles must not be empty")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be between 0 and 100")

    lengths = np.asarray(
        [cycle_data["X_s"].shape[0] for cycle_data in cycles.values()],
        dtype=np.float64,
    )
    return int(np.ceil(np.percentile(lengths, percentile)))


def compute_unit_stats(
    cycles: dict[tuple[int, int], dict],
    unit_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-unit z-score statistics from all cycles of one unit.

    Args:
        cycles: Per-cycle dictionary returned by ``group_by_unit_cycle``.
        unit_id: Unit id whose sensor statistics should be computed.

    Returns:
        Mean, std with numerical epsilon added, and non-dead-sensor mask.
    """
    unit_arrays = [
        np.asarray(cycle_data["X_s"])
        for (unit, _), cycle_data in cycles.items() if unit == unit_id
    ]
    if not unit_arrays:
        raise ValueError(f"unit_id {unit_id} was not found in cycles")

    unit_sensors = np.concatenate(unit_arrays, axis=0)
    if unit_sensors.shape[0] == 0:
        raise ValueError(f"unit_id {unit_id} has no valid timesteps")

    mean = unit_sensors.mean(axis=0)
    raw_std = unit_sensors.std(axis=0)
    valid_mask = raw_std >= _DEAD_SENSOR_STD
    std = raw_std + _STD_EPS
    return mean, std, valid_mask


def compute_W_stats(
    cycles: dict[tuple[int, int], dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std of W across all cycles and timesteps.

    Args:
        cycles: Per-cycle dictionary returned by ``group_by_unit_cycle``.

    Returns:
        Global operating-condition mean and std arrays.
    """
    if not cycles:
        raise ValueError("cycles must not be empty")

    all_W = np.concatenate(
        [cycle_data["W"] for cycle_data in cycles.values()],
        axis=0,
    )
    return all_W.mean(axis=0), all_W.std(axis=0) + _STD_EPS


def process_cycle(
    cycle_data: dict,
    L_max: int,
    mean: np.ndarray,
    std: np.ndarray,
    valid_mask: np.ndarray,
    W_mean_global: np.ndarray | None = None,
    W_std_global: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize one cycle and return padded/truncated model-ready tensors.

    Args:
        cycle_data: One cycle dictionary from ``group_by_unit_cycle``.
        L_max: Target sequence length.
        mean: Per-sensor z-score mean for this unit.
        std: Per-sensor z-score std for this unit.
        valid_mask: Boolean mask marking non-dead sensors.
        W_mean_global: Optional global W mean for normalization.
        W_std_global: Optional global W std for normalization.

    Returns:
        ``x``, ``W_seq``, and boolean timestep mask tensors.
    """
    if L_max <= 0:
        raise ValueError("L_max must be positive")

    x_norm, W = normalize_cycle(cycle_data, mean, std, valid_mask, W_mean_global, W_std_global)

    x_padded, W_padded, timestep_mask = _tail_truncate_or_head_pad(
        x_norm,
        W,
        L_max,
    )
    return (
        torch.from_numpy(x_padded),
        torch.from_numpy(W_padded),
        torch.from_numpy(timestep_mask),
    )


def normalize_cycle(cycle_data: dict, mean: np.ndarray, std: np.ndarray,
                    valid_mask: np.ndarray, W_mean_global: np.ndarray | None = None,
                    W_std_global: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shared float32 normalization BEFORE any truncation or padding."""
    X_s = np.asarray(cycle_data["X_s"], dtype=np.float32)
    W = np.asarray(cycle_data["W"], dtype=np.float32)
    _validate_cycle_arrays(X_s, W)
    if (W_mean_global is None) != (W_std_global is None):
        raise ValueError("W_mean_global and W_std_global must be provided together")
    if W_mean_global is not None and W_std_global is not None:
        W_mean_array = np.asarray(W_mean_global, dtype=np.float32)
        W_std_array = np.asarray(W_std_global, dtype=np.float32)
        if W_mean_array.shape != (W.shape[1],):
            raise ValueError("W_mean_global shape must match W feature dimension")
        if W_std_array.shape != (W.shape[1],):
            raise ValueError("W_std_global shape must match W feature dimension")
        W = (W - W_mean_array) / W_std_array

    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    sensor_mask = np.asarray(valid_mask, dtype=bool)
    _validate_stat_shapes(X_s, mean_array, std_array, sensor_mask)

    x_norm = np.zeros_like(X_s, dtype=np.float32)
    if X_s.shape[0] > 0 and np.any(sensor_mask):
        x_norm[:, sensor_mask] = (
            X_s[:, sensor_mask] - mean_array[sensor_mask]
        ) / std_array[sensor_mask]

    return x_norm, W


def construct_u(
    W_cycle: np.ndarray,
    Fc: int,
    n_fc: int = 3,
    W_mean_global: np.ndarray | None = None,
    W_std_global: np.ndarray | None = None,
) -> torch.FloatTensor:
    """Build the auxiliary variable from optionally normalized W stats and Fc.

    Args:
        W_cycle: Raw operating-condition array for valid timesteps only.
        Fc: One-indexed flight condition id.
        n_fc: Number of flight-condition classes.
        W_mean_global: Optional global W mean for normalization.
        W_std_global: Optional global W std for normalization.

    Returns:
        Float tensor ``[W_mean, W_std, Fc_onehot]``.
    """
    if n_fc <= 0:
        raise ValueError("n_fc must be positive")
    if Fc < 1 or Fc > n_fc:
        raise ValueError("Fc must be in the 1-indexed range [1, n_fc]")

    W_array = np.asarray(W_cycle, dtype=np.float32)
    if W_array.ndim != 2:
        raise ValueError("W_cycle must be a 2D array")
    if W_array.shape[0] == 0:
        raise ValueError("W_cycle must contain at least one valid timestep")
    if (W_mean_global is None) != (W_std_global is None):
        raise ValueError("W_mean_global and W_std_global must be provided together")

    if W_mean_global is not None and W_std_global is not None:
        W_mean_array = np.asarray(W_mean_global, dtype=np.float32)
        W_std_array = np.asarray(W_std_global, dtype=np.float32)
        if W_mean_array.shape != (W_array.shape[1],):
            raise ValueError("W_mean_global shape must match W feature dimension")
        if W_std_array.shape != (W_array.shape[1],):
            raise ValueError("W_std_global shape must match W feature dimension")
        W_array = (W_array - W_mean_array) / W_std_array

    W_mean = W_array.mean(axis=0)
    W_std = W_array.std(axis=0) + _STD_EPS
    Fc_onehot = np.eye(n_fc, dtype=np.float32)[Fc - 1]
    u = np.concatenate((W_mean, W_std, Fc_onehot)).astype(np.float32)
    return torch.from_numpy(u)


def _resolve_h5_key(h5_file: object, base_key: str, split: str) -> str:
    """Return the h5 dataset key for a base array and split."""
    candidate_keys = (f"{base_key}_{split}", f"{split}/{base_key}", base_key)
    for candidate_key in candidate_keys:
        if candidate_key in h5_file:
            return candidate_key
    raise KeyError(
        f"could not find dataset for {base_key!r}; tried {candidate_keys}"
    )


def _validate_aligned_rows(*arrays: np.ndarray) -> None:
    """Validate that all arrays are 2D and timestep-aligned."""
    row_count: int | None = None
    for array in arrays:
        if array.ndim != 2:
            raise ValueError("all raw arrays must be 2D")
        if row_count is None:
            row_count = array.shape[0]
        elif array.shape[0] != row_count:
            raise ValueError("A, W, and X_s must have the same number of rows")


def _validate_cycle_arrays(X_s: np.ndarray, W: np.ndarray) -> None:
    """Validate cycle arrays before per-cycle processing."""
    if X_s.ndim != 2 or W.ndim != 2:
        raise ValueError("cycle X_s and W arrays must be 2D")
    if X_s.shape[0] != W.shape[0]:
        raise ValueError("cycle X_s and W must have the same length")


def _validate_stat_shapes(
    X_s: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Validate per-sensor statistic shapes against a cycle sensor array."""
    sensor_dim = X_s.shape[1]
    expected_shape = (sensor_dim,)
    if mean.shape != expected_shape:
        raise ValueError("mean shape must match the sensor dimension")
    if std.shape != expected_shape:
        raise ValueError("std shape must match the sensor dimension")
    if valid_mask.shape != expected_shape:
        raise ValueError("valid_mask shape must match the sensor dimension")


def _tail_truncate_or_head_pad(
    X_s: np.ndarray,
    W: np.ndarray,
    L_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tail-truncate or head-pad aligned cycle arrays to ``L_max``."""
    length = X_s.shape[0]
    x_out = np.zeros((L_max, X_s.shape[1]), dtype=np.float32)
    W_out = np.zeros((L_max, W.shape[1]), dtype=np.float32)
    mask = np.zeros((L_max,), dtype=bool)

    if length >= L_max:
        x_out[...] = X_s[-L_max:]
        W_out[...] = W[-L_max:]
        mask[...] = True
        return x_out, W_out, mask

    start = L_max - length
    x_out[start:] = X_s
    W_out[start:] = W
    mask[start:] = True
    return x_out, W_out, mask


# Increment when preprocessing computations or the stored schema change.
PROCESSED_VERSION = 1
PROCESSED_DIR = Path(__file__).resolve().parents[1] / 'data/processed'


def prepare_data(h5_path: str, config: dict) -> Path:
    """Reuse or build model-ready dev/test tensors without overwriting a cache.

    The cache identity excludes model settings and unit split seeds. Test uses
    dev W statistics and L_max, and dev sensor statistics for matching unit ids.
    Labels are kept separate from model inputs. Source identity uses file stats,
    not a full content hash, to avoid rereading the raw file on every launch.
    """
    mode = config.get('input_mode', 'flight')
    if mode == 'short_window':
        return prepare_window_sequences(h5_path, config)
    if mode != 'flight':
        raise ValueError(f'unknown input_mode: {mode}')
    source = Path(h5_path).resolve(strict=True)
    parameters = {}
    for key in ('downsample_factor', 'n_fc'):
        value = config.get(key)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f'data.{key} must be a positive integer')
        parameters[key] = int(value)
    stat = source.stat()
    metadata = dict(
        version=PROCESSED_VERSION, source=str(source), size=stat.st_size,
        mtime_ns=stat.st_mtime_ns, **parameters,
    )
    encoded = json.dumps(metadata, sort_keys=True)
    identity = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    directory = Path(config.get('processed_dir', PROCESSED_DIR)).resolve()
    destination = directory / f'{source.stem}_{identity}.h5'
    if destination.exists():
        with h5py.File(destination, 'r') as cached:
            if cached.attrs.get('metadata') != encoded or not cached.attrs.get('complete', False):
                raise ValueError(f'invalid processed data: {destination}')
        return destination

    directory.mkdir(parents=True, exist_ok=True)
    print(f'Preparing N-CMAPSS data: {source} -> {destination}', flush=True)
    # Publish only a completed file; exclusive linking never replaces an existing
    # cache, including one created concurrently by another process.
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with h5py.File(temporary, 'w') as target:
            target.attrs['metadata'] = encoded
            dev_stats = {}
            W_mean = W_std = None
            L_max = None
            for split in ('dev', 'test'):
                arrays = load_h5(str(source), split)
                rul = cycle_rul(arrays['A'], arrays['Y'])
                cycles = group_by_unit_cycle(
                    arrays['A'], arrays['W'], arrays['X_s'],
                    factor=parameters['downsample_factor'],
                )
                del arrays
                keys = sorted(cycles)
                units = sorted({unit for unit, _ in keys})
                if split == 'dev':
                    L_max = compute_Lmax(cycles)
                    W_mean, W_std = compute_W_stats(cycles)
                    dev_stats = {unit: compute_unit_stats(cycles, unit) for unit in units}
                    stats = dev_stats
                    target.attrs['L_max'] = L_max
                    target.create_dataset('W_mean_global', data=W_mean)
                    target.create_dataset('W_std_global', data=W_std)
                else:
                    stats = {
                        unit: dev_stats[unit] if unit in dev_stats
                        else compute_unit_stats(cycles, unit)
                        for unit in units
                    }
                group = target.create_group(split)
                group.create_dataset('keys', data=np.asarray(keys, dtype=np.int64))
                ordered_rul = rul.set_index(['unit', 'cycle']).loc[keys, 'rul'].to_numpy()
                group.create_dataset('rul', data=ordered_rul)
                stat_group = group.create_group('sensor_stats')
                stat_group.create_dataset('units', data=units)
                for index, name in enumerate(('mean', 'std', 'valid_mask')):
                    stat_group.create_dataset(name, data=np.stack([stats[unit][index] for unit in units]))
                for index, key in enumerate(keys):
                    cycle = cycles[key]
                    mean, std, valid_mask = stats[key[0]]
                    x, W_seq, mask = process_cycle(
                        cycle, L_max, mean, std, valid_mask, W_mean, W_std,
                    )
                    u = construct_u(cycle['W'], cycle['Fc'], parameters['n_fc'], W_mean, W_std)
                    for name, tensor in (('x', x), ('W_seq', W_seq), ('mask', mask), ('u', u)):
                        array = tensor.numpy()
                        if index == 0:
                            group.create_dataset(
                                name, shape=(len(keys), *array.shape), dtype=array.dtype,
                                chunks=(1, *array.shape), compression='lzf',
                            )
                        group[name][index] = array
                del cycles
            if source.stat().st_size != metadata['size'] or source.stat().st_mtime_ns != metadata['mtime_ns']:
                raise ValueError('raw data changed during preprocessing')
            target.attrs['complete'] = True
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    print(f'Prepared data saved: {destination}', flush=True)
    return destination


def prepare_window_sequences(h5_path: str, config: dict) -> Path:
    """Cache full normalized flights using EXISTING common statistics, never tail tensors.

    Only sequence storage and window identity are new. The common flight cache,
    its statistics, old identity and all existing results remain unchanged.
    """
    for name in ('window_length', 'stride'):
        value = config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'data.{name} must be a positive integer')
    base_config = dict(config, input_mode='flight')
    base = prepare_data(h5_path, base_config)
    with h5py.File(base, 'r') as common:
        base_metadata = common.attrs['metadata']
    metadata = dict(version=1, input_mode='short_window', window_length=config['window_length'],
                    stride=config['stride'], rule='within_flight_before_truncation_no_padding',
                    normalization_source=str(base), common_metadata=json.loads(base_metadata))
    encoded = json.dumps(metadata, sort_keys=True)
    identity = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    destination = base.parent / f'{Path(h5_path).stem}_short_window_{identity}.h5'
    if destination.exists():
        with h5py.File(destination, 'r') as cached:
            if cached.attrs.get('metadata') != encoded or not cached.attrs.get('complete', False):
                raise ValueError(f'invalid short-window sequences: {destination}')
        return destination
    print(f'Preparing full pre-truncation sequences: {destination}', flush=True)
    with tempfile.NamedTemporaryFile(dir=base.parent, suffix='.tmp', delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with h5py.File(temporary, 'w') as target, h5py.File(base, 'r') as common:
            target.attrs['metadata'] = encoded
            target.attrs['L_max'] = common.attrs['L_max']  # provenance only, never a window limit
            for split in ('dev', 'test'):
                arrays = load_h5(h5_path, split)
                cycles = group_by_unit_cycle(arrays['A'], arrays['W'], arrays['X_s'],
                                             factor=config['downsample_factor'])
                del arrays
                keys = sorted(cycles)
                expected_keys = common[split]['keys'][()]
                if not np.array_equal(expected_keys, keys):
                    raise ValueError('raw flight keys differ from normalization cache')
                lengths = np.asarray([len(cycles[k]['X_s']) for k in keys])
                short = [keys[i] for i in np.flatnonzero(lengths < config['window_length'])]
                if short:
                    raise ValueError(f'{split} flights shorter than window_length={config["window_length"]}: {short}')
                group = target.create_group(split)
                group.create_dataset('keys', data=expected_keys)
                group.create_dataset('rul', data=common[split]['rul'][()])
                group.create_dataset('u', data=common[split]['u'][()])
                offsets = np.r_[0, lengths.cumsum()]
                group.create_dataset('offsets', data=offsets)
                x_store = group.create_dataset('x', shape=(int(offsets[-1]), 14), dtype='f4', compression='lzf')
                w_store = group.create_dataset('W_seq', shape=(int(offsets[-1]), 4), dtype='f4', compression='lzf')
                stats = common[split]['sensor_stats']
                unit_index = {int(u): i for i, u in enumerate(stats['units'][()])}
                for i, key in enumerate(keys):
                    j = unit_index[key[0]]
                    x, w = normalize_cycle(cycles[key], stats['mean'][j], stats['std'][j],
                                            stats['valid_mask'][j], common['W_mean_global'][()],
                                            common['W_std_global'][()])
                    x_store[offsets[i]:offsets[i + 1]] = x
                    w_store[offsets[i]:offsets[i + 1]] = w
                del cycles
            source = Path(h5_path).resolve()
            original = metadata['common_metadata']
            if source.stat().st_size != original['size'] or source.stat().st_mtime_ns != original['mtime_ns']:
                raise ValueError('raw data changed during short-window preprocessing')
            target.attrs['complete'] = True
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return destination


def cycle_rul(a: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Validate original rows and retain one unchanged RUL value per flight."""
    if y.shape not in [(len(a),), (len(a), 1)]:
        raise ValueError('Y must contain one original RUL value per row')
    rows = pd.DataFrame({'unit': a[:, 0], 'cycle': a[:, 1], 'rul': y.ravel()})
    if not np.isfinite(rows.to_numpy()).all():
        raise ValueError('nonfinite original A keys or Y RUL')
    if not np.equal(rows[['unit', 'cycle']], np.floor(rows[['unit', 'cycle']])).all().all():
        raise ValueError('fractional original flight keys')
    grouped = rows.groupby(['unit', 'cycle'], sort=True).rul.agg(['min', 'max'])
    if (grouped['min'] != grouped['max']).any():
        raise ValueError('original Y RUL is not constant within flight')
    return grouped['min'].rename('rul').reset_index().astype({'unit': int, 'cycle': int})


def main() -> None:
    """Prepare or reuse the same processed file consumed by training/evaluation."""
    import yaml
    from train import _resolve_main_h5_path

    parser = argparse.ArgumentParser(description='Prepare N-CMAPSS model inputs once.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get('data'), dict):
        raise ValueError('config must include a data mapping')
    data = config['data']
    path = prepare_data(str(_resolve_main_h5_path(data)), data)
    print(f'Processed data: {path}')


if __name__ == '__main__':
    main()
