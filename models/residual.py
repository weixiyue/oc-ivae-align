"""Residual-A/B, UnsupervisedHI 157ebfae; adapted to shared OC-iVAE tensors."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader


class ResidualModel(nn.Module):
    """A reconstructs from X,W; B predicts X from W alone, including padding."""
    def __init__(self, config: dict):
        super().__init__()
        self.variant = config['variant']
        if self.variant not in ('A', 'B'):
            raise ValueError('Residual variant must be A or B')
        if (config['channels'], config['kernel_size'], config['stride'], config['padding']) != (
                [64, 64, 64, 64], 11, 1, 'same'):
            raise ValueError('Residual models require four 64-channel layers, kernel 11, stride 1, same')
        if (config['in_features'], config['condition_dim']) != (14, 4):
            raise ValueError('Residual models require 14 sensors and 4 conditions')
        previous = config['condition_dim'] + (config['in_features'] if self.variant == 'A' else 0)
        layers = []
        for channels in config['channels']:
            layers.extend([nn.Conv1d(previous, channels, 11, padding=5),
                           nn.Tanh() if self.variant == 'A' else nn.ReLU()])
            previous = channels
        self.convolutions = nn.Sequential(*layers)
        self.output = (nn.Linear(previous, 14) if self.variant == 'A'
                       else nn.Conv1d(previous, 14, 11, padding=5))
        # Keras Conv1D and Dense defaults; no claim of identical cross-framework RNG.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((x, w), dim=-1) if self.variant == 'A' else w
        hidden = self.convolutions(inputs.transpose(1, 2))
        if self.variant == 'A':
            return self.output(hidden.transpose(1, 2))
        return self.output(hidden).transpose(1, 2)


def fit_residual_pca(residuals: np.ndarray, keys, seed: int) -> dict:
    """Ordinary sklearn PCA on complete dev; orient BEFORE any flight aggregation.

    Keys are (unit,cycle) for legacy flights or (unit,cycle,start) for windows.
    The first/last ten entries per unit are ordered by cycle and window start.
    Use transform on dev too, so saved numeric parameters suffice for reload.
    """
    if residuals.ndim != 2 or len(residuals) != len(keys):
        raise ValueError('PCA requires flattened residuals matching dev keys')
    for start in range(0, len(residuals), 8192):
        if not np.isfinite(residuals[start:start + 8192]).all():
            raise ValueError('PCA requires finite residuals')
    key_array = np.asarray(keys, dtype=np.int64)
    if key_array.ndim != 2 or key_array.shape[1] not in (2, 3):
        raise ValueError('PCA requires unit/cycle[/window-start] keys')
    if len(np.unique(key_array, axis=0)) != len(keys):
        raise ValueError('PCA requires unique fit keys')
    # Keep normal PCA and its auto solver. Never fall back to IncrementalPCA.
    # For the large N, small window dimension case, sklearn auto uses covariance
    # eigendecomposition; the input itself is an on-disk residual memmap.
    pca = PCA(n_components=1, random_state=seed).fit(residuals)
    state = dict(mean=pca.mean_, components=pca.components_, direction=1, fit_split='dev')
    scores = np.empty(len(keys), dtype=pca.components_.dtype)
    for start in range(0, len(keys), 8192):
        scores[start:start + 8192] = residual_scores(residuals[start:start + 8192], state)
    differences = []
    for unit in np.unique(key_array[:, 0]):
        indices = np.flatnonzero(key_array[:, 0] == unit)
        order = (np.lexsort((key_array[indices, 2], key_array[indices, 1]))
                 if key_array.shape[1] == 3 else np.argsort(key_array[indices, 1]))
        unit_scores = scores[indices[order]]
        differences.append(float(unit_scores[:10].mean() - unit_scores[-10:].mean()))
    difference = float(np.mean(differences))
    windows = key_array.shape[1] == 3
    state.update(direction=-1 if difference < 0 else 1, early_minus_late_mean=difference,
                 fit_keys=(np.unique(key_array[:, :2], axis=0) if windows else key_array).tolist(),
                 n_samples_seen=len(keys), input_mode='short_window' if windows else 'flight',
                 flatten_order='time_then_channels_no_padding' if windows
                 else 'time_then_channels_including_padding', n_components=1,
                 solver=pca._fit_svd_solver, random_state=seed,
                 explained_variance=pca.explained_variance_, labels_used=False)
    if windows:
        state.update(fit_key_columns=['unit', 'cycle', 'window_start'],
                     fit_window_key_sha256=hashlib.sha256(key_array).hexdigest(),
                     direction_entries='first10_minus_last10_windows_per_dev_unit',
                     window_aggregation='mean_per_unit_cycle_after_direction')
    return state


def residual_scores(residuals: np.ndarray, state: dict) -> np.ndarray:
    """Restore sklearn non-whitened PCA transform from numerical parameters."""
    components, mean = state['components'], state['mean']
    if (components.shape != (1, residuals.shape[1]) or mean.shape != (residuals.shape[1],)
            or state['direction'] not in (-1, 1) or state['fit_split'] != 'dev'):
        raise ValueError('residual PCA state does not match fixed-length residuals/dev protocol')
    # sklearn projects before subtracting projected mean to avoid a centered copy.
    scores = (residuals @ components.T - mean.reshape(1, -1) @ components.T).ravel()
    scores = scores * state['direction']
    if not np.isfinite(scores).all():
        raise FloatingPointError('nonfinite residual PCA scores')
    return scores


@contextmanager
def window_residual_matrix(model: ResidualModel, dataset, device: str, batch_size: int):
    """Generate one disk-backed residual matrix in batches; always remove it on exit."""
    model.to(device).eval()
    with tempfile.TemporaryDirectory(prefix='residual_window_matrix_') as directory:
        matrix = np.memmap(Path(directory) / 'residuals.f32', dtype=np.float32, mode='w+',
                           shape=(len(dataset), dataset.window_length * 14))
        try:
            offset = 0
            with torch.no_grad():
                for x, _, _, w, _, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                    x, w = x.to(device), w.to(device)
                    residual = (model(x, w) - x).abs().reshape(len(x), -1)
                    if not torch.isfinite(residual).all():
                        raise FloatingPointError('nonfinite window residuals')
                    matrix[offset:offset + len(x)] = residual.cpu().numpy()
                    offset += len(x)
            matrix.flush()
            yield matrix
        finally:
            matrix._mmap.close()


def predict_window_flights(model: ResidualModel, dataset, state: dict,
                           device: str, batch_size: int) -> pd.DataFrame:
    """Apply saved window PCA/direction, then average windows to one score per flight."""
    if (state.get('input_mode') != 'short_window' or state['window_length'] != dataset.window_length
            or state['stride'] != dataset.stride):
        raise ValueError('saved PCA input mode/window settings differ from window dataset')
    with window_residual_matrix(model, dataset, device, batch_size) as matrix:
        scores = np.empty(len(dataset), dtype=np.float64)
        for start in range(0, len(dataset), 8192):
            scores[start:start + 8192] = residual_scores(matrix[start:start + 8192], state)
    windows = pd.DataFrame(dataset.samples, columns=['unit', 'cycle', 'window_start']).assign(score=scores)
    flights = windows.groupby(['unit', 'cycle'], sort=True, as_index=False).score.mean()
    if flights[['unit', 'cycle']].values.tolist() != [list(k) for k in dataset.flight_keys]:
        raise ValueError('window aggregation does not cover every original flight exactly once')
    return flights
