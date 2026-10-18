"""Residual-A/B, UnsupervisedHI 157ebfae; adapted to shared OC-iVAE tensors."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn


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


def fit_residual_pca(residuals: np.ndarray, keys: list[tuple[int, int]], seed: int) -> dict:
    """Fit on ALL dev flights; orient by unit early-minus-late ten-cycle means.

    Residual columns are time then sensor, including all common padding positions.
    Keys, not HI/RUL labels, determine ordering and orientation. PCA transform is
    also used on dev so independent restoration uses the identical scoring path.
    """
    if residuals.ndim != 2 or len(residuals) != len(keys) or not np.isfinite(residuals).all():
        raise ValueError('PCA requires finite flattened residuals matching dev keys')
    key_array = np.asarray(keys, dtype=np.int64)
    if key_array.shape != (len(residuals), 2) or len(set(map(tuple, key_array))) != len(keys):
        raise ValueError('PCA requires unique unit/cycle keys')
    pca = PCA(n_components=1, random_state=seed).fit(residuals)
    scores = pca.transform(residuals).ravel()
    differences = []
    for unit in np.unique(key_array[:, 0]):
        indices = np.flatnonzero(key_array[:, 0] == unit)
        indices = indices[np.argsort(key_array[indices, 1])]
        unit_scores = scores[indices]
        differences.append(float(unit_scores[:10].mean() - unit_scores[-10:].mean()))
    difference = float(np.mean(differences))
    return dict(mean=pca.mean_, components=pca.components_, direction=-1 if difference < 0 else 1,
                early_minus_late_mean=difference, fit_split='dev', fit_keys=key_array.tolist(),
                n_samples_seen=len(keys), flatten_order='time_then_channels_including_padding',
                n_components=1, solver=pca._fit_svd_solver, random_state=seed,
                explained_variance=pca.explained_variance_, labels_used=False)


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
