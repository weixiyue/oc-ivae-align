"""Proposal-C CNN from UnsupervisedHI 157ebfae, using KGLD input tensors."""
from __future__ import annotations

import torch
from torch import nn

SOURCE_COMMIT = '157ebfae7baeee1e0cd52180f8e58ead3dcb448e'


class ProposalCNN(nn.Module):
    """Keras channels-last Flatten, scalar Z, repeated Z plus per-step W."""
    def __init__(self, config: dict):
        super().__init__()
        self.window_size = config['window_size']
        if config['channels'] != [128, 64, 32, 16] or config['kernel_size'] != 85:
            raise ValueError('Proposal-C requires channels [128,64,32,16], kernel 85')
        if (config['stride'], config['padding'], config['activation'], config['z_dim'],
                config['flatten_order']) != (1, 'same', 'relu', 1, 'time_then_channels'):
            raise ValueError('Proposal-C requires stride 1, same/ReLU, scalar Z and time-first Flatten')
        encoder = []
        previous = config['in_features']
        for channels in config['channels']:
            encoder.extend([nn.Conv1d(previous, channels, 85, padding=42), nn.ReLU()])
            previous = channels
        self.encoder = nn.Sequential(*encoder)
        self.z = nn.Linear(self.window_size * previous, 1)
        decoder = []
        previous = 1 + config['condition_dim']
        for channels in reversed(config['channels']):
            decoder.extend([nn.Conv1d(previous, channels, 85, padding=42), nn.ReLU()])
            previous = channels
        self.decoder = nn.Sequential(*decoder)
        self.output = nn.Linear(previous, config['in_features'])
        # Keras Conv1D/Dense defaults: Glorot uniform kernels, zero biases.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.window_size:
            raise ValueError('Proposal-C fixed window_size differs from prepared L_max')
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return self.z(h.reshape(len(x), -1)).squeeze(-1)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        repeated = z[:, None, None].expand(-1, x.shape[1], -1)
        h = self.decoder(torch.cat((repeated, w), dim=-1).transpose(1, 2))
        return dict(z=z, x_hat=self.output(h.transpose(1, 2)))


def proposal_loss(x: torch.Tensor, output: dict, cycles: torch.Tensor,
                  correlation_weight: float = 1.) -> dict[str, torch.Tensor]:
    """Unmasked mean MAE + batch Pearson(Z, cycle); no epsilon or substitution."""
    z = output['z']
    t = cycles.to(dtype=z.dtype).reshape(-1)
    centered_z, centered_t = z - z.mean(), t - t.mean()
    numerator = (centered_z * centered_t).sum()
    denominator = centered_z.square().sum().pow(.5) * centered_t.square().sum().pow(.5)
    correlation = numerator / denominator
    mae = (output['x_hat'] - x).abs().mean()
    loss = mae + correlation_weight * correlation
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite Proposal-C MAE/Pearson loss (check zero batch variance); no epsilon added')
    return dict(loss=loss, mae=mae, correlation=correlation)
