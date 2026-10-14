"""Non-causal TCN decoder for reconstructing sensor sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.backbone import MaskedLayerNorm, TCNResBlock
from model.constants import (
    C_HIDDEN,
    D_ZC,
    D_ZD,
    DECODER_DILATIONS,
    N_DECODER_LAYERS,
    W_DIM,
    W_SUMMARY_DIM,
)

_DEFAULT_KERNEL_SIZE = 3
_DEFAULT_DILATION = 1
_DEFAULT_DROPOUT = 0.1
_N_SENSOR_FEATURES = 14
_VAR_EPS = 1e-8


class NonCausalResBlock(nn.Module):
    """Non-causal dilated TCN residual block for decoder features."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = _DEFAULT_KERNEL_SIZE,
        dilation: int = _DEFAULT_DILATION,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        """Initialize a two-layer non-causal residual block.

        Args:
            channels: Number of input and output channels.
            kernel_size: Temporal convolution kernel size.
            dilation: Dilation used by the first convolution.
            dropout: Dropout probability after GELU activations.
        """
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        if dilation < 1:
            raise ValueError("dilation must be at least 1")

        padding = dilation * (kernel_size - 1) // 2
        second_padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = MaskedLayerNorm(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=_DEFAULT_DILATION,
            padding=second_padding,
        )
        self.norm2 = MaskedLayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        """Apply the block to ``h`` with shape ``(B, T, C)``."""
        _validate_sequence_and_mask(h, mask)
        mask_float = mask.float().unsqueeze(-1)

        y = self.conv1(h.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm1(y, mask)))
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm2(y, mask)))
        return (h + y) * mask_float


class TCNDecoder(nn.Module):
    """Reconstruct sensor sequences from latent variables and W summary."""

    def __init__(self, config: dict) -> None:
        """Initialize the non-causal TCN decoder.

        Args:
            config: Model config with ``c_hidden``, ``n_decoder_layers``,
                ``d_zd``, ``d_zc``, and ``w_summary_dim``.
        """
        super().__init__()
        c_hidden = int(config.get("c_hidden", C_HIDDEN))
        n_decoder_layers = int(config.get("n_decoder_layers", N_DECODER_LAYERS))
        d_zd = int(config.get("d_zd", D_ZD))
        d_zc = int(config.get("d_zc", D_ZC))
        w_summary_dim = int(config.get("w_summary_dim", W_SUMMARY_DIM))
        if n_decoder_layers < 1:
            raise ValueError("n_decoder_layers must be at least 1")
        if n_decoder_layers > len(DECODER_DILATIONS):
            raise ValueError("n_decoder_layers exceeds available decoder dilations")

        self.d_zd = d_zd
        self.d_zc = d_zc
        self.w_summary_dim = w_summary_dim
        self.input_projection = nn.Linear(d_zd + d_zc + w_summary_dim, c_hidden)
        self.layers = nn.ModuleList(
            [
                NonCausalResBlock(
                    c_hidden,
                    kernel_size=_DEFAULT_KERNEL_SIZE,
                    dilation=dilation,
                )
                for dilation in DECODER_DILATIONS[:n_decoder_layers]
            ]
        )
        self.output_projection = nn.Linear(c_hidden, _N_SENSOR_FEATURES)

    def forward(
        self,
        z_d: Tensor,
        z_c: Tensor,
        W_seq: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Decode latents into ``x_hat`` with shape ``(B, T, 14)``."""
        _validate_latent(z_d, self.d_zd, "z_d")
        _validate_latent(z_c, self.d_zc, "z_c")
        _validate_sequence_and_mask(W_seq, mask)
        if W_seq.shape[-1] != W_DIM:
            raise ValueError("W_seq feature dimension does not match W_DIM")
        if z_d.shape[0] != W_seq.shape[0] or z_c.shape[0] != W_seq.shape[0]:
            raise ValueError("z_d, z_c, and W_seq must share batch size")

        w_summary = _masked_w_summary(W_seq, mask)
        if w_summary.shape[-1] != self.w_summary_dim:
            raise ValueError("computed W summary does not match w_summary_dim")

        z = torch.cat((z_d, z_c, w_summary), dim=-1)
        h = self.input_projection(z)
        h = h.unsqueeze(1).expand(-1, W_seq.shape[1], -1)
        mask_float = mask.float().unsqueeze(-1)
        h = h * mask_float
        for layer in self.layers:
            h = layer(h, mask)

        x_hat = self.output_projection(h)
        return x_hat * mask_float


def _masked_w_summary(W_seq: Tensor, mask: Tensor) -> Tensor:
    """Return concatenated masked mean and std for ``W_seq``."""
    W_mean = _masked_avg(W_seq, mask)
    W_std = _masked_std(W_seq, mask, W_mean)
    return torch.cat((W_mean, W_std), dim=-1)


def _masked_avg(sequence: Tensor, mask: Tensor) -> Tensor:
    """Average valid timesteps in ``sequence``."""
    mask_float = mask.float().unsqueeze(-1)
    count = mask.float().sum(dim=1, keepdim=True)
    return (sequence * mask_float).sum(dim=1) / count.clamp(min=1)


def _masked_std(sequence: Tensor, mask: Tensor, mean: Tensor) -> Tensor:
    """Compute valid-timestep standard deviation in ``sequence``."""
    mask_float = mask.float().unsqueeze(-1)
    count = mask.float().sum(dim=1, keepdim=True)
    centered = sequence - mean.unsqueeze(1)
    var = ((centered * centered) * mask_float).sum(dim=1) / count.clamp(min=1)
    return torch.sqrt(var + _VAR_EPS)


def _validate_sequence_and_mask(sequence: Tensor, mask: Tensor) -> None:
    """Validate sequence and mask dimensions."""
    if sequence.dim() != 3:
        raise ValueError("sequence must have shape (B, T, C)")
    if mask.dim() != 2:
        raise ValueError("mask must have shape (B, T)")
    if sequence.shape[:2] != mask.shape:
        raise ValueError("sequence and mask must share batch/time dimensions")


def _validate_latent(latent: Tensor, expected_dim: int, name: str) -> None:
    """Validate a batch of latent vectors."""
    if latent.dim() != 2:
        raise ValueError(f"{name} must have shape (B, D)")
    if latent.shape[1] != expected_dim:
        raise ValueError(f"{name} feature dimension does not match config")
