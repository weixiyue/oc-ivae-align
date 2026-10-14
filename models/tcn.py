"""Causal encoder and non-causal decoder TCN components."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_INPUT_PROJECTION_KERNEL_SIZE = 1
_DEFAULT_KERNEL_SIZE = 3
_DEFAULT_DILATION = 1
_DEFAULT_DROPOUT = 0.1
_BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
_DECODER_DILATIONS = [1, 2, 4, 8]
_N_SENSOR_FEATURES = 14
_VAR_EPS = 1e-8
W_SUMMARY_DIM = 8


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left-only causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        """Initialize a causal convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Temporal convolution kernel size.
            dilation: Temporal dilation factor.
        """
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        if dilation < 1:
            raise ValueError("dilation must be at least 1")

        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal convolution to ``x`` with shape ``(B, C, T)``."""
        if x.dim() != 3:
            raise ValueError("x must have shape (B, C, T)")

        padded = F.pad(x, (self.padding, 0))
        return self.conv(padded)


class MaskedLayerNorm(nn.Module):
    """LayerNorm that keeps padded timesteps zeroed."""

    def __init__(self, num_features: int) -> None:
        """Initialize masked normalization over the feature dimension.

        Args:
            num_features: Number of features in the final tensor dimension.
        """
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Normalize ``x`` with shape ``(B, T, C)`` while preserving padding."""
        _validate_sequence_and_mask(x, mask)
        mask_float = mask.float().unsqueeze(-1)
        x_masked = x * mask_float
        return self.norm(x_masked) * mask_float


class TCNResBlock(nn.Module):
    """Causal TCN residual block with masked Pre-LN-style operations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = _DEFAULT_KERNEL_SIZE,
        dilation: int = _DEFAULT_DILATION,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        """Initialize a two-layer causal residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Temporal convolution kernel size.
            dilation: Dilation used by the first convolution.
            dropout: Dropout probability after GELU activations.
        """
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm1 = MaskedLayerNorm(out_channels)
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=_DEFAULT_DILATION,
        )
        self.norm2 = MaskedLayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply the residual block to ``x`` with shape ``(B, T, C)``."""
        _validate_sequence_and_mask(x, mask)

        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm1(y, mask)))

        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm2(y, mask)))

        if isinstance(self.residual, nn.Identity):
            residual = x
        else:
            residual = self.residual(x.transpose(1, 2)).transpose(1, 2)

        return (y + residual) * mask.float().unsqueeze(-1)


class SharedTCNBackbone(nn.Module):
    """Shared causal TCN encoder backbone for sensor sequences."""

    def __init__(self, in_features: int, config: dict) -> None:
        """Initialize input projection and causal residual TCN stack.

        Args:
            in_features: Number of input sensor features.
            config: Model config with ``c_hidden`` and ``n_backbone_layers``.
        """
        super().__init__()
        c_hidden = int(config.get("c_hidden", 64))
        n_backbone_layers = int(config.get("n_backbone_layers", 6))
        if n_backbone_layers < 1:
            raise ValueError("n_backbone_layers must be at least 1")
        if n_backbone_layers > len(_BACKBONE_DILATIONS):
            raise ValueError("n_backbone_layers exceeds available backbone dilations")

        self.input_conv = CausalConv1d(
            in_features,
            c_hidden,
            kernel_size=_INPUT_PROJECTION_KERNEL_SIZE,
            dilation=_DEFAULT_DILATION,
        )
        self.input_norm = MaskedLayerNorm(c_hidden)
        self.activation = nn.GELU()
        self.layers = nn.ModuleList(
            [
                TCNResBlock(
                    c_hidden,
                    c_hidden,
                    kernel_size=_DEFAULT_KERNEL_SIZE,
                    dilation=dilation,
                )
                for dilation in _BACKBONE_DILATIONS[:n_backbone_layers]
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode ``x`` with shape ``(B, T, S)`` into ``(B, T, C_hidden)``."""
        _validate_sequence_and_mask(x, mask)

        h = self.input_conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.activation(self.input_norm(h, mask))
        h = h * mask.float().unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, mask)

        return h


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
                ``d_zd``, ``d_zc``, ``w_summary_dim``, and
                ``decoder_use_W_summary`` (defaults to True for old configs).
        """
        super().__init__()
        c_hidden = int(config.get("c_hidden", 64))
        n_decoder_layers = int(config.get("n_decoder_layers", 4))
        d_zd = int(config.get("d_zd", 2))
        d_zc = int(config.get("d_zc", 8))
        w_summary_dim = int(config.get("w_summary_dim", W_SUMMARY_DIM))
        if n_decoder_layers < 1:
            raise ValueError("n_decoder_layers must be at least 1")
        if n_decoder_layers > len(_DECODER_DILATIONS):
            raise ValueError("n_decoder_layers exceeds available decoder dilations")

        self.d_zd = d_zd
        self.d_zc = d_zc
        self.w_summary_dim = w_summary_dim
        self.use_W_summary = config.get("decoder_use_W_summary", True)
        if not isinstance(self.use_W_summary, bool):
            raise ValueError("decoder_use_W_summary must be a boolean")
        self.input_projection = nn.Linear(d_zd + d_zc + w_summary_dim, c_hidden)
        self.layers = nn.ModuleList(
            [
                NonCausalResBlock(
                    c_hidden,
                    kernel_size=_DEFAULT_KERNEL_SIZE,
                    dilation=dilation,
                )
                for dilation in _DECODER_DILATIONS[:n_decoder_layers]
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
        if W_seq.shape[-1] != 4:
            raise ValueError("W_seq must have four operating-condition features")
        if z_d.shape[0] != W_seq.shape[0] or z_c.shape[0] != W_seq.shape[0]:
            raise ValueError("z_d, z_c, and W_seq must share batch size")

        w_summary = _masked_w_summary(W_seq, mask)
        if w_summary.shape[-1] != self.w_summary_dim:
            raise ValueError("computed W summary does not match w_summary_dim")

        if not self.use_W_summary:
            # Keep identical projection shape/initialization; only remove its W input.
            w_summary = torch.zeros_like(w_summary)
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
    """Concatenate valid-timestep W means and standard deviations."""
    mask_float = mask.float().unsqueeze(-1)
    count = mask.float().sum(dim=1, keepdim=True)
    mean = (W_seq * mask_float).sum(dim=1) / count.clamp(min=1)
    centered = W_seq - mean.unsqueeze(1)
    var = ((centered * centered) * mask_float).sum(dim=1) / count.clamp(min=1)
    std = torch.sqrt(var + _VAR_EPS)
    return torch.cat((mean, std), dim=-1)


def _validate_sequence_and_mask(x: Tensor, mask: Tensor) -> None:
    """Validate sequence and mask dimensions."""
    if x.dim() != 3:
        raise ValueError("x must have shape (B, T, C)")
    if mask.dim() != 2:
        raise ValueError("mask must have shape (B, T)")
    if x.shape[:2] != mask.shape:
        raise ValueError("x and mask must have matching batch/time dimensions")


def _validate_latent(latent: Tensor, expected_dim: int, name: str) -> None:
    """Validate a batch of latent vectors."""
    if latent.dim() != 2:
        raise ValueError(f"{name} must have shape (B, D)")
    if latent.shape[1] != expected_dim:
        raise ValueError(f"{name} feature dimension does not match config")
