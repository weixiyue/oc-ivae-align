"""Shared causal TCN backbone for OC-iVAE encoders."""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F

from model.constants import BACKBONE_DILATIONS, C_HIDDEN, N_BACKBONE_LAYERS

_INPUT_PROJECTION_KERNEL_SIZE = 1
_DEFAULT_KERNEL_SIZE = 3
_DEFAULT_DILATION = 1
_DEFAULT_DROPOUT = 0.1


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
        c_hidden = int(config.get("c_hidden", C_HIDDEN))
        n_backbone_layers = int(config.get("n_backbone_layers", N_BACKBONE_LAYERS))
        if n_backbone_layers < 1:
            raise ValueError("n_backbone_layers must be at least 1")
        if n_backbone_layers > len(BACKBONE_DILATIONS):
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
                for dilation in BACKBONE_DILATIONS[:n_backbone_layers]
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


def _validate_sequence_and_mask(x: Tensor, mask: Tensor) -> None:
    """Validate sequence and mask dimensions."""
    if x.dim() != 3:
        raise ValueError("x must have shape (B, T, C)")
    if mask.dim() != 2:
        raise ValueError("mask must have shape (B, T)")
    if x.shape[:2] != mask.shape:
        raise ValueError("x and mask must have matching batch/time dimensions")
