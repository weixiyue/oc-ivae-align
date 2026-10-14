"""Reconstruction losses for OC-iVAE."""

import torch


def masked_recon_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.BoolTensor,
) -> torch.Tensor:
    """Masked MSE reconstruction loss over valid timesteps only.

    Args:
        x_hat: Reconstructed sensor data with shape ``(B, T, S)``.
        x: Original normalized sensor data with shape ``(B, T, S)``.
        mask: Boolean timestep mask with shape ``(B, T)``.

    Returns:
        Scalar mean squared error over valid timesteps and sensors.
    """
    if x_hat.shape != x.shape:
        raise ValueError("x_hat and x must have the same shape")
    if x_hat.dim() != 3:
        raise ValueError("x_hat and x must have shape (B, T, S)")
    if mask.shape != x_hat.shape[:2]:
        raise ValueError("mask must have shape (B, T)")

    squared_err = (x_hat - x) ** 2
    mask_3d = mask.unsqueeze(-1).float()
    masked_err = squared_err * mask_3d
    sensor_dim = x_hat.shape[-1]
    return masked_err.sum() / (mask_3d.sum() * sensor_dim).clamp(min=1)
