"""HSIC independence loss for latent variables."""

import torch

_BANDWIDTH_EPS = 1e-6


def hsic_loss(
    z_d: torch.Tensor,
    z_c: torch.Tensor,
) -> torch.Tensor:
    """Mini-batch HSIC with RBF kernel and median bandwidth heuristic.

    Args:
        z_d: Degradation latent tensor with shape ``(B, d_zd)``.
        z_c: Operating-condition latent tensor with shape ``(B, d_zc)``.

    Returns:
        Scalar non-negative HSIC estimate.
    """
    if z_d.dim() != 2 or z_c.dim() != 2:
        raise ValueError("z_d and z_c must have shape (B, D)")
    if z_d.shape[0] != z_c.shape[0]:
        raise ValueError("z_d and z_c must share batch size")

    batch_size = z_d.shape[0]
    if batch_size <= 1:
        return z_d.new_tensor(0.0)

    K = _rbf_kernel(z_d)
    L = _rbf_kernel(z_c)
    eye = torch.eye(batch_size, device=z_d.device, dtype=z_d.dtype)
    ones = torch.ones(batch_size, batch_size, device=z_d.device, dtype=z_d.dtype)
    H = eye - ones / batch_size
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = torch.trace(Kc @ Lc) / ((batch_size - 1) ** 2)
    return hsic.clamp(min=0)


def pairwise_sq_dist(X: torch.Tensor) -> torch.Tensor:
    """Return pairwise squared Euclidean distances for rows of ``X``."""
    squared_norm = (X * X).sum(dim=1, keepdim=True)
    distances = squared_norm + squared_norm.transpose(0, 1) - 2 * (X @ X.T)
    return distances.clamp(min=0)


def _rbf_kernel(X: torch.Tensor) -> torch.Tensor:
    """Return an RBF kernel matrix using the median bandwidth heuristic."""
    distances = pairwise_sq_dist(X)
    sigma = _median_bandwidth(distances)
    return torch.exp(-distances / (2 * sigma * sigma))


def _median_bandwidth(distances: torch.Tensor) -> torch.Tensor:
    """Return median-heuristic RBF bandwidth from pairwise distances."""
    batch_size = distances.shape[0]
    upper_indices = torch.triu_indices(
        batch_size,
        batch_size,
        offset=1,
        device=distances.device,
    )
    dist_flat = distances[upper_indices[0], upper_indices[1]]
    if dist_flat.numel() == 0:
        return distances.new_tensor(0.0)

    median_distance = dist_flat.median().clamp(min=_BANDWIDTH_EPS)
    return torch.sqrt(median_distance / 2)
