"""Reconstruction, latent regularization and staged HI training losses."""
from __future__ import annotations

import torch

_BANDWIDTH_EPS = 1e-6
_CORR_EPS = 1e-8


def compute_total_loss(
    out: dict,
    x: torch.Tensor,
    mask: torch.BoolTensor,
    cycle_id: torch.LongTensor,
    unit_id: torch.LongTensor,
    epoch: int,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    """Compute staged total loss with KL annealing.

    Args:
        out: Forward output dict from ``KGLD``.
        x: Original input tensor with shape ``(B, T, S)``.
        mask: Valid timestep mask with shape ``(B, T)``.
        cycle_id: Cycle indices with shape ``(B,)``.
        unit_id: Unit indices with shape ``(B,)``.
        epoch: Current zero-indexed epoch.
        config: Flat loss schedule and weight config.

    Returns:
        Tuple of scalar tensor total loss and logging dictionary.
    """
    warmup_end_epoch = config["warmup_end_epoch"]
    transition_end_epoch = config["transition_end_epoch"]
    beta_d = config["beta_d"]
    beta_c = config["beta_c"]
    lambda_hsic = config["lambda_hsic"]
    lambda_mono = config["lambda_mono"]
    lambda_spearman = config["lambda_spearman"]
    ranking_margin = config["ranking_margin"]
    ranking_delta_t = config["ranking_delta_t"]
    spearman_tau = config["spearman_tau"]

    recon = masked_recon_loss(out["x_hat"], x, mask)
    kl_d = gaussian_kl_loss(
        out["mu_d"],
        out["log_var_d"],
        out["mu_d_prior"],
        out["log_var_d_prior"],
    )
    kl_c = gaussian_kl_loss(
        out["mu_c"],
        out["log_var_c"],
        out["mu_c_prior"],
        out["log_var_c_prior"],
    )
    hsic = hsic_loss(out["z_d"], out["z_c"])
    mono = monotonicity_loss(
        out["hi_raw"],
        cycle_id,
        unit_id,
        delta_t=int(ranking_delta_t),
        margin=float(ranking_margin),
    )
    spearman = soft_spearman_loss(
        out["hi_raw"],
        cycle_id,
        unit_id,
        tau=float(spearman_tau),
    )

    if epoch < int(warmup_end_epoch):
        total = recon
    elif epoch < int(transition_end_epoch):
        kl_weight = _compute_kl_weight(
            epoch,
            int(warmup_end_epoch),
            int(transition_end_epoch),
        )
        total = (
            recon
            + kl_weight * (float(beta_d) * kl_d + float(beta_c) * kl_c)
            + float(lambda_hsic) * hsic
        )
    else:
        total = (
            recon
            + float(beta_d) * kl_d
            + float(beta_c) * kl_c
            + float(lambda_hsic) * hsic
            + float(lambda_mono) * mono
            + float(lambda_spearman) * spearman
        )

    loss_dict = {
        "recon": recon.item(),
        "kl_d": kl_d.item(),
        "kl_c": kl_c.item(),
        "hsic": hsic.item(),
        "mono": mono.item(),
        "spearman": spearman.item(),
        "total": total,
    }
    return total, loss_dict


def _compute_kl_weight(
    epoch: int,
    warmup_end_epoch: int,
    transition_end_epoch: int,
) -> float:
    """Return the linear KL annealing weight for phase 2."""
    denominator = transition_end_epoch - warmup_end_epoch
    if denominator <= 0:
        return 1.0
    return (epoch - warmup_end_epoch) / denominator


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


def gaussian_kl_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    mu_prior: torch.Tensor,
    log_var_prior: torch.Tensor,
) -> torch.Tensor:
    """KL divergence KL(q(z|x,u) || p(z|u)) for Gaussian distributions.

    Args:
        mu: Posterior mean with shape ``(B, D)``.
        log_var: Posterior log-variance with shape ``(B, D)``.
        mu_prior: Prior mean with shape ``(B, D)``.
        log_var_prior: Prior log-variance with shape ``(B, D)``.

    Returns:
        Scalar mean KL over batch and latent dimensions.
    """
    if not (mu.shape == log_var.shape == mu_prior.shape == log_var_prior.shape):
        raise ValueError("all KL inputs must have the same shape")
    if mu.dim() != 2:
        raise ValueError("KL inputs must have shape (B, D)")

    log_var_prior_safe = log_var_prior.clamp(min=-10, max=10)
    kl = 0.5 * (
        log_var_prior_safe
        - log_var
        + (torch.exp(log_var) + (mu - mu_prior) ** 2)
        / torch.exp(log_var_prior_safe)
        - 1
    )
    return kl.mean()


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


def monotonicity_loss(
    hi: torch.Tensor,
    cycle_id: torch.LongTensor,
    unit_id: torch.LongTensor,
    delta_t: int = 3,
    margin: float = 0.01,
) -> torch.Tensor:
    """Pairwise ranking loss enforcing HI decreases over time within each unit.

    Args:
        hi: Scalar HI values with shape ``(B,)``.
        cycle_id: Cycle indices with shape ``(B,)``.
        unit_id: Unit indices with shape ``(B,)``.
        delta_t: Minimum cycle gap to form a valid pair.
        margin: Hinge margin.

    Returns:
        Scalar mean hinge loss over valid pairs.
    """
    _validate_vector_inputs(hi, cycle_id, unit_id)
    if delta_t < 0:
        raise ValueError("delta_t must be non-negative")

    unit_eq = unit_id.unsqueeze(1) == unit_id.unsqueeze(0)
    gap = cycle_id.unsqueeze(0) - cycle_id.unsqueeze(1)
    valid = unit_eq & (gap >= delta_t)
    if not torch.any(valid):
        return torch.tensor(0.0, device=hi.device, dtype=hi.dtype, requires_grad=True)

    later_minus_earlier = hi.unsqueeze(0) - hi.unsqueeze(1)
    losses = torch.relu(later_minus_earlier + margin)
    return losses[valid].mean()


def _validate_vector_inputs(
    hi: torch.Tensor,
    cycle_id: torch.Tensor,
    unit_id: torch.Tensor,
) -> None:
    """Validate aligned one-dimensional batch vectors."""
    if hi.dim() != 1:
        raise ValueError("hi must have shape (B,)")
    if cycle_id.dim() != 1 or unit_id.dim() != 1:
        raise ValueError("cycle_id and unit_id must have shape (B,)")
    if hi.shape[0] != cycle_id.shape[0] or hi.shape[0] != unit_id.shape[0]:
        raise ValueError("hi, cycle_id, and unit_id must share batch size")


def soft_spearman_loss(
    hi: torch.Tensor,
    cycle_id: torch.LongTensor,
    unit_id: torch.LongTensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Soft Spearman loss enforcing HI decreases over time within each unit.

    Args:
        hi: Scalar HI values with shape ``(B,)``.
        cycle_id: Cycle indices with shape ``(B,)``.
        unit_id: Unit indices with shape ``(B,)``.
        tau: Temperature for soft ranking.

    Returns:
        Scalar mean ``1 - corr`` over valid units.
    """
    _validate_vector_inputs(hi, cycle_id, unit_id)
    if tau <= 0:
        raise ValueError("tau must be positive")

    unit_losses: list[torch.Tensor] = []
    for current_unit in torch.unique(unit_id):
        unit_mask = unit_id == current_unit
        if int(unit_mask.sum().item()) < 2:
            continue

        hi_unit = hi[unit_mask]
        cycle_unit = cycle_id[unit_mask].float()
        r_hi = _soft_rank(-hi_unit, tau)
        r_cycle = _soft_rank(cycle_unit, tau)
        corr = _pearson_corr(r_hi, r_cycle)
        unit_losses.append(1 - corr)

    if not unit_losses:
        return torch.tensor(0.0, device=hi.device, dtype=hi.dtype, requires_grad=True)

    return torch.stack(unit_losses).mean()


def _soft_rank(v: torch.Tensor, tau: float) -> torch.Tensor:
    """Return differentiable soft ranks for a one-dimensional tensor."""
    pairwise_diff = (v.unsqueeze(1) - v.unsqueeze(0)) / tau
    return torch.sigmoid(pairwise_diff).sum(dim=1) + 0.5


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return Pearson correlation between two one-dimensional tensors."""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    numerator = (a_centered * b_centered).sum()
    denominator = torch.norm(a_centered) * torch.norm(b_centered) + _CORR_EPS
    return numerator / denominator
