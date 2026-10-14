"""KL divergence losses for OC-iVAE."""

import torch


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
