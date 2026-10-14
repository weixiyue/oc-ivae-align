"""Staged aggregate loss for OC-iVAE training."""

import torch

from loss.hsic import hsic_loss
from loss.kl import gaussian_kl_loss
from loss.monotonicity import monotonicity_loss
from loss.recon import masked_recon_loss
from loss.spearman import soft_spearman_loss


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
        out: Forward output dict from ``OC_iVAE``.
        x: Original input tensor with shape ``(B, T, S)``.
        mask: Valid timestep mask with shape ``(B, T)``.
        cycle_id: Cycle indices with shape ``(B,)``.
        unit_id: Unit indices with shape ``(B,)``.
        epoch: Current zero-indexed epoch.
        config: Flat loss schedule and weight config.

    Returns:
        Tuple of scalar tensor total loss and logging dictionary.
    """
    warmup_end_epoch = _get_config_value(config, "warmup_end_epoch")
    transition_end_epoch = _get_config_value(config, "transition_end_epoch")
    beta_d = _get_config_value(config, "beta_d")
    beta_c = _get_config_value(config, "beta_c")
    lambda_hsic = _get_config_value(config, "lambda_hsic")
    lambda_mono = _get_config_value(config, "lambda_mono")
    lambda_spearman = _get_config_value(config, "lambda_spearman")
    ranking_margin = _get_config_value(config, "ranking_margin")
    ranking_delta_t = _get_config_value(config, "ranking_delta_t")
    spearman_tau = _get_config_value(config, "spearman_tau")

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


def _get_config_value(config: dict, key: str) -> object:
    """Read a required loss config value."""
    if key not in config:
        raise ValueError(f"config must include {key!r}")
    return config[key]
