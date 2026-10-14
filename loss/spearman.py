"""Soft Spearman rank losses for health indicator learning."""

import torch

_CORR_EPS = 1e-8


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
