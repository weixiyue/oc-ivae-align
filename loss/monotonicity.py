"""Monotonicity losses for health indicator learning."""

import torch


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
