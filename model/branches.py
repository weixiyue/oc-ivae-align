"""Latent posterior branches for OC-iVAE."""

from __future__ import annotations

from torch import Tensor, nn

from model.backbone import TCNResBlock
from model.constants import C_HIDDEN, D_ZC, D_ZD, U_DIM

_BRANCH_KERNEL_SIZE = 3
_BRANCH_DILATION = 1


class MaskedAvgPool(nn.Module):
    """Average valid timesteps from a masked sequence."""

    def forward(self, h: Tensor, mask: Tensor) -> Tensor:
        """Pool ``h`` with shape ``(B, T, C)`` into ``(B, C)``."""
        _validate_sequence_and_mask(h, mask)
        mask_float = mask.float().unsqueeze(-1)
        numerator = (h * mask_float).sum(dim=1)
        denominator = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return numerator / denominator


class FiLMGenerator(nn.Module):
    """Generate feature-wise FiLM scale and shift from auxiliary variables."""

    def __init__(self, u_dim: int = U_DIM, c_hidden: int = C_HIDDEN) -> None:
        """Initialize the FiLM MLP.

        Args:
            u_dim: Auxiliary variable dimension.
            c_hidden: Hidden channel dimension to modulate.
        """
        super().__init__()
        self.c_hidden = c_hidden
        self.net = nn.Sequential(
            nn.Linear(u_dim, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c_hidden * 2),
        )

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """Return FiLM ``gamma`` and ``beta`` tensors with shape ``(B, C)``."""
        if u.dim() != 2:
            raise ValueError("u must have shape (B, U)")

        out = self.net(u)
        gamma = out[:, : self.c_hidden]
        beta = out[:, self.c_hidden :]
        return gamma, beta


class ZdBranch(nn.Module):
    """Posterior branch for degradation latent variable ``z_d``."""

    def __init__(self, config: dict) -> None:
        """Initialize the degradation branch.

        Args:
            config: Model config with ``c_hidden`` and ``d_zd``.
        """
        super().__init__()
        c_hidden = int(config.get("c_hidden", C_HIDDEN))
        d_zd = int(config.get("d_zd", D_ZD))
        self.d_zd = d_zd
        self.tcn_resblock = TCNResBlock(
            c_hidden,
            c_hidden,
            kernel_size=_BRANCH_KERNEL_SIZE,
            dilation=_BRANCH_DILATION,
        )
        self.pool = MaskedAvgPool()
        self.head = nn.Linear(c_hidden, d_zd * 2)

    def forward(self, h: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior ``mu_d`` and ``log_var_d`` from backbone features."""
        h_branch = self.tcn_resblock(h, mask)
        pooled = self.pool(h_branch, mask)
        out = self.head(pooled)
        mu = out[:, : self.d_zd]
        log_var = out[:, self.d_zd :]
        return mu, log_var


class ZcBranch(nn.Module):
    """Posterior branch for operating-condition latent variable ``z_c``."""

    def __init__(self, config: dict) -> None:
        """Initialize the operating-condition branch.

        Args:
            config: Model config with ``c_hidden``, ``d_zc``, and ``u_dim``.
        """
        super().__init__()
        c_hidden = int(config.get("c_hidden", C_HIDDEN))
        d_zc = int(config.get("d_zc", D_ZC))
        u_dim = int(config.get("u_dim", U_DIM))
        self.d_zc = d_zc
        self.film_generator = FiLMGenerator(u_dim, c_hidden)
        self.tcn_resblock = TCNResBlock(
            c_hidden,
            c_hidden,
            kernel_size=_BRANCH_KERNEL_SIZE,
            dilation=_BRANCH_DILATION,
        )
        self.pool = MaskedAvgPool()
        self.head = nn.Linear(c_hidden, d_zc * 2)

    def forward(self, h: Tensor, mask: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior ``mu_c`` and ``log_var_c`` from FiLM features."""
        h_branch = self.tcn_resblock(h, mask)
        gamma, beta = self.film_generator(u)
        h_film = gamma.unsqueeze(1) * h_branch + beta.unsqueeze(1)
        pooled = self.pool(h_film, mask)
        out = self.head(pooled)
        mu = out[:, : self.d_zc]
        log_var = out[:, self.d_zc :]
        return mu, log_var


def _validate_sequence_and_mask(h: Tensor, mask: Tensor) -> None:
    """Validate sequence and mask dimensions."""
    if h.dim() != 3:
        raise ValueError("h must have shape (B, T, C)")
    if mask.dim() != 2:
        raise ValueError("mask must have shape (B, T)")
    if h.shape[:2] != mask.shape:
        raise ValueError("h and mask must have matching batch/time dimensions")
