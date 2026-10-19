"""KGLD and its latent branches, conditional prior and health head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.tcn import (
    SharedTCNBackbone,
    TCNDecoder,
    TCNResBlock,
    _validate_sequence_and_mask,
)

_BRANCH_KERNEL_SIZE = 3
_BRANCH_DILATION = 1


class KGLD(nn.Module):
    """Operating Condition-aware identifiable VAE for HI construction."""

    def __init__(self, config: dict) -> None:
        """Initialize all KGLD submodules.

        Args:
            config: Model config with backbone, latent, prior, and decoder
                dimensions.
        """
        super().__init__()
        if "in_features" not in config:
            raise ValueError("config must include 'in_features'")

        in_features = int(config["in_features"])
        self.backbone = SharedTCNBackbone(in_features, config)
        self.zd_branch = ZdBranch(config)
        self.zc_branch = ZcBranch(config)
        self.prior = PriorNetwork(config)
        self.decoder = TCNDecoder(config)
        self.hi_head = HIHead(config)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample during training and return ``mu`` directly during eval."""
        if not self.training:
            return mu

        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * log_var)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        u: Tensor,
        W_seq: Tensor,
    ) -> dict[str, Tensor]:
        """Run KGLD and return the locked intermediate-output dictionary."""
        h = self.backbone(x, mask)
        mu_d, log_var_d = self.zd_branch(h, mask)
        mu_c, log_var_c = self.zc_branch(h, mask, u)
        (
            mu_d_prior,
            log_var_d_prior,
            mu_c_prior,
            log_var_c_prior,
        ) = self.prior(u)
        z_d = self.reparameterize(mu_d, log_var_d)
        z_c = self.reparameterize(mu_c, log_var_c)
        x_hat = self.decoder(z_d, z_c, W_seq, mask)
        hi_raw = self.hi_head(z_d)

        return {
            "x_hat": x_hat,
            "mu_d": mu_d,
            "log_var_d": log_var_d,
            "mu_c": mu_c,
            "log_var_c": log_var_c,
            "mu_d_prior": mu_d_prior,
            "log_var_d_prior": log_var_d_prior,
            "mu_c_prior": mu_c_prior,
            "log_var_c_prior": log_var_c_prior,
            "z_d": z_d,
            "z_c": z_c,
            "hi_raw": hi_raw,
        }


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

    def __init__(self, u_dim: int = 11, c_hidden: int = 64) -> None:
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
        c_hidden = int(config.get("c_hidden", 64))
        d_zd = int(config.get("d_zd", 2))
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
        c_hidden = int(config.get("c_hidden", 64))
        d_zc = int(config.get("d_zc", 8))
        u_dim = int(config.get("u_dim", 11))
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


class PriorNetwork(nn.Module):
    """Conditional prior ``p(z_d, z_c | u)`` with separate latent heads."""

    def __init__(self, config: dict) -> None:
        """Initialize the shared MLP stem and dual Gaussian heads.

        Args:
            config: Model config with ``u_dim``, ``d_zd``, ``d_zc``, and
                ``c_hidden``.
        """
        super().__init__()
        u_dim = int(config.get("u_dim", 11))
        c_hidden = int(config.get("c_hidden", 64))
        d_zd = int(config.get("d_zd", 2))
        d_zc = int(config.get("d_zc", 8))

        self.u_dim = u_dim
        self.d_zd = d_zd
        self.d_zc = d_zc
        self.stem = nn.Sequential(
            nn.Linear(u_dim, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
        )
        self.zd_head = nn.Linear(c_hidden, d_zd * 2)
        self.zc_head = nn.Linear(c_hidden, d_zc * 2)
        nn.init.zeros_(self.zd_head.weight)
        nn.init.zeros_(self.zd_head.bias)

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return conditional Gaussian parameters for ``z_d`` and ``z_c``."""
        if u.dim() != 2:
            raise ValueError("u must have shape (B, u_dim)")
        if u.shape[1] != self.u_dim:
            raise ValueError("u feature dimension does not match config['u_dim']")

        h = self.stem(u)
        zd_out = self.zd_head(h)
        zc_out = self.zc_head(h)

        mu_d_prior = zd_out[:, : self.d_zd]
        log_var_d_prior = zd_out[:, self.d_zd :]
        mu_c_prior = zc_out[:, : self.d_zc]
        log_var_c_prior = zc_out[:, self.d_zc :]
        return mu_d_prior, log_var_d_prior, mu_c_prior, log_var_c_prior


class HIHead(nn.Module):
    """Map ``z_d`` to a raw scalar health indicator."""

    def __init__(self, config: dict) -> None:
        """Initialize the no-bias linear HI projection.

        Args:
            config: Model config with ``d_zd``.
        """
        super().__init__()
        self.d_zd = int(config.get("d_zd", 2))
        self.linear = nn.Linear(self.d_zd, 1, bias=False)

    def forward(self, z_d: Tensor) -> Tensor:
        """Return raw HI values with shape ``(B,)``."""
        if z_d.dim() != 2:
            raise ValueError("z_d must have shape (B, d_zd)")
        if z_d.shape[1] != self.d_zd:
            raise ValueError("z_d feature dimension does not match config['d_zd']")

        return self.linear(z_d).squeeze(-1)
