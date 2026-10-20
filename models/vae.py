"""Existing VAE baselines and isolated knowledge ablations of the KGLD architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.kgld import MaskedAvgPool, KGLD
from models.tcn import TCNResBlock


class StandardNormalPrior(nn.Module):
    """Return fixed standard-normal parameters, ignoring auxiliary values."""

    def __init__(self, d_zd: int, d_zc: int) -> None:
        """Store latent dimensions without parameters or buffers."""
        super().__init__()
        self.d_zd = int(d_zd)
        self.d_zc = int(d_zc)

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return both latent priors with u's batch size, device and dtype."""
        return (
            u.new_zeros((u.shape[0], self.d_zd)),
            u.new_zeros((u.shape[0], self.d_zd)),
            u.new_zeros((u.shape[0], self.d_zc)),
            u.new_zeros((u.shape[0], self.d_zc)),
        )


class PassthroughZcBranch(nn.Module):
    """Standard VAE z_c branch.

    Accepts u for interface compatibility but ignores it. No FiLM conditioning
    applied.
    """

    def __init__(self, c_hidden: int, d_zc: int) -> None:
        """Initialize a no-FiLM z_c posterior branch.

        Args:
            c_hidden: Hidden channel dimension from the shared backbone.
            d_zc: Operating-condition latent dimension.
        """
        super().__init__()
        self.d_zc = int(d_zc)
        self.tcn_resblock = TCNResBlock(
            int(c_hidden),
            int(c_hidden),
            kernel_size=3,
            dilation=1,
        )
        self.pool = MaskedAvgPool()
        self.head = nn.Linear(int(c_hidden), 2 * self.d_zc)

    def forward(self, h: Tensor, mask: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        """Standard VAE z_c branch.

        Accepts u for interface compatibility but ignores it. No FiLM
        conditioning applied.
        """
        del u
        h_branch = self.tcn_resblock(h, mask)
        pooled = self.pool(h_branch, mask)
        out = self.head(pooled)
        mu = out[:, : self.d_zc]
        log_var = out[:, self.d_zc :]
        return mu, log_var


def build_cvae_model(config: dict) -> KGLD:
    """Build KGLD with a fixed standard normal prior for cVAE baseline."""
    model = KGLD(config)
    d_zd = int(config.get("d_zd", 2))
    d_zc = int(config.get("d_zc", 8))
    model.prior = StandardNormalPrior(d_zd=d_zd, d_zc=d_zc)
    return model


def build_vae_model(config: dict) -> KGLD:
    """Standard VAE baseline.

    No FiLM conditioning, standard normal prior. Used to measure total
    contribution of KGLD conditioning mechanism.
    """
    model = KGLD(config)
    c_hidden = int(config.get("c_hidden", 64))
    d_zd = int(config.get("d_zd", 2))
    d_zc = int(config.get("d_zc", 8))
    model.zc_branch = PassthroughZcBranch(c_hidden=c_hidden, d_zc=d_zc)
    model.prior = StandardNormalPrior(d_zd=d_zd, d_zc=d_zc)
    return model


def build_knowledge_ablation(config: dict, variant: str) -> KGLD:
    """Replace only the requested knowledge modules after ordinary KGLD initialization."""
    if variant not in ('no_prior', 'no_film', 'no_prior_no_film'):
        raise ValueError(f'unknown knowledge ablation: {variant}')
    model = KGLD(config)
    if variant in ('no_film', 'no_prior_no_film'):
        original = model.zc_branch
        # All retained weights and the post-KGLD RNG state stay shared across variants.
        with torch.random.fork_rng(devices=[]):
            branch = PassthroughZcBranch(config.get('c_hidden', 64), config.get('d_zc', 8))
        branch.tcn_resblock.load_state_dict(original.tcn_resblock.state_dict())
        branch.head.load_state_dict(original.head.state_dict())
        model.zc_branch = branch
    if variant in ('no_prior', 'no_prior_no_film'):
        model.prior = StandardNormalPrior(config.get('d_zd', 2), config.get('d_zc', 8))
    return model
