"""Standard VAE baseline model builder."""

from __future__ import annotations

from torch import Tensor, nn

from baselines.cvae import StandardNormalPrior
from model.backbone import TCNResBlock
from model.branches import MaskedAvgPool, ZcBranch
from model.oc_ivae import OC_iVAE


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


def build_vae_model(config: dict) -> OC_iVAE:
    """Standard VAE baseline.

    No FiLM conditioning, standard normal prior. Used to measure total
    contribution of OC-iVAE conditioning mechanism.
    """
    model = OC_iVAE(config)
    if not isinstance(model.zc_branch, ZcBranch):
        raise TypeError(
            "expected OC_iVAE.zc_branch to be a ZcBranch before replacement"
        )

    c_hidden = int(config.get("c_hidden", 64))
    d_zd = int(config.get("d_zd", 2))
    d_zc = int(config.get("d_zc", 8))
    model.zc_branch = PassthroughZcBranch(c_hidden=c_hidden, d_zc=d_zc)
    model.prior = StandardNormalPrior(d_zd=d_zd, d_zc=d_zc)
    return model
