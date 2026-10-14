"""Top-level OC-iVAE model assembly."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from model.backbone import SharedTCNBackbone
from model.branches import ZcBranch, ZdBranch
from model.decoder import TCNDecoder
from model.hi_head import HIHead
from model.prior import PriorNetwork


class OC_iVAE(nn.Module):
    """Operating Condition-aware identifiable VAE for HI construction."""

    def __init__(self, config: dict) -> None:
        """Initialize all OC-iVAE submodules.

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
        """Run OC-iVAE and return the locked intermediate-output dictionary."""
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

    @torch.no_grad()
    def get_hi(
        self,
        x: Tensor,
        mask: Tensor,
        u: Tensor,
        W_seq: Tensor,
    ) -> np.ndarray:
        """Return raw HI values as a NumPy array using deterministic inference."""
        self.eval()
        out = self.forward(x, mask, u, W_seq)
        return out["hi_raw"].cpu().numpy()
