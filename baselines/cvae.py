"""cVAE baseline model builder."""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor, nn

from model.constants import D_ZC, D_ZD
from model.oc_ivae import OC_iVAE


class _PriorOutput(dict[str, Tensor]):
    """Mapping that also supports OC_iVAE's existing tuple-style unpacking."""

    def __iter__(self) -> Iterator[Tensor]:
        return iter(
            (
                self["mu_d_prior"],
                self["log_var_d_prior"],
                self["mu_c_prior"],
                self["log_var_c_prior"],
            )
        )


class StandardNormalPrior(nn.Module):
    """cVAE baseline prior.

    Ignores u and returns standard normal parameters. Used to isolate the
    contribution of iVAE conditional prior.
    """

    def __init__(self, d_zd: int, d_zc: int) -> None:
        """Store latent dimensions without registering parameters or buffers.

        Args:
            d_zd: Degradation latent dimension.
            d_zc: Operating-condition latent dimension.
        """
        super().__init__()
        self.d_zd = int(d_zd)
        self.d_zc = int(d_zc)

    def forward(self, u: Tensor) -> dict[str, Tensor]:
        """Return standard normal parameters with batch size inferred from u."""
        batch_size = u.shape[0]
        return _PriorOutput(
            {
                "mu_d_prior": torch.zeros(
                    batch_size,
                    self.d_zd,
                    device=u.device,
                    dtype=u.dtype,
                ),
                "log_var_d_prior": torch.zeros(
                    batch_size,
                    self.d_zd,
                    device=u.device,
                    dtype=u.dtype,
                ),
                "mu_c_prior": torch.zeros(
                    batch_size,
                    self.d_zc,
                    device=u.device,
                    dtype=u.dtype,
                ),
                "log_var_c_prior": torch.zeros(
                    batch_size,
                    self.d_zc,
                    device=u.device,
                    dtype=u.dtype,
                ),
            }
        )


def build_cvae_model(config: dict) -> OC_iVAE:
    """Build OC-iVAE with a fixed standard normal prior for cVAE baseline."""
    model = OC_iVAE(config)
    d_zd = int(config.get("d_zd", D_ZD))
    d_zc = int(config.get("d_zc", D_ZC))
    model.prior = StandardNormalPrior(d_zd=d_zd, d_zc=d_zc)
    return model
