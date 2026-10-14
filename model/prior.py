"""Conditional Gaussian prior network for OC-iVAE latents."""

from __future__ import annotations

from torch import Tensor, nn

from model.constants import C_HIDDEN, D_ZC, D_ZD, U_DIM


class PriorNetwork(nn.Module):
    """Conditional prior ``p(z_d, z_c | u)`` with separate latent heads."""

    def __init__(self, config: dict) -> None:
        """Initialize the shared MLP stem and dual Gaussian heads.

        Args:
            config: Model config with ``u_dim``, ``d_zd``, ``d_zc``, and
                ``c_hidden``.
        """
        super().__init__()
        u_dim = int(config.get("u_dim", U_DIM))
        c_hidden = int(config.get("c_hidden", C_HIDDEN))
        d_zd = int(config.get("d_zd", D_ZD))
        d_zc = int(config.get("d_zc", D_ZC))

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
