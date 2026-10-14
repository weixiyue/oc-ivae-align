"""Health indicator head for degradation latent variables."""

from __future__ import annotations

from torch import Tensor, nn

from model.constants import D_ZD


class HIHead(nn.Module):
    """Map ``z_d`` to a raw scalar health indicator."""

    def __init__(self, config: dict) -> None:
        """Initialize the no-bias linear HI projection.

        Args:
            config: Model config with ``d_zd``.
        """
        super().__init__()
        self.d_zd = int(config.get("d_zd", D_ZD))
        self.linear = nn.Linear(self.d_zd, 1, bias=False)

    def forward(self, z_d: Tensor) -> Tensor:
        """Return raw HI values with shape ``(B,)``."""
        if z_d.dim() != 2:
            raise ValueError("z_d must have shape (B, d_zd)")
        if z_d.shape[1] != self.d_zd:
            raise ValueError("z_d feature dimension does not match config['d_zd']")

        return self.linear(z_d).squeeze(-1)
