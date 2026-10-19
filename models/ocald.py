"""OCALD (kgld 4f9e53cd), adapted only to the shared KGLD input protocol."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

SOURCE_COMMIT = '4f9e53cd3623b5cf0311d594fd7efcd4bec88206'
REPRESENTATION_MODULES = ('sensor_encoder', 'conditional_prior', 'decoder')


def to_valid_prefix(x: torch.Tensor, w: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move valid timesteps, in order, to a prefix; zero only the new padding."""
    if mask.dtype != torch.bool or x.shape[:2] != mask.shape or w.shape[:2] != mask.shape:
        raise ValueError('OCALD requires aligned X/W and bool mask')
    lengths = mask.sum(1)
    if (lengths == 0).any():
        raise ValueError('OCALD cannot encode an empty flight')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask, mask.shape[1]).argsort(dim=1, stable=True)
    prefix = positions < lengths[:, None]
    x = x.gather(1, indices[:, :, None].expand_as(x)).masked_fill(~prefix[:, :, None], 0)
    w = w.gather(1, indices[:, :, None].expand_as(w)).masked_fill(~prefix[:, :, None], 0)
    return x, w, prefix


def valid_lengths(mask: torch.Tensor) -> torch.Tensor:
    """Check the packed GRU contract at the adapted input boundary."""
    lengths = mask.sum(1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if mask.dtype != torch.bool or (lengths == 0).any() or not torch.equal(mask, expected):
        raise ValueError('packed GRU requires a nonempty valid prefix')
    return lengths


def time_positions(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Source time coordinate: 0..1 on the valid prefix, zero on padding."""
    lengths = valid_lengths(mask)
    positions = torch.arange(mask.shape[1], device=mask.device, dtype=dtype)[None, :]
    return (positions / (lengths - 1).clamp_min(1)[:, None]).masked_fill(~mask, 0)


def last_valid_state(gru: nn.GRU, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Extract the last state without feeding padded values through the GRU."""
    packed = pack_padded_sequence(sequence, valid_lengths(mask).cpu(), batch_first=True,
                                  enforce_sorted=False)
    _, hidden = gru(packed)
    return hidden[-1]


class SensorEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_i_dim: int, latent_c_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu_i = nn.Linear(hidden_dim, latent_i_dim)
        self.logvar_i = nn.Linear(hidden_dim, latent_i_dim)
        self.mu_c = nn.Linear(hidden_dim, latent_c_dim)
        self.logvar_c = nn.Linear(hidden_dim, latent_c_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = last_valid_state(self.gru, x, mask)
        return self.mu_i(state), self.logvar_i(state), self.mu_c(state), self.logvar_c(state)


class ConditionalPrior(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_c_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_c_dim)
        self.logvar = nn.Linear(hidden_dim, latent_c_dim)

    def forward(self, w: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = last_valid_state(self.gru, w, mask)
        return self.mu(state), self.logvar(state)


class Decoder(nn.Module):
    def __init__(self, latent_i_dim: int, latent_c_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        self.use_w = False
        self.condition_dim = 0
        self.layers = nn.Sequential(
            nn.Linear(latent_i_dim + latent_c_dim + 1, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def enable_w_input(self, condition_dim: int) -> None:
        """Preserve source initialization: resize only this layer under fork_rng."""
        with torch.random.fork_rng(devices=[]):
            old = self.layers[0]
            self.layers[0] = nn.Linear(old.in_features + condition_dim, old.out_features)
        self.use_w = True
        self.condition_dim = condition_dim

    def forward(self, z_i: torch.Tensor, z_c: torch.Tensor, positions: torch.Tensor,
                w: torch.Tensor | None = None) -> torch.Tensor:
        latents = torch.cat((z_i, z_c), dim=-1)
        parts = [latents[:, None, :].expand(-1, positions.shape[1], -1)]
        if self.use_w:
            if w is None or w.shape != (*positions.shape, self.condition_dim):
                raise ValueError('W-conditioned decoder requires aligned per-step W')
            parts.append(w)
        parts.append(positions[:, :, None])
        return self.layers(torch.cat(parts, dim=-1))


class HIHead(nn.Module):
    def __init__(self, latent_i_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_i_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, mu_i: torch.Tensor) -> torch.Tensor:
        return self.layers(mu_i).squeeze(-1)


class OCALD(nn.Module):
    """Source global I/C (8/8) structure; no scalar I or temporal C variants."""
    def __init__(self, config: dict):
        super().__init__()
        if config['gru_layers'] != 1 or config['bidirectional']:
            raise ValueError('source OCALD uses one unidirectional GRU layer')
        if (config['latent_i_dim'], config['latent_c_dim']) != (8, 8):
            raise ValueError('this integration supports only 8D I and 8D global C')
        if config.get('c_mode', 'global') != 'global' or config.get('score_mode', 'hi_head') != 'hi_head':
            raise ValueError('only global C and HIHead are supported')
        self.decoder_use_w = config['decoder_use_w']
        if not isinstance(self.decoder_use_w, bool):
            raise ValueError('decoder_use_w must be boolean')
        self.sensor_encoder = SensorEncoder(config['in_features'], config['sensor_hidden_dim'], 8, 8)
        self.conditional_prior = ConditionalPrior(config['condition_dim'], config['condition_hidden_dim'], 8)
        self.decoder = Decoder(8, 8, config['decoder_hidden_dims'], config['in_features'])
        self.hi_head = HIHead(8, config['hi_hidden_dim'])
        if self.decoder_use_w:
            self.decoder.enable_w_input(config['condition_dim'])

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.sensor_encoder(x, mask)

    def score(self, mu_i: torch.Tensor) -> torch.Tensor:
        return self.hi_head(mu_i)

    def forward(self, x: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stochastic representation forward on already-adapted prefix inputs."""
        mu_i, logvar_i, mu_c, logvar_c = self.encode(x, mask)
        prior_mu_c, prior_logvar_c = self.conditional_prior(w, mask)
        z_i = mu_i + torch.exp(0.5 * logvar_i) * torch.randn_like(mu_i)
        z_c = mu_c + torch.exp(0.5 * logvar_c) * torch.randn_like(mu_c)
        decoder_w = w.masked_fill(~mask[:, :, None], 0) if self.decoder_use_w else None
        return dict(mu_i=mu_i, logvar_i=logvar_i, mu_c=mu_c, logvar_c=logvar_c,
                    prior_mu_c=prior_mu_c, prior_logvar_c=prior_logvar_c,
                    x_hat=self.decoder(z_i, z_c, time_positions(mask, x.dtype), decoder_w))

    def health_score(self, x: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decreasing-health score; sign conversion is evaluation-only."""
        x, _, mask = to_valid_prefix(x, w, mask)
        return -self.score(self.encode(x, mask)[0])


def diagonal_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """Source KL summed over latent dimensions."""
    return 0.5 * (logvar_p - logvar_q + torch.exp(logvar_q - logvar_p)
                  + (mu_q - mu_p).square() * torch.exp(-logvar_p) - 1).sum(dim=-1)


def training_loss(head: HIHead, weights: dict, *, x=None, mask=None, output=None,
                  r_early=None, r_late=None, sigma_x_squared=1., kappa=1.) -> dict[str, torch.Tensor]:
    """Flight-summed Gaussian reconstruction/KL and source softplus/head L2."""
    keys = ('reconstruction', 'kl_i', 'kl_c', 'order', 'l2')
    if set(weights) != set(keys) or any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in weights.values()):
        raise ValueError('five finite nonnegative loss weights are required')
    zero = next(head.parameters()).new_zeros(())
    terms = dict.fromkeys(keys, zero)
    active = []
    if output is not None:
        if sigma_x_squared <= 0:
            raise ValueError('sigma_x_squared must be positive')
        residual = x.masked_fill(~mask[:, :, None], 0) - output['x_hat'].masked_fill(~mask[:, :, None], 0)
        reconstruction = residual.square().sum(dim=(1, 2)) / (2 * sigma_x_squared)
        kl_i = diagonal_gaussian_kl(output['mu_i'], output['logvar_i'],
                                   torch.zeros_like(output['mu_i']), torch.zeros_like(output['logvar_i']))
        kl_c = diagonal_gaussian_kl(*(output[k] for k in ('mu_c', 'logvar_c', 'prior_mu_c', 'prior_logvar_c')))
        terms.update(reconstruction=reconstruction.mean(), kl_i=kl_i.mean(), kl_c=kl_c.mean())
        if all(weights[k] == 1 for k in keys[:3]):
            active.append((reconstruction + kl_i + kl_c).mean())
        else:
            active.extend(weights[k] * terms[k] for k in keys[:3] if weights[k])
    elif any(weights[k] for k in keys[:3]):
        raise ValueError('representation inputs required')
    if r_early is not None and r_early.numel():
        if kappa <= 0:
            raise ValueError('kappa must be positive')
        terms['order'] = F.softplus((r_early - r_late) / kappa).mean()
    elif weights['order']:
        raise ValueError('nonzero order weight requires valid pairs')
    terms['l2'] = torch.stack([p.square().sum() for p in head.parameters()]).sum()
    active.extend(weights[k] * terms[k] for k in ('order', 'l2') if weights[k])
    loss = sum(active[1:], active[0]) if active else zero
    return dict(loss=loss, **terms)


def freeze_representation(model: OCALD) -> None:
    """Only HIHead is trainable in the second stage."""
    for name in REPRESENTATION_MODULES:
        module = getattr(model, name)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
