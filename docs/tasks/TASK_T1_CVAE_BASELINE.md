# Task T1 · cVAE Baseline

> Context file: AGENTS.md (auto-loaded)  
> Scope: ONLY create `baselines/cvae.py` and `configs/ds03_cvae_baseline.yaml`  
> Do NOT modify any existing file in `model/`, `loss/`, `train/`, or `eval/`

---

## Background

The cVAE baseline is the **single most important comparison** for the OC-iVAE paper.
It differs from OC-iVAE in exactly one component: the prior.

| Component | OC-iVAE (full) | cVAE baseline |
|---|---|---|
| Encoder (z_d branch) | TCN, no u | identical |
| Encoder (z_c branch) | TCN + FiLM conditioned on u | identical |
| Prior for z_d | `PriorNetwork(u)` → conditional Gaussian | fixed `N(0, I)` |
| Prior for z_c | `PriorNetwork(u)` → conditional Gaussian | fixed `N(0, I)` |
| KL target | `KL(q ‖ p(z|u))` | `KL(q ‖ N(0, I))` |
| All other losses | HSIC, Mono, Spearman, Recon | identical |

This isolates the contribution of the **iVAE conditional prior** — the theoretical
core of OC-iVAE. All other hyperparameters must match `ds03_default.yaml` exactly.

---

## Task

### Step 1 — Create `baselines/cvae.py`

Implement `StandardNormalPrior`, a drop-in replacement for `model/prior.py:PriorNetwork`.

```
File location: baselines/cvae.py
```

Implement this single class:

**`StandardNormalPrior(nn.Module)`**
- `__init__(self, d_zd: int, d_zc: int)`: store dims only, no learnable parameters
- `forward(self, u: Tensor) -> dict`: ignore `u` entirely; return a dict with the
  same four keys as `PriorNetwork.forward()`, all filled with zeros (mean) and
  zeros (log-variance, i.e. unit variance), shaped correctly from `u.shape[0]` (batch size)
- The returned dict must have exactly these keys (matching `oc_ivae.py` forward interface):
  - `"mu_d_prior"`: shape `(B, d_zd)`, all zeros
  - `"log_var_d_prior"`: shape `(B, d_zd)`, all zeros  
  - `"mu_c_prior"`: shape `(B, d_zc)`, all zeros
  - `"log_var_c_prior"`: shape `(B, d_zc)`, all zeros
- Device must match input `u` — use `torch.zeros(..., device=u.device)`
- Add a docstring explaining: "cVAE baseline prior. Ignores u and returns standard
  normal parameters. Used to isolate the contribution of iVAE conditional prior."

Also implement a convenience function at module level:

**`build_cvae_model(config: dict) -> OC_iVAE`**
- Imports `OC_iVAE` from `model.oc_ivae`
- Constructs a normal `OC_iVAE` from config
- Replaces `model.prior` with `StandardNormalPrior(d_zd, d_zc)`
- Returns the modified model
- This is the only entry point scripts should use for the cVAE baseline

```python
# Expected usage in a training script:
from baselines.cvae import build_cvae_model
model = build_cvae_model(config)
# Everything else (trainer, loss, eval) is identical to full OC-iVAE
```

### Step 2 — Create `configs/ds03_cvae_baseline.yaml`

Copy `configs/ds03_default.yaml` exactly. Change only these two fields:

```yaml
# ── ADD this section at the top of the file ──────────────────────────────
experiment:
  name: ds03_cvae_baseline
  description: "cVAE baseline: conditional encoder, standard normal prior"
  baseline_type: cvae                 # used by run scripts to select build_cvae_model

# ── KEEP all other sections identical to ds03_default.yaml ───────────────
# data: ...        unchanged
# model: ...       unchanged
# training: ...    unchanged
# loss_weights: ... unchanged
# postprocess: ... unchanged
# logging:
#   checkpoint_dir: checkpoints/ds03_cvae_baseline   ← only this line changes
#   log_dir: runs/ds03_cvae_baseline                 ← and this line
```

Do not change any numeric hyperparameter. The only differences from
`ds03_default.yaml` are `experiment.*` and the checkpoint/log paths.

---

## Implementation constraints

- `baselines/cvae.py` must import only from `model.*` and standard library / torch.
  Do not import from `loss/`, `train/`, or `eval/`.
- Do not add `__init__.py` to `baselines/` — Codex will handle imports directly.
- `StandardNormalPrior` must pass `isinstance(prior, nn.Module)` — it is an
  `nn.Module` even though it has no parameters.
- Do not register any buffers or parameters in `StandardNormalPrior.__init__`.
  The zeros are created fresh in `forward` each call.
- All type hints required on public methods.

---

## Verification

After implementing, run the following smoke test mentally (do not run training):

```python
import torch
from baselines.cvae import StandardNormalPrior, build_cvae_model

# Test StandardNormalPrior output shapes and device
prior = StandardNormalPrior(d_zd=2, d_zc=8)
u = torch.zeros(16, 11)
out = prior(u)
assert out["mu_d_prior"].shape == (16, 2)
assert out["log_var_d_prior"].shape == (16, 2)
assert out["mu_c_prior"].shape == (16, 8)
assert out["log_var_c_prior"].shape == (16, 8)
assert (out["mu_d_prior"] == 0).all()
assert (out["log_var_d_prior"] == 0).all()

# Test device propagation
u_gpu = torch.zeros(4, 11).cuda()  # only if CUDA available
out_gpu = prior(u_gpu)
assert out_gpu["mu_d_prior"].device == u_gpu.device

# Test build_cvae_model returns OC_iVAE with replaced prior
config = {...}  # minimal valid config
model = build_cvae_model(config)
from baselines.cvae import StandardNormalPrior
assert isinstance(model.prior, StandardNormalPrior)
```

Then run:
```bash
python -m pytest tests/ -v
```

---

## Task completion checklist

- [ ] `baselines/cvae.py` created with `StandardNormalPrior` and `build_cvae_model`
- [ ] `configs/ds03_cvae_baseline.yaml` created, all numerics identical to ds03_default
- [ ] `StandardNormalPrior.forward()` returns correct shapes for any batch size
- [ ] Device propagation correct (`u.device` used for output tensors)
- [ ] Type hints on all public methods
- [ ] Docstrings explain baseline purpose
- [ ] No modification to any existing file
- [ ] `python -m pytest tests/ -v` passes
