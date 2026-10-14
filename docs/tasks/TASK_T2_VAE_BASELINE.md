# Task T2 · Standard VAE Baseline

> Context file: AGENTS.md (auto-loaded)
> Scope: ONLY create `baselines/vae.py` and `configs/ds03_vae_baseline.yaml`
> Prerequisite: T1 and T1b must be complete
> Do NOT modify any existing file in `model/`, `loss/`, `train/`, `eval/`, or `scripts/`

---

## Background

The standard VAE baseline removes **all operating-condition conditioning**,
answering the question: "What is the total contribution of the entire
OC-iVAE conditioning mechanism (FiLM + conditional prior combined)?"

This forms a three-step ablation ladder with T1:

```
Full Model (OC-iVAE)
  └─ conditional prior p(z|u)          ← T1 removes this → cVAE
      └─ FiLM conditioning on u         ← T2 also removes this → standard VAE
```

| Component | Full Model | T1 cVAE | T2 Std VAE |
|-----------|:---------:|:-------:|:----------:|
| z_d branch (no u) | ✅ | ✅ | ✅ |
| z_c branch with FiLM | ✅ | ✅ | ❌ → no FiLM |
| Conditional prior p(z\|u) | ✅ | ❌ → N(0,I) | ❌ → N(0,I) |
| HSIC loss | ✅ | ✅ | ✅ |
| Monotonicity loss | ✅ | ✅ | ✅ |
| Spearman loss | ✅ | ✅ | ✅ |

The z_c branch is **kept** (so the latent space capacity is identical),
but its FiLM generator is bypassed — it receives no operating condition
information. Combined with a standard normal prior, z_c has no reason to
encode operating conditions, so the model effectively becomes a standard VAE
with a split latent space but no conditioning signal.

**Why keep z_c?** Removing z_c would make T2 identical to T5-A6 (w/o z_c
ablation). Keeping z_c with the same total dimension (D_ZD + D_ZC = 10)
ensures fair capacity comparison: any metric difference is due to conditioning,
not model size.

---

## Task

### Step 1 — Create `baselines/vae.py`

Implement two classes and one builder function.

```
File location: baselines/vae.py
```

**`PassthroughZcBranch(nn.Module)`**

A drop-in replacement for `model/branches.py:ZcBranch` that bypasses FiLM.

- `__init__(self, c_hidden: int, d_zc: int)`:
  - Creates a single `TCNResBlock(c_hidden, c_hidden, kernel_size=3, dilation=1)`
    — import `TCNResBlock` from `model.backbone`
  - Creates a `MaskedAvgPool` — import from `model.branches`
  - Creates `Linear(c_hidden, 2 * d_zc)` as the output head
- `forward(self, h: Tensor, mask: Tensor, u: Tensor) -> tuple[Tensor, Tensor]`:
  - Accepts `u` in the signature (same interface as `ZcBranch`) but **ignores it entirely**
  - Applies `TCNResBlock(h)`, then `MaskedAvgPool`, then `Linear`
  - Returns `(mu_c, log_var_c)` with shapes `(B, d_zc)` each
  - Docstring must state: "Standard VAE z_c branch. Accepts u for interface
    compatibility but ignores it. No FiLM conditioning applied."

**`StandardNormalPrior(nn.Module)`**

This class already exists in `baselines/cvae.py`. Do NOT duplicate it.
Instead, import it:

```python
from baselines.cvae import StandardNormalPrior
```

**`build_vae_model(config: dict) -> OC_iVAE`**

- Imports `OC_iVAE` from `model.oc_ivae`
- Imports `ZcBranch` from `model.branches` (to verify replacement target)
- Constructs a normal `OC_iVAE` from config
- Replaces `model.zc_branch` with `PassthroughZcBranch(c_hidden, d_zc)`
- Replaces `model.prior` with `StandardNormalPrior(d_zd, d_zc)`
- Returns the modified model
- Reads `c_hidden`, `d_zd`, `d_zc` from config dict using these keys:
  `config.get("c_hidden", 64)`, `config.get("d_zd", 2)`, `config.get("d_zc", 8)`
- Docstring: "Standard VAE baseline. No FiLM conditioning, standard normal
  prior. Used to measure total contribution of OC-iVAE conditioning mechanism."

```python
# Expected usage:
from baselines.vae import build_vae_model
model = build_vae_model(config)
assert isinstance(model.prior, StandardNormalPrior)
# model.zc_branch is now PassthroughZcBranch — no FiLM
```

### Step 2 — Register in `_build_model()` router

**This is the only permitted modification to `scripts/run_train.py` and
`scripts/run_eval.py`.** Add one `elif` branch to the existing `_build_model()`
function in both files:

```python
# In the existing _build_model() function, add after the cvae branch:
elif baseline_type == "vae":
    from baselines.vae import build_vae_model
    return build_vae_model(model_config)
```

Do not change anything else in either script.

### Step 3 — Create `configs/ds03_vae_baseline.yaml`

Copy `configs/ds03_default.yaml` exactly. Change only these fields:

```yaml
experiment:
  name: ds03_vae_baseline
  description: "Standard VAE baseline: no FiLM, standard normal prior"
  baseline_type: vae

logging:
  checkpoint_dir: checkpoints/ds03_vae_baseline
  log_dir: runs/ds03_vae_baseline
```

Do not change any numeric hyperparameter. All loss weights, lr, epochs,
batch construction — identical to `ds03_default.yaml`.

---

## Implementation constraints

- `PassthroughZcBranch` must import `TCNResBlock` from `model.backbone` and
  `MaskedAvgPool` from `model.branches` — do not reimplement these.
- `StandardNormalPrior` must be imported from `baselines.cvae`, not redefined.
- All public methods must have type hints and docstrings.
- `PassthroughZcBranch.forward()` signature must be
  `(self, h: Tensor, mask: Tensor, u: Tensor) -> tuple[Tensor, Tensor]`
  to match `ZcBranch`'s interface exactly — even though `u` is unused.
- Do not add `__init__.py` to `baselines/`.
- Do not modify `model/branches.py` or any other file in `model/`.

---

## Verification

After implementing, run the following smoke test mentally:

```python
import torch
from baselines.vae import PassthroughZcBranch, build_vae_model
from baselines.cvae import StandardNormalPrior

# Test PassthroughZcBranch ignores u
branch = PassthroughZcBranch(c_hidden=64, d_zc=8)
B, T, C = 4, 63, 64
h = torch.randn(B, T, C)
mask = torch.ones(B, T, dtype=torch.bool)
u = torch.randn(B, 11)           # should be ignored
u_zeros = torch.zeros(B, 11)     # different u
mu1, lv1 = branch(h, mask, u)
mu2, lv2 = branch(h, mask, u_zeros)
assert mu1.shape == (B, 8)
assert lv1.shape == (B, 8)
# Output must be identical regardless of u (u is ignored)
assert torch.allclose(mu1, mu2), "PassthroughZcBranch must ignore u"

# Test build_vae_model returns OC_iVAE with correct replacements
config = {...}  # minimal valid config
model = build_vae_model(config)
assert isinstance(model.prior, StandardNormalPrior)
assert isinstance(model.zc_branch, PassthroughZcBranch)

# Test forward() output interface is unchanged
x = torch.randn(B, T, 14)
u = torch.randn(B, 11)
W_seq = torch.randn(B, T, 4)
out = model(x, mask, u, W_seq)
required_keys = {
    'x_hat', 'mu_d', 'log_var_d', 'mu_c', 'log_var_c',
    'mu_d_prior', 'log_var_d_prior', 'mu_c_prior', 'log_var_c_prior',
    'z_d', 'z_c', 'hi_raw'
}
assert required_keys.issubset(out.keys()), f"Missing keys: {required_keys - out.keys()}"
```

Then run:
```bash
python -m pytest tests/ -v
```

---

## What to expect from results

After training, compare against the cVAE baseline (T1):

```
Full Model:    Mono=0.5444  Trend=0.9102  Prog=0.9079
cVAE (T1):     Mono=0.3495  Trend=0.8745  Prog=0.7107  ← prior only removed
Std VAE (T2):  Mono=?       Trend=?       Prog=?        ← prior + FiLM removed
```

Expected direction: T2 metrics ≤ T1 metrics, because T2 removes an additional
conditioning mechanism (FiLM) on top of what T1 already removed.

If T2 is unexpectedly close to T1 (delta < 0.01 on all metrics), it suggests
FiLM's contribution is marginal and the conditional prior is doing most of the
work — this is still a valid and publishable finding.

---

## Task completion checklist

- [ ] `baselines/vae.py` created with `PassthroughZcBranch` and `build_vae_model`
- [ ] `StandardNormalPrior` imported from `baselines.cvae`, not redefined
- [ ] `PassthroughZcBranch.forward()` output is **identical** regardless of `u` value
- [ ] `build_vae_model()` replaces both `model.zc_branch` and `model.prior`
- [ ] `elif baseline_type == "vae"` added to `_build_model()` in both run scripts
- [ ] `configs/ds03_vae_baseline.yaml` created, all numerics identical to ds03_default
- [ ] Full model `forward()` interface still returns all 12 required keys
- [ ] Type hints and docstrings on all public methods
- [ ] No modification to any file in `model/`, `loss/`, `train/`, `eval/`
- [ ] `python -m pytest tests/ -v` passes
