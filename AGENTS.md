# AGENTS.md — OC-iVAE Project

> 本文件是 Codex CLI 的项目指令文件。每次会话开始时自动读取。
> 文档版本：v2.0 | 对应模型设计文档：OC_iVAE_model_design.md v1.0
> **v2.0 变更**：核心模型实现已完成；当前阶段重心转移至实验扩展（Baselines + Ablations）

---

## Current Phase: Experiment Extension

> ⚠️ 核心模型代码已完成并通过验证。当前阶段的任务是补充实验，为 IEEE TII 投稿提供对比证据。
> 不要修改 `model/`、`loss/`、`train/`、`eval/` 中已完成的代码。

### 实验任务状态

| 编号  | 任务                     | 状态      | 输出位置                                 |
| ----- | ------------------------ | --------- | ---------------------------------------- |
| T1    | cVAE Baseline (DS03)     | 🔲 待完成  | `baselines/cvae.py`                      |
| T2    | 标准 VAE Baseline (DS03) | 🔲 待完成  | `baselines/vae.py`                       |
| T3    | LSTM-AE Baseline (DS03)  | 🔲 待完成  | `baselines/lstm_ae.py`                   |
| T4    | AE Baseline (DS03)       | 🔲 待完成  | `baselines/ae.py`                        |
| T5-A1 | Ablation: w/o HSIC       | 🔲 待完成  | `configs/ds03_ablation_no_hsic.yaml`     |
| T5-A2 | Ablation: w/o Mono       | 🔲 待完成  | `configs/ds03_ablation_no_mono.yaml`     |
| T5-A3 | Ablation: w/o Spearman   | 🔲 待完成  | `configs/ds03_ablation_no_spearman.yaml` |
| T5-A4 | Ablation: w/o FiLM       | 🔲 待完成  | `configs/ds03_ablation_no_film.yaml`     |
| T5-A5 | Ablation: w/o Cond Prior | ♻️ 复用 T1 | —                                        |
| T5-A6 | Ablation: w/o z_c        | 🔲 待完成  | `configs/ds03_ablation_no_zc.yaml`       |
| T6    | DS02 完整实验            | 🔲 待完成  | `configs/ds02_oc_ivae.yaml`              |

---

## Project Overview

This project implements **OC-iVAE** (Operating Condition-aware Identifiable VAE),
an unsupervised health indicator (HI) construction model for turbofan engines,
evaluated on **N-CMAPSS DS03** (and DS02 for cross-dataset validation).

**Key objective**: Learn a disentangled latent space where `z_d` (degradation)
and `z_c` (operating condition) are independent, then derive a monotonically
decreasing scalar HI from `z_d`.

**Framework**: PyTorch | **Python**: 3.10+ | **Datasets**: N-CMAPSS DS03, DS02

---

## Commands

```bash
# 安装依赖
pip install torch numpy scipy pandas h5py pyyaml

# 运行单元测试（任务完成后必须执行）
python -m pytest tests/ -v

# 测试单个文件（优先用这个，快速反馈）
python -m pytest tests/test_backbone.py -v

# 检查 shape 正确性（快速冒烟测试）
python scripts/smoke_test.py

# 运行训练（Full Model）
python scripts/run_train.py --config configs/ds03_default.yaml

# 运行 baseline 训练（以 cVAE 为例）
python scripts/run_train.py --config configs/ds03_cvae_baseline.yaml

# 运行评估
python scripts/run_eval.py --config configs/ds03_default.yaml

# 运行消融实验（批量）
for cfg in configs/ds03_ablation_*.yaml; do
    python scripts/run_train.py --config $cfg
done
```

---

## Project Structure

```
oc_ivae/
├── data/
│   ├── dataset.py          # NCMAPSSDataset: returns (x, mask, u, W_seq, unit_id, cycle_id, hi)
│   ├── sampler.py          # UnitStratifiedSampler: K units × M cycles per batch
│   └── preprocess.py       # Downsampling, P90 truncation, z-score, u construction
├── model/                  # ✅ COMPLETE — DO NOT MODIFY
│   ├── backbone.py         # SharedTCNBackbone
│   ├── branches.py         # ZdBranch, ZcBranch, FiLMGenerator
│   ├── prior.py            # PriorNetwork (conditional Gaussian prior)
│   ├── decoder.py          # TCNDecoder
│   ├── hi_head.py          # HIHead
│   └── oc_ivae.py          # OC_iVAE main model
├── loss/                   # ✅ COMPLETE — DO NOT MODIFY
│   ├── recon.py
│   ├── kl.py
│   ├── hsic.py
│   ├── monotonicity.py
│   ├── spearman.py
│   └── total.py
├── train/                  # ✅ COMPLETE — DO NOT MODIFY
│   ├── trainer.py
│   └── scheduler.py        # NOTE: currently empty; KL schedule is in loss/total.py
├── eval/                   # ✅ COMPLETE — DO NOT MODIFY
│   ├── inference.py
│   ├── postprocess.py
│   └── metrics.py
├── baselines/              # 🔲 NEW — baseline implementations go here
│   ├── cvae.py             # T1: StandardNormalPrior + build_cvae_model()
│   ├── vae.py              # T2: single-branch VAE, standard normal prior
│   ├── lstm_ae.py          # T3: LSTM encoder-decoder, recon-error HI
│   └── ae.py               # T4: TCN-AE, recon-error HI
├── scripts/
│   ├── run_train.py        # ✅ reads experiment.baseline_type to select model builder
│   ├── run_eval.py         # ✅ supports cycle_hi_path: null for DS02
│   └── smoke_test.py       # currently empty
├── configs/
│   ├── ds03_default.yaml           # ✅ canonical Full Model config — DO NOT MODIFY
│   ├── ds03_cvae_baseline.yaml     # T1 — to be created
│   ├── ds03_vae_baseline.yaml      # T2 — to be created
│   ├── ds03_lstm_ae_baseline.yaml  # T3 — to be created
│   ├── ds03_ae_baseline.yaml       # T4 — to be created
│   ├── ds03_ablation_no_hsic.yaml  # T5-A1 — to be created
│   ├── ds03_ablation_no_mono.yaml  # T5-A2 — to be created
│   ├── ds03_ablation_no_spearman.yaml # T5-A3 — to be created
│   ├── ds03_ablation_no_film.yaml  # T5-A4 — to be created
│   ├── ds03_ablation_no_zc.yaml    # T5-A6 — to be created
│   └── ds02_oc_ivae.yaml           # T6 — to be created
├── docs/
└── tests/                  # currently empty placeholders
```

**Key rule**: All new baseline code goes in `baselines/`. All new configs go in `configs/`.
Never modify files in `model/`, `loss/`, `train/`, or `eval/`.

---

## Architecture Constants (DO NOT MODIFY)

These values are fixed by design. Never change them without explicit instruction:

```python
# Encoder
C_HIDDEN = 64
N_BACKBONE_LAYERS = 6          # dilation = [1, 2, 4, 8, 16, 32]
N_BRANCH_LAYERS = 1            # one extra TCN layer per branch
D_ZD = 2                       # degradation latent dim
D_ZC = 8                       # operating condition latent dim
U_DIM = 11                     # auxiliary variable dim

# Decoder
N_DECODER_LAYERS = 4           # dilation = [1, 2, 4, 8]
DECODER_CAUSAL = False         # non-causal (bidirectional receptive field)

# Training
K_UNITS_PER_BATCH = 7
M_CYCLES_PER_UNIT = 12
DELTA_T_RANKING = 3
RANKING_MARGIN = 0.01
SPEARMAN_TAU = 0.1
```

---

## Baseline Design Principles

All baselines must follow these rules to ensure fair comparison:

1. **Same hyperparameters**: lr, weight_decay, epochs, batch construction,
   loss weights — all identical to `ds03_default.yaml` unless the component
   being ablated requires a change.
2. **Same evaluation pipeline**: use `eval/inference.py`, `eval/postprocess.py`,
   `eval/metrics.py` unchanged. Baselines output the same metrics format.
3. **Same HI post-processing**: per-unit min-max + EWMA(alpha=0.3).
4. **Recon-error HI direction**: for AE/LSTM-AE, high reconstruction error =
   more degraded, so `HI = 1 - normalize(recon_error)` to match the
   "1=healthy, 0=failing" convention.
5. **Monotonicity and Spearman losses**: keep these in all VAE-based baselines
   (cVAE, standard VAE) so the only variable is the prior, not the training signal.

---

## Baseline Model Interface

All baselines that wrap `OC_iVAE` (T1, T2) must provide a `build_*_model(config)`
function that returns a model whose `forward()` output matches the locked interface:

```python
# The forward() return dict is LOCKED — all fields required:
{
    'x_hat', 'mu_d', 'log_var_d', 'mu_c', 'log_var_c',
    'mu_d_prior', 'log_var_d_prior', 'mu_c_prior', 'log_var_c_prior',
    'z_d', 'z_c', 'hi_raw'
}
```

AE/LSTM-AE baselines (T3, T4) do not use this interface — they have their own
`forward()` returning `(x_hat, z)` and compute HI from reconstruction error
in a separate eval script `scripts/run_eval_ae.py` (create if needed).

---

## Data Conventions

| Variable   | Shape           | Description                                           |
| ---------- | --------------- | ----------------------------------------------------- |
| `x`        | `(B, L_max, S)` | sensor time series, z-score normalized per unit       |
| `mask`     | `(B, L_max)`    | bool, True = valid timestep                           |
| `u`        | `(B, 11)`       | auxiliary variable: [W_mean×4, W_std×4, Fc_onehot×3]  |
| `W_seq`    | `(B, L_max, 4)` | operating condition time series (alt, XM, TRA, T2)    |
| `unit_id`  | `(B,)`          | int64                                                 |
| `cycle_id` | `(B,)`          | int64                                                 |
| `hi`       | `(B,)`          | float32; NaN when no ground-truth HI available (DS02) |

**Padding convention**: short cycles are zero-padded at the **front** (head).
Long cycles are truncated from the **front** (keep tail).

---

## DS02 Specifics

DS02 has no ground-truth HI. The evaluation pipeline handles this gracefully:

- `dataset.py`: `has_hi=False`, `__getitem__` returns `hi=NaN`
- `inference.py`: sets `hi_gt=None` when all GT values are NaN
- `run_eval.py`: displays `N/A` in the MAE column when `hi_gt=None`
- Config must set `cycle_hi_path: null` explicitly

DS02 may have only 4 units in the train split. The sampler uses
`effective_k = min(k, n_units)` automatically — set `k_units_per_batch: 4`
in `configs/ds02_oc_ivae.yaml` to avoid confusion.

---

## Known Issues (do not fix without explicit task card)

| Issue                              | Location             | Impact                               |
| ---------------------------------- | -------------------- | ------------------------------------ |
| `N_SENSOR_FEATURES=14` hardcoded   | `model/decoder.py`   | Fails if S≠14                        |
| `train/scheduler.py` empty         | `train/scheduler.py` | KL schedule lives in `loss/total.py` |
| `scripts/smoke_test.py` empty      | —                    | No smoke test available              |
| `tests/` all empty                 | —                    | No regression tests                  |
| `sensor_threshold` YAML key unused | `data/preprocess.py` | Dead-sensor threshold always 1e-3    |

Do not fix these unless a task card explicitly instructs it.

---

## Model Forward Interface (LOCKED)

`oc_ivae.py` 的 `forward` 必须返回如下 dict，字段名和类型不可更改：

```python
def forward(self, x, mask, u, W_seq):
    """
    Returns:
        {
            'x_hat':          Tensor (B, L_max, S),
            'mu_d':           Tensor (B, D_ZD),
            'log_var_d':      Tensor (B, D_ZD),
            'mu_c':           Tensor (B, D_ZC),
            'log_var_c':      Tensor (B, D_ZC),
            'mu_d_prior':     Tensor (B, D_ZD),
            'log_var_d_prior':Tensor (B, D_ZD),
            'mu_c_prior':     Tensor (B, D_ZC),
            'log_var_c_prior':Tensor (B, D_ZC),
            'z_d':            Tensor (B, D_ZD),
            'z_c':            Tensor (B, D_ZC),
            'hi_raw':         Tensor (B,),
        }
    """
```

---

## Coding Style

- Python 3.10+, type hints on all public functions
- No global state; pass config as dataclass or dict
- All loss functions: pure functions (no side effects), return scalar Tensor
- LayerNorm before activation (Pre-LN style)
- Mask operations: always use `mask.float()` for arithmetic, keep bool for indexing
- No `print()` in model/loss code; use `logging` in trainer only
- Docstrings: one-line summary + Args/Returns for public classes and functions

---

## Permissions

### Codex may do autonomously:

- Write/edit files in `baselines/` as specified in the task card
- Create new YAML configs in `configs/` as specified in the task card
- Add type hints, docstrings, and inline comments
- Fix obvious bugs within the task scope
- Run `python -m pytest tests/` after completing a task

### Codex must ask before:

- Installing new packages
- Modifying any file in `model/`, `loss/`, `train/`, `eval/`
- Changing any Architecture Constant listed above
- Adding new fields to the `forward()` return dict
- Modifying `configs/ds03_default.yaml`

### Codex must never:

- Run training or long-running scripts
- Modify git history
- Create files outside the project structure above
- "Refactor" modules that are already complete

---

## Experiment Recording

每次实验完成后必须更新 `EXPERIMENT_LOG.md`，格式：

```markdown
## [YYYY-MM-DD] [exp_name]

**Config**: `configs/xxx.yaml`
**Git tag**: `baseline/xxx` or `ablation/xxx`
**Dataset**: DS03 / DS02
**Type**: Full Model / Baseline / Ablation

| Metric | Dev | Test |
|--------|-----|------|
| Monotonicity | | |
| Trendability | | |
| Prognosability | | |
| MAE | | |

**Notes**: (anomalies, deviations from expectation)
```

Git tag convention:

```
baseline/cvae-ds03
baseline/vae-ds03
baseline/lstm-ae-ds03
baseline/ae-ds03
ablation/wo-hsic-ds03
ablation/wo-mono-ds03
ablation/wo-spearman-ds03
ablation/wo-film-ds03
ablation/wo-zc-ds03
dataset/ds02-full
stats/multi-seed-ds03
```

---

## Task Completion Checklist

After completing each task, Codex should verify:

- [ ] File is in the correct location per Project Structure
- [ ] All public functions have type hints and docstrings
- [ ] No import of modules not yet implemented
- [ ] No modification to any file in `model/`, `loss/`, `train/`, `eval/`
- [ ] Config numerics identical to `ds03_default.yaml` (except for the ablated component)
- [ ] `python -m pytest tests/ -v` passes
- [ ] No hardcoded hyperparameter values; all from config dict