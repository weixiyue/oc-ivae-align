## Project: OC-iVAE

OC-iVAE is an Operating Condition-aware identifiable VAE for unsupervised health-indicator construction on turbofan engines. It targets N-CMAPSS DS03, uses PyTorch, separates degradation latent `z_d` from operating-condition latent `z_c`, and derives a monotonically decreasing scalar HI from `z_d`.

## Current State

- All 5 phases complete and passing smoke tests.
- Best checkpoint: `checkpoints/exp_tuning_mono_weight/best_model.pt`
- Latest verified test run: Mono=0.515 Trend=0.891 Prog=0.748 MAE=0.046

## Architecture (locked — do not modify without instruction)

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
K_UNITS_PER_BATCH = 8
M_CYCLES_PER_UNIT = 12
DELTA_T_RANKING = 3
RANKING_MARGIN = 0.01
SPEARMAN_TAU = 0.1
```

## Module Interface Summary

- `data/preprocess.py` | functions | raw `A/W/X_s` -> grouped cycles, normalized `x/W_seq/mask/u`
- `data/dataset.py` | `NCMAPSSDataset` | h5 + cycle-HI + config -> `(x, mask, u, W_seq, unit_id, cycle_id, hi)`
- `data/sampler.py` | `UnitStratifiedSampler` | `dataset.samples` -> batch index lists of `k*m`
- `model/backbone.py` | `SharedTCNBackbone` | `x, mask` -> `h (B,T,C_hidden)`
- `model/branches.py` | `ZdBranch`, `ZcBranch` | `h, mask[, u]` -> posterior `mu/log_var`
- `model/prior.py` | `PriorNetwork` | `u` -> prior params for `z_d` and `z_c`
- `model/decoder.py` | `TCNDecoder` | `z_d, z_c, W_seq, mask` -> `x_hat`
- `model/hi_head.py` | `HIHead` | `z_d` -> `hi_raw`
- `model/oc_ivae.py` | `OC_iVAE` | `x, mask, u, W_seq` -> locked output dict
- `loss/total.py` | `compute_total_loss` | forward dict + ids + epoch -> scalar loss + log dict
- `train/trainer.py` | `Trainer` | model + loaders + config -> training, validation, checkpointing
- `eval/inference.py` | `extract_hi` | model + dataset -> per-unit raw HI and GT HI
- `eval/postprocess.py` | functions | raw per-unit HI -> min-max normalized and EWMA-smoothed HI
- `eval/metrics.py` | `evaluate_all` | extracted HI -> Mono/Trend/Prog/MAE and per-unit details
- `scripts/run_train.py` | `main` | config path -> train, evaluate train HI, log metrics
- `scripts/run_eval.py` | `main` | config/checkpoint/split -> evaluate and save metrics/plots

## Data Flow

```python
# batch = next(train_loader)  # DataLoader uses UnitStratifiedSampler
# x, mask, u, W_seq, unit_id, cycle_id, hi = move_to_device(batch)
# del hi  # training is unsupervised with respect to cycle-level HI
# out = model(x, mask, u, W_seq)
# total_loss, loss_dict = compute_total_loss(out, x, mask, cycle_id, unit_id, epoch, config)
# optimizer.zero_grad(set_to_none=True)
# total_loss.backward()
# clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip_norm"])
# optimizer.step()
# scheduler.step()  # torch CosineAnnealingLR in Trainer
```

## Config System

YAML configs use top-level sections: `data`, `model`, `training`, `loss_weights`, `postprocess`, and `logging`. `scripts/run_train.py` derives `exp_name` from the config filename stem, then sets `checkpoint_dir=checkpoints/{exp_name}` and `log_dir=runs/{exp_name}`. `Trainer` flattens `training`, `loss_weights`, and `logging`; `training.use_early_stopping` defaults to `false`. `scripts/run_eval.py` merges all mapping sections, infers `in_features` from the first dataset sample, and resolves cycle-HI paths from config or sibling `*cycle_dev.h5` / `*cycle_test.h5` files.

## Known Constraints

- Never run training or long-running scripts.
- Never modify git history.
- Never create files outside the project structure unless the task explicitly names them.
- Never refactor modules that are already complete.
- Do not change architecture constants without explicit instruction.
- Do not add fields to `OC_iVAE.forward()` output.
- Do not modify `configs/ds03_default.yaml` without explicit instruction.
- Treat `tests/` and `scripts/smoke_test.py` as currently empty placeholders until tests are implemented.
- Prefer `scripts/run_train.py` and `scripts/run_eval.py` over `test.py` or `scripts/plot_hi.py`.

## Open Tasks

- [ ] Ablation experiments (remove HSIC / mono / spearman)
- [ ] Global normalization for Prognosability improvement
- [ ] Fill in empty test files in `tests/`
