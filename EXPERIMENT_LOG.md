# Experiment Log — OC-iVAE

## 2026-06-04 | scaffold-complete | —
- Git tag: v0.1-scaffold
- 所有模块实现完毕并通过冒烟测试
- 关键设计决策：
  - 使用全生命周期数据（含hs=0），663个training cycles
  - k=7（DS03只有9个unit，val占2个）
  - W_seq和u均用全局统计量归一化
- 待验证：正式训练100epoch后的HI曲线形态


## 2026-06-04 20:34:04 | training-run | configs/ds03_default.yaml
- Epochs completed: 11
- Best val recon: 0.949809
- Checkpoint: checkpoints/best_model.pt
- Train monotonicity: -0.014172
- Train trendability: 0.253116
- Train prognosability: 0.754236

## 2026-06-04 | baseline | configs/ds03_default.yaml
- Git tag: v0.1-baseline
- Monotonicity:  -0.014
- Trendability:   0.253
- Prognosability: 0.754
- 问题：早期振荡严重，hi_raw范围窄(-0.09~-0.01)
- 下一步：exp/tuning-mono-weight，lambda_mono 2→10，lambda_spearman 0.5→2

## 2026-06-08 | tuning-mono-weight | configs/exp_tuning_mono_weight.yaml
- 调整：lambda_mono=10, lambda_spearman=2, warmup缩短至5epoch
- 期望：早期振荡减少，Mono>0.2

## 2026-06-08 09:36:21 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 52
- Best val recon: 0.944035
- Checkpoint: checkpoints/best_model.pt
- Train monotonicity: 0.360256
- Train trendability: 0.867270
- Train prognosability: 0.638577

## 2026-06-08 | tuning-mono-weight | configs/exp_tuning_mono_weight.yaml
- Git tag: v0.2-tuning-mono
- Monotonicity:  0.360
- Trendability:  0.867
- Prognosability: 0.639
- 结论：lambda_mono=10, lambda_spearman=2 显著改善曲线形态
- 残留问题：EOL值不一致（Unit3/7约0.3，Unit1/2/4约0.0），Prog偏低
- 下一步：提高 lambda_mono 或调整 EWMA alpha

## 2026-06-08 10:50:22 | training-run | configs/exp_tuning_mape_prog.yaml
- Epochs completed: 7
- Best val recon: 0.996979
- Checkpoint: checkpoints/exp_tuning_mape_prog/best_model.pt
- Train monotonicity: 0.111151
- Train trendability: 0.681054
- Train prognosability: 0.703521

## 2026-06-08 11:01:34 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 38
- Best val recon: 0.943932
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.243597
- Train trendability: 0.846248
- Train prognosability: 0.707967

## 2026-06-08 11:04:40 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 14
- Best val recon: 0.992187
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.034007
- Train trendability: 0.200472
- Train prognosability: 0.752074

## 2026-06-08 11:14:14 | training-run | configs/exp_tuning_mape_prog.yaml
- Epochs completed: 92
- Best val recon: 0.946047
- Checkpoint: checkpoints/exp_tuning_mape_prog/best_model.pt
- Train monotonicity: 0.406319
- Train trendability: 0.880280
- Train prognosability: 0.720473

## 2026-06-08 11:20:20 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 80
- Best val recon: 0.939700
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.438071
- Train trendability: 0.872562
- Train prognosability: 0.715775

## 2026-06-08 11:37:32 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 82
- Best val recon: 0.955154
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.411648
- Train trendability: 0.885804
- Train prognosability: 0.657994

## 2026-06-08 11:39:22 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 94
- Best val recon: 0.946734
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.450450
- Train trendability: 0.931190
- Train prognosability: 0.673556

## 2026-06-08 11:41:16 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 100
- Best val recon: 0.949574
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.424149
- Train trendability: 0.868048
- Train prognosability: 0.675965

## 2026-06-08 16:46:39 | training-run | configs/exp_tuning_mono_weight.yaml
- Epochs completed: 80
- Best val recon: 0.941424
- Checkpoint: checkpoints/exp_tuning_mono_weight/best_model.pt
- Train monotonicity: 0.454861
- Train trendability: 0.879185
- Train prognosability: 0.667217

## 2026-06-08 16:59:14 | training-run | configs/exp_tuning_mape_prog.yaml
- Epochs completed: 98
- Best val recon: 0.940703
- Checkpoint: checkpoints/exp_tuning_mape_prog/best_model.pt
- Train monotonicity: 0.408781
- Train trendability: 0.866894
- Train prognosability: 0.716992

## 2026-06-08 19:15:29 | training-run | configs/ds02_default.yaml
- Epochs completed: 94
- Best val recon: 0.898116
- Checkpoint: checkpoints/ds02_default/best_model.pt
- Train monotonicity: 0.544424
- Train trendability: 0.910155
- Train prognosability: 0.907932

## 2026-06-08 19:20:19 | training-run | configs/exp_tuning_mape_prog.yaml
- Epochs completed: 98
- Best val recon: 0.949053
- Checkpoint: checkpoints/exp_tuning_mape_prog/best_model.pt
- Train monotonicity: 0.511930
- Train trendability: 0.877569
- Train prognosability: 0.717638

## 2026-06-10 11:03:49 | training-run | configs/ds03_cvae_baseline.yaml
- Epochs completed: 99
- Best val recon: 0.936355
- Checkpoint: checkpoints/ds03_cvae_baseline/best_model.pt
- Train monotonicity: 0.349471
- Train trendability: 0.874502
- Train prognosability: 0.710655

## 2026-06-10 11:11:47 | training-run | configs/ds03_default.yaml
- Epochs completed: 86
- Best val recon: 0.932811
- Checkpoint: checkpoints/ds03_default/best_model.pt
- Train monotonicity: 0.430144
- Train trendability: 0.892060
- Train prognosability: 0.666290

## 2026-06-10 11:32:49 | training-run | configs/ds03_vae_baseline.yaml
- Epochs completed: 12
- Best val recon: 0.939890
- Checkpoint: checkpoints/ds03_vae_baseline/best_model.pt
- Train monotonicity: -0.001911
- Train trendability: 0.216524
- Train prognosability: 0.775265

## 2026-06-10 11:33:55 | training-run | configs/ds03_vae_baseline.yaml
- Epochs completed: 86
- Best val recon: 0.926920
- Checkpoint: checkpoints/ds03_vae_baseline/best_model.pt
- Train monotonicity: 0.238231
- Train trendability: 0.838297
- Train prognosability: 0.761022

## 2026-06-10 16:08:59 | training-run | configs/ds03_ablation_no_hsic.yaml
- Epochs completed: 80
- Best val recon: 0.933088
- Checkpoint: checkpoints/ds03_ablation_no_hsic/best_model.pt
- Train monotonicity: 0.347886
- Train trendability: 0.888520
- Train prognosability: 0.677401

## 2026-06-10 16:20:38 | training-run | configs/ds03_ablation_no_mono.yaml
- Epochs completed: 11
- Best val recon: 0.930429
- Checkpoint: checkpoints/ds03_ablation_no_mono/best_model.pt
- Train monotonicity: -0.023460
- Train trendability: 0.254736
- Train prognosability: 0.719007

## 2026-06-10 16:22:22 | training-run | configs/ds03_ablation_no_mono.yaml
- Epochs completed: 83
- Best val recon: 0.930379
- Checkpoint: checkpoints/ds03_ablation_no_mono/best_model.pt
- Train monotonicity: 0.278650
- Train trendability: 0.874381
- Train prognosability: 0.734160

## [2026-10-14] exp_tuning_mono_weight — HI protocol re-evaluation

**Config**: `configs/exp_tuning_mono_weight.yaml`
**Git tag**: N/A (evaluation only; no new tag)
**Dataset**: DS03
**Type**: Full Model (existing epoch-80 checkpoint; no training)

| Metric | Dev | Test |
|---|---:|---:|
| mae | 0.060226084 | 0.050272103 |
| mape | 20.259381420 | 25.235357429 |
| monotonicity | 0.152312941 | 0.159958350 |
| trendability | 0.940106608 | 0.927277841 |
| fc | 0.854523789 | 0.904810375 |
| mi | 0.727703291 | 0.737175284 |

**Notes**: Protocol `unsupervisedhi-ncmapss-v1`, reference commit `157ebfae7baeee1e0cd52180f8e58ead3dcb448e`; all-dev scaler (663 flights), MI seed 42. Raw scores unchanged bit-for-bit. Per-unit input normalization remains unchanged. FC replaces old prognosability. Differences from previous scores reflect evaluation changes, not model improvement. Output: `runs/exp_tuning_mono_weight/evaluation_unsupervisedhi_v1_20261014_verified`. Validation/limits: `docs/EVALUATION_VERIFICATION.md`.

## [2026-10-14] decoder W-summary single-factor paired ablation

**Config**: `configs/exp_tuning_mono_weight_w_summary_seed42.yaml` / `configs/exp_tuning_mono_weight_no_w_summary_seed42.yaml`
**Git tag**: N/A (local code commit only)
**Dataset**: DS03
**Type**: Full Model / Ablation (seed-42 paired initialization; 100 epochs each)

| Metric | Dev true | Dev false | Test true | Test false |
|---|---:|---:|---:|---:|
| mae | 0.076172616 | 0.066921154 | 0.066169863 | 0.058134065 |
| mape | 21.213460902 | 20.540014224 | 24.555696595 | 24.216003800 |
| monotonicity | 0.152646314 | 0.147190170 | 0.174392708 | 0.199358021 |
| trendability | 0.921813143 | 0.928074947 | 0.925684550 | 0.933264489 |
| fc | 0.831925500 | 0.828742253 | 0.908185653 | 0.909809383 |
| mi | 0.710474693 | 0.721305980 | 0.721354326 | 0.735425004 |

**Notes**: Historical checkpoint initialization cannot be verified, so two new runs were used; selected epochs true=82, false=84. Initialization and actual 100-epoch batch/RNG/LR traces match. False improves all six test metrics; dev monotonicity/FC slightly decrease. Curves mostly overlap, test average absolute difference=0.014187. The current pair does not support direct decoder W-summary as the necessary source of HI performance; no claim about all model advantages or physical disentanglement. Formal output/validation: `runs/w_summary_ablation_seed42_20261014/EXPERIMENT_REPORT.md`. No extra seeds, structures or test-driven changes; legacy checkpoint/results preserved.

## 2026-10-15 11:59:37 | training-run | config/ds03_default.yaml
- Epochs completed: 100
- Checkpoint: runs/ds03_default/best.pt
- Unified all-dev/test evaluation: runs/ds03_default

## 2026-10-15 12:05:56 | training-run | config/exp_tuning_mono_weight.yaml
- Epochs completed: 100
- Checkpoint: runs/exp_tuning_mono_weight/best.pt
- Unified all-dev/test evaluation: runs/exp_tuning_mono_weight

## 2026-10-15 14:27:09 | training-run | config/ds03_vae_baseline.yaml
- Epochs completed: 100
- Checkpoint: runs/ds03_vae_baseline/best.pt
- Unified all-dev/test evaluation: runs/ds03_vae_baseline

## 2026-10-19 15:22:54 | training-run | config/tuning/kgld_hsic_5.yaml
- Epochs completed: 100
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:25:37 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 97
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:28:18 | training-run | config/kgld.yaml
- Epochs completed: 100
- Checkpoint: runs/kgld/best.pt
- Unified all-dev/test evaluation: runs/kgld

## 2026-10-19 15:32:22 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 194
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:34:43 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 41
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:35:42 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 40
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:37:21 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:39:10 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 15:41:44 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 16:20:10 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 16:21:41 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 16:24:46 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning

## 2026-10-19 16:27:04 | training-run | config/tuning/kgld_tuning.yaml
- Epochs completed: 200
- Checkpoint: runs/kgld_tuning/best.pt
- Unified all-dev/test evaluation: runs/kgld_tuning
