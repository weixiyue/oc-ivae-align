# HI evaluation protocol verification — 2026-10-14

Scope: evaluation, plotting entry points, the training-end evaluation call, and evaluation tests only. No changes to model, losses, training algorithm, datasets/input processing, or canonical configs. No training, checkpoint search, package installation, README edits, or push. Initial Git status was clean. Original source revision: `26a4b5f995350ad05fa9d8e9ec90e9132e56be34`.

## Reference and protocol

Protocol: `unsupervisedhi-ncmapss-v1`. Fixed [UnsupervisedHI commit](https://github.com/KBaja/UnsupervisedHI/tree/157ebfae7baeee1e0cd52180f8e58ead3dcb448e). Checked `CMAPSS/eval_utils.py` metric functions and `CMAPSS/03_train_evaluate_proposals.ipynb`: CNN exposes the linear latent layer `Z`; proposed-model evaluation fits a global sklearn MinMaxScaler on development predictions and reuses transform on test, without sigmoid, per-unit min-max, or cycle EWMA. Current model already produces deterministic posterior-mean linear scores in eval mode, so inference/model code was preserved.

- MAE/MAPE: all cycles equally weighted; MAPE excludes zero reference HI and records counts.
- Monotonicity: each unit's absolute nonpositive-versus-positive difference imbalance; zero differences count as nonpositive; units equally weighted.
- Trendability: signed Spearman against actual cycles, NaN -> 0; absolute value after equal-unit mean. Per-unit correlations remain signed.
- FC: exp(-population std(last HI) / mean(first HI - last HI)); only cross-unit FC is saved. Nonpositive denominator is explicitly flagged. The old prognosability formula is not reused.
- MI: per-unit sklearn mutual_info_regression(HI[:,None], RUL), followed by 1-exp(-MI), then equal-unit average. All estimator defaults retained; seed 42 is an explicit reproducibility supplement.
- Nonfinite formulas become JSON null with diagnostic reasons. Undefined unit values are not silently dropped from unit averages. Constant outputs are flagged even when monotonicity is 1. No manual epsilon, clipping, smoothing, or label-based calibration.

All 663 dev flights fit one scaler (range 3.250363290310); test only transforms. Fit keys, sample count, parameters, sklearn version and constant-input behavior are saved separately in scaler.json. sklearn's constant case maps equal dev scores to zero with unit scale, and transforms test by subtracting the dev constant. Synthetic tests verify this behavior. No window aggregation is used here; the explicit optional window interface averages only the same unit/cycle before scaling.

Original unit split is unchanged: train [1, 2, 3, 4, 5, 7, 8], validation [6, 9]. Evaluation is explicitly **all dev**, the union of these groups, never labeled train. Inputs still use per-unit sensor standardization; input preprocessing is **not** aligned with reference preprocessing.

## Re-evaluation

Checkpoint: `/home/xiyue/CodeSpace/oc-ivae-align/checkpoints/exp_tuning_mono_weight/best_model.pt`, epoch 80, SHA256 `f0092274e8238b220589d1bc3e21e5415df4a33aebca461dba00593c6e208f34`. No checkpoint writes. Config: `configs/exp_tuning_mono_weight.yaml`.

| Metric | Dev (663 cycles, 9 units) | Test (438 cycles, 6 units) |
|---|---:|---:|
| mae | 0.060226084 | 0.050272103 |
| mape | 20.259381420 | 25.235357429 |
| monotonicity | 0.152312941 | 0.159958350 |
| trendability | 0.940106608 | 0.927277841 |
| fc | 0.854523789 | 0.904810375 |
| mi | 0.727703291 | 0.737175284 |

MAPE valid counts: dev 654, test 432. FC denominators: dev 0.710261824, test 0.746763820, both positive. Neither split has constant model output or nonfinite aggregate metrics. Test HI range: [0.08012719566533055, 1.018438185531528]; values above 1 are exported and plotted without clipping.

For comparison, recomputed old formulas on the **same full dev/test flights and identical raw scores**, with per-unit min-max + EWMA(alpha=0.3):

| Old metric | All dev | Test |
|---|---:|---:|
| mae | 0.044277478 | 0.046162069 |
| monotonicity | 0.457075864 | 0.515049338 |
| trendability | 0.883034527 | 0.891161144 |
| prognosability | 0.690480769 | 0.747908056 |

Old monotonicity was signed and did not count zero differences; old trendability was per-unit absolute Pearson using array index; old MAE averaged unit MAEs; old prognosability divided endpoint std by endpoint mean plus epsilon. The new FC has a different denominator and is not directly comparable to old prognosability. Score changes between protocols are changes of evaluation definition/postprocessing, **not model training improvement**. Existing legacy results are preserved.

## Commands and validation

```bash
python -m pytest tests/ -v --basetemp=tests/.eval_tmp/pytest_last2
python -m py_compile eval/metrics.py eval/protocol.py eval/postprocess.py scripts/run_eval.py scripts/run_train.py scripts/plot_hi.py tests/test_metrics.py
python scripts/run_eval.py --config configs/exp_tuning_mono_weight.yaml --device cpu --output-dir runs/exp_tuning_mono_weight/evaluation_unsupervisedhi_v1_20261014_verified
PYTHONPATH=. python tests/.eval_tmp/verify_re_evaluation.py
git diff --check
```

10 evaluation tests pass. Existing other test modules are empty placeholders. Tests cover hand calculations, unequal unit lengths, zero differences, signed trend cancellation, zero MAPE references, constant/short sequences, nonpositive FC denominators, keyed association/duplicates/missing/nonfinite labels, raw-flight RUL constancy, global dev-only calibration, no EWMA/per-unit scaling, optional same-flight window means, deterministic posterior scores, wrapped/bare old checkpoints, complete exports, and refusal to overwrite an output directory.

Before modifying code, old extract_hi computed and temporarily saved all development/test raw scores with the specified checkpoint on CPU. After modification, CSV loaded with float_precision=round_trip matches **all 1101 scores bit-for-bit**, maximum difference 0. The one-off verification script loaded the old metric source via git show for the comparison above, and extracted only the six fixed-reference metric functions, adding random_state=42 to reference MI. The six new results match the reference functions within 1e-14 on both real splits. The reference training utility has an upstream syntax typo in its proposed training function; no TensorFlow training utilities were executed. A broad initial compile encountered that temporary reference source, while explicit compilation of all changed Python files passes.

An initial CSV formatting check found that float32 columns print rounded decimals. Scores are now converted losslessly to float64 for CSV serialization. Floating-point cancellation assertions use numeric tolerance without changing metric formulas.

Validation summaries are retained in `runs/exp_tuning_mono_weight/evaluation_unsupervisedhi_v1_20261014_verified/validation.json`; predictions, metrics, scaler, curves and actual resolved evaluation configuration are in the same directory. Temporary verification script/scores/downloads/test directories, initial provisional evaluation output and generated plotting cache are removed after verification; permanent regression tests and final evaluation artifacts are retained. Checkpoint/data/outputs are excluded from the local commit.

## Limits

Reference cycle-HI files have no explicit cycle column. Their unique increasing source index, contiguous per-unit blocks, counts, raw cycle sequence starting at 1, and key coverage were checked before deriving cycle ids from per-unit row order. Semantic row order cannot be independently established from these files; provenance is recorded. Explicit cycle columns use direct key joins.

No GPU run, new training, alternate checkpoint, or real DS02 evaluation was performed. The training-end call is adapted but a real train-to-evaluation run was deliberately not executed. Missing reference HI and both old checkpoint formats are tested with lightweight entry-point fixtures. No claim of complete reference input-preprocessing alignment is made.
