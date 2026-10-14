# OC-iVAE 代码走读

本文面向已经理解 OC-iVAE 高层设计、但需要掌握当前代码实现细节的项目作者。所有说明以当前仓库源码为准；若设计文档与代码不一致，本文会明确标注“当前实现”与“设计意图”的差异。

## 1. 项目总览

目录职责如下：`data/` 负责从 N-CMAPSS h5 原始数组构造 cycle 级样本；`model/` 实现 OC-iVAE 的 encoder、latent branch、prior、decoder 和 HI head；`loss/` 实现重构、KL、HSIC、单调性与 Spearman 损失；`train/` 提供训练循环和 checkpoint；`eval/` 负责 HI 推理、后处理和指标；`scripts/` 是训练、评估和辅助绘图入口；`configs/` 存放 DS03、DS02 和调参实验 YAML；`tests/` 当前全为空占位；`docs/` 存放项目状态和审计文档。`scripts/smoke_test.py` 与 `train/scheduler.py` 目前也是空文件，KL annealing 实际写在 `loss/total.py:_compute_kl_weight()`。

完整数据流为：

```text
raw h5
  -> data/preprocess.py:load_h5()
  -> data/preprocess.py:group_by_unit_cycle()
  -> data/dataset.py:NCMAPSSDataset.__getitem__()
  -> data/sampler.py:UnitStratifiedSampler.__iter__()
  -> DataLoader batch: x, mask, u, W_seq, unit_id, cycle_id, hi
  -> model/oc_ivae.py:OC_iVAE.forward()
  -> loss/total.py:compute_total_loss()
  -> train/trainer.py:Trainer.train_epoch()
  -> eval/inference.py:extract_hi()
  -> eval/postprocess.py:postprocess_unit()
  -> eval/metrics.py:evaluate_all()
```

训练时 `hi` 不参与 loss；有 ground truth HI 时只用于评估 MAE 和绘图，无 ground truth 时用 NaN/None 贯通评估链路。

## 2. 数据层（data/）

`data/preprocess.py:load_h5()` 从 h5 中读取 `A/W/X_s/Y`，支持 `A_dev`、`dev/A`、`A` 三种 key 形式。`group_by_unit_cycle()` 以 `A` 的 unit、cycle、Fc 列分组，并且当前已经移除 `hs` 过滤，所以保留全生命周期数据。每个 cycle 先经 `downsample()` 按 `downsample_factor` 取步长采样，再存为 `{X_s, W, Fc, unit, cycle}`。

归一化分两条线：传感器 `x` 使用 `compute_unit_stats()` 按 unit 计算 mean/std，并将 std 小于 `_DEAD_SENSOR_STD=1e-3` 的 dead sensor 置零；操作条件 `W` 使用 `compute_W_stats()` 在所有 cycle 和 timestep 上计算全局 mean/std，并在 `process_cycle()` 与 `construct_u()` 中执行 z-score。需要注意：当前源码没有把 W 映射或裁剪到固定 `[-0.5, 1.0]` 区间；实际实现是 `(W - W_mean_global) / W_std_global`，数学上无固定上下界。如果论文中要写固定范围，需要先补代码或以实验统计值说明。

`process_cycle()` 同时生成 `mask`：若 cycle 长度大于等于 `L_max`，保留尾部 `[-L_max:]` 且 mask 全 True；若较短，则在头部补 0，真实数据放在尾部，mask 的尾部为 True。`construct_u()` 的输出是 11 维：`[W_mean(4), W_std(4), Fc_onehot(3)]`。

伪代码：

```python
arrays = load_h5(h5_path, split)
cycles = group_by_unit_cycle(A, W, X_s, factor)
W_mean, W_std = compute_W_stats(cycles)
x, W_seq, mask = process_cycle(cycle, L_max, unit_mean, unit_std, valid_mask, W_mean, W_std)
u = construct_u(cycle["W"], Fc, n_fc=3, W_mean_global=W_mean, W_std_global=W_std)
```

`data/dataset.py:NCMAPSSDataset.__init__()` 先读 h5 并分组，然后计算 `L_max=P90`。若 `cycle_hi_path is None`，则 `self.has_hi=False`、`self.hi_lookup={}`；否则读取 h5 key `df`，由 `_build_hi_lookup()` 建立 `(unit_id, cycle_id) -> HI` 的 `cycle_hi_map`。源码变量名是 `has_hi`，不是 `has_gt_hi`。当前 `_validate_hi_lookup()` 只在 `has_hi=True` 时检查所有暴露样本都有 HI；无 GT 的 DS02 可跳过。`__getitem__()` 返回 7 元组 `(x, mask, u, W_seq, unit_id, cycle_id, hi)`，其中 `hi` 无标签时是 `torch.tensor(float("nan"))`。

`data/sampler.py:UnitStratifiedSampler` 先按 unit 建立 `unit_to_indices`，每个 batch 抽 `k` 个 unit、每个 unit 抽 `m` 个 cycle。这样 batch 内有同 unit 的多个 cycle，才能构造单调性 pair 和 Spearman rank。DS03 常用 `k=7, m=12`，AGENTS 设计常量曾写 7/8 存在历史差异；DS02 当前配置 `k=6`，但 train split 可能只有 4 个 unit，所以 sampler 已用 `effective_k=min(k, n_units)` 兜底，避免 0 batch。

## 3. 模型层（model/）

`model/backbone.py:SharedTCNBackbone` 输入 `x(B,T,S)` 和 `mask(B,T)`，先用 kernel=1 的 causal conv 投影到 `C_HIDDEN=64`，再经过 dilation `[1,2,4,8,16,32]` 的 `TCNResBlock`。每个 block 采用 causal padding、MaskedLayerNorm、GELU、Dropout 和 residual，输出 `h(B,T,64)`。`h` 是传感器序列的共享时序表征，后续同时送给 z_d 与 z_c 分支。

`model/branches.py` 有三块：`MaskedAvgPool` 对 valid timestep 求平均；`ZdBranch` 是 TCNResBlock -> masked average -> Linear 输出 `mu_d/log_var_d`；`ZcBranch` 多一个 `FiLMGenerator(u)`，用 `u` 生成 `gamma/beta` 后调制 `h_branch`：`h_film = gamma[:,None,:] * h_branch + beta[:,None,:]`，再池化并输出 `mu_c/log_var_c`。所以 FiLM 当前在 z_c branch，而不是 decoder。

`model/prior.py:PriorNetwork` 用同一个 MLP stem 接收 `u(B,11)`，然后分成 `zd_head` 与 `zc_head` 生成 `mu_d_prior/log_var_d_prior/mu_c_prior/log_var_c_prior`。当前代码没有 `MonotonicMLP`，也没有 Softplus 单调权重约束；唯一非对称点是 `zd_head` 的 weight/bias 被初始化为 0，使 z_d prior 初始接近标准正态，而 z_c prior 默认初始化。

`model/decoder.py:TCNDecoder` 当前也没有 FiLM。它先从 `W_seq` 按 mask 计算 `W_summary=[masked_mean(4), masked_std(4)]`，与 `z_d,z_c` 拼接成 `d_zd+d_zc+8` 维向量，经 Linear 到 64 维，再沿时间维复制到 `(B,T,64)`，经过 dilation `[1,2,4,8]` 的非 causal residual TCN，最后 Linear 输出 `x_hat(B,T,14)`。这里 `_N_SENSOR_FEATURES=14` 是硬编码技术债。

`model/hi_head.py:HIHead` 很简单：`Linear(d_zd,1,bias=False)` 后 squeeze 成 `hi_raw(B,)`。`model/oc_ivae.py:OC_iVAE.forward()` 的顺序是 backbone -> zd_branch -> zc_branch -> prior -> reparameterize -> decoder -> hi_head，返回锁定 dict：`x_hat`、posterior/prior 参数、sampled `z_d/z_c` 和 `hi_raw`。`reparameterize()` 在训练态采样 `mu + eps * exp(0.5*log_var)`，eval 态直接返回 `mu`，保证推理确定性。

伪代码：

```python
h = backbone(x, mask)
mu_d, log_var_d = zd_branch(h, mask)
mu_c, log_var_c = zc_branch(h, mask, u)
prior_params = prior(u)
z_d, z_c = reparameterize(mu_d, log_var_d), reparameterize(mu_c, log_var_c)
x_hat = decoder(z_d, z_c, W_seq, mask)
hi_raw = hi_head(z_d)
```

## 4. z_d / z_c 解耦机制

### 机制一：架构解耦

共享 backbone 只负责提取 `h`；`model/branches.py:ZdBranch.forward()` 与 `ZcBranch.forward()` 是独立参数分支。`ZdBranch` 不接收 `u`，只能从传感器时序 `h` 中提取退化相关信息；`ZcBranch` 接收 `u` 并通过 `FiLMGenerator` 注入工况统计，从架构上鼓励 `z_c` 承载 operating condition。关键变量是 `h_branch`、`gamma`、`beta`、`mu/log_var`。

```python
h = SharedTCNBackbone(x, mask)
mu_d, log_var_d = ZdBranch(h, mask)          # no u
gamma, beta = FiLMGenerator(u)
h_c = gamma[:, None, :] * TCNResBlock(h) + beta[:, None, :]
mu_c, log_var_c = ZcHead(masked_avg(h_c))
```

### 机制二：Loss 解耦

`loss/total.py:compute_total_loss()` 同时计算 KL、HSIC、单调性和 Spearman。HSIC 在 `loss/hsic.py:hsic_loss()`，对 `z_d` 和 `z_c` 分别构造 RBF kernel，带宽用 pairwise squared distance 的 median heuristic，然后用中心化矩阵 `H=I-11^T/B` 得到 `trace(HKH HLH)/(B-1)^2`，惩罚 batch 内统计依赖。KL 在 `loss/kl.py:gaussian_kl_loss()`，公式为：

```text
KL = 0.5 * [ log σ_p^2 - log σ_q^2
             + (σ_q^2 + (μ_q-μ_p)^2) / σ_p^2 - 1 ]
```

当前不是“MonotonicMLP prior vs 标准高斯”的实现，而是 z_d、z_c 都对齐 `PriorNetwork(u)` 给出的条件 Gaussian；非对称性主要来自 `beta_d/beta_c` 权重和 z_d prior head 的零初始化。单调性 hinge 在 `loss/monotonicity.py:monotonicity_loss()`：同一 unit 且 `cycle_later-cycle_earlier >= delta_t` 时，惩罚 `relu(hi_later - hi_earlier + margin)`。

```python
kl_d = KL(q_d(mu_d, log_var_d), p_d(mu_d_prior, log_var_d_prior))
kl_c = KL(q_c(mu_c, log_var_c), p_c(mu_c_prior, log_var_c_prior))
hsic = HSIC_RBF(z_d, z_c)
mono = mean(relu(hi_later - hi_earlier + margin))
total = recon + beta_d*kl_d + beta_c*kl_c + lambda_hsic*hsic + ...
```

## 5. 从 z_d 到 HI 的完整路径

路径如下：`ZdBranch` 输出 `mu_d/log_var_d`；训练时 `OC_iVAE.reparameterize()` 采样 `z_d`，推理时直接使用 `mu_d`；`HIHead(z_d)` 线性投影得到 `hi_raw`。评估阶段 `eval/inference.py:extract_hi()` 按 unit 聚合 `cycle_ids/hi_raw/hi_gt`。随后 `eval/postprocess.py:normalize_hi()` 对每个 unit 做 min-max 到 `[0,1]`，再 `smooth_hi()` 做 EWMA：`s_t = alpha*h_t + (1-alpha)*s_{t-1}`。最后 `eval/metrics.py:evaluate_all()` 计算指标。

四个指标：Monotonicity 为相邻下降比例减上升比例，`(#diff<0)/(n-1) - (#diff>0)/(n-1)`；Trendability 为 `|corr(hi, time)|`；Prognosability 当前实现是先取每个 unit 的末端 `hi[-1]`，再聚合为 `exp(-std(EOL)/(mean(EOL)+eps))`；MAE 为 `mean(|hi_proc-hi_gt|)`，无 GT 时为 NaN，并在 `run_eval.py` 表格中显示 `N/A`。

## 6. 训练流程（train/）

`scripts/run_train.py:main()` 读取 YAML，根据配置文件名推导 `exp_name`，设置 `checkpoints/{exp_name}` 与 `runs/{exp_name}`。它构造 train/val dataset，自动从 `train_dataset[0]` 推断 `in_features`，用 `UnitStratifiedSampler` 给 train DataLoader 组 batch；val DataLoader 使用普通 batch size `k*m`。`train/trainer.py:Trainer` 使用 AdamW、CosineAnnealingLR、gradient clipping 和可选 early stopping。checkpoint 在 `val_recon` 改善时保存为 `best_model.pt`，`load_checkpoint()` 显式 `weights_only=False`，因为 checkpoint 包含 optimizer state。

三阶段训练在 `loss/total.py:compute_total_loss()`：epoch `< warmup_end_epoch` 只用 `recon`；`warmup_end_epoch <= epoch < transition_end_epoch` 加入线性 KL annealing 和 HSIC，其中 `kl_weight=(epoch-warmup)/(transition-warmup)`；之后 full phase 使用 `recon + beta_d*kl_d + beta_c*kl_c + lambda_hsic*hsic + lambda_mono*mono + lambda_spearman*spearman`。权重来自 YAML 的 `loss_weights`，阶段边界来自 `training`。

`loss_dict` 中：`recon` 是 masked MSE；`kl_d/kl_c` 是两个 latent 的 Gaussian KL；`hsic` 是解耦正则；`mono` 是 pairwise hinge；`spearman` 是 soft rank correlation loss；`total` 是用于 backward 的 Tensor。

## 7. 评估与推理（eval/）

`eval/inference.py:extract_hi()` 调用 `model.eval()` 并用 `torch.no_grad()` 关闭 dropout 与梯度；DataLoader batch size 固定 64。每个 batch 取 `out["hi_raw"]`，再按 unit 收集并按 cycle 排序。若 batch 中的 GT HI 全是 NaN，最终 `hi_gt=None`。

`eval/postprocess.py` 的 `alpha` 来自 `postprocess.ewma_alpha` 或 CLI `--alpha`。min-max 是 per-unit 的：`(hi-min)/(max-min+eps)`，不是全局归一化；这会让每个 unit 的曲线都拉到 `[0,1]`，但也可能影响跨 unit 的 Prognosability。

`scripts/run_eval.py` 自动解析 DS03 sibling `*cycle_dev.h5/*cycle_test.h5`；如果配置显式 `cycle_hi_path: null`，则按无 GT 评估。绘图时有 GT 才画蓝色 dashed `HI gt`，否则只画红色 `HI pred`；MAE 为 NaN 时显示 `N/A`。

## 8. 配置系统

YAML 结构固定为六段。`data`：`h5_path` 主 h5；`cycle_hi_path` 可选，`null` 表示无 GT；`downsample_factor` 降采样步长；`sensor_threshold` 当前未接入代码，实际使用 `_DEAD_SENSOR_STD=1e-3`；`n_fc` 飞行条件类别数；`val_split` unit-level 验证比例。`model`：`c_hidden`、`n_backbone_layers`、`n_decoder_layers`、`d_zd`、`d_zc`、`u_dim`、`decoder_use_W_summary`。`training`：`k_units_per_batch`、`m_cycles_per_unit`、`optimizer`、`lr`、`weight_decay`、`total_epochs`、`gradient_clip_norm`、`use_early_stopping`、`early_stopping_patience`、`warmup_end_epoch`、`transition_end_epoch`。`loss_weights`：`beta_d/beta_c/lambda_hsic/lambda_mono/lambda_spearman/ranking_margin/ranking_delta_t/spearman_tau`。`postprocess`：`ewma_alpha`。`logging`：`log_dir/checkpoint_dir/log_every_n_epochs`。

`scripts/run_train.py` 保留 nested config，Trainer 内部只 flatten `training/loss_weights/logging`；模型配置由 `model` section 加上推断的 `in_features/w_summary_dim`。`scripts/run_eval.py` 则用 `_merge_config_sections()` 把所有 mapping section 合并成一个 flat config，并在构造模型前同样推断 `in_features`。

## 9. 已知问题与待办

由于本任务范围要求只新增本文档，未修改 `docs/PROJECT_STATUS.md`；以下是建议写入其 Open Tasks 的当前事项：

| 状态 | 问题 | 影响范围 | 优先级 |
| --- | --- | --- | --- |
| 未开始 | `tests/` 全为空，`scripts/smoke_test.py` 为空 | 所有模块缺少回归保护，pytest 收集 0 tests | 高 |
| 未开始 | `model/decoder.py` 输出传感器维度硬编码 14 | DS02/DS03 以外或传感器维度变化时会失败 | 高 |
| 未开始 | `train/scheduler.py` 为空，KL schedule 实际在 `loss/total.py` | 文档与实现不一致，后续维护易混淆 | 中 |
| 部分完成 | 支持无 GT HI：dataset/eval 已支持，`scripts/plot_hi.py` 仍假设有 GT | DS02 推荐使用 `run_eval.py`，不要用旧 plot 脚本 | 中 |
| 未开始 | `sensor_threshold` 配置未接入 `compute_unit_stats()` | YAML 参数对 dead sensor 阈值无效 | 中 |
| 未开始 | 当前没有 MonotonicMLP/Softplus prior，也没有 decoder FiLM | 若论文主张这些机制，需要补实现或改表述 | 高 |
| 未开始 | train/val normalization stats 仍基于完整 dev cycles 计算 | 严格实验设置下可能有验证泄漏争议 | 中 |
| 未开始 | per-unit min-max 后处理可能削弱跨 unit Prognosability 解释 | 指标与论文解释 | 中 |

答辩时建议把当前实现表述为：“解耦主要来自 branch 架构、z_c 的 u-FiLM 引导、HSIC 正则、KL 条件先验和 HI 单调约束共同作用”；不要声称已实现 MonotonicMLP、Softplus 单调 prior 或 decoder FiLM，除非后续补代码。
