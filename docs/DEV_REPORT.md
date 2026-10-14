# OC-iVAE 开发审计报告

### 1. 项目结构总览

```text
oc_ivae/
├── AGENTS.md                    # 项目级 Codex 指令、架构常量、接口锁定规则和权限约束。
├── configs/
│   ├── ds03_default.yaml         # 默认 DS03 配置，包含 data/model/training/loss/postprocess/logging。
│   ├── exp_tuning_mono_weight.yaml # 单调性权重调参配置。
│   └── exp_tuning_mape_prog.yaml # Prognosability/MAE 调参配置。
├── data/
│   ├── __init__.py               # 空包初始化文件。
│   ├── preprocess.py             # H5 读取、降采样、按 unit/cycle 分组、归一化、padding/truncation、u 构造。
│   ├── dataset.py                # NCMAPSSDataset，生成模型 batch 所需张量和 cycle-level HI。
│   └── sampler.py                # UnitStratifiedSampler，按 K 个 unit、每个 M 个 cycle 采样 batch。
├── model/
│   ├── __init__.py               # 空包初始化文件。
│   ├── constants.py              # 模型和训练常量。
│   ├── backbone.py               # 共享 causal TCN backbone、MaskedLayerNorm、TCNResBlock。
│   ├── branches.py               # z_d/z_c posterior 分支，z_c 分支使用 FiLM。
│   ├── prior.py                  # 条件 Gaussian prior p(z_d,z_c|u)。
│   ├── decoder.py                # 非 causal TCN decoder，从 z_d/z_c/W_summary 重构传感器序列。
│   ├── hi_head.py                # z_d 到标量 HI 的无 bias 线性头。
│   └── oc_ivae.py                # 顶层模型组装和锁定 forward 输出 dict。
├── loss/
│   ├── __init__.py               # 空包初始化文件。
│   ├── recon.py                  # masked MSE 重构损失。
│   ├── kl.py                     # posterior 与 conditional prior 的 Gaussian KL。
│   ├── hsic.py                   # RBF-kernel mini-batch HSIC 独立性损失。
│   ├── monotonicity.py           # 同 unit 内 HI 随 cycle 下降的 pairwise ranking loss。
│   ├── spearman.py               # soft Spearman rank loss。
│   └── total.py                  # warmup/transition/full 三阶段总损失聚合。
├── train/
│   ├── __init__.py               # 空包初始化文件。
│   ├── trainer.py                # 训练循环、验证、checkpoint、LR scheduler。
│   └── scheduler.py              # 当前为空文件，未实现 KL scheduler。
├── eval/
│   ├── __init__.py               # 空包初始化文件。
│   ├── inference.py              # eval 模式批量提取 raw HI。
│   ├── postprocess.py            # per-unit min-max normalization 与 EWMA smoothing。
│   └── metrics.py                # Monotonicity、Trendability、Prognosability、MAE。
├── scripts/
│   ├── run_train.py              # 训练入口：读配置、建 dataset/loader/model/trainer、记录实验。
│   ├── run_eval.py               # 独立评估入口：加载 checkpoint、评估并画图。
│   ├── plot_hi.py                # 旧版/辅助 HI 曲线绘图脚本。
│   └── smoke_test.py             # 当前为空文件。
├── tests/
│   ├── __init__.py               # 空测试包初始化文件。
│   ├── test_preprocess.py        # 空占位。
│   ├── test_dataset.py           # 空占位。
│   ├── test_backbone.py          # 空占位。
│   ├── test_branches.py          # 空占位。
│   ├── test_model_forward.py     # 空占位。
│   ├── test_losses.py            # 空占位。
│   └── test_metrics.py           # 空占位。
└── test.py                       # 顶层临时评估脚本，含硬编码配置和 checkpoint 路径。
```

### 2. 模块接口关系

数据流为：`Dataset -> Sampler -> DataLoader -> OC_iVAE.forward() -> compute_total_loss()`。

- `NCMAPSSDataset.__getitem__()` 输出单个样本：`x (L_max, S)`、`mask (L_max)`、`u (11,)`、`W_seq (L_max, 4)`、`unit_id scalar int64`、`cycle_id scalar int64`、`hi scalar float32`。位置：`data/dataset.py:122`。
- `UnitStratifiedSampler.__iter__()` 输出一个 batch 的样本索引列表，长度为 `k * m`；当前训练配置为 `7 * 12 = 84`。位置：`data/sampler.py:51`、`configs/ds03_default.yaml:20`。
- `DataLoader` 自动 collate 后传递 batch：`x (B, L_max, S)`、`mask (B, L_max)`、`u (B, 11)`、`W_seq (B, L_max, 4)`、`unit_id (B,)`、`cycle_id (B,)`、`hi (B,)`。训练中 `hi` 被丢弃。位置：`train/trainer.py:79`。
- `OC_iVAE.forward(x, mask, u, W_seq)` 返回锁定 dict：`x_hat (B,L_max,14)`、`mu_d/log_var_d/z_d (B,2)`、`mu_c/log_var_c/z_c (B,8)`、prior 参数和 `hi_raw (B,)`。位置：`model/oc_ivae.py:46`。
- `compute_total_loss(out, x, mask, cycle_id, unit_id, epoch, config)` 使用 `x_hat/x/mask` 算重构，用 posterior/prior 参数算 KL，用 `z_d/z_c` 算 HSIC，用 `hi_raw/unit_id/cycle_id` 算单调和 Spearman；不使用 `hi` 监督标签。位置：`loss/total.py:12`。
- 反向传播链路在 trainer 内完成：`total_loss.backward()`、gradient clipping、`optimizer.step()`。位置：`train/trainer.py:86`。

### 3. 关键设计决策记录

- 决策：保留全生命周期数据，已移除 `hs == healthy` 过滤。
  - 位置：`data/preprocess.py:90`、`data/preprocess.py:94`
  - 原因：HI 曲线需要覆盖从健康到退化的完整生命周期，否则只能看到局部状态，无法学习 1 到 0 的下降趋势。

- 决策：`W_seq` 与 `u` 使用同一组全局 `W_mean/W_std` 归一化。
  - 位置：`data/preprocess.py:193`、`data/preprocess.py:245`、`data/preprocess.py:309`、`data/dataset.py:93`
  - 原因：操作条件变量同时进入 z_c branch、prior 和 decoder，统一尺度可降低原始物理量量纲差异对优化的影响。注意当前 stats 在 dataset 初始化时基于 `self.cycles` 计算，而 `self.cycles` 早于 train/val split。

- 决策：传感器 `x` 使用 per-unit z-score，并屏蔽 dead sensors。
  - 位置：`data/preprocess.py:162`、`data/preprocess.py:186`、`data/preprocess.py:259`
  - 原因：不同发动机 unit 的传感器 baseline 不同，按 unit 标准化可突出 unit 内退化趋势；低方差 dead sensor 不参与有效特征归一化。

- 决策：当前实验采用 `k=7` 个 unit per batch。
  - 位置：`model/constants.py:22`、`configs/ds03_default.yaml:21`、`scripts/run_train.py:80`
  - 原因：源码中没有直接注释说明科学原因；从 `UnitStratifiedSampler` 行为推断，`7 * 12 = 84` 的 batch 规模能在同 unit 内形成 ranking/Spearman 对，同时比 AGENTS.md 中的 `8 * 12 = 96` 更省显存。此处与 `AGENTS.md:98` 的 `K_UNITS_PER_BATCH = 8` 不一致。

- 决策：短 cycle 做 head padding，长 cycle 做 tail truncate。
  - 位置：`data/preprocess.py:385`、`data/preprocess.py:396`、`data/preprocess.py:402`
  - 原因：短序列前端补零让有效时间步靠后对齐；长序列保留尾部，因为尾部通常包含更多退化信号。

- 决策：eval 模式下 `reparameterize()` 返回 `mu`，不采样。
  - 位置：`model/oc_ivae.py:38`、`eval/inference.py:31`
  - 原因：推理和 HI 提取需要确定性输出，避免随机采样导致 HI 曲线和指标不可复现。

- 决策：FiLM 只用于 `ZcBranch`，且应用在 branch 的 `TCNResBlock` 之后。
  - 位置：`model/branches.py:101`、`model/branches.py:111`、`model/branches.py:113`
  - 原因：先由 TCN branch 抽取 temporal representation，再用 `u` 生成 `gamma/beta` 调制 z_c 表征；z_d branch 不使用 FiLM，以减少 degradation latent 被 operating condition 直接注入。

- 决策：`z_d` 条件 prior head 以 0 初始化，`z_c` prior head 保持默认初始化。
  - 位置：`model/prior.py:35`、`model/prior.py:37`
  - 原因：让 z_d prior 初始接近标准正态，降低 degradation latent 在训练早期被 auxiliary variable 过强解释的风险；z_c prior 保留对 operating condition 的表达能力。

- 决策：decoder 是非 causal TCN，并拼接 `W_summary = [W_mean, W_std]`。
  - 位置：`model/decoder.py:83`、`model/decoder.py:107`、`model/decoder.py:136`
  - 原因：重构任务不要求在线因果性，可使用双向上下文；W summary 给 decoder 明确操作条件上下文，减少把 operating condition 压入 z_d 的压力。

- 决策：总损失分三阶段启用。
  - 位置：`loss/total.py:74`、`loss/total.py:76`、`loss/total.py:87`
  - 原因：先重构稳定 encoder/decoder，再逐步加入 KL/HSIC，最后加入 HI 单调性和 rank 约束，降低训练初期多目标冲突。

- 决策：HI 由 `z_d` 经无 bias 线性层得到。
  - 位置：`model/hi_head.py:20`、`model/oc_ivae.py:66`
  - 原因：HI 只能依赖 degradation latent，且无 bias 线性投影更接近“从 z_d 方向读出健康度”的设计。

### 4. 已知问题和技术债

近期已修复：

- 修复：`Trainer.load_checkpoint()` 已显式设置 `weights_only=False`。
  - 位置：`train/trainer.py:191`
  - 说明：checkpoint dict 包含 optimizer state，不是纯模型权重；显式传参用于消除 PyTorch FutureWarning，并保留 optimizer restore 逻辑。

- 修复：`run_eval.py` 已移除 `in_features=14` 与 DS03 cycle-HI 文件名硬编码。
  - 位置：`scripts/run_eval.py:60`、`scripts/run_eval.py:75`、`scripts/run_eval.py:181`
  - 说明：`in_features` 现在从 `train_dataset[0][0].shape[-1]` 推断；cycle-HI 路径优先读 config key，否则从主 h5 同目录解析 `*cycle_dev.h5` / `*cycle_test.h5`。`python scripts/run_eval.py --config configs/exp_tuning_mono_weight.yaml --split test` 已成功运行。

剩余问题和技术债：

- 问题：`train/scheduler.py` 是 0 行空文件，且未被任何模块 import；KL annealing 实现在 `loss/total.py` 私有函数内，trainer 使用的是 PyTorch `CosineAnnealingLR`。
  - 位置：`train/scheduler.py`、`loss/total.py:_compute_kl_weight`、`train/trainer.py:52`
  - 严重程度：中
  - 建议修复方式：要么把 KL schedule 正式迁入 `train/scheduler.py` 并在 loss/trainer 使用，要么删除文档中对该文件职责的承诺。

- 问题：decoder 输出传感器维度硬编码为 14，未绑定 `config["in_features"]`。
  - 位置：`model/decoder.py:22`、`model/decoder.py:118`
  - 严重程度：高
  - 建议修复方式：增加 decoder 输出维度配置，默认使用 `in_features`；保持 forward 返回字段不变。

- 问题：`AGENTS.md` 锁定 `K_UNITS_PER_BATCH = 8`，但 `model/constants.py` 和所有 YAML 配置使用 7。
  - 位置：`AGENTS.md:98`、`model/constants.py:22`、`configs/ds03_default.yaml:21`
  - 严重程度：中
  - 建议修复方式：在明确用户指令下统一文档与代码；当前不应擅自修改架构常量。

- 问题：`UnitStratifiedSampler` 构造函数默认 `k=8`，与当前 constants/config 的 `k=7` 不一致。
  - 位置：`data/sampler.py:19`
  - 严重程度：低
  - 建议修复方式：默认值从 `model.constants.K_UNITS_PER_BATCH` 读取，或移除默认值并强制由配置传入。

- 问题：`sensor_threshold` 在 YAML 中配置，但 dead sensor 阈值实际使用硬编码 `_DEAD_SENSOR_STD = 1e-3`。
  - 位置：`configs/ds03_default.yaml:7`、`data/preprocess.py:12`、`data/preprocess.py:188`
  - 严重程度：中
  - 建议修复方式：把阈值作为 config 参数传入 `compute_unit_stats()` 或预处理入口。

- 问题：train/val normalization stats 在 `NCMAPSSDataset` 初始化时从全量 dev `self.cycles` 计算，而不是仅从 train split 计算；validation dataset 也未接收 `train_stats`。
  - 位置：`data/dataset.py:80`、`data/dataset.py:93`、`scripts/run_train.py:61`、`scripts/run_train.py:68`
  - 严重程度：高
  - 建议修复方式：先 split unit，再只用 train units 计算 train stats，并显式传给 val/test dataset。

- 问题：test split 中如果某个 unit 不在 `train_stats`，`__getitem__()` 会回退到用该 test unit 自身 cycle 计算 sensor stats，可能引入评估泄漏。
  - 位置：`data/dataset.py:137`
  - 严重程度：中
  - 建议修复方式：定义 test normalization 策略；若必须 per-unit 标准化，应在报告中说明这是无监督全生命周期后处理假设。

- 问题：AGENTS.md 的 Dataset 返回接口少写了 `hi`，实际返回 7 个字段。
  - 位置：`AGENTS.md:48`、`data/dataset.py:165`
  - 严重程度：低
  - 建议修复方式：在获得修改 AGENTS.md 的明确任务后同步文档；训练中 `hi` 当前只用于 eval，不用于 loss。

- 问题：`trainer.py` 使用 `print()` 输出日志，不符合 AGENTS.md “trainer only logging”的约束。
  - 位置：`train/trainer.py:153`、`AGENTS.md:157`
  - 严重程度：低
  - 建议修复方式：改为 `logging`，并保留相同的 epoch 指标内容。

- 问题：无有效 pair/unit 时，`monotonicity_loss()` 和 `soft_spearman_loss()` 返回新的 leaf Tensor，未与输入 `hi` 建立计算图。
  - 位置：`loss/monotonicity.py:32`、`loss/spearman.py:42`
  - 严重程度：低
  - 建议修复方式：返回 `hi.sum() * 0.0`，保持 device/dtype 和计算图一致。

- 问题：`scripts/plot_hi.py` 和顶层 `test.py` 含多处硬编码路径、维度和绘图布局，不应作为 canonical 训练/评估入口。
  - 位置：`scripts/plot_hi.py:25`、`scripts/plot_hi.py:52`、`scripts/plot_hi.py:58`、`test.py:7`、`test.py:11`
  - 严重程度：中
  - 建议修复方式：保留为临时脚本时加注释；生产评估统一走已修复路径解析的 `scripts/run_eval.py`。

- 问题：`scripts/smoke_test.py` 是空文件，但 AGENTS.md 把它列为 shape smoke test。
  - 位置：`scripts/smoke_test.py`、`AGENTS.md:35`
  - 严重程度：中
  - 建议修复方式：实现最小 forward shape smoke test，覆盖 Dataset-free synthetic batch。

- 问题：TODO/FIXME 搜索无命中。
  - 位置：全项目 `rg -n "TODO|FIXME|HACK|XXX"` 无结果
  - 严重程度：低
  - 建议修复方式：无需修复；后续技术债应显式写入 issue/report，而不是散落注释。

### 5. 测试覆盖现状

当前 `tests/` 下所有测试文件均为 0 行，占位未实现；没有已实现的 pytest 断言。

| 测试文件 | 状态 | 应覆盖内容 |
| --- | --- | --- |
| `tests/test_preprocess.py` | 空文件 | `hs` 不过滤、W 全局归一化、head pad/tail truncate、u 构造、dead sensor mask。 |
| `tests/test_dataset.py` | 空文件 | split、train_stats/L_max 传递、返回 7 字段及 shape/dtype。 |
| `tests/test_backbone.py` | 空文件 | causal conv shape、mask 后 padding 为 0、dilation 层数。 |
| `tests/test_branches.py` | 空文件 | `ZdBranch` 无 FiLM、`ZcBranch` FiLM after TCNResBlock、输出维度。 |
| `tests/test_model_forward.py` | 空文件 | locked forward dict keys、shape、eval reparameterize deterministic。 |
| `tests/test_losses.py` | 空文件 | recon/kl/hsic/mono/spearman 数值和无有效 pair 边界。 |
| `tests/test_metrics.py` | 空文件 | per-unit postprocess、四项指标、空序列/单点序列边界。 |

建议优先补充：

1. `test_model_forward.py`：用 synthetic batch 验证 forward locked keys 和所有 shape，这是模型接口风险最高处。
2. `test_preprocess.py`：覆盖 full lifecycle、padding/truncation、W/u 同源归一化，防止数据约定回退。
3. `test_dataset.py`：验证 train/val/test stats 使用策略，当前存在潜在泄漏。
4. `test_losses.py`：验证三阶段 loss 开关和 KL annealing，确保 scheduler/total loss 修改不会改变训练目标。
5. `test_metrics.py`：固定评估指标定义，避免 postprocess 或 Prognosability 公式被误改。
