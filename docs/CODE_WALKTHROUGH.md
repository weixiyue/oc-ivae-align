# 当前代码走读

更新：2026-10-14。本文描述简化重构后的实现，旧走读保留在
[history/CODE_WALKTHROUGH_pre_refactor.md](history/CODE_WALKTHROUGH_pre_refactor.md)。
入口与环境准备见 [README](../README.md)。

## 阅读顺序与职责

| 文件 | 主要职责 |
| --- | --- |
| dataset.py | NCMAPSSDataset、unit split、HI lookup、UnitStratifiedSampler |
| utils/preprocess.py | HDF5 读取、cycle 分组、统计、归一化、padding/truncation、u |
| models/tcn.py | causal encoder、masked LayerNorm、non-causal decoder、W summary |
| models/oc_ivae.py | OC-iVAE 装配、两个 posterior、FiLM、conditional prior、HI head |
| models/vae.py | 标准正态先验与两个 baseline 的组件替换 |
| models/__init__.py | build_model 的三个显式条件分支 |
| utils/losses.py | 单项数学损失、三阶段聚合及 KL annealing |
| train.py | 训练/验证、AdamW/LR、best selection、checkpoint、history/审计 |
| evaluate.py | 推理、flight 关联、dev-only scaler、评估和导出 |
| utils/metrics.py | 六项指标、聚合方式、非有限值诊断 |
| experiments/ | 配对初始化、采样记录、调度、比较与现有产物核验 |

## 数据链路

```text
A/W/X_s/Y HDF5
  → unit/cycle 分组与逐 cycle 下采样
  → L_max、传感器/W 统计
  → (x, mask, u, W_seq, unit_id, cycle_id, hi)
  → Sampler + DataLoader
  → model.forward()
```

实际按完整生命周期处理，不过滤 hs。L_max 是下采样后 cycle 长度 P90 的 ceil。
传感器逐 unit 标准化，dead sensor 保留列但置零，decoder 仍固定重构 14 个传感器。
短 cycle 在前端补零，长 cycle 去前端保留尾部；mask=True 表示有效时刻。

W 的全局统计和 L_max 沿用原实现的统计范围，不改成只计算 selected train units。
评估 test 复用 dev 的 W 统计和 L_max，未见 unit 的传感器统计按 test cycles 单独计算。

u 是完整下采样 cycle 的 W mean/std 与 Fc one-hot；
W_seq 经同源归一化、padding/truncation 后送入 decoder。
decoder 根据有效 mask 再计算 mean/std，不能把它改为直接取 u 前八维。
HI lookup 在 Dataset 中沿用 per-unit 行序；评估另做严格 flight-key 关联。

Sampler 用 random_seed + epoch_counter 的局部 NumPy RNG。
有效 K 为 min(K, unit 数)，不足 M 个 cycle 时有放回采样；
unit 数大于 K 时只产生完整 K-unit 组，末尾不足 K 的余数沿用原丢弃语义。

## 模型链路

```text
x/mask → SharedTCNBackbone
        ├→ ZdBranch → mu_d/log_var_d
        └→ ZcBranch + FiLM(u) → mu_c/log_var_c
u → PriorNetwork → 两个 latent 的条件 prior
posterior → train 时重参数采样 / eval 时直接取 mean
z_d/z_c + masked W summary → TCNDecoder → x_hat
z_d → 无 bias 线性 HIHead → hi_raw
```

forward 保留十二个字段：x_hat；两个 posterior 的 mu/log_var；
两个 prior 的 mu/log_var；z_d、z_c、hi_raw。
模块属性和参数注册顺序保持，旧 state_dict 无需迁移。

cVAE 仅替换标准正态 prior；VAE 再替换无 FiLM 的 z_c 分支。
两者先构造 OC-iVAE 再替换组件，保持初始化/RNG 消耗；没有另写训练循环。
关闭 decoder W-summary 只将八个输入置零，投影形状不变。

## 训练链路

train_model() 创建新 run 目录并写实际配置和初始化出处。
普通入口不固定 seed，配对脚本负责 seed/shared state/RNG 恢复。

每轮按下列顺序执行：

1. train_epoch() 执行 zero_grad → forward → 零起始 epoch 的 loss → backward → clip → step；检查非有限值。
2. validate() 在 eval/no_grad 下计算 reconstruction，训练与验证均按 batch 数平均。
3. 记录 scheduler.step 前的 LR 和 history，可选记录实际 batch/RNG 审计。
4. scheduler.step()；以严格更小的 val_recon 保存 best.pt，之后判断 early stopping。

compute_total_loss() 始终计算各分量，阶段只控制总目标的组合：

- warmup：reconstruction。
- transition：reconstruction + 线性 anneal 的 KL + HSIC。
- 最终阶段：reconstruction + KL + HSIC + ranking + soft Spearman。

不使用 hi 标签监督，不根据 test 选择模型。各阶段边界、reduction 和 zero-weight 行为未改。
checkpoint 保存模型/optimizer、model_config、W 开关、run_metadata、epoch 和 best loss。
load_checkpoint() 只恢复原有状态，不完整恢复 scheduler、sampler 或 RNG。

## 评估链路

predict_cycles() 使用 batch size 64、posterior mean，直接生成 float64 unit/cycle/score 表。
read_rul()/read_reference()/associate() 检查标签、真实 cycle 键和一对一覆盖；
没有 window 平均分支，重复 flight 明确报错。

scale_dev_test() 只在全 dev score 上拟合 sklearn MinMaxScaler，transform 两个 split，
不使用 HI 标签拟合、不 clip、不 EWMA。evaluate_all() 在排序后的表上计算六指标。
MAE/MAPE 按 cycle 聚合；structural/MI 按原公式跨 unit 聚合，MI random_state=42。
常量输出、短序列、非正 FC 分母和缺少 HI 标签保持明确诊断，非有限指标转成 JSON null。

scores.csv 保留 split/unit/cycle/score/hi/hi_gt/rul；同时保存 scaler、配置出处和曲线。
匹配的新训练目录可接受第一次评估，重复评估必须用新目录。
历史 per-unit/EWMA 工具已删除，相关历史报告和结果未删除。

## 版本证据

数学和执行顺序的数值对照见 REFACTOR_STEP1.md 至 REFACTOR_STEP5.md。
没有从结构重构推导模型性能提升、严格物理解耦或完整实验复现结论。
