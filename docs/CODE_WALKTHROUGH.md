# 当前代码走读

更新：2026-10-19。本文描述简化重构后的实现，旧走读保留在
[history/CODE_WALKTHROUGH_pre_refactor.md](history/CODE_WALKTHROUGH_pre_refactor.md)。
入口与环境准备见 [README](../README.md)。

## 阅读顺序与职责

| 文件 | 主要职责 |
| --- | --- |
| dataset.py | NCMAPSSDataset/ShortWindowDataset、flight 内索引切窗、unit split、sampler |
| utils/preprocess.py | 离线/首次自动预处理、HDF5 缓存、cycle 分组、统计、归一化、u、RUL 校验 |
| models/tcn.py | causal encoder、masked LayerNorm、non-causal decoder、W summary |
| models/kgld.py | KGLD 装配、两个 posterior、FiLM、conditional prior、HI head |
| models/vae.py | 标准正态先验与两个 baseline 的组件替换 |
| models/__init__.py | 原三模型及 OCALD/Proposal-C/Residual 的显式构建分支 |
| models/ocald.py | 有效前缀适配、原 GRU/MLP/HIHead、flight 求和 ELBO/排序/L2 |
| models/proposal_cnn.py | 时间优先 Flatten 的 Proposal-C、unmasked MAE 和 batch Pearson |
| models/residual.py | Residual-A/B、磁盘窗口残差、普通 PCA、10窗口定向、flight 均值 |
| experiments/train_residual.py | train-unit 健康段 flight/window MAE、完整 dev PCA、公共 flight 评价 |
| utils/optimizers.py | Proposal-C/Residual 共用 Keras Adam epsilon-hat 更新 |
| utils/experiment.py | 新模型实际配置快照、history、完整 checkpoint 保存 |
| utils/losses.py | 单项数学损失、三阶段聚合及 KL annealing |
| train.py | 训练/验证、AdamW/LR、best selection、checkpoint、history/审计 |
| evaluate.py | 推理、flight 关联、dev-only scaler、评估和导出 |
| experiments/evaluate_linear_probe.py | 独立带标签线性探测、训练设备拟合、测试设备评价 |
| utils/linear_probe.py | mu_d/mu_I/Z确定性提取、键表/划分验证、StandardScaler+OLS及R²诊断 |
| utils/metrics.py | 六项指标、聚合方式、非有限值诊断 |
| experiments/ | 配对初始化、内存采样记录、调度、一致性检查与比较 |

## 数据链路

```text
A/W/X_s/Y HDF5
  → unit/cycle 分组与逐 cycle 下采样
  → L_max、传感器/W 统计
  → 归一化、padding/truncation、完整 cycle 的 u 和原始周期 RUL
  → data/processed/<原文件名>_<处理标识>.h5
  → Dataset 读取张量、选择 train/val units、独立关联 HI
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

prepare_data() 由独立 CLI 或 Dataset 调用，命中时只核对原文件信息和缓存元信息。
先写临时文件，全部成功后以不覆盖已有文件的方式发布；失败清理本次临时文件。
Dataset 不再保留原始 cycles，也不在 __getitem__ 中重复处理。采样所需 keys/order 保持不变。
预处理同时准备 dev/test，原始 Y 的周期内恒定性检查提前到首次生成，不影响有效数据的结果。
旧 Dataset 的 train_stats/L_max 注入接口由缓存中的固定 dev→test 统计关系取代，当前调用处已更新。

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
两者先构造 KGLD 再替换组件，保持初始化/RNG 消耗；没有另写训练循环。
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
周期 RUL 已由预处理检查并保存；read_reference()/associate() 检查标签、真实 cycle 键和一对一覆盖；
没有 window 平均分支，重复 flight 明确报错。

scale_dev_test() 只在全 dev score 上拟合 sklearn MinMaxScaler，transform 两个 split，
不使用 HI 标签拟合、不 clip、不 EWMA。evaluate_all() 在排序后的表上计算六指标。
MAE/MAPE 按 cycle 聚合；structural/MI 按原公式跨 unit 聚合，MI random_state=42。
常量输出、短序列、非正 FC 分母和缺少 HI 标签保持明确诊断，非有限指标转成 JSON null。

scores.csv 保留 split/unit/cycle/score/hi/hi_gt/rul；metrics.json 的 metadata 保存 scaler 和配置出处。
run_evaluation() 默认生成 dev/test 曲线；配对脚本传入 save_plots=False，改为绘制两组合并对比图。
不再单独输出 scaler.json 和 evaluation_config.yaml，历史文件原地保留。
匹配的新训练目录可接受第一次评估，重复评估必须用新目录。
历史 per-unit/EWMA 工具已删除，相关历史报告和结果未删除。

## 两个迁移模型入口

experiments.train_ocald 保留独立两阶段与联合流程，experiments.train_proposal_cnn 保留独立 CNN 目标；
不经过原 train_model 的 KGLD loss、scheduler、分层 sampler 或验证重构选模。
Dataset 的数据缓存和 unit split 共用，OCALD 仅重排左右 padding，CNN 原样读取固定长度 X/W。
评价 predict_cycles 按具体模型提取 -score(mu_I)、Z 或原 hi_raw，不要求伪造统一 forward 字段。
新 checkpoint 保存完整实际配置，评价拒绝外部结构/数据冲突并核对预处理文件来源；
旧 state_dict checkpoint 仍走原加载规则，W-summary 校验只对原三模型生效。
源 commit、默认预算、初始化/Adam 约定和 smoke 命令均见 README。

## 版本证据

数学和执行顺序的数值对照见 REFACTOR_STEP1.md 至 REFACTOR_STEP5.md。
没有从结构重构推导模型性能提升、严格物理解耦或完整实验复现结论。

Residual 短窗口由独立 YAML 的 data.input_mode=short_window 选择；公共预处理使用旧缓存统计，
从原文件重新读取逐 flight 下采样序列并存储截断/padding之前的归一化结果。窗口缓存身份独立，
Dataset 仅保存起点索引。PCA/方向在完整 dev 窗口层面确定，随后 cycle 均值才进入公共标签关联/缩放。
旧 flight 模式及其他模型仍读取原固定长度缓存；旧配置数值和历史产物保留。
