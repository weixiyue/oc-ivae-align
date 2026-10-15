# 当前项目状态

更新：2026-10-15。以源码、独立 YAML 和运行快照为准；设计和历史审计不覆盖当前 AGENTS.md。
当前使用命令见 [README](../README.md)，实现说明见 [代码走读](CODE_WALKTHROUGH.md)。

## 已完成的结构重构

- 模型整合为四个文件，保留 OC-iVAE/cVAE/VAE 的实际计算与 checkpoint keys。
- train.py 使用普通函数，CLI 与配对实验共用训练循环；没有 Trainer 继承层。
- evaluate.py 直接处理 cycle 分数表，独立 metrics.py 保留六项指标公式。
- dataset.py 合并数据加载/取样，utils/preprocess.py 保留 N-CMAPSS 处理。
- 全部十份 YAML 位于 config/，内容不变；实验调度位于 experiments/。
- 新运行使用独立 runs 目录，记录配置、best.pt、history、scores 和评估出处。
- 按用户批准删除 tests/、一次性 test.py 和废弃空目录；删除前最后检查为 15 passed。

## 模型与实验

现有三种模型均可选择：缺省为 OC-iVAE，baseline_type 为 cvae 或 vae。
VAE 仍保留双 latent 分支及 decoder W 路径。当前未增加其他模型或消融。
各 YAML 的训练权重、预算和早停设置独立保留，没有为比较而顺便统一参数。

普通入口仍不设置初始化 seed；已知 seed 与 configured_seed 分开记录。
配对脚本设置 seed=42，复用共享初始化与实际 batch/RNG/LR 审计。
两个独立 verify 脚本已删除；配对一致性检查在实验入口完成，允许两组提前停止于不同 epoch。
新实验不依赖历史 checkpoint/Git 源码，不生成独立审计目录、初始权重副本或 batch 明细。
实验输出精简两步已完成；新配对实验保存 13 个文件，清单见 README。
归一化参数和评估出处统一保存在 metrics.json，配对实验只生成两张合并对比图。
普通训练及独立评估仍默认绘制 dev/test 曲线，已有实验产物不删除或改写。

## 数据与评价

- data/raw 为现有链接，当前目标为 /home/xiyue/CodeSpace/raw/；只读取，不移动目标数据。
- 当前可用 DS03，DS02 主 HDF5 缺失。processed/ 未生成缓存，没有新增输入格式。
- 输入标准化、P90、完整生命周期、padding/truncation、W/u 和 unit 采样语义保持。
- 全 dev 拟合全局 MinMaxScaler，test 只 transform，不平滑、不裁剪。
- MAE、MAPE、Monotonicity、Trendability、FC、MI；无 HI 参考时 MAE/MAPE 为 null。
- 旧 per-unit/EWMA/Prognosability 结果属于历史协议，不能直接用于当前指标比较。

## 验证范围

CPU 固定输入下模型、loss、必要梯度、参数更新、数据张量及 sampler 序列对照通过。
11 个既有模型 checkpoint 严格加载并保持原文件 hash。
DS03 epoch-80 历史 checkpoint 的 663 dev/438 test flights 重评，分数、缩放和指标保持一致。
短合成数据覆盖训练到评估及配对调度，不代表真实数据重新训练。
没有 GPU、真实 DS02 或新增正式实验验证。

2026-10-15 配对调度精简检查：隔离合成 HDF5 上，两组各训练 4 轮，修改前后
history、最佳权重、optimizer 状态、scores、scaler、指标、对比 CSV 和对比图完全一致。
旧 runner 对照跳过已完成的一次性 smoke，实际训练与评估均执行原共享函数。
另外使用受控验证损失确认两组分别在第 2/3 轮早停可正常汇总，已有目录拒绝覆盖。
上述为 CPU 短合成检查，不代表正式实验复现；临时脚本和产物位于 /tmp。

评估产物精简检查：固定两个已有合成 checkpoint 重评，scores、完整 metrics 内容及默认曲线
与修改前一致；被移除的 scaler/评估配置文件内容完整保留于 metrics.metadata。
关闭绘图不改变分数或指标；配对短训练验证输出恰好 13 个文件，history、最佳权重和对比结果不变。
独立评估 CLI 默认曲线与重复目录拒绝覆盖检查通过；未运行 GPU 或正式数据训练。

历史 runs/checkpoints、实验日志、源码快照和设计结论保留。
逐步证据见 REFACTOR_STEP1.md 至 REFACTOR_STEP5.md；旧状态原文保留在
[history/PROJECT_STATUS_pre_refactor.md](history/PROJECT_STATUS_pre_refactor.md)。
