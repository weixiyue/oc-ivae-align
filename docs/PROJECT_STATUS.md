# 当前项目状态

更新：2026-10-14。以源码、独立 YAML 和运行快照为准；设计和历史审计不覆盖当前 AGENTS.md。
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
正式配对产物核验仍针对现有 DS03 的 100 轮和固定 flight 数量，不是通用多数据集核验工具。

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

历史 runs/checkpoints、实验日志、源码快照和设计结论保留。
逐步证据见 REFACTOR_STEP1.md 至 REFACTOR_STEP5.md；旧状态原文保留在
[history/PROJECT_STATUS_pre_refactor.md](history/PROJECT_STATUS_pre_refactor.md)。
