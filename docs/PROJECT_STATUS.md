# 当前项目状态

更新：2026-10-18。以源码、独立 YAML 和运行快照为准；设计和历史审计不覆盖当前 AGENTS.md。
当前使用命令见 [README](../README.md)，实现说明见 [代码走读](CODE_WALKTHROUGH.md)。

## 已完成的结构重构

- 原三模型的计算与 checkpoint keys 保持；新增 ocald.py/proposal_cnn.py/residual.py 独立实现。
- train.py 使用普通函数，CLI 与配对实验共用训练循环；没有 Trainer 继承层。
- evaluate.py 直接处理 cycle 分数表，独立 metrics.py 保留六项指标公式。
- dataset.py 合并数据加载/取样，utils/preprocess.py 保留 N-CMAPSS 处理。
- 原十份 YAML 内容不变；config/ 新增 OCALD/Proposal-C 四份及 Residual 四份完整独立配置（含两份短窗口），实验入口位于 experiments/。
- 新运行使用独立 runs 目录，记录配置、best.pt、history、scores 和评估出处。
- 此前按批准删除旧 tests/、test.py 和空目录；本次新增迁移模型的永久数学与 smoke 测试。

## 模型与实验

现有三种模型均可选择：缺省为 OC-iVAE，baseline_type 为 cvae 或 vae。
VAE 仍保留双 latent 分支及 decoder W 路径。
新增采用公共数据协议的 OCALD 两阶段/联合/decoder W 和 Proposal-C CNN；来源 commit 及命令见 README。
只迁入 8D I/C 原结构和 CNN C 约束，未迁入标量 I、时序 C 或 NG/F。
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
- 当前可用 DS03，DS02 主 HDF5 缺失。原始输入仍为 N-CMAPSS HDF5。
- DS03 预处理结果已保存到 data/processed/，约 65 MB；十份 YAML 数值不变。
- 首次自动生成或使用 utils.preprocess CLI，之后训练/评估/配对实验读取同一缓存。
- 缓存保存最终张量、周期 RUL、统计量和来源信息；HI 参考独立关联，不进入训练目标。
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

预处理缓存检查：变长周期、dead sensor、dev/test 重叠 unit ID 的合成数据上，
全部输入张量、统计量、RUL、unit split 和采样序列与原在线处理一致。
4 组模型输出/损失/梯度及 4 轮训练 history/最佳权重对照完全一致。
缓存命中时禁用所有原始预处理函数仍可加载和取样；来源/参数变化生成新文件，
不完整缓存报错，已有缓存未覆盖，离线 CLI 检查通过。
短配对实验仍输出 13 个文件，两组配置和评估均记录相同 processed_data 路径。
真实 DS03 缓存为 67,796,442 bytes。使用原 epoch-80 checkpoint 分别运行旧在线流程和缓存流程，
663 dev/438 test 的 scores 和曲线逐字节一致，原 metrics/scaler/评估出处完全一致，
仅新增 processed_data 来源字段；checkpoint hash 保持不变。未进行真实训练或 GPU 验证。

历史 runs/checkpoints、实验日志、源码快照和设计结论保留。
逐步证据见 REFACTOR_STEP1.md 至 REFACTOR_STEP5.md；旧状态原文保留在
[history/PROJECT_STATUS_pre_refactor.md](history/PROJECT_STATUS_pre_refactor.md)。

本次迁移的永久测试入口为 `python -m unittest discover -s tests -v`，8 项通过，
包含四个 CPU CLI smoke（合成数据，每阶段 1 次更新，完整 dev 缩放和评价）。
OCALD 与固定源 commit 的前向/损失/梯度/联合采样完全一致；CNN 使用独立公式和梯度对照。
旧 OC-iVAE 真实 DS03 的 scores、完整 metrics 和曲线均不变。所有本次临时权重、配置和输出已清理。
无 TensorFlow/源 CNN 权重，未做跨框架逐位复现、完整预算训练或 GPU 验证。

2026-10-18 新增 Residual-A/B 和独立完整配置，共用健康段训练入口、缓存与公共六指标。
网络仅使用 train 设备 cycle<=20；PCA/方向/缩放使用完整 dev，test 只变换，状态保存可独立重载。
10 项测试通过，包含六份配置 CPU smoke；Residual smoke 每次7条更新、144条 dev 拟合、48条 test。
数值/梯度、sklearn PCA/方向、重载、dev-only 缩放及既有入口回归通过。无完整训练或 TensorFlow/GPU 对照。
本次临时数据、配置、权重和结果自动清理；命令与限制见 README。

2026-10-18 Residual 短窗口接入：新增两份独立50/stride1配置，旧 flight 配置数值保留。
变长有效序列复用原统计，在 L_max 截断/padding前切窗；普通 PCA/10窗口方向后按flight均值再公共缩放。
13项测试通过，含八份配置 CPU smoke、源窗口对照、旧 OC-iVAE/flight A/B 真实checkpoint等价重评。
真实DS03 A/B各1步/512窗口；健康段140flight/112402窗口，完整dev 524907窗口、test 403868窗口。
永久缓存/测试保留；本次临时权重、残差矩阵和输出清理；未完整训练或验证GPU/TensorFlow，不作性能结论。

独立线性探测已接入：OC-iVAE mu_d、OCALD mu_I、Proposal-C Z及各自公共最终HI，
冻结网络，严格按已记录训练设备拟合StandardScaler+OLS，测试cycle合并报告R²/MAE/RMSE。
18项测试通过；实际DS03三模型使用516 fit/147 val/438 test，公共六指标产物保持逐字节一致。
探测产物使用独立目录和probe_metrics.json，不覆盖无监督结果；未知划分要求显式YAML。
仅CPU评价验证，无网络训练或GPU验证；临时产物清理，命令及协议见README。
