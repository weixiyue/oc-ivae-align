# OC-iVAE

使用 PyTorch 在 N-CMAPSS 上构建无 HI/RUL 标签监督的健康指标，提供 OC-iVAE、cVAE、VAE，
以及采用 OC-iVAE 数据协议的 OCALD 和 Proposal-C CNN 适配版。
原三模型使用 `train.py`，新模型使用独立实验入口；全部共用 `evaluate.py`。
项目规范见 [AGENTS.md](AGENTS.md)，当前状态见 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)。

## 环境

本轮验证环境为 Python 3.11.15、PyTorch 2.5.1+cu121，实际检查均使用 CPU。
[requirements.txt](requirements.txt) 固定已验证的运行依赖；Torch 固定发行版本 2.5.1，
CUDA wheel 的构建需与目标环境匹配。PyTables (`tables`) 用于 pandas 的 cycle-HI HDF 读取。

```bash
python -m pip install -r requirements.txt
```

依赖安装命令仅供环境准备；重构过程中没有安装或升级包，也未验证 GPU。
精确数值复现应保持同一依赖版本、数据、checkpoint、线程和随机状态。

## 目录

```text
train.py                       # 训练 CLI 和共享普通函数
evaluate.py                    # cycle 推理、标签关联、dev-only 缩放与导出
dataset.py                     # Dataset、unit/cycle 划分与 sampler
models/
    oc_ivae.py                 # 模型装配、分支、FiLM、conditional prior、HI head
    tcn.py                     # causal encoder 与 non-causal decoder
    vae.py                     # 两个已有 baseline 的差异组件
    ocald.py                   # 原 8D I/C OCALD、padding 适配及模型专属损失
    proposal_cnn.py            # Proposal-C CNN 与 batch Pearson 约束
    __init__.py                # 显式模型选择
utils/
    preprocess.py             # N-CMAPSS 数据处理
    losses.py                 # 重构、KL、HSIC、ranking、soft Spearman
    metrics.py                # 独立指标公式
    experiment.py             # 新实验共用的配置快照、history 与 checkpoint 保存
config/                       # 全部独立实验 YAML
experiments/                  # 两个独立新模型训练入口及 W-summary 配对实验
tests/                        # 模型数学、采样、重载与端到端合成 smoke
data/raw/                     # 原有符号链接
data/processed/               # 可复用的预处理 HDF5，Git 忽略
runs/                         # 新运行配置、checkpoint、history 和评估结果
checkpoints/                  # 既有 checkpoint，原地保留
docs/                         # 当前说明、设计和历史验证资料
```

当前 `data/raw` 链接到 `/home/xiyue/CodeSpace/raw/`。保留链接及其目标，不递归移动、删除或复制原始数据。
当前可用的是 DS03 主文件与 dev/test cycle-HI 文件；DS02 主文件缺失。

## 数据预处理

可在训练前预处理一次：

```bash
python -m utils.preprocess --config config/exp_tuning_mono_weight.yaml
```

结果保存为 `data/processed/<原文件名>_<处理标识>.h5`，包含 dev/test 的
`x`、`mask`、`W_seq`、`u`、unit/cycle、周期 RUL、归一化统计量和 `L_max`。
HI 参考仍独立读取，不写入模型输入缓存。预处理不使用模型参数，也不消耗训练随机状态。

未手动预处理时，训练或评估首次启动会提示并自动生成；之后相同数据和处理参数直接复用。
Dataset 只加载所选 unit 的已处理张量到内存，取样时不再重复归一化或构造 `u`。
训练、评估、不同模型及配对实验共享处理文件，train/val 的 seed 和比例不影响缓存标识。
首次生成同时准备 dev/test，test 不参与训练目标、模型选择或 dev 统计量计算。

处理标识包含原文件实际路径、大小、纳秒修改时间、降采样因子、`n_fc` 和预处理版本。
数据或参数变化生成新的文件，不覆盖旧文件；缓存损坏或不匹配会报错，不自动重建。
文件信息检查不扫描原始文件内容，无法识别刻意保持路径、大小和修改时间的内容替换。
预处理算法变化时需更新 `utils/preprocess.py` 的 `PROCESSED_VERSION`。
可在 YAML 的 `data.processed_dir` 指定另一处理目录；默认目录及生成数据均不提交 Git。
运行快照记录实际 `processed_data` 路径，仍需保留原文件以核对来源信息。

## 训练与评估

从仓库根目录运行。每个新训练必须使用不存在的目录；默认名称来自配置文件名。
既有同名历史 runs 不会被覆盖，使用 `--run-dir` 明确指定新名称。

```bash
python train.py --config config/exp_tuning_mono_weight.yaml \
  --run-dir runs/exp_tuning_mono_weight_new
```

训练按 validation reconstruction 选择最佳 checkpoint，再自动评估全部 dev 和 test。
HI 标签不进入训练目标，test 不参与模型选择。

独立重评使用新的输出目录；配置快照可定位同一运行的 `best.pt`：

```bash
python evaluate.py --config runs/exp_tuning_mono_weight_new/config.yaml \
  --output-dir runs/exp_tuning_mono_weight_new/reevaluation --device cpu --threads 2
```

旧 checkpoint 使用显式路径，无需修改或迁移权重：

```bash
python evaluate.py --config config/exp_tuning_mono_weight.yaml \
  --checkpoint checkpoints/exp_tuning_mono_weight/best_model.pt \
  --output-dir runs/exp_tuning_mono_weight_legacy_check --device cpu --threads 2
```

查询参数不会启动训练或推理：

```bash
python train.py --help
python evaluate.py --help
python -m experiments.run_w_summary_ablation --help
```

## 模型、配置和后续实验

| 模型 | 先验 | z_c 分支 | decoder W summary |
| --- | --- | --- | --- |
| OC-iVAE | u 条件 Gaussian | FiLM(u) | 由现有配置开关控制 |
| cVAE | 标准正态 | 保留 FiLM(u) | 保留同一配置控制 |
| VAE | 标准正态 | 无 FiLM | 保留同一配置控制 |

当前 VAE 仍是双 latent 分支，不能解释为完全移除了 W/u 所有路径的单分支模型。
模型由 `experiment.baseline_type` 的缺省值、`cvae` 或 `vae` 显式选择。
现有 YAML 全部保持数值与独立实验身份，包括各自的早停和 loss 权重差异；没有额外统一或调参。
历史 `postprocess.ewma_alpha` 不应用到当前评估，部分 logging 字段保留原记录用途。

后续新增模型放在 `models/`，新增实验放在独立 `config/*.yaml`；
共享训练/评估部分继续调用普通函数。模型特有目标和 HI 构造按实际差异扩展，
不要求新模型伪造 OC-iVAE 的 12 个字段，不提前引入 registry 或通用 Trainer。
批量脚本只调度、审计和汇总，不复制训练循环与指标计算。

## OCALD 与 Proposal-C

两者均直接复用公共 Dataset/预处理缓存、seed=42 的 train/val 设备划分和完整 dev/test 评价，
不使用 HI/RUL 训练或选模；不是源仓库的数据协议或完整实验复现。

| 配置 | 训练方式与保留项 |
| --- | --- |
| [ocald_original](config/ocald_original.yaml) | 表示 100 轮后冻结，只训练 HIHead 100 轮；I/C 各 8 维 |
| [ocald_joint](config/ocald_joint.yaml) | 联合 100 轮；KL 各 1、排序 10、头部 L2=1e-4 |
| [ocald_joint_decoder_w](config/ocald_joint_decoder_w.yaml) | 同上，显式开启 decoder W 输入 |
| [proposal_cnn_c](config/proposal_cnn_c.yaml) | CNN + batch Pearson C；Adam 1e-4、batch 20、50 轮 |

实际迁移来源固定为 [kgld 4f9e53cd3623b5cf0311d594fd7efcd4bec88206](https://github.com/weixiyue/kgld/tree/4f9e53cd3623b5cf0311d594fd7efcd4bec88206)
的 model.py、losses.py、train.py 与对应三份配置，以及
[UnsupervisedHI 157ebfae7baeee1e0cd52180f8e58ead3dcb448e](https://github.com/KBaja/UnsupervisedHI/tree/157ebfae7baeee1e0cd52180f8e58ead3dcb448e)
的 CMAPSS/03_train_evaluate_proposals.ipynb、CMAPSS/eval_utils.py；每份 YAML 记录完整 commit 和文件。

OCALD 仅把有效 X/W 按原序移到前端、右补零，再送入 packed GRU；训练和推理共用该适配。
高斯重构按 flight 求和、KL 按维求和后取 batch 均值，排序仍为鼓励 score 随周期增加的 softplus。
保留源 Adam 1e-3、裁剪 5（HI 阶段不裁剪）、原均匀设备/配对及联合 4x4 分组采样。
默认 decoder 不读取 W；conditional prior 仍读取 W。公共评价才取 `-score(mu_I)`，训练不改符号。

Proposal-C 保留通道 [128,64,32,16]、kernel 85、same/ReLU，按 Keras 时间优先顺序 Flatten 得到 Z，
重复 Z 拼接逐时刻 W，再用反向通道序列卷积及逐时刻 Dense 输出 14 维。
平均 MAE 包含公共补零位置；C 是整个 batch 的 Pearson(Z, cycle)，权重 1，最小化鼓励负相关。
先顺序组 batch 再 shuffle batch，最后不足一个 batch 也保留；零方差导致非有限 C 时明确报错。
使用 Glorot uniform/零偏置；Keras Adam 显式 beta=(0.9,0.999)、epsilon=1e-7（uncorrected sqrt(v) 后的 epsilon-hat）。
源 Notebook 未指定 seed，此适配显式用 42；PyTorch RNG/浮点实现不保证与 TensorFlow 逐位相同。
源 eval_utils.py 的函数签名缺逗号，只迁移已核对算法，不复制其训练/评价工具。

```bash
python -m experiments.train_ocald --config config/ocald_original.yaml \
  --run-dir runs/ocald_original_seed42 --device cpu --threads 2
python -m experiments.train_ocald --config config/ocald_joint.yaml \
  --run-dir runs/ocald_joint_seed42 --device cpu --threads 2
python -m experiments.train_ocald --config config/ocald_joint_decoder_w.yaml \
  --run-dir runs/ocald_joint_decoder_w_seed42 --device cpu --threads 2
python -m experiments.train_proposal_cnn --config config/proposal_cnn_c.yaml \
  --run-dir runs/proposal_cnn_c_seed42 --device cpu --threads 2
python evaluate.py --config runs/proposal_cnn_c_seed42/config.yaml \
  --output-dir runs/proposal_cnn_c_seed42/reevaluation --device cpu --threads 2
```

新入口默认 CPU，可显式选择 cuda；每次必须使用新目录。加 `--smoke` 后每阶段 1 轮、最多 1 次更新，
OCALD 仅取各训练设备前 8 个 flight；公共 scaler 和六指标始终使用完整 dev/test，不缩小评价集合。
新模型按最后轮保存 `best.pt`，元数据 selection=last_epoch；两阶段另存 `representation.pt`。
checkpoint 包含模型类型、阶段、完整实际配置、seed 和预处理出处。
重评使用运行快照，结构/数据不一致的外部配置会报错，其余设置优先使用 checkpoint 保存配置。
每轮 history 记录真实损失项、权重、优化步数、抽取样本数和耗时；CNN 不记录伪造的 KL/HSIC。

既有 W-summary 配对实验命令：

```bash
python -m experiments.run_w_summary_ablation \
  --output-dir runs/w_summary_pair_new --device cpu --threads 2
```

该命令会执行两次真实训练，仍使用现有两份 seed-42 YAML；应在明确安排实验后运行。
本轮只在隔离的短合成配置上核验调度，没有重跑正式配对训练。

每次配对实验保存 13 个文件：

```text
runs/<experiment_name>/
    comparison.csv
    comparison_curves_dev.png
    comparison_curves_test.png
    true/
        config.yaml
        best.pt
        history.csv
        metrics.json
        scores.csv
    false/
        config.yaml
        best.pt
        history.csv
        metrics.json
        scores.csv
```

`comparison.csv` 包含两组 dev/test 六项指标和 false 减 true 的差值；
两张图按发动机展示两组 HI 与参考 HI。每组不再单独生成曲线。
共同完成 epochs 的配对审计结果保存在各组 `config.yaml` 的 `run_metadata.paired_audit` 中。

## 数据与评估协议

输入为 N-CMAPSS HDF5 的 `A/W/X_s/Y`，按 unit/cycle 分组并覆盖完整生命周期。
保持逐 cycle 下采样、P90 长度、逐 unit 传感器标准化、dead-sensor 置零、
头部 padding 和保留 tail 的截断。W 全局归一化后分别构造完整 cycle 的 u 和模型 W_seq；
不改变先验、FiLM 或 decoder 的条件路径。未见 test unit 的传感器统计按现有实现单独计算。

协议 `unsupervisedhi-ncmapss-v1` 按真实 flight 键关联分数：OC-iVAE 保持 posterior mean 的线性 raw HI，
OCALD 使用 -score(mu_I)，Proposal-C 使用 Z。
仅在全部 dev 分数上拟合全局 MinMaxScaler，再 transform dev/test；
不逐机缩放、不 EWMA、不裁剪。test HI 可以超出 `[0, 1]`。
六项指标为 MAE、MAPE、Monotonicity、Trendability、FC、MI；MI random_state=42。
无 HI 参考时 MAE/MAPE 为 null，RUL 仍用于 MI；FC 与旧 Prognosability 不能直接混用。
参考 HI 若无 cycle 列，按现有严格行序检查关联，其语义行序的独立核实限制保留。

## 运行产物与复现范围

训练并自动评估后保存 `config.yaml`、`best.pt`、`history.csv`、`metrics.json` 和 `scores.csv`。
普通训练和独立评估仍生成 dev/test 曲线；独立评估只保存评估结果及曲线。
归一化参数保存在 `metrics.json` 的 `metadata.scaling`，完整评估出处和配置保存在 `metadata`，
不再额外生成 `scaler.json` 或 `evaluation_config.yaml`。已有运行文件保持原样。
OC-iVAE snapshot 记录实际维度、路径、源配置、已知 seed、初始权重/RNG hash、设备和线程。
普通入口保留原未固定初始化 seed 的行为：实际 seed 未指定时记录 null，
YAML 中的 configured_seed 单独记录；配对脚本明确设置 seed=42。
hash 是审计依据，不等同于保存完整初始化/RNG 状态。

checkpoint 保存 state_dict 字典。新文件保留已有参数 keys 和保存字段；
旧 checkpoint 原地保留。训练加载函数不完整恢复 scheduler、sampler 和 RNG，
没有精确续训 CLI。decoder 仍固定输出 14 个传感器，不支持任意 S。

## 验证与文档

旧 tests/ 已在此前清理中删除；本次按迁移验证要求新增永久 unittest 测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

本次 8 项测试通过，包含四份配置的 CPU CLI smoke、保存重载、冻结/联合梯度、
卷积/Flatten/损失及 Adam 独立公式对照、完整 dev 缩放与六指标/CSV 检查。
smoke 在测试创建的合成数据上自动加 --smoke 执行两个入口；临时配置、数据、权重和输出由 TemporaryDirectory 清理。
另对照固定 kgld commit：三份 OCALD 的初始化、随机前向、损失、参数梯度及联合配对序列完全一致；
旧 OC-iVAE checkpoint 的真实 DS03 1101 个周期分数、完整 metrics 和曲线与修改前一致。
没有 TensorFlow 环境或源 CNN 权重，未做跨框架逐位复现，也未执行完整 50/100 轮训练或 GPU 验证。

W-summary 配对脚本在运行时检查配置、共享初始权重和共同训练 epochs 的 batch/RNG/学习率摘要，
摘要保存在 history 和配置中。不再提供独立 verify 命令，也不依赖旧 Git revision 或历史 checkpoint。
历史审计报告保留原有记录；新实验不生成额外的历史对照或验证目录。

CUDA 配对实验须在启动 Python 前设置确定性计算环境变量：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.run_w_summary_ablation \
  --output-dir runs/w_summary_pair_cuda_new --device cuda --threads 2
```

重构证据见 [第一步](docs/REFACTOR_STEP1.md)、[第二步](docs/REFACTOR_STEP2.md)、
[第三步](docs/REFACTOR_STEP3.md)、[第四步](docs/REFACTOR_STEP4.md)、[第五步](docs/REFACTOR_STEP5.md)。
当前代码说明见 [CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)。
原状态和走读保留在 [docs/history/](docs/history/)，设计、旧审计和评估验证文档保留历史结论。
CPU 上模型/损失/梯度/采样对照通过；DS03 旧 checkpoint 的 1101 个 flight 重评一致。
没有真实数据重新训练、GPU 验证或真实 DS02 实验，不将这些检查称为完整实验复现。
