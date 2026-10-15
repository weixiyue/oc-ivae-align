# OC-iVAE

使用 PyTorch 在 N-CMAPSS 上构建无 HI 标签监督的健康指标，提供 OC-iVAE、cVAE 和 VAE 对比模型。
训练统一使用 `train.py`，评估统一使用 `evaluate.py`；实验调度放在 `experiments/`。
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
    __init__.py                # 显式模型选择
utils/
    preprocess.py             # N-CMAPSS 数据处理
    losses.py                 # 重构、KL、HSIC、ranking、soft Spearman
    metrics.py                # 独立指标公式
config/                       # 全部独立实验 YAML
experiments/                  # 配对实验及现有核验工具
data/raw/                     # 原有符号链接
data/processed/               # 预留的生成数据目录，Git 忽略
runs/                         # 新运行配置、checkpoint、history 和评估结果
checkpoints/                  # 既有 checkpoint，原地保留
docs/                         # 当前说明、设计和历史验证资料
```

当前 `data/raw` 链接到 `/home/xiyue/CodeSpace/raw/`。保留链接及其目标，不递归移动、删除或复制原始数据。
当前可用的是 DS03 主文件与 dev/test cycle-HI 文件；DS02 主文件缺失。
预处理由 Dataset 在线执行，没有独立预处理 CLI，也没有新增 processed 数据格式。

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

既有 W-summary 配对实验命令：

```bash
python -m experiments.run_w_summary_ablation \
  --output-dir runs/w_summary_pair_new --device cpu --threads 2
```

该命令会执行两次真实训练，仍使用现有两份 seed-42 YAML；应在明确安排实验后运行。
本轮只在隔离的短合成配置上核验调度，没有重跑正式配对训练。

## 数据与评估协议

输入为 N-CMAPSS HDF5 的 `A/W/X_s/Y`，按 unit/cycle 分组并覆盖完整生命周期。
保持逐 cycle 下采样、P90 长度、逐 unit 传感器标准化、dead-sensor 置零、
头部 padding 和保留 tail 的截断。W 全局归一化后分别构造完整 cycle 的 u 和模型 W_seq；
不改变先验、FiLM 或 decoder 的条件路径。未见 test unit 的传感器统计按现有实现单独计算。

协议 `unsupervisedhi-ncmapss-v1`：posterior mean → 线性 raw HI → 按真实 flight 键关联。
仅在全部 dev 分数上拟合全局 MinMaxScaler，再 transform dev/test；
不逐机缩放、不 EWMA、不裁剪。test HI 可以超出 `[0, 1]`。
六项指标为 MAE、MAPE、Monotonicity、Trendability、FC、MI；MI random_state=42。
无 HI 参考时 MAE/MAPE 为 null，RUL 仍用于 MI；FC 与旧 Prognosability 不能直接混用。
参考 HI 若无 cycle 列，按现有严格行序检查关联，其语义行序的独立核实限制保留。

## 运行产物与复现范围

新运行保存 `config.yaml`、`best.pt`、`history.csv`、`metrics.json`、`scores.csv`、
`scaler.json`、`evaluation_config.yaml` 和 dev/test 曲线。
snapshot 记录实际维度、路径、源配置、已知 seed、初始权重/RNG hash、设备和线程。
普通入口保留原未固定初始化 seed 的行为：实际 seed 未指定时记录 null，
YAML 中的 configured_seed 单独记录；配对脚本明确设置 seed=42。
hash 是审计依据，不等同于保存完整初始化/RNG 状态。

checkpoint 保存 state_dict 字典。新文件保留已有参数 keys 和保存字段；
旧 checkpoint 原地保留。训练加载函数不完整恢复 scheduler、sampler 和 RNG，
没有精确续训 CLI。decoder 仍固定输出 14 个传感器，不支持任意 S。

## 验证与文档

按用户批准已删除整个 tests/，没有保留 pytest 运行入口；删除前最后一次为 15 passed。
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
