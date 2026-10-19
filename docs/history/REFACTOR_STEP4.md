# 第四步：数据、配置与运行产物

日期：2026-10-14。重构前基线：`c1f6293`。
仅完成本步骤；README、依赖清单、tests 删除和最终文档清理留到第五步审核。

## 目录与代码

- Dataset/Sampler 合并到顶层 `dataset.py`；将 split 校验放在初始化边界，
  删除只给固定字符串命名的常量和冗余 HI lookup 检查。
- 预处理迁移到 `utils/preprocess.py`，删除未使用常量及只过滤某 unit 的包装函数。
  HDF5 布局、生命周期、下采样、P90、dead-sensor 置零、归一化、mask、
  head padding、保留 tail 的截断及完整 cycle 的 u 构造不变。
- 全部 10 份 YAML 移至 `config/`，文件内容逐字节保留，
  包括 exp_tuning_mono_weight 及 W-summary 配对配置。
- 三个实验/验证脚本移至 `experiments/`；用 `python -m experiments.<name>` 执行，
  不再修改 sys.path。实验仍调用同一训练和评估流程。
- 删除旧 `configs/`、`scripts/` 及缓存，删除已确认的空 smoke 文件。
  `data/` 保留为数据目录；原 raw 符号链接与目标不移动、不复制、不删除。
- 创建空的 `data/processed/` 并将其加入忽略规则；没有新增缓存格式或离线预处理 CLI。

## 新运行布局与保护

```text
runs/<experiment_name>/
    config.yaml
    best.pt
    history.csv
    metrics.json
    scores.csv
    scaler.json
    evaluation_config.yaml
    hi_curves_dev.png
    hi_curves_test.png
```

`train_model(..., run_dir=Path(...), audit=False)` 创建新的独立目录，
目录存在时直接报错，不覆盖历史运行。CLI 可用 `--run-dir` 指定新名字。
每轮 history 保存原来的 batch 平均损失、val_recon 和 scheduler.step 前的 LR；
audit=True 时在同一 CSV 中增加配对审计字段。

config.yaml 记录解析后的数据路径、实际模型输入维度和 W-summary 维度、
运行目录、源配置、已知种子、初始权重/RNG hash、设备、线程和确定性设置。
规范 YAML 数值不变；快照只补实际运行信息及必要路径。
普通入口仍不设置初始化 seed：seed 为 null；configured_seed 单独记录 YAML 声明，
不将未应用的 seed 声称为实际初始化 seed。配对脚本由调用方明确设置 seed=42。

checkpoint 仍是原七个字段的 state-dict 字典，参数 keys/张量和 optimizer 行为不变。
新模型信息和运行出处只写新 checkpoint；既有 checkpoint 保持原样。

训练结束自动用 config.yaml 评估 best.pt，第一次评估写入同一运行目录。
只有配置、checkpoint、history 匹配且所有评估产物尚不存在时才接受该目录；
重复评估或其他已存在目录仍拒绝覆盖。独立重评应使用新的 --output-dir。
默认 checkpoint 查找改为新运行的 best.pt；旧 checkpoint 用 --checkpoint 显式指定。

配对实验的每个 true/false 子目录采用相同布局，保留 stdout.log、batch trace、
初始化证据、summary 和比较产物。stdout 文件在训练创建目录后首次写入时打开，
用排他创建模式防止覆盖。产物核验仅区分本项目实际存在的旧布局和 runs-v1，
不建立通用迁移框架，不改写历史 manifest 或 checkpoint。

## 验证

CPU、2 个 Torch 线程。修改前源码及验证产物位于临时目录
`/tmp/oc_ivae_step4_reference/`，未加入 Git。

- 合成 N-CMAPSS HDF5 包含不同长度、dead sensor、不同 Fc/hs 和未见 test unit。
  六种 Dataset 场景逐项比较 cycles、划分、L_max、统计和七个返回字段，完全一致；
  包括有/无 HI、共享 dev 统计、新 unit 的统计缓存、padding 和长 cycle 截断。
- 六种 sampler 场景各比较 100 轮：空集、少于 K 的 unit、有放回采样、
  remainder 丢弃及实际默认 K；batch 序列完全一致。
- 全部 YAML 逐字节一致，解析内容一致；没有调参、新模型或新消融。
- 三种模型 × 普通/审计模式共六组，各进行三轮短合成训练。
  history、最终参数、checkpoint 模型/optimizer/epoch/best loss、batch trace 和 RNG
  与冻结的上一版实现完全一致；审计 CSV 只改文件名，内容逐字节一致。
- 合成数据跑通训练 CLI 到自动评估；验证配置快照、默认 best.pt 加载和全套产物。
  重复训练/评估均拒绝覆盖，已有文件和 RNG 不变。
- 完整配对调度在隔离的三轮合成配置上跑通，true/false 的 batch/RNG/LR 匹配，
  新路径、stdout 和比较产物齐全。规范配置及正式训练预算未修改。
- 现有 15 项测试通过；所有入口及三个 experiments 模块 --help 通过。
- 在历史实验临时副本上核验旧布局，以及仅重排文件名的 runs-v1 布局；
  初始化、checkpoint、源 hash、scaler 和指标核验通过。没有迁移真实历史产物。

### 真实 DS03 重评

使用未修改的 exp_tuning_mono_weight epoch-80 checkpoint，SHA256：
`f0092274e8238b220589d1bc3e21e5415df4a33aebca461dba00593c6e208f34`。
663 个 dev flight 和 438 个 test flight，共 1101 个。
新 scores.csv 与第二步对照的 predictions.csv 逐字节一致；scaler 与两张曲线也逐字节一致。
dev/test 六项指标及诊断字典完全相等。出处中的配置路径按迁移后的位置变化。
旧 checkpoint hash 不变，重评产物只写临时独立目录。

## 当前命令

```bash
python train.py --config config/exp_tuning_mono_weight.yaml \
  --run-dir runs/exp_tuning_mono_weight_new

python evaluate.py --config runs/exp_tuning_mono_weight_new/config.yaml \
  --output-dir runs/exp_tuning_mono_weight_new/reevaluation --device cpu --threads 2

python evaluate.py --config config/exp_tuning_mono_weight.yaml \
  --checkpoint checkpoints/exp_tuning_mono_weight/best_model.pt \
  --output-dir runs/exp_tuning_mono_weight_new_legacy_check --device cpu --threads 2

python -m experiments.run_w_summary_ablation --output-dir runs/new_paired_experiment
```

上述训练/配对命令仅为使用说明，本步骤没有执行真实数据训练或正式 100 轮实验。
GPU 与真实 DS02 数据仍未验证。产物核验脚本保留针对正式 DS03 配对实验的
100 轮/flight 数量检查；三轮合成调度由独立临时断言核验，不冒充正式实验。

用户已有 AGENTS.md、EXPERIMENT_LOG.md 修改及三份 tasks 文档删除均保留，
不纳入本步骤提交。历史 runs/checkpoints、源码快照和 raw 链接保持原样。
