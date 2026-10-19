# 第二步：评估简化与目录清理

日期：2026-10-14。修改前基线：`9bd2f3b`。
仅完成第二步；训练主循环与数据重构继续等待逐步审核。

## 变更

- 顶层 `evaluate.py` 合并原评估入口、推理与协议处理，按加载、推理、关联、
  dev 缩放、指标和保存组织普通函数；删除三个单独的配置读取/合并包装函数。
- 推理直接生成 unit/cycle/score 表；删除原嵌套字典、重复排序和无用的标签收集。
  标签继续从参考 HDF5 按键关联，模型推理不使用标签。
- 删除未被实际 cycle 模型使用的 window 均值聚合分支；重复 flight 仍明确报错。
- 指标公式原样保留在 `utils/metrics.py`，不塞入入口；未建立通用配置模块。
- 删除没有调用的逐机 min-max/EWMA 旧后处理及仅转发 CLI 的 `plot_hi.py`。
- 训练末尾和配对实验直接调用 `evaluate.run_evaluation()`；实验产物核验改用新指标模块。
- 清理旧 `eval/`，以及第一步遗留的 `model/`、`baselines/` 目录。
  删除目录前检查内容，仅清理确认的旧字节码缓存；没有递归触及原始数据。

原评估相关 7 个文件（含空包文件）整合为一个入口与一个数学模块；
新入口不再修改 sys.path。保留指标定义、异常诊断、出处与 scaler 信息。
历史文档与实验产物保留，不为新目录结构修改历史记录。

## 保持的协议与接口

- 仍评估 train/validation unit 并集，即全部 dev，以及独立 test。
- posterior mean、batch size 64、实际 cycle 排序、float64 score 导出精度保持。
- N-CMAPSS HDF5、特征、统计范围、归一化、padding、W/u 路径未修改。
- 仅 dev 拟合 MinMaxScaler，test 只 transform；不平滑、不裁剪、不逐机缩放。
- 保留原 RUL flight 内恒定检查、参考 HI 的明确键或严格行序关联、完整键覆盖检查。
- MAE/MAPE/Monotonicity/Trendability/FC/MI 及 MI seed 42 不变。
- 无参考 HI 时 MAE/MAPE 为 null；重复键、非法键和不完整标签仍报错。
- 旧包装 checkpoint 与裸 state_dict 均可加载；W-summary 开关冲突检查不变。
- 本步骤保持 `predictions.csv` 和原产物布局；统一新运行目录留到第四步。
- `run_evaluation(config_path, checkpoint_path, output_dir, device)` 参数保持。
  显式输出目录存在时拒绝覆盖；默认仍创建带时间戳的独立目录。

## 验证

修改前评估源码冻结在 `/tmp/oc_ivae_step2_reference/`，与新实现独立运行。
临时对照脚本和产物不纳入运行时模块或 Git。

### 真实 DS03

配置：`configs/exp_tuning_mono_weight.yaml`。
checkpoint：`checkpoints/exp_tuning_mono_weight/best_model.pt`，epoch 80。
SHA256：`f0092274e8238b220589d1bc3e21e5415df4a33aebca461dba00593c6e208f34`。
CPU、2 个 Torch 线程；修改前后分别执行一次全量 dev/test 推理。

663 个 dev flight、438 个 test flight，共 1101 个 flight。
以下六个产物逐字节一致：

- `predictions.csv`
- `metrics.json`
- `scaler.json`
- `evaluation_config.yaml`
- `hi_curves_dev.png`
- `hi_curves_test.png`

预测 CSV 也与既有 `evaluation_unsupervisedhi_v1_20261014_verified` 的导出逐字节一致。
checkpoint hash 未变；前后产物均写到临时独立目录，未覆盖历史 runs。

### 合成数据与现有检查

- 三种模型 × 裸/包装 checkpoint，共六种完整评估对照。
  覆盖真实小型 HDF5 加载、参考 HI、无标签、test 新 unit 和完整导出；六个产物均逐字节一致。
- 70 个乱序 cycle 跨两个推理 batch，直接表输出与旧推理结果逐元素一致。
- 现有 15 项测试通过；仅适配新入口和表接口，删除已批准移除的 window 聚合断言，
  保留 duplicate 拒绝等实际协议断言。
- 新评估、现有训练、配对实验和产物核验 CLI 的 `--help` 通过。
- 在历史配对实验产物的临时副本上执行核验，源码 hash、初始化、batch/RNG/LR、
  checkpoint 选择、缩放与六指标重算均通过；未覆盖历史核验报告。
- 指标源文件与修改前副本逐字节一致；全部现有 YAML、用户已有修改和 raw 链接保持。

当前命令：

```bash
python evaluate.py --config configs/exp_tuning_mono_weight.yaml \
  --checkpoint checkpoints/exp_tuning_mono_weight/best_model.pt \
  --output-dir runs/exp_tuning_mono_weight/new_evaluation \
  --device cpu --threads 2
```

## 限制与下一步

没有运行训练；没有验证 GPU 或其他 checkpoint 的真实全量评估。
DS02 原始数据缺失，无标签行为通过合成数据验证，不代表真实 DS02 实验复现。
`tests/` 仍按既定第五步计划统一删除；本步骤没有提前执行后续训练、数据或配置重构。
用户已有 `AGENTS.md`、`EXPERIMENT_LOG.md` 修改保持原样，不纳入本步骤提交。

下一步待批准：训练主流程简化，移除 Trainer/AuditedTrainer 间接层并整合损失模块。
