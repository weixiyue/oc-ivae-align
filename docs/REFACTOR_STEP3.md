# 第三步：直接训练流程与损失整合

日期：2026-10-14。重构前基线：`3a22d24`。
本步骤只完成训练与损失简化；数据、配置、产物布局及最终清理仍等待逐步审核。

## 变更

- 顶层 `train.py` 提供 `train_model()`，由 CLI 和配对实验直接调用。
  训练、验证、学习率更新、checkpoint 选择和早停按顺序写在普通函数中。
- 删除 Trainer/AuditedTrainer。`train_epoch()` 负责一次训练 epoch，`validate()`
  负责验证均值；不引入基类、回调框架或新 Trainer 状态容器。
- 配对审计通过 `audit_dir` 显式启用，要求原 RecordedSampler；仍记录初始化后的
  RNG、实际 batch 序列、每轮 LR/损失及 checkpoint RNG，不复制训练循环。
- `train_model()` 返回完成 epoch 数、最佳验证损失、checkpoint 路径与 history。
  普通训练不新增固定 seed；配对脚本保持原共享初始化和 RNG 恢复。
- 将六个数学实现文件合并为 `utils/losses.py`，保留各独立函数和公式。
  合并相同的向量校验，删除只包装 config 字典索引、loss 累加/平均和 batch 移动的函数。
- `save_checkpoint()`/`load_checkpoint()` 保持现有 state-dict 字典字段及 W 开关检查；
  load 返回已保存 epoch/best loss。没有新增 resume CLI 或通用 checkpoint 迁移。
- 删除旧 `scripts/run_train.py`、空 scheduler，并清理 `train/`、`loss/` 及旧字节码。
  配对和验证脚本只更新实际调用与源码 hash 位置。

## 保持的训练行为

1. 模型、参数及构造初始化由已有 `build_model()` 保持。
2. AdamW 的 lr/weight_decay 与 CosineAnnealingLR 的 T_max 不变。
3. loss 使用零起始 epoch；训练后 epoch 加一，验证后 scheduler.step，再选择 checkpoint。
4. loss 的 warmup、KL transition、HSIC、ranking 和 soft Spearman 的计算、reduction、
   梯度及启用边界保持；零权重不改为跳过计算。
5. zero_grad、有限值检查、backward、clip_grad_norm、step 和参数检查顺序保持。
6. 训练和验证仍按 batch 数平均，包括最后一个不足 batch；HI 标签不进入 loss。
7. checkpoint 仍以 val_recon 严格小于最佳值选择；相等值计入 patience。
8. 审计 row 在 scheduler.step 前记录 LR，checkpoint 保存 scheduler.step 后的 optimizer。
9. 配置仍按 training/loss_weights/logging 合并，再接受 scalar 顶层覆盖。
10. 现阶段仍写 `checkpoints/<experiment>/best_model.pt`，训练结束后评估最佳模型；
    新 runs 布局、配置快照和 history.csv 落盘留到第四步。

## 验证

环境：CPU、2 个 Torch 线程。修改前源码冻结在 `/tmp/oc_ivae_step3_reference/`。
临时对照脚本与结果不加入 Git 或运行时代码。

- 全部 10 份现有 YAML，各覆盖 epoch 0、warmup 前一轮、warmup 起点、
  transition 前一轮和最终阶段起点，共 50 个 epoch 对照。
  每个 epoch 使用三个大小不等的合成 batch，比较损失均值、参数、optimizer state、
  裁剪后梯度、验证均值和 RNG；另比较原始损失图的参数及 x/u/W 输入梯度。
- 三种模型 × 两种 W 开关，共六组各五轮的合成训练控制流对照。
  覆盖三个 loss 阶段，逐项比较最终参数、所有审计 row、checkpoint 内容和 RNG。
  `training_metrics.csv` 与 `batch_sequences.jsonl` 前后逐字节一致。
- 使用受控验证序列 `[3, 2, 2, 4, 1]` 对照早停：patience=2 时第 4 轮停止，
  最佳 checkpoint 为第 2 轮；关闭早停时完成第 5 轮并选择第 5 轮。
  前后参数与 checkpoint optimizer/LR 状态一致。
- 11 个现有 checkpoint 的模型、optimizer 和 epoch/best loss 恢复与旧 Trainer 一致；
  读取前后 SHA256 不变。

全部张量数值对照采用 `rtol=0, atol=0`，均通过。
上述短合成循环只用于验证；没有运行真实数据训练或 100-epoch 配对实验。

其他检查：

- 现有 15 项测试通过；仅将 checkpoint/非有限值检查改为调用普通函数。
- 新训练、评估、配对实验与 W-summary 验证 CLI 的 `--help` 通过。
- 历史配对产物的临时副本通过源码 hash、初始化、batch/RNG/LR、
  checkpoint 选择和指标核验；历史日志、manifest 和报告未修改。
- 全部 YAML 逐字节未变；公开函数类型标注、docstring 和编译检查通过。
- 用户已有 AGENTS.md/EXPERIMENT_LOG.md 修改和 data/raw 链接保持原样。

## 当前调用与范围限制

```bash
python train.py --config configs/exp_tuning_mono_weight.yaml
python evaluate.py --config configs/exp_tuning_mono_weight.yaml \
  --checkpoint checkpoints/exp_tuning_mono_weight/best_model.pt
```

实验脚本调用 `train_model(model, train_loader, val_loader, config, device)`；
配对实验另传 `audit_dir=run_dir`，复用同一训练流程。
本步骤只支持三种已有模型，不提前新增未来模型的训练分支。

未验证 GPU 或真实数据 train-to-evaluation 全流程。load_checkpoint 仍没有完整恢复
LR scheduler、sampler 或 RNG，不宣称支持精确续训。
DS02 原始数据仍缺失；验证加载其 checkpoint 不代表真实 DS02 训练复现。
tests/ 按批准的第五步安排统一删除。

下一步待批准：合并 Dataset/Sampler、迁移 preprocess/config，以及整理新运行的产物布局。
