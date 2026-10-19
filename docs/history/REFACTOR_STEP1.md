# 第一步：模型简化与验证

日期：2026-10-14。重构前基线：`ca933c4ef0292e25d60582360fd21e9f657e2fbb`。
本步骤仅完成模型整合与调用方适配；后续评估、训练、数据与清理步骤等待逐步审核。

## 模型布局

| 文件 | 内容 |
| --- | --- |
| `models/oc_ivae.py` | OC-iVAE forward、两个 posterior 分支、FiLM、conditional prior、HI head |
| `models/tcn.py` | causal encoder、non-causal decoder、masked normalization、W summary |
| `models/vae.py` | cVAE/VAE 构造、标准正态先验、无 FiLM 的 posterior 分支 |
| `models/__init__.py` | 三种已有模型的显式条件选择 |

原 `model/` 与 `baselines/` 的 10 个 Python 文件整合为 4 个，共从 884 行减至 737 行。
训练与评估入口共用 `build_model()`，删除两处重复选择逻辑。

删除的内部冗余：自定义 `_PriorOutput` 字典迭代适配、没有项目调用的 `get_hi()`、
未使用常量、刚构造 OC-iVAE 后对其分支类型的冗余检查。
标准正态先验直接返回四个 Tensor，与 conditional prior 一致。
三处相同的 sequence/mask 校验合并；decoder 的 W 均值和标准差在一个函数中计算，
保持原来的运算及 reduction 顺序。

## 保持的行为

- 模型模块属性、参数注册顺序、默认维度、初始化顺序和 RNG 消耗不变。
- baseline 仍先构造 OC-iVAE 再替换组件；VAE 仍保留双分支和 decoder W 路径。
- forward 的 12 个字段、train 采样与 eval posterior mean 语义不变。
- W-summary false 仍仅将 8 个输入置零，投影形状和 checkpoint keys 不变。
- 保留 decoder 固定 14 个传感器输出的现有行为，未修复或扩展该限制。
- 配置、loss、数据处理、训练控制和评估公式未修改；Trainer/inference 仅调整 import。
- 没有移动或改写原始数据、旧 checkpoint、历史 runs、manifest 或实验日志。

历史 W-summary smoke 从指定 Git revision 加载 constants/backbone/decoder，
在局部验证命名空间解析依赖；不需要保留旧 import 路径或引入兼容层。
历史产物核验用其 `code_commit.txt` 中的执行版本读取已删除 decoder 的源码，
仍核对原 manifest SHA256，不替换预期值或跳过检查。
新实验的源码 hash 清单改为覆盖全部四个新模型文件。

## 已执行验证

环境：Python 3.11.15、PyTorch 2.5.1，CPU；数值对照使用 2 个 Torch 线程。

1. 将修改前模型源码冻结在临时目录，分别构造旧/新模型。
   对全部 10 份 YAML，比较初始化 state_dict、参数顺序和构造后 RNG。
2. 固定 seed、输入和 mask，对照全部 12 个 eval/train 输出字段。
   输入为 6 个样本、12 个时间步、14 个传感器，包含 head padding；
   unit/cycle 为两个 unit 各三个间隔 cycle。
3. 每份配置覆盖 epoch 0、warmup 起点、transition 最后一个 epoch 和最终阶段起点。
   对照总损失、各项损失、参数梯度、x/u/W 输入梯度、采样后 RNG、
   梯度裁剪后一次 AdamW 更新及 optimizer state。
4. 额外覆盖三种模型 × 两个 W 开关 × 默认/小型非默认架构，共 12 种构造。
5. 严格加载 9 个旧模型 checkpoint 和配对实验 true/false 的 2 个 checkpoint，
   对照固定输入推理。state keys 分别保持 OC-iVAE 121、cVAE 113、VAE 109；
   读取前后 checkpoint SHA256 不变。

以上所有张量比较通过 `rtol=0, atol=0` 检查。
临时数值对照脚本与结果位于 `/tmp/oc_ivae_step1_reference/`，不加入运行时模块或 Git。

已运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ -v -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python scripts/run_train.py --help
PYTHONDONTWRITEBYTECODE=1 python scripts/run_eval.py --help
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_w_summary.py \
  --work-dir /tmp/oc_ivae_step1_reference/switch_verification \
  --original-revision 4c38401d0a5fd0f2b536c26da93fd60d42d3e6c4
```

现有测试 15 passed；W-summary smoke 全部通过。
另在历史配对实验产物的临时副本上运行 `verify_artifacts()`，
验证原源码 hash、初始化、batch/RNG/LR 审计、最佳 checkpoint 选择及导出指标；通过。
核验脚本产生的报告只写到副本，未覆盖历史报告。
新模型函数的类型标注、docstring 与编译检查通过。

## 范围限制与下一步

本步骤没有运行训练或真实 HDF5 全量推理，没有声称完成实验复现。
数值对照范围是 CPU 合成输入、checkpoint 加载及历史导出数据；未验证 GPU。
DS02 checkpoint 已加载验证，但原始 DS02 数据当前缺失。

`tests/` 按批准计划留到第五步统一删除；本次仅更新必要导入和模型选择 patch 点。
根目录训练/评估入口、utils 简化、配置迁移与 README 均尚未执行。
用户已有 `AGENTS.md`、`EXPERIMENT_LOG.md` 修改保持原样，不纳入本步骤提交。
