# 第五步：最终清理、使用文档与依赖

日期：2026-10-14。修改前基线：`4c62682`；全项目数值对照基线：
`ca933c4ef0292e25d60582360fd21e9f657e2fbb`。
本步骤完成五步计划，不继续增加模型、消融或正式训练。

## 清理与文档

- 按用户批准删除整个 tests/：六个空测试、两个有效测试及空包文件。
  删除前最后一次现有测试为 15 passed；删除后不运行空 pytest，也不将零收集当作通过。
- 删除硬编码 DS02 路径的一次性 test.py，清理 tests 字节码、pytest cache 和空 notebooks/。
  原始数据和历史结果目录未参与递归清理。
- 新增 README.md，说明统一入口、独立 YAML、新运行目录、模型差异、数据/评估协议、
  旧 checkpoint 使用、种子记录、现有核验工具和复现范围。
- 新增 requirements.txt，固定本地已验证运行版本；移除测试后不列 pytest 运行依赖。
  pandas cycle-HI HDF 读取所需的 tables 明确列入。
- 原 PROJECT_STATUS/CODE_WALKTHROUGH 原文逐字节存入 docs/history/，
  当前同名文档重写为实际代码说明。
- 设计、旧开发审计和评估验证只添加历史/版本说明，原内容和结论保留。
  REFACTOR_STEP1–4 和 EXPERIMENT_LOG 不改写。

## 最终验证

第五步没有修改任何运行时 Python 实现；全部 14 个模块与第四步提交逐字节一致。
所有模块编译、导入通过；训练、评估及三个 experiments 模块的 --help 通过。
全部十份 config YAML 与第四步逐字节一致。
requirements 的固定版本与当前安装元数据逐项核对，未安装或升级包。
当前 README/状态/走读的导航链接存在；旧路径只留在明确标记的历史资料中。

独立临时脚本从重构前 Git revision 读取 model/baselines/loss 的 Python 源码，
不复制数据、不创建运行时兼容层。与最终实现比较：

- 十份现有配置的初始化 state_dict、参数 keys 和 RNG 消耗一致。
- 全部十二个 eval/train forward 字段一致。
- 每份配置覆盖 warmup、transition 和最终阶段，共四十组 loss/梯度对照。
  比较总损失、组件损失、参数梯度及 x/u/W 输入梯度；rtol=0、atol=0，全部通过。
- 十一个旧模型 checkpoint 严格加载后的固定输入输出一致；读取前后文件 hash 不变。
- 当前 W-summary smoke 通过：旧 decoder 输出、false W 不变性及 W 列零梯度、
  条件路径、checkpoint roundtrip、开关冲突、初始化/采样 RNG 和一百轮 sampler 序列。

临时源码、脚本和结果保留在 /tmp/oc_ivae_step5_reference/，不纳入 Git 或项目框架。
新依赖安装和真实训练没有执行。
本步骤不重复全量 DS03 推理：运行时与第四步相同，第四步已验证 1101 个 flight 的
score CSV、scaler、曲线与六指标一致。GPU、真实 DS02 和正式重新训练仍未验证。

## 验证环境

| 依赖 | 本地版本 |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.5.1+cu121，实际使用 CPU |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.3 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.10.9 |
| tables | 3.11.1 |

requirements 中 Torch 固定发行版本 2.5.1，未规定 CUDA wheel 来源或验证新的干净安装环境。
现有核验脚本不依赖 pytest；正式产物核验仍只覆盖既有 DS03 配对实验的范围。

## 完成状态

模型、训练、评估、数据和配置已采用批准的简化结构。
新运行记录配置/best/history/scores/metrics，拒绝覆盖历史结果。
没有通用 Trainer、registry、配置继承、checkpoint 迁移框架或新增数据格式。
新增对比模型与消融仍需按实际需求单独实施，不属于本次重构。

data/raw 符号链接及目标保持，旧 checkpoints/runs、源码快照和实验记录保持。
用户 AGENTS.md、EXPERIMENT_LOG.md 修改及三份任务文档删除未纳入本步骤提交。
五步变更均仅在本地提交，没有推送。
