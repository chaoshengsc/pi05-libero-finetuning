# π0.5 在 LIBERO 上的 LoRA 微调

单 GPU 训练与检查点配对评测

[English](README.md) · [实验协议](docs/experiment.md) · [补充结果](docs/results.md) · [数据](results/README.md) · [复现说明](docs/reproduce.md)

## 摘要

本项目研究在单张 RTX A5000 上，将官方预训练策略 `pi05_base` 通过参数高效微调适配至 LIBERO 的表现。实验采用全线性层 LoRA，训练 2,000 次参数更新并保存六个检查点。闭环评测使用固定的任务—初态组合，开发集与留出集的初态互不重叠。开发集成功次数由 800 步的 89/100 增至 2,000 步的 97/100，预训练基线在相同开发协议下为 0/100。首个完整留出评测来自 800 步检查点，结果为 346/400（86.5%）。本仓库提供实验配置、逐回合结果、原始录像与分析代码。

以下结果对应**北京时间 2026 年 11 月 16 日 13:56** 的公开快照。其余留出评测组待完整后纳入。

## 实验设置

![LoRA 微调与检查点配对闭环评测的实验设计](assets/experiment-design.png)

**图 1. 实验设计。**（a）一条训练轨迹产生六个检查点。（b）每个检查点使用相同的开发集和留出集任务—初态组合；两个评测集合覆盖相同的十个任务，初态互不重叠。[PDF](assets/experiment-design.pdf) · [SVG](assets/experiment-design.svg)

**表 1. 训练与评测配置。**

| 项目 | 设置 |
|:--|:--|
| 初始化 | 官方 OpenPI `pi05_base` |
| 微调方式 | 冻结基础权重；全线性层 LoRA，rank 64、alpha 32 |
| 示范数据 | 40 个任务；1,608 条训练、85 条离线验证示范 |
| 优化设置 | 有效批大小 256（8 × 32）；种子 0；2,000 次更新 |
| 硬件 | 单张 NVIDIA RTX A5000 |
| 开发评测 | 10 个 LIBERO-Spatial 任务 × 10 个初态，共 100 回合 |
| 留出评测 | 相同 10 个任务 × 另外 40 个初态，共 400 回合 |
| 成功判定 | 仿真器原生任务条件；最多 220 个策略动作步 |

示范数据划分与仿真初态划分具有不同用途。归一化统计仅由训练示范拟合。在每个评测集合内，六个检查点使用相同初态与固定采样流程。完整参数、版本及运行记录见[实验协议](docs/experiment.md)。

## 实验结果

![六个检查点的开发集成功率与 800 步检查点的逐任务留出成绩](assets/evaluation-results.png)

**图 2. 闭环任务成功率。**（a）开发评测，每个检查点 100 回合，纵轴从 80% 开始；预训练基线 0/100 单列于表 2。（b）800 步检查点的留出评测，每任务 40 回合，柱上数字为成功次数，虚线为总体成功率 86.5%。任务编号见[数据字典](results/README.md#task-dictionary)。[PDF](assets/evaluation-results.pdf) · [SVG](assets/evaluation-results.svg)

**表 2. 固定初态上的成功次数。** 仅报告完整评测组，加粗表示六个训练检查点中最高的开发集观测成绩。

| 检查点 | 开发集（n = 100） | 留出集（n = 400） |
|:--|--:|--:|
| 预训练 `pi05_base` | 0 / 100 | — |
| 800 | 89 / 100 | 346 / 400 |
| 1,000 | 90 / 100 | 待完整结果 |
| 1,250 | 95 / 100 | 待完整结果 |
| 1,500 | 91 / 100 | 待完整结果 |
| 1,750 | 96 / 100 | 待完整结果 |
| 2,000 | **97 / 100** | 待完整结果 |

开发成绩并非随训练步数单调上升。对 1,750 与 2,000 步逐回合配对，可得共同成功 93 次、仅 1,750 步成功 3 次、仅 2,000 步成功 4 次，净差为一个百分点。这是开发集上的描述性比较，尚不能据此确定留出集上的优势。逐任务结果与配对列联表见[补充结果](docs/results.md)。

## 定性结果

![800 步检查点在成功及未成功留出回合上的外部相机关键帧](assets/qualitative-rollouts.png)

**图 3. 策略执行录像。** 任务 00，800 步检查点。（a）初态 02 在第 78 个策略动作步满足成功条件。（b）初态 16 到 220 步上限仍未成功。每行展示该回合的首帧、中间帧与末帧，仅取外部相机视角；两行使用不同初态与时间索引。这些选定示例用于展示行为，成功率由全部评测回合计算。[PDF](assets/qualitative-rollouts.pdf) · [录像与选取规则](docs/demos.md)

[成功回合 MP4](assets/demos/step800-task00-init02.mp4) · [未成功回合 MP4](assets/demos/step800-task00-init16.mp4) · [另一任务示例](docs/demos.md#task-09--on-the-wooden-cabinet)

## 可复现性

公开快照包含 1,100 条逐回合结果。数据核验和配对统计仅依赖 Python 标准库。

```sh
git clone https://github.com/chaoshengsc/pi05-libero-finetuning.git
cd pi05-libero-finetuning
python scripts/analyze_results.py
python scripts/analyze_results.py --pair 1750 2000 --split development
```

| 材料 | 位置 |
|:--|:--|
| 训练设置与数据划分 | [`configs/`](configs/) |
| 逐回合结果与任务统计 | [`results/`](results/README.md) |
| 录像与来源记录 | [`assets/demos/`](docs/demos.md) |
| 数据校验与绘图脚本 | [`scripts/`](scripts/) |
| 环境及复现命令 | [`docs/reproduce.md`](docs/reproduce.md) |

图表由核验后的记录生成，提供 PNG、SVG 和 PDF。当前版本支持复现已公开的数据分析，尚未分发模型权重及完整训练、评测启动程序。

## 局限性

当前结果来自单个训练种子、已知 LIBERO-Spatial 任务与仿真环境。留出初态不能代表新任务或真实机器人的表现。本次快照尚无六个检查点的完整留出排名。800 步留出评测跨越一次内核及显卡驱动更新，重启前完成 352 回合，恢复后补齐 48 回合，详见[运行记录](docs/experiment.md#runtime-history)。

## 基础项目

- [OpenPI](https://github.com/Physical-Intelligence/openpi) 提供预训练策略与模型实现。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 提供仿真任务与初态。
- [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero/tree/9dfa69510ea9e1613fc54112bc706444b686a231) 提供训练示范。

本仓库记录独立开展的微调实验。
