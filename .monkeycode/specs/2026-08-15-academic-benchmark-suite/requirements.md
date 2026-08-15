# Requirements Document

## Introduction

构建 Yogacara Agent 的学术基准测试套件。当前 `exp_automator.py` 已具备多轮实验与 95% 置信区间图表能力，但缺少标准化的基准测试协议、可复现的实验配置、结果归档与多维度指标报告。本特性建立一套可命令行驱动、可配置、可复现、可生成学术报告的基准测试套件，支撑"转识成智"效果的定量评估。

## Glossary

- **基准测试 (Benchmark Suite)**：一组标准化的实验协议与可复现的测试命令。
- **Episode**：一次从 reset 到 done 的完整 Agent 运行周期。
- **四智指标**：大圆镜智 / 平等性智 / 妙观察智 / 成所作智的量化得分。
- **Seed 配置 (Seed Config)**：可复现的随机种子，用于控制实验随机性。
- **报告 (Report)**：由实验结果自动生成的学术风格摘要（表格 + 图表 + 结论文本）。

## Requirements

### Requirement 1: 基准测试 CLI

**User Story:** AS 研究人员，I want 通过一条命令行运行完整基准测试，SO THAT 无需手动编写实验脚本。

#### Acceptance Criteria

1. WHEN 用户在终端执行 `yogacara-bench`，系统 SHALL 运行默认基准协议（30 episodes × 60 步）。
2. WHEN 用户提供 `--episodes` 参数，系统 SHALL 按指定 episode 数运行实验。
3. WHEN 用户提供 `--max-steps` 参数，系统 SHALL 按指定每轮最大步数运行实验。
4. WHEN 用户提供 `--seeds` 参数（多个随机种子），系统 SHALL 对每个随机种子运行一组实验并分别报告结果。
5. WHEN 用户提供 `--output` 参数，系统 SHALL 将结果写入指定目录而非默认 `./experiments`。
6. IF 参数不合法（episodes < 1 或 max-steps < 1），系统 SHALL 报错并给出用法说明。

### Requirement 2: 可复现随机性

**User Story:** AS 研究人员，I want 相同的实验配置产生相同的结果，SO THAT 实验可复现、可验证。

#### Acceptance Criteria

1. WHEN 基准测试启动，系统 SHALL 以指定随机种子初始化 `random` 与 `numpy`。
2. WHEN 用户未提供随机种子，系统 SHALL 使用时间戳派生种子并记录该种子到结果文件。
3. WHEN 使用相同种子重跑相同配置，系统 SHALL 产生相同的累计奖励序列。
4. WHEN 每个 episode 运行前，系统 SHALL 重置环境状态以保证 episode 间独立。

### Requirement 3: 多维度指标报告

**User Story:** AS 研究人员，I want 获得覆盖性能、效率与认知品质的多维度指标，SO THAT 全面评估 Agent 表现。

#### Acceptance Criteria

1. WHEN 基准测试完成，系统 SHALL 报告以下性能指标：累计奖励均值与 95% 置信区间、资源发现率、末那拦截率。
2. WHEN 基准测试完成，系统 SHALL 报告以下效率指标：平均每 episode 步数、单位步数累计奖励。
3. WHEN 基准测试完成，系统 SHALL 报告以下认知指标：四智得分、种子库平均重要性、三性分布比例。
4. WHEN 报告生成，系统 SHALL 输出机器可读 JSON 摘要与人类可读文本报告。
5. WHEN 报告生成，系统 SHALL 生成累计奖励随步数的学习曲线图（含 95% CI 带状区间）。

### Requirement 4: 结果归档与历史对比

**User Story:** AS 研究人员，I want 将实验结果归档并对比多次实验，SO THAT 观察框架迭代带来的改进。

#### Acceptance Criteria

1. WHEN 基准测试完成，系统 SHALL 将原始 step 级日志以 CSV 写入输出目录。
2. WHEN 基准测试完成，系统 SHALL 将实验配置与指标摘要以 JSON 写入输出目录。
3. WHEN 输出目录存在历史结果文件，系统 SHALL 生成与最近一次结果的对比表。
4. IF 输出目录无历史结果，系统 SHALL 跳过对比表并在报告中注明"无历史对比"。

### Requirement 5: 报告生成自动化

**User Story:** AS 研究人员，I want 自动生成学术风格图表，SO THAT 可直接用于论文或演示。

#### Acceptance Criteria

1. WHEN 基准测试完成，系统 SHALL 生成累计奖励学习曲线图（PNG + PDF）。
2. WHEN 存在多种子对比，系统 SHALL 生成不同种子间的指标对比条形图。
3. WHEN 报告生成失败（缺少 matplotlib 等依赖），系统 SHALL 输出降级文本报告而非中断退出。
4. WHEN 用户提供 `--no-figures` 参数，系统 SHALL 跳过图表生成仅输出数据文件。

### Requirement 6: 与现有实验模块集成

**User Story:** AS 开发者，I want 基准测试套件复用现有实验自动化能力，SO THAT 避免重复实现。

#### Acceptance Criteria

1. WHEN 基准套件运行，系统 SHALL 复用 `exp_automator.py` 的 episode 运行逻辑与置信区间计算。
2. WHEN 基准套件运行，系统 SHALL 复用现有 `yogacara_langgraph.build_graph` 构建图实例。
3. WHEN 四智指标需要计算，系统 SHALL 复用 `ego_monitor.four_wisdoms_report` 现有实现。
4. WHEN 随机种子控制需要接入 numpy，系统 SHALL 同时设置 `random.seed` 与 `numpy.random.seed`。
