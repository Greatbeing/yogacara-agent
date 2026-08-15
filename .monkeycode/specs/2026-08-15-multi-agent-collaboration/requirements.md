# Requirements Document

## Introduction

为 Yogacara Agent 增加进程内多智能体协作模式。当前框架为单 Agent 单进程设计：`yogacara_langgraph.py` 使用模块级共享单例（env/alaya/manas），多个请求并发会互相踩踏。本特性引入进程内多 Agent 实例能力，使多个 Agent 在共享 Alaya 种子库的前提下独立运行，通过种子互熏习（种子共享与传播）实现协作式进化，为后续跨进程分布式协作打下基础。

## Glossary

- **协作 Agent (Collaborative Agent)**：在共享种子库基础上独立决策的 Agent 实例。
- **共享种子库 (Shared Alaya)**：多个 Agent 共用的阿赖耶记忆（AlayaMemory 实例）。
- **互熏习 (Mutual Perfuming)**：一个 Agent 产生的种子被另一个 Agent 检索并影响其决策的机制。
- **Agent ID**：标识单个 Agent 实例的唯一字符串。
- **协作标签 (Collaboration Tag)**：种子中标记其来源 Agent 的元数据字段。

## Requirements

### Requirement 1: 进程内多 Agent 会话

**User Story:** AS 开发者，I want 在同一进程内创建多个独立的 Agent 会话，SO THAT 各 Agent 可以独立运行而不互相踩踏环境状态。

#### Acceptance Criteria

1. WHEN 调用方请求创建协作会话，系统 SHALL 返回携带唯一 Agent ID 的会话对象。
2. WHEN 创建多个会话，系统 SHALL 为每个会话提供独立的 env 实例（agent_pos、resources、traps 互不影响）。
3. WHEN 创建多个会话，系统 SHALL 共享同一个 AlayaMemory 实例作为协作记忆层。
4. WHEN Agent 会话运行时，系统 SHALL 保证其决策状态（step、recent_rewards、pos_history）互不干扰。
5. WHEN 会话被释放，系统 SHALL 提供清理接口以回收独立资源。

### Requirement 2: 协作种子互熏习

**User Story:** AS 开发者，I want 一个 Agent 的经验种子能被其他 Agent 检索使用，SO THAT 实现协作式进化。

#### Acceptance Criteria

1. WHEN Agent A 存储种子，系统 SHALL 在种子元数据中记录来源 Agent ID。
2. WHEN Agent B 检索记忆，系统 SHALL 返回共享种子库中包括 Agent A 来源在内的种子。
3. WHEN 检索结果返回，系统 SHALL 在状态中标注每条种子的来源 Agent ID 供展示与统计。
4. WHEN 共享种子被检索使用，系统 SHALL 记录一次跨 Agent 协作事件。
5. WHILE 共享种子库运行，系统 SHALL 保持现有 retrieve 的最近邻语义（按状态嵌入距离取 top-k）。

### Requirement 3: 协作统计与监控

**User Story:** AS 研究人员，I want 观察到多 Agent 之间的协作效应，SO THAT 评估互熏习是否带来整体性能提升。

#### Acceptance Criteria

1. WHEN 协作运行期间，系统 SHALL 统计各 Agent 的累计奖励、末那拦截次数、资源发现数。
2. WHEN 协作运行期间，系统 SHALL 统计跨 Agent 种子检索次数与种子贡献占比。
3. WHEN 用户请求协作摘要，系统 SHALL 返回按 Agent ID 分组的性能表与种子贡献表。
4. WHEN 种子因来源不同 Agent 存在，系统 SHALL 在种子分布统计中按来源 Agent 分类。

### Requirement 4: 协作模式 API

**User Story:** AS 开发者，I want 通过公开 API 驱动多 Agent 协作实验，SO THAT 可集成到基准套件与演示。

#### Acceptance Criteria

1. WHEN 调用方创建协作实验，系统 SHALL 提供 `create_collaborative_session(agent_count, seed=...)` 入口。
2. WHEN 调用方运行协作实验，系统 SHALL 串行或交错执行各 Agent 的 episode 并共享种子库。
3. WHEN 协作实验结束，系统 SHALL 返回汇总指标（各 Agent 性能 + 协作效应度量）。
4. WHEN 协作实验的 Agent 数量超过阈值，系统 SHALL 提供上限保护并明确报错。
5. WHEN 与单 Agent 基线对比，系统 SHALL 提供协作模式与单 Agent 模式的指标对比输出。

### Requirement 5: 协作效应度量

**User Story:** AS 研究人员，I want 定量度量互熏习的协作增益，SO THAT 验证多 Agent 协作是否优于单 Agent 独立演化。

#### Acceptance Criteria

1. WHEN 协作实验完成，系统 SHALL 计算协作模式平均累计奖励与单 Agent 基线平均累计奖励。
2. WHEN 协作实验完成，系统 SHALL 计算协作增益 = (协作均值 - 基线均值) / 基线均值，并在基线为 0 时安全处理。
3. WHEN 协作效应度量完成，系统 SHALL 将协作增益值纳入实验报告。
4. WHEN 多个随机种子下运行协作实验，系统 SHALL 报告协作增益的均值与方差。
