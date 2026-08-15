# Requirements Document

## Introduction

构建 Yogacara Agent 的 Web 实时监控面板与种子演化可视化工具。当前项目已有 Streamlit 演示（demo_app.py）与 pywebview 桌面端（desktop/index.html），但缺少可通过浏览器独立访问的实时监控页面。本特性在现有 FastAPI 服务基础上新增静态页面托管与实时数据端点，实现八识运行状态、四智指标、末那拦截事件与种子演化时间线的可视化，并沉淀为可独立访问的 `/dashboard` 页面。

## Glossary

- **监控面板 (Dashboard)**：浏览器可访问的 Web 页面，实时展示 Agent 运行状态。
- **种子演化 (Seed Evolution)**：阿赖耶种子库中种子的产生、重要性变化、类型分布变化与淘汰过程。
- **四智指标**：大圆镜智 / 平等性智 / 妙观察智 / 成所作智的量化得分。
- **末那拦截事件 (Manas Interception)**：第七识对环境风险的过滤拦截记录。
- **轮询 (Polling)**：前端周期性地向后端请求最新状态的模式，避免引入 WebSocket 复杂度。

## Requirements

### Requirement 1: Dashboard 页面托管

**User Story:** AS 研究人员，I want 在浏览器中打开一个监控页面，SO THAT 无需启动桌面端即可观察 Agent 运行状态。

#### Acceptance Criteria

1. WHEN 用户在浏览器访问 `GET /dashboard`，系统 SHALL 返回包含完整监控 UI 的 HTML 页面。
2. WHEN Dashboard 页面加载完成，系统 SHALL 自动开始按固定间隔轮询后端实时端点。
3. WHEN 后端 API 不可达时，系统 SHALL 在页面显示连接失败提示并持续重试。
4. WHEN 用户刷新页面，系统 SHALL 重新加载最新状态而不产生重复的后台运行任务。

### Requirement 2: 实时状态监控

**User Story:** AS 研究人员，I want 实时看到 Agent 的决策与记忆状态，SO THAT 观察八识架构的在线运行情况。

#### Acceptance Criteria

1. WHEN Dashboard 加载，系统 SHALL 通过 `GET /health` 展示八识记忆系统摘要（种子总数、类型分布、平均重要性、末那拦截次数、慢循环状态）。
2. WHEN Agent 运行中，系统 SHALL 每 2 秒刷新一次网格世界视图、当前步数、累计奖励、不确定性。
3. WHEN 发生末那拦截，系统 SHALL 在事件面板追加一条含拦截原因与步数的记录。
4. WHEN 四智指标可用，系统 SHALL 以进度条与百分比形式展示四项智慧得分。
5. WHILE 无实时事件发生时，系统 SHALL 保持现有数据显示并在 UI 标注数据新鲜度时间戳。

### Requirement 3: 种子演化可视化

**User Story:** AS 研究人员，I want 看到种子库随时间的演化过程，SO THAT 理解阿赖耶识"种子生现行、现行熏种子"的动态机制。

#### Acceptance Criteria

1. WHEN Dashboard 加载，系统 SHALL 通过种子历史快照数据渲染种子库演化时间线。
2. WHEN 新增种子，系统 SHALL 在演化图中体现该种子的产生时刻、类型（名言种/业种/异熟种）与重要性。
3. WHEN 种子重要性因熏习更新而变化，系统 SHALL 在演化图中体现其数值波动。
4. WHEN 种子被淘汰或衰减，系统 SHALL 在演化图中体现其消失，而非保留在活跃集合中。
5. WHEN 种子数量少于阈值，系统 SHALL 在演化图中显示空状态提示而非渲染错误。
6. IF 后端缺少历史快照数据，系统 SHALL 提供手动触发快照的机制并提示用户。

### Requirement 4: Episode 运行控制

**User Story:** AS 研究人员，I want 从 Dashboard 发起一次完整 episode 运行，SO THAT 无需命令行即可测试 Agent。

#### Acceptance Criteria

1. WHEN 用户在 Dashboard 点击"运行 Episode"按钮，系统 SHALL 调用现有 `POST /run_episode` 端点。
2. WHEN Episode 运行完成，系统 SHALL 在页面展示步骤数、累计奖励、末那拦截数、资源发现数与耗时。
3. WHILE Episode 运行中，系统 SHALL 显示运行中状态并禁用重复提交按钮。
4. IF Episode 运行失败，系统 SHALL 展示错误信息并允许用户重新发起。
5. WHEN Episode 完成，系统 SHALL 自动刷新四智指标与记忆统计面板。

### Requirement 5: 种子详情浏览

**User Story:** AS 研究人员，I want 查看种子库中各条种子的详细信息，SO THAT 理解记忆内容与质量。

#### Acceptance Criteria

1. WHEN 用户查看种子列表，系统 SHALL 通过 `GET /memory/seeds` 展示种子类型、位置、动作、奖励、重要性、三性与时间戳。
2. WHEN 用户按种子类型筛选，系统 SHALL 仅展示匹配类型（名言种/业种/异熟种）的种子。
3. WHEN 种子数量超过 100，系统 SHALL 分页展示，每页最多 50 条。

### Requirement 6: 种子演化历史快照存储

**User Story:** AS 系统设计者，I want 周期性记录种子库演化快照，SO THAT 演化可视化有数据支撑。

#### Acceptance Criteria

1. WHEN slow_loop 执行熏习更新，系统 SHALL 记录一次种子库状态快照（含种子数量、类型分布、平均重要性、时间戳）。
2. WHEN 快照累计超过 200 条，系统 SHALL 裁剪最旧快照以控制内存与响应体积。
3. WHEN 用户请求演化数据，系统 SHALL 返回按时间排序的快照序列。
4. WHEN 内存快照为空，系统 SHALL 返回空数组而非报错。
5. WHEN Agent 进程重启，系统 SHALL 从空快照重新开始累积演化历史。
