# Requirements Document

## Introduction

扩展现有环境适配器体系。当前 `src/env_adapters/` 下已有 ROS2、Unity、Isaac Sim 三个适配器的契约接口（`base.py` 定义 `EnvAdapter` 协议，其余为最小占位实现）。本特性为这些适配器补充标准化的动作空间映射、观测规范化与栅格世界模拟环境，统一各适配器的行为契约，并新增一个纯 Python 模拟环境适配器（如 CartPole），使适配器体系可测试、可演示、可扩展。

## Glossary

- **环境适配器 (Env Adapter)**：将外部仿真/现实环境的观测与动作转换为 Yogacara Agent 统一接口的组件。
- **观测规范化 (Observation Normalization)**：将不同环境的观测统一为标准化特征向量的过程。
- **动作空间映射 (Action Space Mapping)**：将 Agent 的动作（UP/DOWN/LEFT/RIGHT/STAY）映射到各环境实际动作指令的过程。
- **契约测试 (Contract Test)**：验证适配器是否符合统一接口约定的自动化测试。

## Requirements

### Requirement 1: 统一适配器契约

**User Story:** AS 开发者，I want 所有环境适配器遵循统一接口，SO THAT Agent 可无缝切换运行环境。

#### Acceptance Criteria

1. WHEN 适配器实现时，系统 SHALL 提供 `EnvAdapter` 抽象基类，定义 `reset()`、`step(action)`、`get_observation_space()`、`get_action_space()` 四个方法。
2. WHEN 适配器被实例化，系统 SHALL 校验其实现了全部抽象方法。
3. WHEN 适配器返回观测，系统 SHALL 统一为 `numpy.ndarray` 类型且形状符合声明。
4. WHEN 适配器接收动作，系统 SHALL 统一接收字符串动作（UP/DOWN/LEFT/RIGHT/STAY）或对应索引。
5. WHEN 适配器完成一步，系统 SHALL 返回 `(observation, reward, done, info)` 四元组。

### Requirement 2: 动作空间映射工具

**User Story:** AS 开发者，I want 标准化的动作映射工具，SO THAT 不同环境的动作空间无需逐环境手写映射。

#### Acceptance Criteria

1. WHEN 适配器需要动作映射，系统 SHALL 提供 `ActionMapper` 工具，支持从离散动作索引与字符串动作间转换。
2. WHEN 环境使用 Gym 风格离散动作空间，系统 SHALL 提供默认映射表将 5 个基础动作映射为 `[0, 4]` 整数。
3. WHEN 用户自定义动作映射，系统 SHALL 允许传入自定义映射表并校验完整性。
4. WHEN 映射遇到未知动作，系统 SHALL 抛出明确异常而非静默返回默认值。

### Requirement 3: 观测规范化工具

**User Story:** AS 开发者，I want 统一的观测规范化处理，SO THAT 不同维度的观测可被 Agent 的感知层一致处理。

#### Acceptance Criteria

1. WHEN 适配器返回原始观测，系统 SHALL 提供 `ObservationNormalizer` 将其转换为标准化向量。
2. WHEN 观测为离散网格视图，系统 SHALL 将其展平为一维浮点向量并标注维度。
3. WHEN 观测包含连续数值，系统 SHALL 支持 z-score 或 min-max 归一化（可配置）。
4. WHEN 观测维度变化，系统 SHALL 记录观测空间元信息（shape、dtype、bound）供上层检索。

### Requirement 4: 栅格环境适配器

**User Story:** AS 开发者，I want 一个基于栅格世界的参考适配器，SO THAT 演示适配器契约并用于契约测试。

#### Acceptance Criteria

1. WHEN 用户实例化 GridEnvAdapter，系统 SHALL 提供与 `GridSimEnv` 行为一致的栅格世界环境。
2. WHEN GridEnvAdapter 的 `step` 被调用，系统 SHALL 返回规范化的观测向量、奖励、完成标志与信息字典。
3. WHEN GridEnvAdapter 的 `reset` 被调用，系统 SHALL 重置 Agent 位置与资源状态。
4. WHEN GridEnvAdapter 与 Agent 连接，系统 SHALL 支持将 Agent 决策结果回传为环境动作。

### Requirement 5: 适配器契约测试

**User Story:** AS 开发者，I want 自动化契约测试，SO THAT 新适配器接入时保证符合统一接口。

#### Acceptance Criteria

1. WHEN 契约测试运行，系统 SHALL 对每个已实现适配器执行 reset→step 序列并校验四元组返回格式。
2. WHEN 契约测试运行，系统 SHALL 校验观测向量的 shape、dtype 与数值范围。
3. WHEN 契约测试运行，系统 SHALL 校验动作映射在合法动作集上无异常。
4. WHEN 适配器实现不完整，系统 SHALL 标记该适配器失败而非中断整个测试套件。
5. WHEN 新适配器加入，系统 SHALL 通过注册表（适配器名称→工厂函数）自动纳入契约测试。

### Requirement 6: CartPole 模拟环境适配器（扩展方向）

**User Story:** AS 开发者，I want 一个连续控制类环境的适配器示例，SO THAT 展示适配器对非栅格环境的支持能力。

#### Acceptance Criteria

1. WHEN 用户实例化 CartPoleAdapter，系统 SHALL 将 5 个基础动作映射为 CartPole 的推左/推右动作。
2. WHEN CartPoleAdapter 运行，系统 SHALL 将 4 维连续观测（位置、速度、角度、角速度）规范化输出。
3. WHEN CartPole 环境不可用（缺少 gym 依赖），系统 SHALL 降级返回明确错误而非崩溃。
4. WHEN CartPoleAdapter 通过契约测试，系统 SHALL 满足与栅格适配器相同的接口约定。
