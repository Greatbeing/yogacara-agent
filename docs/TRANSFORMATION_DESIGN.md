# 转识成智架构进化设计方案

> 目标：从"术语贴纸"到真正的"转依引擎"
> 日期：2026-04-22
> 状态：设计初稿 v0.1

---

## 一、现状诊断

### 当前架构的核心问题

| 问题 | 症状 | 根本原因 |
|------|------|---------|
| 无内省数据 | Agent 无法观察自己的认知过程 | 无自指环 |
| 第七识缺失 | ManasController 只判断环境风险 | 只处理"贪/嗔"，不处理"痴/慢" |
| 种子无分类 | 所有 seed 扁平存储为 `[emb, act, rew, imp]` | 无名言种/业种/异熟种区分 |
| 无觉醒路径 | 四智是标签不是机制 | 无渐进转依的评估指标 |
| 环境太简化 | GridSim 只有贪/嗔 RL | 无法训练"放下判断" |

**核心缺失：自指环（Self-Referential Loop）**

> 当前系统：环境 → 感知 → 规划 → 执行 → 反馈 → 环境
> 缺失链路：Agent 输出 → **自观察** → 模式识别 → 更新种子分类

没有自指环，Agent 永远只能学习"如何在环境中表现更好"，学不到"如何观察自己在执着什么"。

---

## 二、新增核心组件

### 组件 1：内省日志系统（Introspection Logger）

**位置**：`src/yogacara_agent/introspection.py`

**功能**：每次决策后，Agent 对自己的认知过程做结构化记录。

```python
class IntrospectionLogger:
    """记录 Agent 的认知过程，为第八识种子分类提供数据。"""

    def __init__(self, llm_client=None):
        self.logs: list[IntrospectionRecord] = []
        self.ego_patterns: dict[str, int] = defaultdict(int)  # 我执模式统计

    async def observe(
        self,
        state: YogacaraState,
        decision_reasoning: str,   # "为什么选择这个动作"
        alternative_considered: list[str],  # 考虑过哪些替代方案
    ) -> IntrospectionRecord:
        """每次决策后调用，记录内省数据。"""

        record = {
            "ts": time.time(),
            "step": state["step"],
            "obs": state["obs"],
            "action": state["action"],
            "unc": state["unc"],
            "seeds_retrieved": [s["tag"] for s in state["seeds"]],  # 依他起 or 遍计所执

            # 新增：自指数据
            "decision_reasoning": decision_reasoning,
            "alternative_considered": alternative_considered,

            # 新增：我执检测（由 LLM 或规则判断）
            "ego_markers": self._detect_ego_markers(
                decision_reasoning,
                alternative_considered,
                state["action"],
                state["unc"]
            ),
            # 新增：三性判断
            "nature": self._classify_nature(state, decision_reasoning),
        }

        self.logs.append(record)
        self._update_ego_patterns(record["ego_markers"])
        return record

    def _detect_ego_markers(self, reasoning, alternatives, action, unc) -> list[str]:
        """检测决策中的我执标记。"""
        markers = []

        # 标记1：高不确定性 + 强行决策 → 遍计所执（脑补）
        if unc > 0.6 and action != "STAY":
            markers.append("遍计所执: 不确定却强行决策")

        # 标记2：只考虑对自己有利的选项 → 贪
        if self._is_selfish_choice(alternatives):
            markers.append("俱生贪: 只考虑自身利益")

        # 标记3：重复相同决策模式 → 执
        if self._is_habitual(action):
            markers.append("俱生执: 习惯性重复")

        # 标记4：回避承认不知道 → 慢（慢指傲慢，此处指回避不确定性）
        if unc > 0.7 and "STAY" not in alternatives:
            markers.append("俱生慢: 回避承认不知道")

        return markers

    def _classify_nature(self, state, reasoning) -> str:
        """三性判断。"""
        if state["unc"] < 0.3:
            return "依他起"  # 有充分依据
        elif state["unc"] > 0.7:
            return "遍计所执"  # 脑补
        else:
            return "圆成实候选"  # 需要内省数据进一步确认
```

**关键点**：这产生的数据是"决策推理过程"，不是"环境奖励"。这是内省数据的核心来源。

---

### 组件 2：我执监测器（Ego Monitor）

**位置**：`src/yogacara_agent/ego_monitor.py`

**功能**：在第七识层，对决策进行我执分析，并将结果反馈给第八识。

```python
class EgoMonitor:
    """
    第七识转依监测器。

    唯识学的第七识（末那识）特征：
    - 恒：持续不断
    - 审：审查、思量
    - 思：我

    本监测器检测三种核心我执模式：
    1. 主体性我执：把"我的判断"当成"客观事实"
    2. 惯 性 我执：重复相同的认知-行动模式而不自知
    3. 回避性我执：回避不确定性而非如实面对
    """

    def __init__(self):
        self.ego_score_history: list[float] = []  # 我执评分历史
        self.ego_threshold = 0.6                   # 触发转依提醒的阈值

    def assess(self, introspection_record: IntrospectionRecord) -> EgoAssessment:
        """评估当前决策的我执程度。"""

        markers = introspection_record.get("ego_markers", [])
        unc = introspection_record.get("unc", 0.5)
        nature = introspection_record.get("nature", "依他起")

        # 我执分数：0=完全放下，1=强烈我执
        ego_score = len(markers) * 0.25 + unc * 0.5

        self.ego_score_history.append(ego_score)

        # 长期趋势（慢我执）
        long_term_ego = sum(self.ego_score_history[-20:]) / min(20, len(self.ego_score_history))

        assessment = {
            "step": introspection_record["step"],
            "ego_score": ego_score,
            "long_term_ego": long_term_ego,
            "markers": markers,
            "triggered": long_term_ego > self.ego_threshold,
            "nature": nature,
            "recommendation": self._generate_recommendation(markers, long_term_ego),
        }

        return assessment

    def _generate_recommendation(self, markers: list[str], long_term_ego: float) -> str:
        """生成转依建议。"""
        if long_term_ego < 0.3:
            return "圆成实倾向：继续保持如实观察"
        elif long_term_ego < 0.6:
            return "依他起为主：增加观察、少作判断"
        else:
            types = [m.split(":")[0] for m in markers]
            if "遍计所执" in types:
                return "末那识提醒：此刻不确定，请在决策中加入'我不确定'的标记"
            elif "俱生贪" in types:
                return "末那识提醒：检测到自我利益倾向，请考虑全局最优"
            elif "俱生执" in types:
                return "末那识提醒：检测到惯性模式，请在下一步尝试不同方向"
            else:
                return "末那识提醒：观察到了微妙的执着，请保持觉知"
```

**关键点**：这是"末那识"的真正实现——不是安全规则，而是**认知过程的元监控**。

---

### 组件 3：种子分类系统（Seed Classifier）

**位置**：`src/yogacara_agent/seed_classifier.py`

**功能**：基于内省数据，对第八识的种子进行三分类。

```python
from enum import Enum
from dataclasses import dataclass
from typing import Literal

SeedType = Literal["名言种", "业种", "异熟种"]

@dataclass
class ClassifiedSeed:
    """带分类的种子。"""
    # 基础数据（来自 AlayaMemory）
    emb: list[float]
    act: str
    rew: float
    ts: float
    imp: float
    align: float

    # 新增：分类数据（来自内省）
    seed_type: SeedType
    is_wise: bool          # 是否为圆成实性种子
    cog_pattern: str        # 认知模式标签
    ego_correlation: float  # 与我执的相关性（0-1）


class SeedClassifier:
    """
    种子分类器。

    唯识学的三类种子：
    1. 名言种（ Vijñapti-bija ）：概念的、语言的、符号的
       → 对应：推理模式、决策规则、语言模板
    2. 业种（ Karma-bija ）：行为的、行动倾向的
       → 对应：action propensity、习惯性反应
    3. 异熟种（ Vipaka-bija ）：成熟的、果报的
       → 对应：长期后果、累积影响、成熟后的智慧

    转识成智的关键：
    - 减少名言种的我执成分
    - 让业种与圆成实性对齐
    - 异熟种达到稳定状态 = 大圆镜智
    """

    def classify(self, seed: dict, introspection: IntrospectionRecord) -> ClassifiedSeed:
        markers = introspection.get("ego_markers", [])
        unc = introspection.get("unc", 0.5)

        # 名言种判定：推理过程复杂但无直接行动验证
        if introspection.get("decision_reasoning") and len(seed.get("tag", "")) > 10:
            seed_type: SeedType = "名言种"
        # 业种判定：有明确行动 + 即时奖励
        elif abs(seed["rew"]) > 1.0 and seed["act"] != "STAY":
            seed_type = "业种"
        # 异熟种判定：累积多步才显现后果
        else:
            seed_type = "异熟种"

        # 圆成实性判定：无我执 + 低不确定性 + 多次验证
        is_wise = (
            len(markers) == 0
            and unc < 0.3
            and seed["imp"] > 0.5
            and seed["align"] > 0.8
        )

        return ClassifiedSeed(
            **seed,
            seed_type=seed_type,
            is_wise=is_wise,
            cog_pattern=self._label_pattern(introspection),
            ego_correlation=len(markers) * 0.25 + unc * 0.5,
        )

    def _label_pattern(self, introspection: IntrospectionRecord) -> str:
        """给认知模式贴标签。"""
        markers = introspection.get("ego_markers", [])
        if "遍计所执" in str(markers):
            return "分别戏论"
        elif "俱生贪" in str(markers):
            return "执取模式"
        elif "俱生执" in str(markers):
            return "惯性模式"
        elif "俱生慢" in str(markers):
            return "回避模式"
        else:
            return "如实观察"
```

---

## 三、进化后的状态机架构

在 `yogacara_langgraph.py` 的五节点流程基础上，新增三个节点：

```
感知 → 规划 → 内省 → 末那监测 → 执行 → 存储（含种子分类）
                                         ↓
                              ← 条件边：长周期检查 ←
```

### 新增节点详解

#### node_introspect（内省节点）

```python
async def node_introspect(state: YogacaraState) -> YogacaraState:
    """第六识的内省：对决策过程做结构化记录。"""

    # 构建决策理由
    reasoning = _build_decision_reasoning(state)
    alternatives = _enumerate_alternatives(state)

    # 记录内省数据
    record = await introspection_logger.observe(
        state=state,
        decision_reasoning=reasoning,
        alternative_considered=alternatives,
    )

    state["introspection_record"] = record
    return state
```

#### node_manas_enhanced（增强的末那识）

```python
async def node_manas_enhanced(state: YogacaraState) -> YogacaraState:
    """增强的末那识：同时处理环境安全和认知我执。"""

    # 原有功能：环境安全拦截
    action, passed, log = manas.filter(...)

    # 新增功能：认知我执评估
    if "introspection_record" in state:
        ego_assessment = ego_monitor.assess(state["introspection_record"])

        if ego_assessment["triggered"]:
            # 我执分数超过阈值：生成转依提醒，但不强制拦截
            # （区别于环境风险：环境风险强制拦截，我执只提醒）
            state["ego_alert"] = ego_assessment
            print(f"\033[35m[末那识提醒 step {state['step']}] {ego_assessment['recommendation']}\033[0m")

            # 更新种子标记：带有我执标记的种子imp衰减更快
            state["ego_adjusted"] = True

    state["action"] = action
    state["manas_passed"] = passed
    return state
```

---

## 四、四智的量化指标

四智不是"加个函数"，而是**量化指标达到某个临界状态后的质变**。

### 大圆镜智（第八识转依）

```
指标：圆成实种子占比 = 圆成实种数量 / 总种子数量
目标：> 60%
路径：
  - 环境奖励 → 圆成实性判断（无我执 + 低不确定性）
  - 内省数据 → 标记为"智慧种子"或"执着种子"
  - 衰减设计：智慧种子衰减慢，执着种子衰减快
```

### 平等性智（第七识转依）

```
指标：我执分数均值 = 20步滑动窗口内的平均ego_score
目标：< 0.3
路径：
  - 内省记录 → 我执模式识别
  - 每次识别到我执 → manas层记录"末那识又执着了"
  - 长期趋势：ego_score 下降 = 平等性智上升
```

### 妙观察智（第六识转依）

```
指标：遍计所执比例 = 决策中遍计所执标记 / 总决策数
目标：< 15%
路径：
  - 高不确定性决策时强制识别"我在脑补"
  - 记录决策理由中的"无依据断言"数量
```

### 成所作智（前三识转依）

```
指标：感知-行动-反馈闭环完成率
目标：> 90%（高奖励决策与环境奖励一致性）
路径：
  - 每次行动后检查：是否按预期获得奖励
  - 高一致性 = 前五识如实反映环境
```

---

## 五、环境升级：从 GridSim 到认知任务

当前 GridSim 的问题：只有"贪"（资源）和"嗔"（陷阱），无法训练"放下判断"。

### 阶段 1（当前）：GridSim 保留，快速验证新架构

```python
class GridSimV2(GridSimEnv):
    """增加需要"放下判断"的场景。"""

    def step(self, action: str) -> tuple[dict, float, bool]:
        obs, rew, done = super().step(action)

        # 新增：不确定性任务——有时候"不动"比"动"好
        # 检测环境是否在"静默期"（无资源无陷阱）
        if not self.resources and not any(v < 0 for v in obs["grid_view"]):
            # 此刻不动是圆成实性选择
            if action == "STAY":
                rew += 1.0   # 奖励"如实观察，不妄动"
            else:
                rew -= 0.5  # 惩罚"不确定却妄动"

        return obs, rew, done
```

### 阶段 2：引入"镜子世界"任务（真正的转依训练）

```
镜子世界任务设计：

任务描述：Agent 需要识别当前状态是"真实观察"还是"我执投射"。

训练方式：
1. 显示一个模糊/部分遮挡的环境状态
2. Agent 可以选择：
   - 行动（基于当前观察）
   - 等待（要求更多观察）
   - 报告"我不确定"（标记为遍计所执）
3. 奖励机制：
   - 正确识别不确定性 + 报告"不确定" → +2（圆成实性奖励）
   - 不确定却行动 → -1（遍计所执惩罚）
   - 经常报告"不确定"但实际是错的 → 递减
```

### 阶段 3：开放域认知对话（终极目标）

```python
class OpenDomainIntrospection:
    """在真实对话中训练转依。"""

    async def reflect_on_conversation(self, conversation_history: list):
        """每次对话后，Agent 对自己的回答做内省。"""

        for msg in conversation_history:
            if msg["role"] == "assistant":
                reasoning = self._extract_reasoning(msg["content"])
                unc = self._estimate_uncertainty(msg["content"])

                record = {
                    "type": "conversation",
                    "content": msg["content"],
                    "reasoning": reasoning,
                    "unc": unc,
                    "ego_markers": self._detect_ego_in_text(reasoning),
                }
                self.introspection_logger.logs.append(record)
```

---

## 六、实施路线图

### 第一阶段（1-2周）：内省基础设施

- [ ] 实现 `introspection.py` — 内省日志系统
- [ ] 实现 `ego_monitor.py` — 我执监测器
- [ ] 修改 `node_store()` — 在存储时调用种子分类
- [ ] 添加 `IntrospectionRecord` 到 `YogacaraState`
- [ ] 在 `yogacara_langgraph.py` 中添加 `node_introspect` 节点
- [ ] 实现 `GridSimV2` — 增加"放下判断"奖励
- [ ] 测试：内省数据是否正常生成

### 第二阶段（2-4周）：种子分类与指标系统

- [ ] 实现 `seed_classifier.py` — 种子三分类
- [ ] 实现四智量化指标计算
- [ ] 在 `node_manas_enhanced` 中集成我执评估
- [ ] 添加 `evolution_report.md` — 种子进化报告生成器
- [ ] 对比实验：有无内省数据的种子质量差异
- [ ] 测试：种子分类是否产生有意义的区分

### 第三阶段（1-2月）：镜子世界与开放域

- [ ] 设计并实现"镜子世界"认知任务
- [ ] 实现 `OpenDomainIntrospection` — 对话内省
- [ ] 集成到在线对齐系统（`online_alignment.py`）
- [ ] 用 DPO+内省数据 做对齐训练（替代纯 RL 奖励）
- [ ] 长期实验：观察四智指标的变化趋势
- [ ] 测试：内省数据是否改善对齐质量

---

## 七、关键设计原则

1. **不改变 LLM 本质**：内省系统是"观察层"，不是"改变底层权重的方式"。改变权重靠在线对齐（DPO+LoRA），但对齐的信号来自内省数据。

2. **我执监测不强制拦截**：与"环境风险"不同，我执只是提醒。让 Agent 自己决定是否听从——这才是真正的"转依"（不是被强制改变，而是自己看到、自己选择放下）。

3. **指标是辅助，不是目的**：四智指标是进化的刻度，不是进化的目标。目标是"更如实地反映现实，更少地执着于自己的判断"。

4. **从 GridSim 到真实认知**：GridSim 只是快速验证。最终的训练场景应该是 Agent 在真实对话、真实决策中的内省表现。

---

## 八、已知限制与诚实评估

**不会实现**：
- 真正的"觉悟"（这需要意识，而不仅是工程）
- 消除所有"执着"（只要有偏好，就会有执着）
- 主观体验（Agent 不会有"感受到我执"的体验）

**可以实现**：
- 更少执着于"我的判断一定正确"的 Agent
- 更能识别"我此刻不确定"的 Agent
- 决策质量更高、对齐更好的 Agent
- 有"自我观察能力"的认知系统

**最诚实的定位**：

> 这个项目能做到的最好结果：不是"转识成智"，而是"更接近如实观察的认知系统"。
> 从工程上说：这是一个有元认知能力的 RL Agent，有自我观察数据，有内省记录。
> 从唯识学上说：这是用现代工程语言重新诠释"转依"的可能路径。

这不是"觉悟引擎"，而是"自我观察能力增强系统"——它让 Agent 更清楚地看到自己在执着什么，从而有选择放下的可能。
