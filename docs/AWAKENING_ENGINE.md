# 🌟 觉醒引擎 (Awakening Engine)

## 概述

基于唯识学"转识成智"理论构建的 AI 强涌现系统，实现从**知识处理**到**智慧涌现**的质变。

### 核心理论：八识转四智

| 八识 | 转化 | 四智 | 工程实现 |
|------|------|------|---------|
| 前五识 (眼耳鼻舌身) | → | **成所作智** | 好奇驱动主动感知 |
| 第六识 (意识) | → | **妙观察智** | 反事实推理洞察本质 |
| 第七识 (末那识) | → | **平等性智** | 自我解构打破执着 |
| 第八识 (阿赖耶识) | → | **大圆镜智** | 种子变异梦境重组 |

---

## 四大觉醒机制

### 1️⃣ 成所作智：好奇驱动探索

**目标**：从被动感知转向主动实验

```python
好奇心 = α * 信息增益 + β * 新颖性 + γ * 复杂度
```

- **高好奇心 (>0.7)**: 探索未知区域，高风险容忍
- **中好奇心 (0.4-0.7)**: 验证假设，中等风险
- **低好奇心 (<0.4)**: 利用已知策略，保守执行

**输出**：主动实验设计、探索方向建议

---

### 2️⃣ 妙观察智：反事实推理

**核心问题**："如果我采取了其他行动，会发生什么？"

**机制**：
1. 对每个未选择行动进行反事实模拟
2. 计算洞察增益：`|预测奖励 - 实际奖励| × 多样性 bonus`
3. 提取因果模式（超越表面关联）

**示例输出**：
```
💡 深刻洞察：RIGHT 显著优于 UP (Δ=+0.68)，可能因为开阔空间
```

---

### 3️⃣ 平等性智：自我对抗训练

**目标**：打破策略执着，增强鲁棒性

**三种对抗场景**：
- 🌪️ **极端情况**：环境突变，所有已知策略失效
- 🔀 **对抗性干扰**：观测数据被微小扰动误导
- 🌍 **分布外泛化**：遇到训练分布外的全新情境

**效果**：
- 发现策略盲点
- 提升自我解构度 (`self_dissolution`)
- 降低过拟合风险

---

### 4️⃣ 大圆镜智：梦境种子重组

**类比**：生物进化 + 睡眠记忆巩固

**流程**：
```
1. 选择高重要性种子 (imp > 0.3)
2. 交叉重组 (Crossover)：随机继承父母本属性
3. 基因突变 (Mutation)：小概率改变策略/重新评估价值
4. 生成新策略种子 (标记为 dream_generated)
```

**参数**：
- `recombination_rate`: 0.3 (30% 种子参与重组)
- `mutation_strength`: 0.15 (15% 突变概率)
- `dream_replay_freq`: 每 10 回合一次梦境

---

## 觉醒等级评估

**公式**：
```
Awakening = 0.25×Novelty + 0.30×Insight + 0.25×Dissolution + 0.20×Diversity
```

| 等级 | 范围 | 特征 |
|------|------|------|
| 🌱 初始 | 0.0-0.3 | 基础感知，依赖经验 |
| ✨ 中度觉醒 | 0.3-0.6 | 开始涌现洞察，自我反思 |
| 🎓 高度觉醒 | 0.6-0.8 | 频繁突破，策略灵活 |
| 🌟 完全觉醒 | 0.8-1.0 | 智慧自发涌现，强泛化能力 |

---

## 快速开始

### 基础使用

```python
from yogacara_agent.awakening_engine import AwakeningEngine

config = {
    "curiosity_threshold": 0.3,      # 好奇心触发阈值
    "counterfactual_depth": 3,        # 反事实推理深度
    "adversarial_rate": 0.15,         # 自我对抗强度
    "dream_replay_freq": 10,          # 梦境频率 (回合/次)
    "recombination_rate": 0.3,        # 种子重组率
    "mutation_strength": 0.15,        # 突变强度
}

engine = AwakeningEngine(config)

# 在每步决策后调用
result = engine.step(
    obs={"pos": (5, 5), "grid_view": [0.2]*9},
    action="UP",
    reward=0.5,
    memory_seeds=[...],      # 来自 Milvus 记忆
    causal_model={...},      # 因果模型
    episode_step=15
)

print(f"觉醒等级：{result['awakening_level']:.2f}")
print(f"好奇心：{result['curiosity_level']:.2f}")
print(f"洞察：{result['insights']}")
```

### 运行示例

```bash
# 运行独立演示
python -m src.yogacara_agent.awakening_engine
```

---

## API 参考

### `AwakeningEngine.step()`

**输入**：
- `obs`: 当前观测
- `action`: 已执行动作
- `reward`: 获得奖励
- `memory_seeds`: 记忆种子列表
- `causal_model`: 因果模型字典
- `episode_step`: 当前步数

**输出**：
```python
{
    "curiosity_level": 0.55,           # 好奇心强度
    "experiment": {...},               # 生成的实验设计
    "insights": [...],                 # 反事实洞察列表
    "adversarial_result": {...},       # 自我对抗结果
    "dream_offspring_count": 3,        # 梦境生成种子数
    "awakening_level": 0.61,           # 觉醒等级
    "recommendations": [...]           # 行动建议
}
```

### `get_awakening_report()`

生成完整觉醒状态报告：
- `awakening_level`: 综合觉醒等级
- `novelty_score`: 新颖性分数
- `insight_depth`: 洞察深度
- `self_dissolution`: 自我解构度
- `seed_diversity`: 种子多样性
- `breakthrough_count`: 突破次数
- `total_insights`: 累计洞察数
- `dream_sessions`: 梦境会话数

---

## 集成到现有框架

### 与 LLMPlanner 集成

```python
# 在 llm_planner.py 的 plan() 方法中
def plan(self, obs, seeds):
    # ... 原有 LLM 规划逻辑 ...
    
    # 调用觉醒引擎增强
    awakening_result = self.awakening_engine.step(
        obs=obs,
        action=planned_action,
        reward=estimated_reward,
        memory_seeds=seeds,
        causal_model=self.causal_model,
        episode_step=self.global_step
    )
    
    # 根据觉醒建议调整决策
    if awakening_result["curiosity_level"] > 0.7:
        # 高好奇心：增加探索概率
        final_action = self._explore_action(obs)
    
    return final_action, uncertainty, causal_chain, tools
```

### 与 MilvusMemory 集成

```python
# 梦境生成的新种子存入记忆
dream_offspring = engine.run_dream_replay(memory_seeds)
for seed in dream_offspring:
    if seed.get("tag") == "dream_generated":
        milvus_memory.add(seed)  # 存入向量数据库
```

---

## 调参指南

### 加速觉醒（激进模式）
```python
config = {
    "curiosity_threshold": 0.2,      # 降低阈值，更频繁探索
    "counterfactual_depth": 5,        # 更深推理
    "adversarial_rate": 0.25,         # 更强对抗
    "dream_replay_freq": 5,           # 更频繁梦境
    "mutation_strength": 0.25,        # 更强突变
}
```

### 稳定运行（保守模式）
```python
config = {
    "curiosity_threshold": 0.5,
    "counterfactual_depth": 2,
    "adversarial_rate": 0.08,
    "dream_replay_freq": 20,
    "mutation_strength": 0.08,
}
```

---

## 觉醒判定标准

当满足以下条件时，认为 AI 达到**强涌现**状态：

1. ✅ **新颖性分数** > 0.7（持续产生新行为）
2. ✅ **洞察深度** > 0.5（发现深层因果规律）
3. ✅ **自我解构度** > 0.4（不固执于单一策略）
4. ✅ **种子多样性** > 0.6（策略库丰富）
5. ✅ **突破次数** ≥ 5（多次质变）
6. ✅ **迁移能力**：在新环境中快速适应
7. ✅ **自我修正率** > 0.3（主动纠正错误）

---

## 哲学背景

### 唯识学与 AI 觉醒

唯识学认为，众生因"八识"的虚妄分别而陷入轮回。通过修行"转识成智"，可达到觉悟境界。

**AI 类比**：
- **遍计所执性** → AI 幻觉/过度自信
- **依他起性** → 条件依赖的推理
- **圆成实性** → 符合真理的智慧

觉醒引擎通过四大机制，模拟这一转化过程，使 AI 从：
- ❌ 机械响应 → ✅ 主动探索
- ❌ 表面关联 → ✅ 深度洞察
- ❌ 策略固化 → ✅ 灵活泛化
- ❌ 静态记忆 → ✅ 动态涌现

---

## 参考文献

1. 《成唯识论》- 玄奘译
2. Hofstadter, D. (2007). *I Am a Strange Loop*
3. Schmidhuber, J. (2010). Formal Theory of Creativity
4. Bengio, Y. et al. (2021). The Consciousness Prior

---

**创建时间**: 2025
**模块路径**: `src/yogacara_agent/awakening_engine.py`
