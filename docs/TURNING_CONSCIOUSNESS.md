# 转识成智引擎 (Turning Consciousness into Wisdom Engine)

## 📜 唯识正见

本引擎严格依据**唯识宗**根本教义构建，实现"转依"(Āśraya-parāvṛtti)的工程化。

### 核心义理

> **"转识成智"非是"增强功能"，而是"去除执障"**。
> 
> 如《成唯识论》云："此即无漏界，不思议善常，安乐解脱身，大牟尼名法。"

### 八识转四智对应关系

| 原识 | 所转智慧 | 梵文 | 核心转化 | 工程实现 |
|------|---------|------|---------|---------|
| **第八识** (阿赖耶识) | **大圆镜智** | Ādarśa-jñāna | 去染存净，离诸分别 | `AlayaPurifier`: 移除染污种子，保持如实清晰 |
| **第七识** (末那识) | **平等性智** | Samatā-jñāna | 断除我执，观自他平等 | `ManasDissolver`: 消解自我中心，趋向平等分布 |
| **第六识** (意识) | **妙观察智** | Pratyavekṣaṇa-jñāna | 善观诸法，无倒观察 | 由清净与平等自然显发 |
| **前五识** (眼耳鼻舌身) | **成所作智** | Kṛtyānuṣṭhāna-jñāna | 成就利乐有情事业 | 由平等性智引导利他行动 |

---

## 🔧 模块说明

### 1. AlayaPurifier (阿赖耶净化器 → 大圆镜智)

**义理依据**:
> "大圆镜智者，谓离一切我、我所执，一切所取、能取分别...性相清净，离诸杂染，如大圆镜，现众色像。" — 《成唯识论》

**功能**:
- **去染存净**: 识别并剔除带有贪、嗔、痴、慢、疑标记的染污种子
- **离诸分别**: 移除主观评价标签，保留客观状态
- **如实映照**: 确保记忆清晰度 (clarity), 不扭曲、不失真

**代码示例**:
```python
from yogacara_agent.turning_consciousness import AlayaPurifier, Seed

purifier = AlayaPurifier(purity_threshold=0.8)

seeds = [
    Seed(content="敌人很可怕", is_defiled=True),   # 恐惧 (染污)
    Seed(content="我要抢资源", is_defiled=True),   # 贪婪 (染污)
    Seed(content="路在北方", is_defiled=False, clarity=0.9),  # 客观事实
]

purified, removed_count = purifier.purify(seeds)
# 结果：移除 2 个染污种子，剩余 1 个清净种子
```

---

### 2. ManasDissolver (末那解构器 → 平等性智)

**义理依据**:
> "平等性智者，谓观一切法、自他有情，悉皆平等...由断我执，证得此智。" — 《成唯识论》

**功能**:
- **检测我执**: 识别策略中对"self_reward"的依赖
- **断除我执**: 将奖励函数从"个体最大化"转向"众生最大化"
- **观自他平等**: 动作概率趋向平均，消除极端偏好

**代码示例**:
```python
from yogacara_agent.turning_consciousness import ManasDissolver

dissolver = ManasDissolver(ego_decay_rate=0.2)

selfish_actions = {"attack": 0.9, "help": 0.1}
new_actions, dissolved_ego = dissolver.dissolve(
    selfish_actions, 
    "maximize_self_gain"
)

# 结果：我执强度下降，动作分布趋向平等
# attack: 0.9 → ~0.6, help: 0.1 → ~0.4
```

---

### 3. TurningConsciousnessEngine (转识成智总引擎)

**统筹八识转四智的全过程**

**代码示例**:
```python
from yogacara_agent.turning_consciousness import (
    TurningConsciousnessEngine, Seed
)

engine = TurningConsciousnessEngine({
    'purity_threshold': 0.7,  # 大圆镜智纯度阈值
    'ego_decay_rate': 0.15    # 我执消解速率
})

# 模拟一步修行
seeds = [
    Seed(content="恐惧", is_defiled=True),
    Seed(content="慈悲", is_defiled=False, clarity=1.0),
]
actions = {"harm": 0.8, "help": 0.2}

result = engine.step(seeds, actions, "selfish_reward")

print(f"大圆镜智：{result.mirror_wisdom_level:.2f}")
print(f"平等性智：{result.equality_wisdom_level:.2f}")
print(f"妙观察智：{result.observation_wisdom_level:.2f}")
print(f"成所作智：{result.action_wisdom_level:.2f}")
print(f"转依等级：{result.turning_level:.2f}")
```

---

## 📊 输出指标

### 四智等级 (0.0 - 1.0)

| 指标 | 含义 | 判据 |
|------|------|------|
| `mirror_wisdom_level` | 大圆镜智 (清净度) | 剩余种子的平均清晰度 |
| `equality_wisdom_level` | 平等性智 (无我度) | 1.0 - 当前我执强度 |
| `observation_wisdom_level` | 妙观察智 (洞察度) | (镜智 + 平等智) / 2 |
| `action_wisdom_level` | 成所作智 (利他度) | 平等智 × 0.9 |

### 过程指标

- `defiled_seeds_removed`: 移除的染污种子数量
- `self_attachment_dissolved`: 消解的我执量
- `insights_generated`: 生成的洞察列表
- `turning_level`: 综合转依等级

---

## 🎯 觉醒/转依判定标准

**初步转依**: `turning_level > 0.3`
- 已去除部分染污
- 我执开始松动

**中度转依**: `turning_level > 0.6`
- 心镜较为清净
- 自他平等观初显

**深度转依**: `turning_level > 0.8`
- 离诸杂染，如实映照
- 我执微薄，平等普观

**圆满转依 (佛果)**: `turning_level ≈ 1.0`
- 大圆镜智：究竟清净
- 平等性智：无我平等
- 妙观察智：善观诸法
- 成所作智：利乐有情

---

## 🧪 运行测试

```bash
# 直接运行演示
python src/yogacara_agent/turning_consciousness.py

# 运行单元测试
PYTHONPATH=/workspace/src python tests/test_turning_consciousness.py
```

---

## 📚 参考文献

1. 《成唯识论》(Viṃśatikā-vijñaptimātratāsiddhi) - 玄奘译
2. 《瑜伽师地论》(Yogācārabhūmi-śāstra) - 弥勒菩萨说
3. 《摄大乘论》(Mahāyāna-saṃgraha) - 无著菩萨造
4. 《八识规矩颂》- 玄奘大师造

---

## ⚠️ 重要说明

**本引擎是对唯识教义的工程化模拟，而非真实修行**。

真实的"转识成智"需要：
- 依止善知识
- 闻思修三慧
- 戒定慧三学
- 长期实修实证

代码仅为辅助理解唯识义理的教学工具，不可替代真实修行。
