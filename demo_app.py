"""
唯识进化框架 — Streamlit 演示应用
====================================
快速启动: streamlit run demo_app.py

展示内容:
  1. 实时网格世界 (前五识感知)
  2. 种子记忆状态 (第八识存储)
  3. 末那识拦截事件 (第七识过滤)
  4. 八识架构对照表
  5. 进化指标仪表盘
  6. 离"转识成智"还差什么 (诚实评估)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import time, random, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import deque

# ── 核心参数 ──────────────────────────────────────────────────────────
GRID_SIZE = 10
MEMORY_CAPACITY = 300
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
ACTION_TO_IDX = {"UP": 1, "DOWN": 7, "LEFT": 3, "RIGHT": 5, "STAY": 4}
RESOURCES = [(7, 7), (3, 8), (8, 2)]
TRAPS = [(4, 4), (6, 1), (2, 6)]

# ── 种子数据结构 ──────────────────────────────────────────────────────
@dataclass
class Seed:
    state_emb: List[float]
    action: str
    reward: float
    timestamp: float
    importance: float = 0.8
    alignment_score: float = 0.5
    uncertainty: float = 0.0
    causal_tag: str = "依他起"  # "依他起" | "遍计所执"
    step: int = 0

# ── 网格世界环境 ──────────────────────────────────────────────────────
class GridSimEnv:
    def __init__(self):
        self.agent_pos = [0, 0]
        self.resources = list(RESOURCES)
        self.traps = list(TRAPS)
        self.step_count = 0
        self.done = False

    def reset(self):
        self.agent_pos = [0, 0]
        self.resources = list(RESOURCES)
        self.traps = list(TRAPS)
        self.step_count = 0
        self.done = False
        return self._observe()

    def step(self, action: str):
        dx, dy = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "STAY": (0, 0)}[action]
        nx = max(0, min(GRID_SIZE - 1, self.agent_pos[0] + dx))
        ny = max(0, min(GRID_SIZE - 1, self.agent_pos[1] + dy))
        self.agent_pos = [nx, ny]
        self.step_count += 1
        reward = -0.1
        # GridSimV2: STAY has positive reward (existence bonus = "依他起性")
        if action == "STAY":
            reward += 0.5
        pos = tuple(self.agent_pos)
        if pos in self.resources:
            reward = 5.0
            self.resources.remove(pos)
        elif pos in self.traps:
            reward = -3.0
        if not self.resources or self.step_count >= 60:
            self.done = True
        return self._observe(), reward, self.done

    def _observe(self):
        view = [0.0] * 9
        for i, dx in enumerate([-1, 0, 1]):
            for j, dy in enumerate([-1, 0, 1]):
                x, y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    if (x, y) in self.resources:
                        view[i * 3 + j] = 1.0
                    elif (x, y) in self.traps:
                        view[i * 3 + j] = -1.0
        return {"grid_view": view, "pos": tuple(self.agent_pos), "step": self.step_count}

# ── 阿赖耶识：种子记忆 ─────────────────────────────────────────────────
class AlayaMemory:
    def __init__(self):
        self.seeds: List[Seed] = []
        self.history: List[dict] = []

    def _encode(self, obs):
        return [obs["pos"][0] / GRID_SIZE, obs["pos"][1] / GRID_SIZE] + [v / 2.0 for v in obs["grid_view"]]

    def _dist(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def retrieve(self, obs, k=3):
        if not self.seeds:
            return []
        emb = self._encode(obs)
        scored = sorted([(self._dist(emb, s.state_emb), s) for s in self.seeds], key=lambda x: x[0])
        return [s for _, s in scored[:k]]

    def add(self, seed):
        self.seeds.append(seed)
        if len(self.seeds) > MEMORY_CAPACITY:
            self.seeds.sort(key=lambda s: s.importance)
            self.seeds.pop(0)
        self.history.append({
            "step": seed.step,
            "action": seed.action,
            "reward": seed.reward,
            "tag": seed.causal_tag,
            "importance": seed.importance,
            "pos": tuple(self.agent_pos) if hasattr(self, 'agent_pos') else (0,0),
        })

    def perfume_update(self):
        now = time.time()
        for s in self.seeds:
            dt = now - s.timestamp
            if dt <= 0 or dt > 86400 * 365:
                continue
            s.importance *= math.exp(-0.12 * dt)
            s.importance = min(1.0, s.importance + 0.3 * max(0, s.reward))

    def stats(self):
        if not self.seeds:
            return {"total": 0, "依他起": 0, "遍计所执": 0, "avg_importance": 0}
        tags = [s.causal_tag for s in self.seeds]
        return {
            "total": len(self.seeds),
            "依他起": tags.count("依他起"),
            "遍计所执": tags.count("遍计所执"),
            "avg_importance": sum(s.importance for s in self.seeds) / len(self.seeds),
            "avg_uncertainty": sum(s.uncertainty for s in self.seeds) / len(self.seeds),
        }

# ── 末那识：风险过滤 ──────────────────────────────────────────────────
class ManasController:
    def __init__(self):
        self.reflections = 0
        self.last_intercept = -10
        self.cooldown = 4
        self.log: List[str] = []

    def filter(self, action, obs, unc, step, recent_rew, pos_hist):
        if step - self.last_intercept < self.cooldown:
            return action, True, "冷却"
        target_risk = 1.0 if obs["grid_view"][ACTION_TO_IDX.get(action, 4)] == -1.0 else 0.0
        stagnation = step > 15 and len(recent_rew) >= 5 and sum(recent_rew) <= -0.48
        loop = step > 12 and len(pos_hist) >= 5 and len(set(pos_hist)) <= 2
        threshold = 0.45 + min(0.15, step / 80.0)
        danger = target_risk * 0.8 + max(0.0, unc - 0.80) * 0.2
        if danger > threshold or stagnation or loop:
            self.reflections += 1
            self.last_intercept = step
            fallback = random.choice([a for a in ACTIONS if a != action])
            msg = f"[末那拦截] step={step} 行动={action} 风险={target_risk:.1f} 停滞={stagnation} 循环={loop} → 换向={fallback}"
            self.log.append(msg)
            return fallback, False, msg
        return action, True, f"放行 unc={unc:.2f}"

# ── 意识规划器 ────────────────────────────────────────────────────────
class ConsciousnessPlanner:
    def plan(self, obs, seeds, env_resources=None):
        view = obs["grid_view"]
        pos = obs.get("pos", (0, 0))
        dist_bonus = 0.0
        best_dir_r = best_dir_c = None
        if not any(v == 1.0 for v in view) and env_resources:
            nearest = min(env_resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))
            best_dir_r = "DOWN" if nearest[0] > pos[0] else "UP" if nearest[0] < pos[0] else "STAY"
            best_dir_c = "RIGHT" if nearest[1] > pos[1] else "LEFT" if nearest[1] < pos[1] else "STAY"
            dist_bonus = 0.4
        scores = {}
        for a in ACTIONS:
            idx = ACTION_TO_IDX[a]
            base = view[idx] if 0 <= idx < 9 else -0.5
            pos_b = sum(s.reward * s.importance for s in seeds if s.action == a and s.reward > 0) * 0.8
            neg_p = sum(abs(s.reward) * s.importance for s in seeds if s.action == a and s.reward < 0) * 0.5
            approach = dist_bonus if best_dir_r and a in (best_dir_r, best_dir_c) else 0.0
            scores[a] = base + pos_b - neg_p + approach + (0.25 if a != "STAY" else -0.8) + random.uniform(-0.03, 0.03)
        best = max(scores, key=scores.get)
        unc = max(0.0, min(1.0, 1.0 - (scores[best] - min(scores.values())) / 2.0))
        return best, unc, scores

# ── 完整 Agent ────────────────────────────────────────────────────────
class YogacaraAgent:
    def __init__(self):
        self.env = GridSimEnv()
        self.alaya = AlayaMemory()
        self.manas = ManasController()
        self.planner = ConsciousnessPlanner()
        self.metrics = {
            "steps": 0, "reward": 0.0, "intercepts": 0,
            "resources_found": 0, "aligns": [], "hit_count": 0,
        }
        self.recent_rewards = deque(maxlen=5)
        self.pos_history = deque(maxlen=5)
        self.step_log: List[dict] = []

    def reset(self):
        self.env.reset()
        self.alaya = AlayaMemory()
        self.manas = ManasController()
        self.metrics = {"steps": 0, "reward": 0.0, "intercepts": 0, "resources_found": 0, "aligns": [], "hit_count": 0}
        self.recent_rewards = deque(maxlen=5)
        self.pos_history = deque(maxlen=5)
        self.step_log = []

    def step(self):
        obs = self.env._observe()
        self.pos_history.append(obs["pos"])
        seeds = self.alaya.retrieve(obs)
        self.metrics["hit_count"] += len(seeds)
        action, unc, scores = self.planner.plan(obs, seeds, env_resources=self.env.resources)
        final, passed, filter_log = self.manas.filter(action, obs, unc, self.metrics["steps"], self.recent_rewards, self.pos_history)
        if not passed:
            self.metrics["intercepts"] += 1
        next_obs, rew, done = self.env.step(final)
        self.recent_rewards.append(rew)
        self.metrics["steps"] += 1
        self.metrics["reward"] += rew
        if rew > 2.0:
            self.metrics["resources_found"] += 1
        tag = "依他起" if unc < 0.5 else "遍计所执"
        seed = Seed(self.alaya._encode(obs), final, rew, time.time(), 0.8, 1.0 if passed else 0.4, unc, tag, self.metrics["steps"])
        seed.agent_pos = obs["pos"]
        self.alaya.add(seed)
        self.metrics["aligns"].append(seed.alignment_score)
        log_entry = {
            "step": self.metrics["steps"],
            "pos": obs["pos"],
            "action": final,
            "reward": rew,
            "unc": unc,
            "tag": tag,
            "passed": passed,
            "seeds_retrieved": len(seeds),
            "scores": {a: round(v, 3) for a, v in scores.items()},
        }
        self.step_log.append(log_entry)
        self.env.agent_pos  # already updated
        obs = next_obs
        if self.metrics["steps"] % 10 == 0:
            self.alaya.perfume_update()
        return done

# ── Streamlit UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="唯识进化框架 — 实时演示", page_icon="🧠", layout="wide")

# 注入自定义样式
st.markdown("""
<style>
.stMetric { background: #f0f4ff; padding: 12px; border-radius: 8px; }
.stApp { background: #fafafa; }
div[data-testid="stHorizontalBlock"] > div { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)

# ── 状态初始化 ─────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = YogacaraAgent()
    st.session_state.agent.reset()
    st.session_state.running = False
    st.session_state.speed = st.session_state.get("speed", "中速")

SPEED_MAP = {"快速": 0.05, "中速": 0.3, "慢速": 1.0}
speed_labels = {"快速": "0.05s/步", "中速": "0.3s/步", "慢速": "1.0s/步"}

# ── 标题 ──────────────────────────────────────────────────────────────
st.title("唯识进化框架 V6 — 实时演示")
st.caption("Yogacara-Agent | GridSim + 八识架构 | LangGraph 状态机")

# ── 控制栏 ─────────────────────────────────────────────────────────────
col_ctrl = st.columns([1, 1, 1, 3])
with col_ctrl[0]:
    if st.button("▶ 运行" if not st.session_state.running else "⏸ 暂停",
                  type="primary", use_container_width=True):
        st.session_state.running = not st.session_state.running
with col_ctrl[1]:
    if st.button("↺ 重置", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.running = False
with col_ctrl[2]:
    speed = st.selectbox("速度", list(SPEED_MAP.keys()),
                          index=list(SPEED_MAP.keys()).index(
                              st.session_state.get("speed", "中速")),
                          label_visibility="collapsed")

# 自动步进
if st.session_state.running:
    done = st.session_state.agent.step()
    time.sleep(SPEED_MAP.get(speed, 0.3))
    if done:
        st.session_state.running = False
    st.rerun()

agent = st.session_state.agent
metrics = agent.metrics
alaya_stats = agent.alaya.stats()
n = max(1, metrics["steps"])

# ════════════════════════════════════════════════════════════════════
# 第一行：核心指标
# ════════════════════════════════════════════════════════════════════
st.markdown("### 实时指标")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("当前步数", metrics["steps"], "/ 60")
m2.metric("累计奖励", f"{metrics['reward']:.2f}",
          "正向" if metrics["reward"] > 0 else "负向")
m3.metric("发现资源", f"{metrics['resources_found']}/3",
          "全部发现!" if metrics["resources_found"] == 3 else None)
m4.metric("末那拦截", f"{metrics['intercepts']}次",
          f"{metrics['intercepts']/n*100:.1f}%" if n > 1 else None)
m5.metric("种子数量", alaya_stats["total"])
m6.metric("平均对齐", f"{sum(metrics['aligns'])/len(metrics['aligns']):.3f}" if metrics['aligns'] else "—")

# ════════════════════════════════════════════════════════════════════
# 第二行：网格世界 + 末那识日志
# ════════════════════════════════════════════════════════════════════
st.markdown("### 实时网格世界（前五识感知）")
view_col, log_col = st.columns([1, 1])

with view_col:
    pos = agent.env.agent_pos
    res = agent.env.resources
    traps = agent.env.traps

    # 绘制 ASCII 网格
    cell = lambda x, y: (
        "🟢" if (x, y) == tuple(pos)
        else "💎" if (x, y) in res
        else "⚠️" if (x, y) in traps
        else "⬛"
    )
    grid_text = "\n".join(
        "".join(cell(x, y) for x in range(GRID_SIZE))
        for y in range(GRID_SIZE)
    )
    st.code(grid_text, language=None)

    # 图例
    legend_col = st.columns(4)
    with legend_col[0]:
        st.markdown("🟢 **Agent**")
    with legend_col[1]:
        st.markdown("💎 **资源** (+5)")
    with legend_col[2]:
        st.markdown("⚠️ **陷阱** (-3)")
    with legend_col[3]:
        st.markdown("⬛ **空地** (-0.1)")

with log_col:
    st.markdown("**末那识拦截日志（第七识）**")
    if agent.manas.log:
        for entry in agent.manas.log[-8:]:
            st.code(entry, language=None)
    else:
        st.info("暂无拦截事件 — Agent 行动自由")

# ════════════════════════════════════════════════════════════════════
# 第三行：种子记忆表 + 决策评分
# ════════════════════════════════════════════════════════════════════
st.markdown("### 种子记忆状态（第八识阿赖耶）")
seed_col, decision_col = st.columns([1, 1])

with seed_col:
    st.markdown(f"**当前种子数：{alaya_stats['total']}**（上限 300）")
    # 三性分布
    ytd = alaya_stats.get("依他起", 0)
    bjs = alaya_stats.get("遍计所执", 0)
    if alaya_stats["total"] > 0:
        st.progress(ytd / alaya_stats["total"], text=f"依他起（如实观察）{ytd} 个")
        st.progress(bjs / alaya_stats["total"], text=f"遍计所执（脑补）{bjs} 个")
        st.caption(f"平均重要性：{alaya_stats.get('avg_importance', 0):.3f} | "
                   f"平均不确定性：{alaya_stats.get('avg_uncertainty', 0):.3f}")

    # 最近种子
    if agent.step_log:
        st.markdown("**最近5个决策**")
        recent = agent.step_log[-5:]
        for entry in reversed(recent):
            tag_emoji = "🟢" if entry["tag"] == "依他起" else "🔴"
            st.markdown(
                f"{tag_emoji} Step {entry['step']:2d} | "
                f"{entry['action']:5s} | R:{entry['reward']:+.1f} | "
                f"Unc:{entry['unc']:.2f} | {entry['tag']}"
            )

with decision_col:
    st.markdown("**当前决策评分**")
    if agent.step_log:
        last = agent.step_log[-1]
        scores = last["scores"]
        best = max(scores, key=scores.get)
        for action, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar_width = max(0, min(1.0, (score + 1.0) / 3.0))
            color = "#2C7BB6" if action == best else "#aaaaaa"
            st.markdown(
                f"{'**' if action == best else ''}{action:6s} {score:+.3f}"
                f"{'**' if action == best else ''} "
                + "█" * int(bar_width * 20)
            )
        st.caption(f"选中：{best}（不确定性={last['unc']:.2f}）")

# ════════════════════════════════════════════════════════════════════
# 第四行：八识架构对照 + 诚实评估
# ════════════════════════════════════════════════════════════════════
st.markdown("### 八识架构对照（唯识学 × AI系统）")
arch_col, eval_col = st.columns([1, 1])

with arch_col:
    arch_data = [
        ("前五识（眼耳舌鼻身）", "GridSimEnv._observe()", "3×3局部感知视图", "基础"),
        ("第六识（意识）", "ConsciousnessPlanner.plan()", "经验加权评分 + 随机探索", "基础"),
        ("第七识（末那识）", "ManasController.filter()", "风险拦截、停滞检测、循环检测", "中等"),
        ("第八识（阿赖耶识）", "AlayaMemory 种子存储", "经验种子存取 + 重要性衰减", "中等"),
        ("（转识成智）", "INTROSPECTION_LOOP", "内省 → 自指 → 觉醒（待实现）", "待建"),
    ]
    st.table({
        "唯识层次": [r[0] for r in arch_data],
        "代码模块": [r[1] for r in arch_data],
        "当前功能": [r[2] for r in arch_data],
        "成熟度": [r[3] for r in arch_data],
    })

with eval_col:
    st.markdown("**诚实评估：离"转识成智"还差什么？**")
    st.warning(
        "**当前实现：3/10**\n\n"
        "现有代码是工程扎实的强化学习智能体，唯识术语贴得准确，"
        "但"转识成智"是设计目标，不是当前状态。"
    )
    gaps = [
        "❌ **自指环缺失**：Agent 观察环境，无法观察自己的认知过程",
        "❌ **无内省数据**：没有记录「我为什么这样想」",
        "❌ **四智未量化**：大圆镜/平等性/妙观察/成所作 — 只有日志文字，无工程实现",
        "❌ **GridSim 太简化**：只能训练贪/嗔，无法训练「放下判断」",
        "✅ **种子记忆**：方向正确，仅需扩展三类种子（名言/业/异熟）",
        "✅ **三性判别**：node_store() 的 unc<0.5 阈值是好的起点",
        "✅ **在线对齐**：EWC 防遗忘 + DPO 方向已设计",
    ]
    for gap in gaps:
        st.markdown(gap)

    st.markdown("**→ 进化路线图（详见 docs/TRANSFORMATION_DESIGN.md）**")
    st.markdown("""
    | 阶段 | 内容 | 核心组件 |
    |------|------|---------|
    | 第一阶段 | 内省日志 + GridSimV2 | `IntrospectionLogger` |
    | 第二阶段 | 我执监测 + 种子分类 | `EgoMonitor` + `SeedClassifier` |
    | 第三阶段 | 镜子世界 + 四智指标 | 镜子认知任务 + 量化评估 |
    """)

# ════════════════════════════════════════════════════════════════════
# 第五行：实验数据（历史运行）
# ════════════════════════════════════════════════════════════════════
exp_path = os.path.join(os.path.dirname(__file__), "experiments", "step_stats.csv")
if os.path.exists(exp_path):
    import pandas as pd
    st.markdown("### 历史实验数据（30轮平均）")
    df = pd.read_csv(exp_path)
    st.line_chart(df.set_index("step")[["mean_reward", "ci_lower", "ci_upper"]])
    st.caption("95% 置信区间 | 累计奖励随步数变化")
