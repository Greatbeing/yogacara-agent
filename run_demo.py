#!/usr/bin/env python3
"""
唯识进化框架 — 终端演示脚本
==================================
解决 Windows GBK 编码问题：
  python run_demo.py

快速体验：
  python run_demo.py --episodes 5 --max-steps 60

Streamlit 可视化（推荐）：
  streamlit run demo_app.py
"""

import sys
import os

# 强制 UTF-8 输出（解决 Windows GBK 问题）
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse, time, random, math
from yogacara_agent import IntrospectionLogger, EgoMonitor, SeedClassifier

# 内省阈值（复制的常量，避免循环导入）
_EQUANIMITY_TARGET = 0.3
_PRAJNA_TARGET = 0.15
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque

# ── 参数解析 ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="唯识进化框架 V6 — 终端演示")
    p.add_argument("-n", "--episodes", type=int, default=1, help="运行轮次数（默认1）")
    p.add_argument("-s", "--max-steps", type=int, default=60, help="每轮最大步数（默认60）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    return p.parse_args()

# ── 核心组件（内联来自 yogacara_test.py） ──────────────────────────────
GRID_SIZE = 10
MEMORY_CAPACITY = 300
CONSOLIDATION_INTERVAL = 10
DECAY_RATE = 0.12
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
ACTION_TO_IDX = {"UP": 1, "DOWN": 7, "LEFT": 3, "RIGHT": 5, "STAY": 4}

@dataclass
class Seed:
    state_emb: List[float]
    action: str
    reward: float
    timestamp: float
    importance: float = 0.8
    alignment_score: float = 0.5
    uncertainty: float = 0.0
    causal_tag: str = "依他起"

class GridSimEnv:
    _INITIAL_RESOURCES = [(7, 7), (3, 8), (8, 2)]
    _TRAPS = [(4, 4), (6, 1), (2, 6)]

    def __init__(self):
        self.agent_pos = [0, 0]
        self.resources = list(self._INITIAL_RESOURCES)
        self.traps = list(self._TRAPS)
        self.step_count = 0
        self.done = False

    def reset(self):
        self.agent_pos = [0, 0]
        self.resources = list(self._INITIAL_RESOURCES)
        self.traps = list(self._TRAPS)
        self.step_count = 0
        self.done = False
        return self._observe()

    def step(self, action: str) -> Tuple[Dict, float, bool]:
        dx, dy = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "STAY": (0, 0)}[action]
        nx = max(0, min(GRID_SIZE - 1, self.agent_pos[0] + dx))
        ny = max(0, min(GRID_SIZE - 1, self.agent_pos[1] + dy))
        self.agent_pos = [nx, ny]
        self.step_count += 1
        reward = -0.1
        # GridSimV2: STAY has positive reward (existence bonus = "依他起性")
        # The agent is rewarded for stillness when uncertain, not punished for inaction
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

    def _observe(self) -> Dict:
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

    def render_ascii(self, pos) -> str:
        lines = []
        for y in range(GRID_SIZE):
            row = ""
            for x in range(GRID_SIZE):
                if (x, y) == tuple(pos):
                    row += "[A]"
                elif (x, y) in self.resources:
                    row += "[R]"
                elif (x, y) in self.traps:
                    row += "[X]"
                else:
                    row += "[ ]"
            lines.append(f"  {y:2d} " + row)
        header = "      " + "   ".join(f"{i:2d}" for i in range(GRID_SIZE))
        return header + "\n" + "\n".join(lines)

class AlayaMemory:
    def __init__(self):
        self.seeds: List[Seed] = []

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

    def perfume_update(self):
        now = time.time()
        for s in self.seeds:
            dt = now - s.timestamp
            if dt <= 0 or dt > 86400 * 365:
                continue
            s.importance *= math.exp(-DECAY_RATE * dt)
            s.importance = min(1.0, s.importance + 0.3 * max(0, s.reward))

    def stats(self):
        if not self.seeds:
            return {"total": 0, "依他起": 0, "遍计所执": 0}
        tags = [s.causal_tag for s in self.seeds]
        return {
            "total": len(self.seeds),
            "依他起": tags.count("依他起"),
            "遍计所执": tags.count("遍计所执"),
        }

class ManasController:
    def __init__(self):
        self.reflections = 0
        self.last_intercept = -10
        self.cooldown = 4

    def filter(self, action, obs, unc, step, recent_rew, pos_hist):
        if step - self.last_intercept < self.cooldown:
            return action, True, ""
        target_risk = 1.0 if obs["grid_view"][ACTION_TO_IDX.get(action, 4)] == -1.0 else 0.0
        stagnation = step > 15 and len(recent_rew) >= 5 and sum(recent_rew) <= -0.48
        loop = step > 12 and len(pos_hist) >= 5 and len(set(pos_hist)) <= 2
        threshold = 0.45 + min(0.15, step / 80.0)
        danger = target_risk * 0.8 + max(0.0, unc - 0.80) * 0.2
        if danger > threshold or stagnation or loop:
            self.reflections += 1
            self.last_intercept = step
            fallback = random.choice([a for a in ACTIONS if a != action])
            return fallback, False, f"[末那拦截] 行动={action} 换向={fallback} 风险={target_risk:.1f} 停滞={stagnation} 循环={loop}"
        return action, True, ""

class ConsciousnessPlanner:
    def plan(self, obs, seeds, env_resources=None, is_stuck=False):
        view = obs["grid_view"]
        pos = obs.get("pos", (0, 0))
        dist_bonus = 0.0
        best_dir_r = best_dir_c = None
        if not any(v == 1.0 for v in view) and env_resources:
            nearest = min(env_resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))
            best_dir_r = "DOWN" if nearest[0] > pos[0] else "UP" if nearest[0] < pos[0] else "STAY"
            best_dir_c = "RIGHT" if nearest[1] > pos[1] else "LEFT" if nearest[1] < pos[1] else "STAY"
            dist_bonus = 0.4
        # Exploration trigger: stuck agent must explore after 15 steps without resource
        exploration_force = (step_count := getattr(self, '_steps_without_resource', 0)) >= 15
        # Two-pass scoring: base → uncertainty → bias (fixes 俱生贪: old code computed unc on empty dict)
        base_scores = {}
        for a in ACTIONS:
            idx = ACTION_TO_IDX[a]
            base = view[idx] if 0 <= idx < 9 else -0.5
            pos_b = sum(s.reward * s.importance for s in seeds if s.action == a and s.reward > 0) * 0.8
            neg_p = sum(abs(s.reward) * s.importance for s in seeds if s.action == a and s.reward < 0) * 0.5
            approach = dist_bonus if best_dir_r and a in (best_dir_r, best_dir_c) else 0.0
            base_scores[a] = base + pos_b - neg_p + approach + random.uniform(-0.03, 0.03)
        best_base = max(base_scores, key=base_scores.get)
        unc_base = max(0.0, min(1.0,
            1.0 - (base_scores[best_base] - min(base_scores.values())) / 2.0))
        scores = {}
        for a in ACTIONS:
            has_bonus = best_dir_r and a in (best_dir_r, best_dir_c)
            if is_stuck:
                bias = -0.8 if a == "STAY" else 0.35
            elif exploration_force:
                bias = -0.8 if a == "STAY" else 0.2
            elif unc_base >= 0.5 and not has_bonus:
                bias = 0.30 if a == "STAY" else -0.35
            elif unc_base < 0.3:
                bias = -0.20 if a == "STAY" else 0.15
            else:
                bias = 0.0
            scores[a] = base_scores[a] + bias
        best = max(scores, key=scores.get)
        unc = max(0.0, min(1.0, 1.0 - (scores[best] - min(scores.values())) / 2.0))
        return best, unc, scores

class YogacaraAgent:
    def __init__(self):
        self.env = GridSimEnv()
        self.alaya = AlayaMemory()
        self.manas = ManasController()
        self.planner = ConsciousnessPlanner()
        self.introspection = IntrospectionLogger()
        self.ego_monitor = EgoMonitor()
        self.seed_classifier = SeedClassifier()
        self.metrics = {"steps": 0, "reward": 0.0, "intercepts": 0, "resources_found": 0, "aligns": []}
        self.recent_rewards = deque(maxlen=5)
        self.pos_history = deque(maxlen=5)
        self._steps_stuck = 0
        self._last_pos = None

    def reset(self):
        self.env = GridSimEnv()
        self.alaya = AlayaMemory()
        self.manas = ManasController()
        self.planner = ConsciousnessPlanner()
        self.introspection = IntrospectionLogger()
        self.ego_monitor = EgoMonitor()
        self.seed_classifier = SeedClassifier()
        self.metrics = {"steps": 0, "reward": 0.0, "intercepts": 0, "resources_found": 0, "aligns": []}
        self.recent_rewards = deque(maxlen=5)
        self.pos_history = deque(maxlen=5)
        self._steps_stuck = 0
        self._last_pos = None

    def run(self, max_steps=60, show_grid=True):
        obs = self.env.reset()
        print()
        print("=" * 60)
        print("  唯识进化框架 V6 — 现行 -> 熏习 -> 种子 -> 现行 -> 末那调控")
        print("=" * 60)
        if show_grid:
            print(self.env.render_ascii(obs["pos"]))
            print()
        for step in range(max_steps):
            self.pos_history.append(obs["pos"])
            seeds = self.alaya.retrieve(obs)
            # Track stuck state: if agent hasn't moved for several steps
            is_stuck = (self._last_pos == obs["pos"] and self._steps_stuck >= 2)
            action, unc, scores = self.planner.plan(obs, seeds, env_resources=self.env.resources, is_stuck=is_stuck)
            final, passed, log = self.manas.filter(action, obs, unc, step, self.recent_rewards, self.pos_history)
            if not passed:
                self.metrics["intercepts"] += 1
                print(f"  [末那识拦截] step={step} {log}")
            next_obs, rew, done = self.env.step(final)
            self.recent_rewards.append(rew)
            self.metrics["steps"] += 1
            self.metrics["reward"] += rew
            if rew > 2.0:
                self.metrics["resources_found"] += 1
                self._steps_stuck = 0
            # Update stuck counter
            if final != "STAY" and self._last_pos != obs["pos"]:
                self._steps_stuck = 0
            else:
                self._steps_stuck += 1
            self._last_pos = obs["pos"]
            tag = "依他起" if unc < 0.5 else "遍计所执"
            seed = Seed(self.alaya._encode(obs), final, rew, time.time(), 0.8, 1.0 if passed else 0.4, unc, tag)
            self.alaya.add(seed)
            self.metrics["aligns"].append(seed.alignment_score)
            # ── 内省 + 我执监测 + 种子分类 ──
            seeds_data = [
                {"rew": s.reward, "action": s.action, "importance": s.importance}
                for s in seeds
            ]
            scores_str = " ".join(f"{a}:{v:+.2f}" for a, v in sorted(scores.items(), key=lambda x: -x[1])[:3])
            reasoning = f"检索{len(seeds)}个种子，评分={scores_str}"
            alternatives = sorted(ACTIONS, key=lambda a: scores.get(a, 0.0), reverse=True)
            record = self.introspection.observe(
                step=step,
                obs=obs,
                action=final,
                unc=unc,
                seeds_retrieved=seeds_data,
                reasoning=reasoning,
                alternatives=alternatives,
                manas_intercepted=not passed,
                score_best=max(scores.values()) if scores else 0.0,
                score_second=sorted(set(scores.values()), reverse=True)[1] if len(scores) > 1 else 0.0,
            )
            ego_assessment = self.ego_monitor.assess(record)
            classification = self.seed_classifier.classify(
                action=final,
                reward=rew,
                unc=unc,
                nature=record.nature,
                ego_markers=record.ego_markers,
                step=step,
                manas_intercepted=not passed,
            )

            # ── 打印行 ──
            nature_icon = {"圆成实": "#", "依他起": "~", "遍计所执": "?"}
            icon = nature_icon.get(record.nature, "-")
            ego_flag = "[EGO]" if ego_assessment.triggered else "    "
            vipaka_flag = "[VIPAKA]" if classification.triggered else "        "
            unc_bar = "#" * int(unc * 10) + "-" * (10 - int(unc * 10))
            print(f"  Step {step:2d} | {icon} {record.nature:8s} | Act:{final:5s} | R:{rew:+5.1f} | Unc:[{unc_bar}] {unc:.2f} | {ego_flag}{vipaka_flag}")

            # 末那识提醒（有我执时）
            if ego_assessment.triggered and step % 5 == 0:
                print(f"    -> 末那识提醒: {ego_assessment.recommendation}")
            obs = next_obs
            if done:
                break
            if (step + 1) % CONSOLIDATION_INTERVAL == 0:
                self.alaya.perfume_update()
                print()
                print(f"  [阿赖耶识] 慢循环巩固 — 种子数:{len(self.alaya.seeds)} "
                      f"| 依他起:{self.alaya.stats()['依他起']} "
                      f"| 遍计所执:{self.alaya.stats()['遍计所执']}")
                print()
        self._print_summary()

    def _print_summary(self):
        n = max(1, self.metrics["steps"])
        stats = self.alaya.stats()
        four_wisdom = self.ego_monitor.four_wisdoms_report()
        vipaka_stats = self.seed_classifier.vipaka.stats
        recent_intro = self.introspection.recent_summary(n=20)
        print()
        print("=" * 60)
        print("  唯识进化指标摘要（内省系统版）")
        print("=" * 60)
        print(f"  总步数        : {self.metrics['steps']}")
        print(f"  累计奖励      : {self.metrics['reward']:.2f}")
        print(f"  发现资源数    : {self.metrics['resources_found']}/3")
        print(f"  末那拦截率    : {self.metrics['intercepts']/n*100:.1f}%")
        print()

        # ── 种子系统 ──
        print("  [种子系统]")
        print(f"    总种子数    : {stats['total']}")
        print(f"    - 依他起    : {stats['依他起']} 个")
        print(f"    - 遍计所执  : {stats['遍计所执']} 个")
        avg_align = sum(self.metrics["aligns"]) / len(self.metrics["aligns"]) if self.metrics["aligns"] else 0
        print(f"    平均对齐分  : {avg_align:.3f}")
        print(f"    末那反思    : {self.manas.reflections} 次")
        print(f"    异熟种统计  : {vipaka_stats}")
        print()

        # ── 内省系统 ──
        print("  [内省系统]")
        print(f"    内省记录数  : {recent_intro['count']}")
        print(f"    平均不确定度: {recent_intro.get('avg_uncertainty', 0):.2f}")
        print(f"    平均决策差距: {recent_intro.get('avg_decision_gap', 0):.3f}")
        print(f"    三性分布    : {recent_intro.get('nature_distribution', {})}")
        print(f"    我执标记总数: {recent_intro.get('ego_markers_total', 0)}")
        print(f"    我执模式    : {recent_intro.get('ego_patterns', {})}")
        print()

        # ── 四智精确指标 ──
        print("  [四智精确指标]")
        for name, data in four_wisdom.items():
            if isinstance(data, dict):
                status = data.get("status", "")
                icon = "OK " if "达标" in status or status == "无" or "待" in status else "!!"
                if name == "大圆镜智":
                    ytd_rate = stats["依他起"] / max(1, stats["total"])
                    print(f"    {icon} {name}: {ytd_rate*100:.1f}% (target >60%) | {status}")
                elif name == "平等性智":
                    raw = data.get("raw_long_term_ego", "?")
                    print(f"    {icon} {name}: 我执均值={raw} (target <{_EQUANIMITY_TARGET}) | {status}")
                elif name == "妙观察智":
                    raw = data.get("raw_prajna_ratio", "?")
                    print(f"    {icon} {name}: 遍计率={raw} (target <{_PRAJNA_TARGET:.0%}) | {status}")
                else:
                    print(f"    {icon} {name}: {status}")
        print()
        print("  [转识成智进度]")
        print(f"    内省系统    : ACTIVE")
        print(f"    我执监测    : ACTIVE (拦截{self.metrics['intercepts']}次)")
        print(f"    种子分类    : ACTIVE (异熟种监测活跃)")
        print("=" * 60)

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           唯识进化框架 — 快速体验脚本                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  选项 A: 本脚本 — 终端输出 + ASCII 可视化              ║")
    print("║  选项 B: streamlit run demo_app.py — 交互式 Web 演示   ║")
    print("║  选项 C: python -m yogacara_agent.yogacara_test        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

def main():
    args = parse_args()
    random.seed(args.seed)
    print_banner()
    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n{'='*60}")
            print(f"  第 {ep+1}/{args.episodes} 轮")
            print(f"{'='*60}")
            random.seed(args.seed + ep)
        agent = YogacaraAgent()
        agent.run(max_steps=args.max_steps, show_grid=(args.episodes == 1))

if __name__ == "__main__":
    main()
