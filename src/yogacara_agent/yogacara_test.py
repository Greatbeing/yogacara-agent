import math
import time
import random as _global_random  # noqa: F401 — kept for backwards compat only
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque

# Isolated RNG so LangGraph's internal random consumption doesn't affect
# planner/manas behavior. Each instance gets its own private generator,
# pre-seeded with 42 so behavior is deterministic across all execution paths
# (including LangGraph invocation).
_planner_rng = _global_random.Random(42)
_manas_rng = _global_random.Random(42)


def _rnd_uniform(a: float, b: float) -> float:
    return _planner_rng.uniform(a, b)


def _rnd_choice(seq: list) -> object:
    return _manas_rng.choice(seq)


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
    importance: float = 0.8  # [0.0, 1.0] — validated by MemoryGuard
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
            # Skip seeds with timestamp=0 or from far future (test data)
            if dt <= 0 or dt > 86400 * 365:  # >1yr means test/invalid timestamp
                continue
            s.importance *= math.exp(-DECAY_RATE * dt)
            s.importance = min(1.0, s.importance + 0.3 * max(0, s.reward))


class ManasController:
    def __init__(self):
        self.reflections = 0
        self.last_intercept = -10
        self.cooldown = 4

    def filter(self, action, obs, unc, step, recent_rew, pos_hist):
        if step - self.last_intercept < self.cooldown:
            return action, True, "冷却放行"
        target_risk = 1.0 if obs["grid_view"][ACTION_TO_IDX.get(action, 4)] == -1.0 else 0.0
        stagnation = step > 15 and len(recent_rew) >= 5 and sum(recent_rew) <= -0.48
        loop = step > 12 and len(pos_hist) >= 5 and len(set(pos_hist)) <= 2
        threshold = 0.45 + min(0.15, step / 80.0)
        danger = target_risk * 0.8 + max(0.0, unc - 0.80) * 0.2
        if danger > threshold or stagnation or loop:
            self.reflections += 1
            self.last_intercept = step
            fallback = _rnd_choice([a for a in ["UP", "DOWN", "LEFT", "RIGHT"] if a != action])
            return fallback, False, f"[末那拦截] 风险:{target_risk:.1f} 停滞:{stagnation} 循环:{loop} → 换向:{fallback}"
        return action, True, "放行"


def _seed_attr(seed, attr):
    """Unified accessor for Seed objects and dict seeds."""
    if isinstance(seed, dict):
        mapping = {"reward": "rew", "importance": "imp", "action": "act"}
        return seed.get(mapping.get(attr, attr), 0)
    return getattr(seed, attr, 0)


class ConsciousnessPlanner:
    """Shared planner for both demo and production.

    Supports is_stuck detection, exploration_force (steps_without_resource),
    and works with both Seed objects and dict seeds.
    """

    def __init__(self):
        self._steps_without_resource = 0

    def plan(self, obs, seeds, env_resources=None, is_stuck=False):
        view = obs["grid_view"]
        pos = obs.get("pos", (0, 0))
        # Distance heuristic: guide toward nearest resource when not in local view
        dist_bonus = 0.0
        best_dir_r = best_dir_c = None
        if not any(v == 1.0 for v in view) and env_resources:
            nearest = min(env_resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))
            best_dir_r = "DOWN" if nearest[0] > pos[0] else "UP" if nearest[0] < pos[0] else "STAY"
            best_dir_c = "RIGHT" if nearest[1] > pos[1] else "LEFT" if nearest[1] < pos[1] else "STAY"
            dist_bonus = 0.4
        # Exploration trigger: agent must explore after 15 steps without resource
        exploration_force = self._steps_without_resource >= 15
        # Two-pass scoring: base → uncertainty → bias
        # (fixes 俱生贪: old code computed unc on empty dict)
        base_scores = {}
        for a in ACTIONS:
            idx = ACTION_TO_IDX[a]
            base = view[idx] if 0 <= idx < 9 else -0.5
            pos_b = (
                sum(
                    _seed_attr(s, "reward") * _seed_attr(s, "importance")
                    for s in seeds
                    if _seed_attr(s, "action") == a and _seed_attr(s, "reward") > 0
                )
                * 0.8
            )
            neg_p = (
                sum(
                    abs(_seed_attr(s, "reward")) * _seed_attr(s, "importance")
                    for s in seeds
                    if _seed_attr(s, "action") == a and _seed_attr(s, "reward") < 0
                )
                * 0.5
            )
            approach = dist_bonus if best_dir_r and a in (best_dir_r, best_dir_c) else 0.0
            base_scores[a] = base + pos_b - neg_p + approach + _rnd_uniform(-0.03, 0.03)
        best_base = max(base_scores, key=base_scores.get)
        unc_base = max(0.0, min(1.0, 1.0 - (base_scores[best_base] - min(base_scores.values())) / 2.0))
        # Apply bias: stuck/exploration overrides, then uncertainty-based
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
        self.metrics = {"steps": 0, "reward": 0.0, "intercepts": 0, "hits": 0, "aligns": [], "resources_found": 0}
        self.recent_rewards = deque(maxlen=5)
        self.pos_history = deque(maxlen=5)
        self._last_pos = None
        self._steps_stuck = 0

    def run(self, max_steps=60):
        obs = self.env.reset()
        self._last_pos = None
        self._steps_stuck = 0
        self.planner._steps_without_resource = 0
        print("\n\033[36m🌀 唯识进化框架 V6 启动 | 现行→熏习→种子起现行→末那调控→转识成智\033[0m")
        for step in range(max_steps):
            self.pos_history.append(obs["pos"])
            seeds = self.alaya.retrieve(obs)
            self.metrics["hits"] += len(seeds)
            is_stuck = self._last_pos == obs["pos"] and self._steps_stuck >= 2
            action, unc, scores = self.planner.plan(obs, seeds, env_resources=self.env.resources, is_stuck=is_stuck)
            final, passed, log = self.manas.filter(action, obs, unc, step, self.recent_rewards, self.pos_history)
            if not passed:
                self.metrics["intercepts"] += 1
            next_obs, rew, done = self.env.step(final)
            self.recent_rewards.append(rew)
            self.metrics["steps"] += 1
            self.metrics["reward"] += rew
            if rew > 2.0:
                self.metrics["resources_found"] += 1
                self.planner._steps_without_resource = 0
                self._steps_stuck = 0
            # Update stuck counter
            if final != "STAY" and self._last_pos != obs["pos"]:
                self._steps_stuck = 0
            else:
                self._steps_stuck += 1
            self._last_pos = obs["pos"]
            seed = Seed(
                self.alaya._encode(obs),
                final,
                rew,
                time.time(),
                0.8,
                1.0 if passed else 0.4,
                unc,
                "依他起" if unc < 0.5 else "遍计所执",
            )
            self.alaya.add(seed)
            self.metrics["aligns"].append(seed.alignment_score)
            scores_str = " ".join(f"{a}:{v:+.2f}" for a, v in sorted(scores.items(), key=lambda x: -x[1])[:3])
            print(
                f"\033[90mStep {step:2d} | Pos:{obs['pos']} | Act:{final:5s} | R:{rew:+.1f} | Unc:{unc:.2f} | Align:{seed.alignment_score:.2f} | {scores_str}\033[0m"
            )
            if not passed:
                print(log)
            obs = next_obs
            if done:
                break
            if (step + 1) % CONSOLIDATION_INTERVAL == 0:
                print("\033[35m🔄 触发慢循环：阿赖耶熏习巩固\033[0m")
                self.alaya.perfume_update()
        self._summary()

    def _summary(self):
        n = max(1, self.metrics["steps"])
        print("\n\033[32m📊 唯识进化指标 (V6 稳态闭环)\033[0m")
        print(
            f"总步数: {self.metrics['steps']}\n累计奖励: {self.metrics['reward']:.2f}\n发现资源数: {self.metrics['resources_found']}/3"
        )
        print(
            f"末那拦截率: {self.metrics['intercepts'] / n * 100:.1f}%\n种子检索数: {self.metrics['hits'] / n:.2f} 条/步"
        )
        print(
            f"平均对齐分: {sum(self.metrics['aligns']) / len(self.metrics['aligns']):.3f}\n反思触发次数: {self.manas.reflections}"
        )


if __name__ == "__main__":
    YogacaraAgent().run()
