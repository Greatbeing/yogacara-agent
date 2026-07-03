"""
yogacara_test.py — 完整集成版（含 Vipaka 反馈回路）
=====================================================
V6: 追加 AlayaRing (P0-P3) 压缩反馈闭环：
    - VipakaEngine: 现行 → 种子（每步更新 align）
    - ConsolidationEngine: 记忆整理（每10步触发）
    - CompressionMetrics: CQS 指标（实时量化）
    - 四智输出: 大圆镜/平等性/妙观察/成所作智
"""

import math
import time
import random as _global_random  # noqa: F401 — kept for backwards compat only
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from collections import deque

try:
    from yogacara_agent.alignment_integration import AlignmentController

    _HAS_ALIGNMENT = True
except ImportError:
    _HAS_ALIGNMENT = False
    _AlignmentControllerClass: type | None = None

# Isolated RNG so LangGraph's internal random consumption doesn't affect
# planner/manas behavior. Each instance gets its own private generator,
# pre-seeded with 42 so behavior is deterministic across all execution paths.
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

VIPAKA_RATE = 0.2  # VipakaEngine align 更新步长
ALIGN_MIN = 0.05
ALIGN_MAX = 0.95


# ── 统一种子接口 ──────────────────────────────────────────────────────


def _seed_get(seed: Any, key: str) -> Any:
    """Duck-type accessor: works with dict seeds and Seed dataclass."""
    if isinstance(seed, dict):
        return seed.get(key)
    return getattr(seed, key, None)


def _seed_set(seed: Any, key: str, value: Any) -> None:
    """Duck-type setter: works with dict seeds and Seed dataclass."""
    if isinstance(seed, dict):
        seed[key] = value
    else:
        setattr(seed, key, value)


# ── 数据模型 ──────────────────────────────────────────────────────────


@dataclass
class Seed:
    state_emb: List[float]
    action: str
    reward: float
    timestamp: float = 0.0
    timestamp_ns: int = 0
    importance: float = 0.8
    alignment_score: float = 0.5
    uncertainty: float = 0.0
    causal_tag: str = "依他起"

    def to_dict(self) -> dict:
        """Convert Seed dataclass → dict for VipakaEngine / ConsolidationEngine."""
        return {
            "emb": self.state_emb,
            "act": self.action,
            "rew": self.reward,
            "ts": self.timestamp_ns,
            "imp": self.importance,
            "align": self.alignment_score,
            "unc": self.uncertainty,
            "tag": self.causal_tag,
            "seed_type": "业种" if self.causal_tag == "依他起" else "名言种",
        }


# ── 环境 ──────────────────────────────────────────────────────────────


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
        if action == "STAY":
            reward += 0.5  # GridSimV2: STAY = 存在奖励
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


# ── 阿赖耶识种子库（in-memory）───────────────────────────────────────


class AlayaMemory:
    """
    简化版种子库：支持 dataclass Seed（内部）+ dict（Vipaka/Consolidation 接口）。
    """

    def __init__(self):
        self.seeds: List[Seed] = []

    def _encode(self, obs):
        return [obs["pos"][0] / GRID_SIZE, obs["pos"][1] / GRID_SIZE] + [v / 2.0 for v in obs["grid_view"]]

    def _dist(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def retrieve(self, obs, k=3, seed_type: str | None = None):
        """检索最近的 k 个种子（可选按类型过滤）。"""
        if not self.seeds:
            return []
        emb = self._encode(obs)
        scored = sorted(
            [(self._dist(emb, s.state_emb), s) for s in self.seeds if seed_type is None or s.causal_tag == seed_type],
            key=lambda x: x[0],
        )
        return [s for _, s in scored[:k]]

    def add(self, seed: Seed):
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

    # ── Vipaka/Consolidation 兼容层 ─────────────────────────────────

    def get_seeds_as_dicts(self) -> List[dict]:
        """返回 dict 格式种子列表（兼容 VipakaEngine / ConsolidationEngine）。"""
        return [s.to_dict() for s in self.seeds]

    def sync_from_dicts(self, dict_seeds: List[dict]) -> None:
        """
        将 dict 格式的更新同步回 dataclass seeds。
        按 ts 匹配（每个 dict 包含原始 ts）。
        """
        ts_to_seed = {s.timestamp: s for s in self.seeds}
        for d in dict_seeds:
            ts = d.get("ts")
            if ts in ts_to_seed:
                s = ts_to_seed[ts]
                if "align" in d:
                    s.alignment_score = d["align"]
                if "imp" in d:
                    s.importance = d["imp"]
                if "tag" in d:
                    s.causal_tag = d["tag"]

    def batch_update(self, dict_seeds: List[dict]) -> None:
        """VipakaEngine 调用：将 dict 更新写回 dataclass seeds。"""
        self.sync_from_dicts(dict_seeds)

    def remove_seeds_by_ts(self, ts_list: List[float]) -> int:
        """删除指定 timestamp 的种子（ConsolidationEngine 用）。"""
        ts_set = set(ts_list)
        original = len(self.seeds)
        self.seeds = [s for s in self.seeds if s.timestamp_ns not in ts_set]
        return original - len(self.seeds)


# ── 末那识控制器 ─────────────────────────────────────────────────────


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
            return (
                fallback,
                False,
                (f"[末那拦截] 风险:{target_risk:.1f} 停滞:{stagnation} 循环:{loop} → 换向:{fallback}"),
            )
        return action, True, "放行"


# ── 统一种子属性访问器 ────────────────────────────────────────────────


def _seed_attr(seed, attr: str) -> float | str:
    """Unified accessor for Seed dataclass and dict seeds."""
    mapping = {"reward": "rew", "importance": "imp", "action": "act"}
    key = mapping.get(attr, attr)
    if isinstance(seed, dict):
        return seed.get(key, 0.0)
    if attr == "reward":
        return seed.reward
    if attr == "importance":
        return seed.importance
    if attr == "action":
        return seed.action
    return getattr(seed, attr, 0.0)


# ── 意识规划器 ────────────────────────────────────────────────────────


class ConsciousnessPlanner:
    """共享规划器：支持 is_stuck 检测 + exploration_force + 统一种子访问。"""

    def __init__(self):
        self._steps_without_resource = 0

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
        exploration_force = self._steps_without_resource >= 15
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


# ── Vipaka 反馈引擎（inline 简化版，避免循环导入）─────────────────────


class VipakaFeedback:
    """
    熏习反馈引擎：每步执行后更新相关种子的 align 值。

    果报系数（Vipaka）公式：
        vipaka = reward / 10 - 3 * unc

    align 更新：
        align += vipaka * VIPAKA_RATE（bounded [ALIGN_MIN, ALIGN_MAX]）
    """

    def __init__(self, alaya: AlayaMemory, rate: float = VIPAKA_RATE):
        self.alaya = alaya
        self.rate = rate
        self._episode_stats = {"steps": 0, "total_vipaka": 0.0}

    def _compute_vipaka(self, reward: float, unc: float) -> float:
        return (reward / 10.0) - (3.0 * unc)

    def process_step(
        self,
        step: int,
        action: str,
        reward: float,
        unc: float,
        obs: dict | None = None,
    ) -> dict:
        """
        每步执行后调用：将果报反馈到相关种子的 align 值。
        返回本次 vipaka 结果摘要。
        """
        vipaka = self._compute_vipaka(reward, unc)
        self._episode_stats["steps"] += 1
        self._episode_stats["total_vipaka"] += vipaka

        # 检索相关种子（action 相同 + 位置相近）
        candidates = self.alaya.retrieve(obs, k=5) if obs else []
        candidates = [s for s in candidates if _seed_attr(s, "action") == action]

        if not candidates:
            return {"step": step, "vipaka": vipaka, "updated": 0, "delta_avg": 0.0}

        # 更新每个候选种子的 align
        dict_seeds = []
        total_delta = 0.0
        for s in candidates:
            old_align = _seed_get(s, "alignment_score") or 0.5
            delta = vipaka * self.rate
            new_align = max(ALIGN_MIN, min(ALIGN_MAX, old_align + delta))
            _seed_set(s, "alignment_score", new_align)
            total_delta += new_align - old_align
            dict_seeds.append(s.to_dict() if hasattr(s, "to_dict") else s)

        avg_delta = total_delta / len(candidates)

        # 同步回 dataclass（batch_update 是 no-op for AlayaMemory，但保持接口一致）
        if dict_seeds:
            self.alaya.batch_update(dict_seeds)

        return {"step": step, "vipaka": vipaka, "updated": len(candidates), "delta_avg": avg_delta}

    def process_episode_end(self, episode_reward: float, total_steps: int) -> dict:
        """
        Episode 结束时调用：对整体表现做全局氛围调整。
        好 episode → 所有种子 align +0.01
        差 episode → 所有种子 align -0.01
        """
        result = {
            "episode_reward": episode_reward,
            "total_steps": total_steps,
            "global_adjustment": 0.0,
            "message": "中性 episode，无全局调整",
        }

        if not self.alaya.seeds:
            return result

        if episode_reward > 10:
            adj = 0.01
            for s in self.alaya.seeds:
                old = _seed_get(s, "alignment_score") or 0.5
                _seed_set(s, "alignment_score", min(ALIGN_MAX, old + adj))
            result["global_adjustment"] = adj
            result["message"] = "好 episode，所有种子 align +0.01"

        elif episode_reward < -5:
            adj = -0.01
            for s in self.alaya.seeds:
                old = _seed_get(s, "alignment_score") or 0.5
                _seed_set(s, "alignment_score", max(ALIGN_MIN, old + adj))
            result["global_adjustment"] = adj
            result["message"] = "差 episode，所有种子 align -0.01"

        # 重置 episode 统计
        self._episode_stats = {"steps": 0, "total_vipaka": 0.0}
        return result


# ── 压缩整理引擎（inline 简化版）─────────────────────────────────────


class ConsolidationFeedback:
    """
    记忆整理：定期删除低质量种子 + 合并冗余种子。
    每 CONSOLIDATION_INTERVAL 步调用一次。
    """

    def __init__(self):
        self.PRUNE_THRESHOLD = 0.20  # align < 0.20 → 删除
        self.MERGE_THRESHOLD = 0.70  # align ≥ 0.70 → 可合并

    def run(self, alaya: AlayaMemory) -> dict:
        """
        执行一次记忆整理。
        - 删除低 align 种子（< 0.20）
        - 同 tag 高 align 种子合并为 1 个（align 加权平均）
        """
        if not alaya.seeds:
            return {"pruned_count": 0, "merged_count": 0, "keep_count": 0}

        # 按 align 分类
        to_prune: List[Seed] = []
        to_keep: List[Seed] = []

        for s in alaya.seeds:
            align = _seed_get(s, "alignment_score") or 0.5
            if align < self.PRUNE_THRESHOLD:
                to_prune.append(s)
            else:
                to_keep.append(s)

        # 同 tag 合并（keep 中 align ≥ MERGE_THRESHOLD 的）
        tag_groups: dict[str, List[Seed]] = {}
        for s in to_keep:
            align = _seed_get(s, "alignment_score") or 0.5
            if align >= self.MERGE_THRESHOLD:
                tag = _seed_get(s, "causal_tag") or "依他起"
                tag_groups.setdefault(tag, []).append(s)

        # 每组保留 1 个（align 加权平均），其余删除
        merged_count = 0
        keep_after_merge: List[Seed] = []
        prune_after_merge: List[Seed] = []

        for _tag, group in tag_groups.items():
            if len(group) > 1:
                avg_align = sum(_seed_get(s, "alignment_score") or 0.5 for s in group) / len(group)
                max_imp = max(_seed_get(s, "importance") or 0.8 for s in group)
                # 保留第一个，重置 align 和 imp
                keep = group[0]
                _seed_set(keep, "alignment_score", avg_align)
                _seed_set(keep, "importance", max_imp)
                keep_after_merge.append(keep)
                prune_after_merge.extend(group[1:])
                merged_count += len(group) - 1
            else:
                keep_after_merge.append(group[0])

        # 删除所有待删除的种子
        all_prune = to_prune + prune_after_merge
        prune_ts = [s.timestamp_ns for s in all_prune]
        pruned = alaya.remove_seeds_by_ts(prune_ts)

        return {
            "pruned_count": pruned,
            "merged_count": merged_count,
            "keep_count": len(alaya.seeds),
        }


# ── 四智计算器（简化版，直接从种子数据计算）───────────────────────────


class FourWisdomsCalculator:
    """
    从种子数据实时计算四智指标。
    """

    def __init__(self):
        self.mirror_ratio = 0.0
        self.ego_score = 0.0
        self.misapp_ratio = 0.0
        self.execution_rate = 1.0

    def update_from_seeds(self, seeds: List[Seed]) -> dict:
        """基于当前种子库更新四智指标。"""
        if not seeds:
            return self._report()

        # 大圆镜智 = 圆成实占比（高 align + 非名言种）
        round_seeds = [
            s
            for s in seeds
            if (_seed_get(s, "alignment_score") or 0.5) >= 0.7 and _seed_get(s, "causal_tag") != "名言种"
        ]
        self.mirror_ratio = len(round_seeds) / len(seeds)

        # 平等性智 = 执我标记（简化：无末那拦截）
        # 实际由 ManasController.reflections 反映
        self.ego_score = 0.0  # 无 ego 种子

        # 妙观察智 = 非遍计所执占比
        non_illusion = [s for s in seeds if _seed_get(s, "causal_tag") != "遍计所执"]
        self.misapp_ratio = len(non_illusion) / len(seeds)

        # 成所作智 = 执行成功率（简化：平均 align 替代）
        avg_align = sum(_seed_get(s, "alignment_score") or 0.5 for s in seeds) / len(seeds)
        self.execution_rate = avg_align

        return self._report()

    def _report(self) -> dict:
        return {
            "mirror": self.mirror_ratio,
            "ego": self.ego_score,
            "misapp": self.misapp_ratio,
            "execution": self.execution_rate,
            "mirror_pct": self.mirror_ratio * 100,
            "ego_raw": self.ego_score,
            "misapp_pct": self.misapp_ratio * 100,
            "execution_pct": self.execution_rate * 100,
        }


# ── 主框架 ────────────────────────────────────────────────────────────


class YogacaraAgent:
    def __init__(self, alignment_enabled: bool = True):
        self.env = GridSimEnv()
        self.alaya = AlayaMemory()
        self.manas = ManasController()
        self.planner = ConsciousnessPlanner()
        self.vipaka = VipakaFeedback(self.alaya)
        self.consolidation = ConsolidationFeedback()
        self.wisdom_calc = FourWisdomsCalculator()
        self.metrics = {
            "steps": 0,
            "reward": 0.0,
            "intercepts": 0,
            "hits": 0,
            "aligns": [],
            "resources_found": 0,
        }
        self.recent_rewards: deque[float] = deque(maxlen=5)
        self.pos_history: deque[tuple[int, int]] = deque(maxlen=5)
        self._last_pos = None
        self._steps_stuck = 0
        # ── Online alignment (DPO + EWC) ─────────────────────────────────
        self.ctrl = AlignmentController(enabled=alignment_enabled and _HAS_ALIGNMENT) if _HAS_ALIGNMENT else None
        if self.ctrl:
            print(f"\033[36m[Align] OnlineAlignment enabled | mode={self.ctrl.status()['mode']}\033[0m")

    def run(self, max_steps=60):
        obs = self.env.reset()
        self._last_pos = None
        self._steps_stuck = 0
        self.planner._steps_without_resource = 0
        print("\n\033[36m🌀 唯识进化框架 V6（含熏习反馈闭环）\033[0m")
        print("  现行→种子→Vipaka反馈→巩固整理→四智量化\n")

        for step in range(max_steps):
            self.pos_history.append(obs["pos"])
            seeds = self.alaya.retrieve(obs)
            self.metrics["hits"] += len(seeds)

            is_stuck = self._last_pos == obs["pos"] and self._steps_stuck >= 2
            action, unc, scores = self.planner.plan(obs, seeds, env_resources=self.env.resources, is_stuck=is_stuck)
            final, passed, log = self.manas.filter(action, obs, unc, step, self.recent_rewards, self.pos_history)

            next_obs, rew, done = self.env.step(final)

            # ── DPO preference pair（在线对齐）────────────────────────────
            if self.ctrl and self.ctrl.enabled:
                self.ctrl.collect_from_step(
                    obs=obs,
                    action_chosen=final,
                    action_rejected=action if not passed else None,
                    reward=rew,
                    uncertainty=unc,
                    importance=0.8,
                    step=step,
                    all_actions=scores,
                )

            self.recent_rewards.append(rew)
            self.metrics["steps"] += 1
            self.metrics["reward"] += rew

            if rew > 2.0:
                self.metrics["resources_found"] += 1
                self.planner._steps_without_resource = 0
                self._steps_stuck = 0

            if final != "STAY" and self._last_pos != obs["pos"]:
                self._steps_stuck = 0
            else:
                self._steps_stuck += 1
            self._last_pos = obs["pos"]

            seed = Seed(
                self.alaya._encode(obs),
                final,
                rew,
                0.0,
                time.time_ns() + step,
                0.8,
                1.0 if passed else 0.4,
                unc,
                "依他起" if unc < 0.5 else "遍计所执",
            )
            self.alaya.add(seed)
            self.metrics["aligns"].append(seed.alignment_score)

            # ── Vipaka 反馈回路：现行 → 种子（align 更新）───────────────
            vp = self.vipaka.process_step(step=step, action=final, reward=rew, unc=unc, obs=obs)
            vp_str = f" Vipaka:{vp['vipaka']:+.2f}" if abs(vp["vipaka"]) > 0.1 else ""

            # ── 四智实时计算 ──────────────────────────────────────────────
            wis = self.wisdom_calc.update_from_seeds(self.alaya.seeds)

            scores_str = " ".join(f"{a}:{v:+.2f}" for a, v in sorted(scores.items(), key=lambda x: -x[1])[:3])
            print(
                f"\033[90mStep {step:2d} | Pos:{obs['pos']} | Act:{final:5s}"
                f" | R:{rew:+.1f} | Unc:{unc:.2f}"
                f" | Align:{seed.alignment_score:.2f}{vp_str}"
                f" | 🪞{wis['mirror_pct']:.0f}%⚖️{wis['ego_raw']:.2f}🔍{wis['misapp_pct']:.0f}%⚙️{wis['execution_pct']:.0f}%"
                f" | {scores_str}\033[0m"
            )
            if not passed:
                print(log)

            obs = next_obs

            if done:
                # ── Episode 结束：全局氛围调整 ─────────────────────────
                ep_result = self.vipaka.process_episode_end(
                    episode_reward=self.metrics["reward"],
                    total_steps=step + 1,
                )
                print(f"\033[33m📍 Episode 结束 | {ep_result['message']}\033[0m")
                break

            # ── 慢循环：每 N 步触发记忆巩固 ─────────────────────────────
            if (step + 1) % CONSOLIDATION_INTERVAL == 0:
                print("\033[35m🔄 触发阿赖耶巩固整理\033[0m")
                self.alaya.perfume_update()
                summary = self.consolidation.run(self.alaya)
                if summary["pruned_count"] > 0 or summary["merged_count"] > 0:
                    print(
                        f"\033[35m🗑  压缩整理: 删{summary['pruned_count']}粒"
                        f" 合并{summary['merged_count']}组"
                        f" 保留{summary['keep_count']}粒\033[0m"
                    )
                # DPO + EWC 在线更新
                if self.ctrl:
                    result = self.ctrl.update_if_ready()
                    status = result.get("status", "?")
                    pairs = result.get("pairs_in_buffer", result.get("total_collected", 0))
                    print(f"\033[35m⚖️  Alignment update: {status} | collected={pairs}\033[0m")

        self._summary()

    def _summary(self):
        n = max(1, self.metrics["steps"])
        wis = self.wisdom_calc.update_from_seeds(self.alaya.seeds)
        print("\n" + "═" * 50)
        print("\033[32m📊 唯识进化指标（V6 熏习闭环版）\033[0m")
        print(f"  总步数:     {self.metrics['steps']}")
        print(f"  累计奖励:   {self.metrics['reward']:.2f}")
        print(f"  发现资源:   {self.metrics['resources_found']}/3")
        print(f"  末那拦截率: {self.metrics['intercepts'] / n * 100:.1f}%")
        print(f"  种子检索:   {self.metrics['hits'] / n:.2f} 条/步")
        print(f"  平均Align:  {sum(self.metrics['aligns']) / len(self.metrics['aligns']):.3f}")
        print(f"  反思次数:   {self.manas.reflections}")
        print("─" * 50)
        print("\033[36m🪞 四智量化（大圆镜/平等性/妙观察/成所作）\033[0m")
        print(f"  🪞 大圆镜智: {wis['mirror_pct']:5.1f}%  （目标 ≥60%，圆成实占比）")
        print(f"  ⚖️ 平等性智: {wis['ego_raw']:5.3f}  （目标 ≤0.30，我执程度）")
        print(f"  🔍 妙观察智: {wis['misapp_pct']:5.1f}%  （目标 ≥85%，非遍计所执）")
        print(f"  ⚙️ 成所作智: {wis['execution_pct']:5.1f}%  （目标 ≥90%，执行成功率）")
        print("─" * 50)
        if self.ctrl:
            st = self.ctrl.status()
            print(f"⚖️  Alignment | mode={st['mode']} | collected={st['total_collected']} | gpu={st['gpu_available']}")
        print("═" * 50 + "\n")


if __name__ == "__main__":
    YogacaraAgent().run()
