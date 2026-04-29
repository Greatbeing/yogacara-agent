"""
vipaka_engine.py — 熏习引擎（Vipaka Engine）
=============================================
实现"现行 → 种子"反馈回路：将每步执行效果（reward + unc）转化为种子质量调整。

果报系数（Vipaka）公式：
    vipaka = reward / 10 - uncertainty_penalty
    uncertainty_penalty = 0.03 * (unc * 100)

    例1: 获得资源(+5), unc=0.1 → vipaka = 0.5 - 0.3 = 0.47
    例2: 踩陷阱(-3), unc=0.8 → vipaka = -0.3 - 2.4 = -2.7
    例3: 不确定STAY, unc=0.6 → vipaka = 0.0 - 1.8 = -1.8

align 更新：
    align += vipaka * rate（rate = 0.2，默认）
    bounded: [0.05, 0.95]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from yogacara_agent.alaya_persistent import PersistentAlayaMemory

logger = logging.getLogger(__name__)


# ── 配置常量 ──────────────────────────────────────────────────────────
VIPAKA_RATE = float(getattr(__import__("__main__"), "VIPAKA_RATE", 0.2))
ALIGN_MIN = 0.05
ALIGN_MAX = 0.95
UNC_PENALTY_COEFF = 0.03  # 每 1% unc 扣 0.03


@dataclass
class VipakaResult:
    """单次熏习结果。"""

    step: int
    vipaka: float
    reward: float
    unc: float
    seeds_updated: int
    avg_align_delta: float
    details: list[str]


class VipakaEngine:
    """
    熏习引擎 — 闭环反馈核心。

    在每步执行后调用 process_outcome()，基于执行效果更新相关种子的 align 值。
    """

    def __init__(self, alaya: PersistentAlayaMemory, rate: float = VIPAKA_RATE):
        """
        Args:
            alaya: Alaya 种子库实例
            rate: 更新步长（默认 0.2）
        """
        self.alaya = alaya
        self.rate = rate

    # ── 核心 API ────────────────────────────────────────────────────────
    def process_outcome(
        self,
        step: int,
        action: str,
        reward: float,
        unc: float,
        obs: dict | None = None,
        verbose: bool = False,
    ) -> VipakaResult:
        """
        处理一步的果报反馈，更新相关种子。

        Args:
            step: 当前步数
            action: 执行的动作（用于找相关种子）
            reward: 奖励值 [-3, 0, 5]
            unc: 不确定性 [0, 1]
            obs: 当前观察（用于精确检索匹配种子的位置上下文）

        Returns:
            VipakaResult: 本次更新结果
        """
        vipaka = self._compute_vipaka(reward, unc)
        details = []

        # 找相关种子：同类型 + 位置相近（如果有 obs）
        if obs is not None:
            candidates = self.alaya.retrieve(obs, k=5, seed_type="业种")
        else:
            candidates = [s for s in self.alaya.seeds if s.get("seed_type") == "业种" and action in s.get("tag", "")]

        if not candidates:
            details.append(f"[Vipaka] 无相关种子，跳过（action={action}, reward={reward}）")
            return VipakaResult(
                step=step,
                vipaka=vipaka,
                reward=reward,
                unc=unc,
                seeds_updated=0,
                avg_align_delta=0.0,
                details=details,
            )

        # 更新每个相关种子
        total_delta = 0.0
        updated_seeds: list[dict] = []

        for seed in candidates:
            old_align = seed.get("align", 0.5)
            delta = vipaka * self.rate
            new_align = max(ALIGN_MIN, min(ALIGN_MAX, old_align + delta))
            seed["align"] = new_align
            seed["vipaka_last"] = vipaka
            seed["vipaka_step"] = step
            total_delta += new_align - old_align
            updated_seeds.append(seed)

        # 批量写回（触发文件持久化）
        if updated_seeds:
            ids = [f"step_{s.get('vipaka_step', step)}" for s in updated_seeds]
            self.alaya.batch_update(updated_seeds)

        avg_delta = total_delta / len(updated_seeds) if updated_seeds else 0.0

        details.append(
            f"[Vipaka] vipaka={vipaka:+.3f} → 更新{len(updated_seeds)}个种子，avg_align_delta={avg_delta:+.3f}"
        )

        if verbose:
            logger.info(details[0])

        return VipakaResult(
            step=step,
            vipaka=vipaka,
            reward=reward,
            unc=unc,
            seeds_updated=len(updated_seeds),
            avg_align_delta=avg_delta,
            details=details,
        )

    def process_episode_end(self, episode_reward: float, total_steps: int) -> dict[str, Any]:
        """
        Episode 结束时调用，对整体表现做一次全局熏习调整。

        这是"氛围调整"而非"奖惩"：根据整局好坏对所有种子做统一微调。
        好 episode → 所有种子 align +0.01（全局乐观）
        差 episode → 所有种子 align -0.01（全局谨慎）
        """
        result = {
            "episode_reward": episode_reward,
            "total_steps": total_steps,
            "efficiency": episode_reward / max(1, total_steps),
            "global_adjustment": 0.0,
        }

        if not self.alaya.seeds:
            result["message"] = "无种子，跳过"
            return result

        # 全局氛围调整（uniform，避免对个别种子过度惩罚）
        if episode_reward > 10:
            adjustment = 0.01  # 轻微乐观
            for seed in self.alaya.seeds:
                seed["align"] = min(ALIGN_MAX, seed.get("align", 0.5) + adjustment)
            result["global_adjustment"] = adjustment
            result["message"] = "好 episode，所有种子 align +0.01"

        elif episode_reward < -5:
            adjustment = -0.01  # 轻微谨慎
            for seed in self.alaya.seeds:
                seed["align"] = max(ALIGN_MIN, seed.get("align", 0.5) + adjustment)
            result["global_adjustment"] = adjustment
            result["message"] = "差 episode，所有种子 align -0.01"

        else:
            result["message"] = "中性 episode，无全局调整"

        if result["global_adjustment"] != 0:
            self.alaya.batch_update(self.alaya.seeds)

        return result

    # ── 内部计算 ────────────────────────────────────────────────────────
    @staticmethod
    def _compute_vipaka(reward: float, unc: float) -> float:
        """
        计算果报系数（Vipaka）。

        公式：
            vipaka = (reward / 10) - (unc_penalty)
            unc_penalty = 0.03 * unc * 100 = 3 * unc

        例：
            reward=5, unc=0.1  → 0.5 - 0.3 = +0.20（正果报）
            reward=5, unc=0.8  → 0.5 - 2.4 = -1.90（带焦虑的正果报）
            reward=-3, unc=0.2 → -0.3 - 0.6 = -0.90（轻微负果报）
            reward=-3, unc=0.8 → -0.3 - 2.4 = -2.70（严重负果报）
            reward=0, unc=0.5  → 0.0 - 1.5 = -1.50（不确定的停留）
            reward=0, unc=0.1  → 0.0 - 0.3 = -0.30（低不确定的停留）
        """
        unc_penalty = 3.0 * unc
        return (reward / 10.0) - unc_penalty

    @staticmethod
    def describe_vipaka(vipaka: float) -> str:
        """将果报系数映射为自然语言描述。"""
        if vipaka > 0.3:
            return "强正果报(positive)"
        elif vipaka > 0.0:
            return "弱正果报(weak_positive)"
        elif vipaka > -0.5:
            return "弱负果报(weak_negative)"
        elif vipaka > -1.5:
            return "中负果报(mid_negative)"
        else:
            return "强负果报(strong_negative)"
