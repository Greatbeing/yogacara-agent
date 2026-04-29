"""
alaya_ring.py — 阿赖耶识压缩环（完整闭环）
==========================================
串联所有组件，构成"感知→规划→执行→熏习→整理→指标"的完整闭环：

    GridSim感知 → Planner生成 → Manas拦截检查
        → execute执行 → vipaka_engine.process_outcome(每步)
        → alaya_store.add(新种子)
        → [每N步] consolidation_engine.run(整理)
        → [每步] compression_metrics.compute(指标)
        → [episode结束时] vipaka_engine.process_episode_end(全局)

用法：
    ring = AlayaRing(alaya, ego_monitor, initial_tokens=50000)
    ring.step(step_num, obs, action, reward, unc, nature, ego_markers)
    ring.episode_end(episode_reward, total_steps)
    metrics = ring.get_metrics()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from yogacara_agent.alaya_persistent import PersistentAlayaMemory
from yogacara_agent.vipaka_engine import VipakaEngine
from yogacara_agent.consolidation_engine import ConsolidationEngine
from yogacara_agent.compression_metrics import CompressionMetricsCalculator

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONSOLIDATION_INTERVAL = 50  # 每 N 步整理一次


@dataclass
class AlayaRingConfig:
    """阿赖耶识环配置。"""
    consolidation_interval: int = DEFAULT_CONSOLIDATION_INTERVAL
    initial_context_tokens: int = 50_000
    vipaka_rate: float = 0.2
    verbose: bool = False


class AlayaRing:
    """
    阿赖耶识压缩环——串联感知、决策、执行、熏习、整理、指标。

    使用方式：
        ring = AlayaRing(alaya=alaya_instance, ego_monitor=ego_instance)
        ring.step(step=1, obs={...}, action='north', reward=5.0, unc=0.1,
                  nature='圆成实', ego_markers=[])
        ...
        ring.episode_end(episode_reward=21.0, total_steps=36)
        metrics = ring.get_metrics()
    """

    def __init__(
        self,
        alaya: PersistentAlayaMemory,
        ego_monitor: Any = None,
        config: AlayaRingConfig | None = None,
    ):
        self.alaya = alaya
        self.ego = ego_monitor
        self.cfg = config or AlayaRingConfig()

        # 初始化组件
        self.vipaka = VipakaEngine(alaya, rate=self.cfg.vipaka_rate)
        self.consolidator = ConsolidationEngine()
        self.metrics_calc = CompressionMetricsCalculator(
            initial_context_tokens=self.cfg.initial_context_tokens
        )

        # 状态
        self._step_count = 0
        self._last_consolidation_step = 0
        self._latest_metrics: Any = None

    # ── 主循环接口 ──────────────────────────────────────────────────────
    def step(
        self,
        step_num: int,
        obs: dict,
        action: str,
        reward: float,
        unc: float,
        nature: str = "依他起",
        ego_markers: list[str] | None = None,
        manas_intercepted: bool = False,
    ) -> dict[str, Any]:
        """
        主环每步调用。

        Returns:
            dict 含：vipaka_result, new_seeds_count, needs_consolidation, metrics
        """
        self._step_count = step_num
        ego_markers = ego_markers or []

        # ── 1. 熏习反馈（现行→种子）────────────────────────────────
        vipaka_result = self.vipaka.process_outcome(
            step=step_num,
            action=action,
            reward=reward,
            unc=unc,
            obs=obs,
            verbose=self.cfg.verbose,
        )

        # ── 2. 新种子入库（由调用方决定是否添加，这里只记录）────────
        # 新种子由外部 planner 决策后通过 alaya.add() 入库

        # ── 3. 定时整理（每 N 步）────────────────────────────────────
        needs_consolidation = (
            step_num - self._last_consolidation_step >= self.cfg.consolidation_interval
        )

        consolidation_result = None
        if needs_consolidation:
            consolidation_result = self.consolidator.run(
                self.alaya.seeds,
                step=step_num,
                dry_run=False,
                verbose=self.cfg.verbose,
            )
            self._last_consolidation_step = step_num

        # ── 4. 计算压缩指标（每步）──────────────────────────────────
        four_wisdom = self._get_four_wisdom()
        self._latest_metrics = self.metrics_calc.compute(
            seeds=self.alaya.seeds,
            initial_tokens=self.cfg.initial_context_tokens,
            **four_wisdom,
            verbose=False,
        )

        return {
            "step": step_num,
            "vipaka": vipaka_result,
            "consolidation": consolidation_result,
            "needs_consolidation": needs_consolidation,
            "metrics": self._latest_metrics,
        }

    def episode_end(self, episode_reward: float, total_steps: int) -> dict[str, Any]:
        """
        Episode 结束时调用。

        1. 全局熏习调整（氛围调整）
        2. 最终整理
        3. 生成 episode 报告
        """
        # 全局熏习
        ep_result = self.vipaka.process_episode_end(episode_reward, total_steps)

        # 最终整理（每次 episode 结束都整理）
        final_consolidation = self.consolidator.run(
            self.alaya.seeds,
            step=total_steps,
            dry_run=False,
            verbose=False,
        )

        # 最终指标
        four_wisdom = self._get_four_wisdom()
        final_metrics = self.metrics_calc.compute(
            seeds=self.alaya.seeds,
            initial_tokens=self.cfg.initial_context_tokens,
            **four_wisdom,
        )

        return {
            "episode_reward": episode_reward,
            "total_steps": total_steps,
            "vipaka_episode": ep_result,
            "final_consolidation": final_consolidation,
            "final_metrics": final_metrics,
            "total_steps_processed": self._step_count,
        }

    def get_metrics(self) -> Any:
        """获取最新压缩指标。"""
        return self._latest_metrics

    def add_seed(self, seed: dict) -> None:
        """添加新种子到种子库。"""
        self.alaya.add(seed)

    # ── 内部方法 ────────────────────────────────────────────────────────
    def _get_four_wisdom(self) -> dict[str, float]:
        """
        从 ego_monitor 获取四智指标（如果没有则返回默认值）。

        期望 ego_monitor 有：
            .four_wisdom_report() -> {mirror_ratio, ego_score, misapprehension_ratio, execution_rate}
        """
        defaults = {
            "mirror_ratio": 0.75,
            "ego_score": 0.1,
            "misapprehension_ratio": 0.0,
            "execution_rate": 0.99,
        }

        if self.ego is None:
            return defaults

        try:
            report = getattr(self.ego, "four_wisdom_report", lambda: None)()
            if report:
                return {
                    "mirror_ratio": report.get("mirror_ratio", defaults["mirror_ratio"]),
                    "ego_score": report.get("ego_score", defaults["ego_score"]),
                    "misapprehension_ratio": report.get("misapprehension_ratio", defaults["misapprehension_ratio"]),
                    "execution_rate": report.get("execution_rate", defaults["execution_rate"]),
                }
        except Exception:
            pass

        return defaults

    def __repr__(self) -> str:
        stats = self.alaya.get_stats()
        return (
            f"AlayaRing(steps={self._step_count}, "
            f"seeds={stats.get('total_seeds', 0)}, "
            f"last_consolidation_step={self._last_consolidation_step})"
        )
