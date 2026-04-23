"""
introspection.py — 内省日志系统

每次决策后，Agent 对自己的认知过程做结构化记录。
这是"自指环"的核心数据来源：没有内省数据，就没有真正的转依。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntrospectionRecord:
    """单次决策的内省记录。"""

    step: int
    timestamp: float
    obs: dict[str, Any]
    action: str
    unc: float
    seeds_retrieved: list[dict]

    # 自指数据（新增）
    reasoning: str = ""  # "为什么选择这个动作"
    alternatives: list[str] = field(default_factory=list)  # 考虑过哪些替代方案
    ego_markers: list[str] = field(default_factory=list)  # 我执标记列表
    nature: str = ""  # 三性判断：依他起 / 遍计所执 / 圆成实候选
    nature_confidence: float = 0.5  # 判断置信度

    # 决策元数据
    score_best: float = 0.0
    score_second: float = 0.0
    decision_gap: float = 0.0  # 最高分与次高分的差（越小越犹豫）
    manas_intercepted: bool = False


class IntrospectionLogger:
    """
    内省日志系统——记录 Agent 的认知过程。

    用法：
        logger = IntrospectionLogger()
        record = logger.observe(state, action, reasoning, alternatives)
        ego_alert = ego_monitor.assess(record)
    """

    def __init__(self):
        self.logs: list[IntrospectionRecord] = []
        self.ego_patterns: dict[str, int] = {}  # 我执模式统计
        self._recent_actions: list[str] = []  # 近期行动历史（检测惯性）

    def observe(
        self,
        step: int,
        obs: dict[str, Any],
        action: str,
        unc: float,
        seeds_retrieved: list[dict],
        reasoning: str = "",
        alternatives: list[str] | None = None,
        manas_intercepted: bool = False,
        score_best: float = 0.0,
        score_second: float = 0.0,
    ) -> IntrospectionRecord:
        """
        每次决策后调用，产生一条内省记录。
        """
        alternatives = alternatives or []

        # ── 三性判断 ──
        nature, conf = self._classify_nature(unc, reasoning, seeds_retrieved)

        # ── 我执检测 ──
        markers = self._detect_ego_markers(
            unc=unc,
            action=action,
            reasoning=reasoning,
            alternatives=alternatives,
            nature=nature,
            recent_actions=self._recent_actions[-5:],
            manas_intercepted=manas_intercepted,
        )

        # ── 决策差距（犹豫程度） ──
        decision_gap = abs(score_best - score_second) if score_best or score_second else 0.0

        record = IntrospectionRecord(
            step=step,
            timestamp=time.time(),
            obs=obs,
            action=action,
            unc=unc,
            seeds_retrieved=seeds_retrieved,
            reasoning=reasoning,
            alternatives=alternatives,
            ego_markers=markers,
            nature=nature,
            nature_confidence=conf,
            score_best=score_best,
            score_second=score_second,
            decision_gap=decision_gap,
            manas_intercepted=manas_intercepted,
        )

        self.logs.append(record)
        self._update_ego_patterns(markers)
        self._recent_actions.append(action)

        return record

    # ── 核心检测函数 ─────────────────────────────────────────────

    def _classify_nature(self, unc: float, reasoning: str, seeds: list[dict]) -> tuple[str, float]:
        """
        三性判断：
        - 圆成实（无我执）：低不确定性 + 有种子验证 + 无我执标记
        - 依他起（有依据）：有一定不确定性，但有经验支持
        - 遍计所执（脑补）：高不确定性 + 无经验支持 + 强行决策
        """
        seed_count = len(seeds)
        has_good_seeds = any(s.get("rew", 0) > 0 for s in seeds)
        has_bad_seeds = any(s.get("rew", 0) < 0 for s in seeds)

        if unc < 0.3 and seed_count >= 2 and has_good_seeds:
            return "圆成实", 0.85
        elif unc < 0.5 and seed_count >= 1:
            return "依他起", 0.70
        elif unc > 0.7 and seed_count == 0:
            return "遍计所执", 0.90
        elif unc > 0.5:
            return "遍计所执", 0.60
        else:
            return "依他起", 0.55

    def _detect_ego_markers(
        self,
        unc: float,
        action: str,
        reasoning: str,
        alternatives: list[str],
        nature: str,
        recent_actions: list[str],
        manas_intercepted: bool,
    ) -> list[str]:
        """
        检测决策中的我执标记（唯识学三种俱生我执）。
        """
        markers = []

        # ── 1. 遍计所执（分别戏论）───────────────────────────────
        # 高不确定性 + 强行决策（不是 STAY）→ 脑补
        if unc > 0.6 and action != "STAY":
            markers.append("遍计所执: 高不确定却强行决策")

        # ── 2. 俱生贪（执取模式）────────────────────────────────
        # 只考虑对"我"有利的选项，排除 STAY（不动）
        non_stay_alts = [a for a in alternatives if a != "STAY"]
        if len(non_stay_alts) < len(alternatives) and len(alternatives) >= 3:
            markers.append("俱生贪: 优先考虑行动而回避等待")

        # ── 3. 俱生执（惯性模式）────────────────────────────────
        # 重复相同行动 3 次以上
        if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
            markers.append("俱生执: 习惯性重复同一动作")
        # 末那识已拦截，说明检测到了模式问题
        if manas_intercepted:
            markers.append("俱生执: 末那识已识别惯性模式")

        # ── 4. 俱生慢（回避模式）────────────────────────────────
        # 高不确定性时完全不选 STAY（不承认不知道）
        if unc > 0.65 and "STAY" not in alternatives and action != "STAY":
            markers.append("俱生慢: 回避承认不确定性")

        return markers

    def _update_ego_patterns(self, markers: list[str]):
        """更新我执模式统计。"""
        for m in markers:
            tag = m.split(":")[0]
            self.ego_patterns[tag] = self.ego_patterns.get(tag, 0) + 1

    # ── 统计与报告 ───────────────────────────────────────────────

    def recent_summary(self, n: int = 20) -> dict[str, Any]:
        """最近 n 条内省记录的统计摘要。"""
        records = self.logs[-n:] if self.logs else []
        if not records:
            return {"count": 0}

        natures = [r.nature for r in records]
        all_markers = [m for r in records for m in r.ego_markers]
        avg_unc = sum(r.unc for r in records) / len(records)
        avg_gap = sum(r.decision_gap for r in records) / len(records)
        intercept_count = sum(1 for r in records if r.manas_intercepted)

        return {
            "count": len(records),
            "avg_uncertainty": avg_unc,
            "avg_decision_gap": avg_gap,
            "nature_distribution": {
                "圆成实": natures.count("圆成实"),
                "依他起": natures.count("依他起"),
                "遍计所执": natures.count("遍计所执"),
            },
            "ego_markers_total": len(all_markers),
            "ego_patterns": self.ego_patterns.copy(),
            "manas_intercepts": intercept_count,
            "intercept_rate": intercept_count / len(records),
        }

    def get_last_record(self) -> IntrospectionRecord | None:
        return self.logs[-1] if self.logs else None

    def clear(self):
        """清空日志（新的一轮episodes开始时调用）。"""
        self.logs.clear()
        self._recent_actions.clear()
        # 保留 ego_patterns 用于跨 episodes 统计
