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
        # 大圆镜智：圆成实种子计数（由内省系统直接维护）
        self._parinispanna_count = 0
        self._total_classified = 0

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
        # 大圆镜智计数：圆成实 = 无我执 + 决策有依据（seeds支持）
        self._total_classified += 1
        if nature == "圆成实" and not markers:
            self._parinispanna_count += 1

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

        # 宽松版分类：有经验支持时降级处理
        has_good_seed = any(s.get("rew", 0) > 0 for s in seeds)

        if has_good_seed and unc < 0.45 and seed_count >= 1:
            return "圆成实", 0.80
        elif unc < 0.55 and (seed_count >= 1 or has_good_seed):
            return "依他起", 0.75
        elif unc < 0.5 and seed_count >= 1:
            return "依他起", 0.70
        elif unc > 0.7 and seed_count == 0:
            return "遍计所执", 0.90
        elif unc > 0.6:
            return "遍计所执", 0.65
        else:
            return "依他起", 0.60

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
        # 高不确定性时选择行动而非等待（真正执取）
        # 低不确定性时选择行动是理性决策，不是贪
        if unc > 0.55 and action != "STAY" and len(alternatives) >= 2:
            markers.append("俱生贪: 高不确定却执取行动而非等待")

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

    def compute_wisdom_of_action(self) -> dict[str, Any]:
        """
        成所作智——意→行→果→识 反馈闭环完整度。

        三指标：
        1. 闭环完成率：reward ≠ 0 的步数比例（环境有反馈=闭环了）
        2. 意图-行动一致率：(steps - manas_intercepted) / steps（末那未拦截=意行合一）
        3. 果-识吻合度：奖励符合预期的比例
                   - 踩资源：reward ≈ +5.0
                   - 踩陷阱：reward ≈ -3.0
                   - 空走：reward ≈ -0.1~0.6（STAY=+0.5）
        """
        records = self.logs
        if not records:
            return {"status": "无内省数据"}

        total = len(records)
        # 1. 闭环完成率：环境给了非零反馈
        closed_loop = sum(1 for r in records if r.obs.get("reward", 0) != 0)
        loop_rate = closed_loop / total

        # 2. 意图-行动一致率：末那未拦截
        consistent = sum(1 for r in records if not r.manas_intercepted)
        intent_rate = consistent / total

        # 3. 果-识吻合度：reward 在合理范围内
        matched = 0
        for r in records:
            rew = r.obs.get("reward", 0)
            act = r.action
            # 资源格：reward ≈ +5.0
            if rew >= 4.0:
                matched += 1
            # 陷阱格：reward ≈ -3.0
            elif rew <= -2.0:
                matched += 1
            # 空走/STAY：reward 在 [-0.2, +0.7] 范围
            elif -0.2 <= rew <= 0.7:
                matched += 1
            # 移动未踩到东西：-0.2 ~ +0.1 范围也算吻合
            elif rew == -0.1 and act != "STAY":
                matched += 1
        # 吻合度（归一化到 [0,1]，0.8 以上算达标）
        alignment_rate = matched / total
        alignment_score = min(1.0, alignment_rate / 0.8)  # 80%吻合率 = 1.0

        # 成所作智综合分：三个指标加权平均
        wisdom_score = (loop_rate * 0.3 + intent_rate * 0.3 + alignment_score * 0.4)

        # 资源发现率（额外参考）
        resources_found = sum(1 for r in records if r.obs.get("reward", 0) >= 4.0)

        return {
            "score": round(wisdom_score, 3),
            "loop_rate": round(loop_rate, 3),
            "intent_rate": round(intent_rate, 3),
            "alignment_rate": round(alignment_rate, 3),
            "resources_found": resources_found,
            "total_steps": total,
            "status": (
                "达标" if wisdom_score >= 0.7
                else "发展中" if wisdom_score >= 0.4
                else "待提升"
            ),
        }

    def get_last_record(self) -> IntrospectionRecord | None:
        return self.logs[-1] if self.logs else None

    def clear(self):
        """清空日志（新的一轮episodes开始时调用）。"""
        self.logs.clear()
        self._recent_actions.clear()
        # 保留 ego_patterns 用于跨 episodes 统计
