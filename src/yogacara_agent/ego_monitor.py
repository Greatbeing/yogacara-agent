"""
ego_monitor.py — 我执监测器（末那识元认知层）

第七识（末那识）的核心特征：恒、审、思——持续审查"我"在执着什么。

本监测器不强制拦截决策（那是环境安全规则的任务），
只生成"转依提醒"，让 Agent 有机会看到自己在执着什么。

四智目标：
  - 平等性智：长期我执均值 < 0.3
  - 妙观察智：遍计所执比例 < 15%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yogacara_agent.introspection import IntrospectionRecord, IntrospectionLogger


# ── 四智量化阈值 ─────────────────────────────────────────────────
EGO_SCORE_THRESHOLD = 0.6  # 触发转依提醒的长期我执分数
EGO_LONG_TERM_WINDOW = 20  # 长期我执计算窗口
PRAJNA_TARGET = 0.15  # 妙观察智目标：遍计所执 < 15%
EQUANIMITY_TARGET = 0.3  # 平等性智目标：长期我执均值 < 0.3


@dataclass
class EgoAssessment:
    """单次我执评估结果。"""

    step: int
    ego_score: float  # 即时我执分数 [0, 1]
    long_term_ego: float  # 长期我执分数（20步滑动窗口）
    markers: list[str]  # 触发标记
    triggered: bool  # 是否超过阈值
    recommendation: str  # 转依建议
    nature: str  # 三性判断
    # 四智指标
    equanimity_wisdom: float  # 平等性智近似：1 - 长期我执
    prajna_wisdom: float  # 妙观察智近似：1 - 遍计所执比例


class EgoMonitor:
    """
    我执监测器——第七识（末那识）的元认知实现。

    监测三种核心我执模式（对应唯识学的俱生我执）：
    1. 遍计所执：分别戏论，强行判断
    2. 俱生贪：只考虑对我有利的选项
    3. 俱生执：惯性重复，不自知
    4. 俱生慢：回避承认不确定

    不做强制拦截，只生成"末那识提醒"——让 Agent 自己选择是否放下。
    """

    def __init__(
        self,
        ego_threshold: float = EGO_SCORE_THRESHOLD,
        long_term_window: int = EGO_LONG_TERM_WINDOW,
    ):
        self.ego_score_history: list[float] = []
        self.ego_threshold = ego_threshold
        self.long_term_window = long_term_window

        # 跨 episodes 统计
        self.total_assessments = 0
        self.total_triggers = 0

        # 近期的遍计所执计数（用于妙观察智计算）
        self._prajna_history: list[str] = []  # nature 标签历史

    def assess(self, record: IntrospectionRecord) -> EgoAssessment:
        """
        评估一条内省记录的我执程度。
        """
        markers = record.ego_markers
        unc = record.unc
        nature = record.nature

        # ── 即时我执分数 ──
        # 平等性智：只看我执标记，不看不确定性（不确定性是认知局限，不是执著）
        marker_score = min(1.0, len(markers) * 0.25)
        # 只有在有我执标记时，不确定性和决策差距才放大执著程度
        if markers:
            unc_score = unc * 0.2
            gap_score = min(0.2, record.decision_gap * 0.3)
            ego_score = min(1.0, marker_score + unc_score + gap_score)
        else:
            ego_score = 0.0  # 无标记 = 无我执

        self.ego_score_history.append(ego_score)

        # ── 长期我执分数（慢循环）──
        window = self.ego_score_history[-self.long_term_window :]
        long_term_ego = sum(window) / len(window) if window else 0.0

        # ── 遍计所执历史 ──
        self._prajna_history.append(nature)
        self._prajna_history = self._prajna_history[-100:]  # 保留最近100条

        # ── 四智近似指标 ──
        # 平等性智：长期我执越低越好
        equanimity_wisdom = max(0.0, 1.0 - long_term_ego)

        # 妙观察智：遍计所执比例越低越好
        prajna_history = self._prajna_history[-self.long_term_window :]
        prajna_ratio = prajna_history.count("遍计所执") / len(prajna_history) if prajna_history else 0.0
        prajna_wisdom = max(0.0, 1.0 - prajna_ratio / max(0.01, PRAJNA_TARGET))

        # ── 是否触发 ──
        triggered = long_term_ego > self.ego_threshold

        # ── 转依建议 ──
        recommendation = self._generate_recommendation(markers, long_term_ego, nature, unc, record.action)

        self.total_assessments += 1
        if triggered:
            self.total_triggers += 1

        return EgoAssessment(
            step=record.step,
            ego_score=ego_score,
            long_term_ego=long_term_ego,
            markers=markers,
            triggered=triggered,
            recommendation=recommendation,
            nature=nature,
            equanimity_wisdom=equanimity_wisdom,
            prajna_wisdom=prajna_wisdom,
        )

    def _generate_recommendation(
        self,
        markers: list[str],
        long_term_ego: float,
        nature: str,
        unc: float,
        action: str,
    ) -> str:
        """生成转依建议（末那识提醒的文本）。"""
        if long_term_ego < 0.2:
            return "圆成实倾向：继续保持如实观察，不作多余判断"

        marker_types = [m.split(":")[0] for m in markers]

        if "遍计所执" in marker_types:
            if unc > 0.7:
                return f"末那识提醒(step {self.total_assessments})：此刻高度不确定({unc:.0%})，请在决策中加入「我不确定」的标记"
            return "末那识提醒：检测到分别戏论倾向，请如实观察而非脑补"
        elif "俱生贪" in marker_types:
            return "末那识提醒：检测到自我利益倾向，请考虑全局最优而非局部收益"
        elif "俱生执" in marker_types:
            return "末那识提醒：检测到惯性模式，请在下一步尝试不同方向"
        elif "俱生慢" in marker_types:
            return "末那识提醒：回避不确定性是一种执着，请在决策中承认「我不知道」"
        elif nature == "遍计所执":
            return "末那识提醒：决策依据不足，请增加观察或选择等待"
        else:
            return "末那识提醒：观察到微妙的执着，保持觉知即可"

    # ── 四智指标汇总 ─────────────────────────────────────────────

    def four_wisdoms_report(
        self, intro_logger: IntrospectionLogger | None = None, mirror_ratio: float = 0.0
    ) -> dict[str, Any]:
        """
        返回四智的当前量化指标。
        基于最近 20 条评估计算。

        Args:
            intro_logger: 可选，内省记录器（用于计算成所作智）
            mirror_ratio: 可选，大圆镜智 = 圆成实种子 / 总分类种子
        """
        history = self.ego_score_history[-self.long_term_window :]
        prajna_hist = self._prajna_history[-self.long_term_window :]

        if not history:
            return {
                "大圆镜智": {"ratio": mirror_ratio, "status": "无种子数据"},
                "平等性智": {"score": 0.0, "status": "无评估数据"},
                "妙观察智": {"ratio": 0.0, "status": "无评估数据"},
                "成所作智": {"status": "待集成"},
            }

        # 平等性智
        long_term_ego = sum(history) / len(history) if history else 0.0
        equanimity = max(0.0, 1.0 - long_term_ego / max(0.01, EQUANIMITY_TARGET))
        equanimity = min(1.0, equanimity)

        # 妙观察智
        prajna_ratio = prajna_hist.count("遍计所执") / len(prajna_hist) if prajna_hist else 1.0
        prajna = max(0.0, 1.0 - prajna_ratio / max(0.01, PRAJNA_TARGET))
        prajna = min(1.0, prajna)

        return {
            "大圆镜智": {
                "ratio": mirror_ratio,
                "status": "达标" if mirror_ratio >= 0.6 else "未达标",
                "target": ">60%",
            },
            "平等性智": {
                "score": equanimity,
                "raw_long_term_ego": round(long_term_ego, 3),
                "status": "达标" if long_term_ego < EQUANIMITY_TARGET else "未达标",
                "target": f"<{EQUANIMITY_TARGET}",
            },
            "妙观察智": {
                "ratio": prajna,
                "raw_prajna_ratio": f"{prajna_ratio:.1%}",
                "status": "达标" if prajna_ratio < PRAJNA_TARGET else "未达标",
                "target": f"<{PRAJNA_TARGET:.0%}",
            },
            "成所作智": {
                **(
                    (intro_logger.compute_wisdom_of_action())
                    if intro_logger
                    else {"status": "待集成（需反馈闭环数据）"}
                )
            },
        }

    def reset(self):
        """重置当前 session 的数据（新 episodes 开始时调用）。"""
        self.ego_score_history.clear()
        self._prajna_history.clear()
