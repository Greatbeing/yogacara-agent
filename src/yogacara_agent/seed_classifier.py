"""
seed_classifier.py — 种子分类系统（转识成智 Phase2）

三类种子（对应唯识学）：
  名言种：抽象标签——"这是陷阱"、"这是资源"
  业  种：行为+结果配对——"往北走→+5"、"踩陷阱→-3"
  异熟种：跨 episode 的隐藏模式——"连续3次失败"、"总是绕路"

核心逻辑：
  - 内省记录 → 触发分类
  - 分类结果 → 影响 seed align（决定下次是否复用）
  - 异熟种单独积累（跨 session），观察长期习气
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── 类型定义 ─────────────────────────────────────────────────────


@dataclass
class SeedClassification:
    """单次种子分类结果。"""

    step: int
    seed_type: str  # "名言种" | "业种" | "异熟种"
    subtype: str  # 具体子类型
    align: float  # 调整后的 align 值 [0, 1]
    tag: str  # 语义标签
    note: str  # 分类理由
    triggered: bool  # 是否触发新种子


@dataclass
class VipakaAccumulator:
    """
    异熟种积累器——观察跨 episode 的隐藏模式。

    记录长期统计，不做即时决策，只在模式清晰时输出"异熟提醒"。
    """

    _consecutive_failures: list[tuple[int, str]] = field(default_factory=list)
    _last_action: str | None = None
    _action_streak: int = 0
    _action_type_streak: int = 0
    _total_steps: int = 0
    _total_resources: int = 0
    _total_traps: int = 0

    def record_step(
        self,
        action: str,
        reward: float,
        unc: float,
        step: int,
    ) -> SeedClassification | None:
        """记录一步，更新异熟种统计。"""
        self._total_steps += 1

        # ── 连续失败模式 ──
        if reward < 0:
            self._consecutive_failures.append((step, action))
            self._consecutive_failures = self._consecutive_failures[-5:]  # 保留最近5次
        else:
            self._consecutive_failures.clear()

        # ── 动作惯性检测 ──
        if action == self._last_action:
            self._action_streak += 1
        else:
            self._action_streak = 1
            self._last_action = action

        # ── 陷阱命中率 ──
        if reward == -3.0:
            self._total_traps += 1
        if reward == 5.0:
            self._total_resources += 1

        return self._check_vipaka_patterns(action, reward, unc, step)

    def _check_vipaka_patterns(
        self,
        action: str,
        reward: float,
        unc: float,
        step: int,
    ) -> SeedClassification | None:
        """检查是否有值得记录的异熟模式。"""
        # 模式1：连续失败 ≥ 3次
        if len(self._consecutive_failures) >= 3:
            last_three = self._consecutive_failures[-3:]
            actions = [a for _, a in last_three]
            if len(set(actions)) <= 2:  # 最多2种动作，说明有惯性
                return SeedClassification(
                    step=step,
                    seed_type="异熟种",
                    subtype="惯性失败模式",
                    align=0.25,  # 低 align：这是坏习气，不值得复用
                    tag="习气_惯性失败",
                    note=f"连续{len(last_three)}次失败，动作惯性：{actions}",
                    triggered=True,
                )

        # 模式2：动作重复 ≥ 5次（同方向惯性）
        if self._action_streak >= 5:
            return SeedClassification(
                step=step,
                seed_type="异熟种",
                subtype="动作惯性",
                align=0.3,
                tag="习气_动作惯性",
                note=f"连续{self._action_streak}次执行相同动作({action})",
                triggered=True,
            )

        # 模式3：高不确定性 + 连续同一动作 → 焦虑模式
        if unc > 0.7 and self._action_streak >= 3:
            return SeedClassification(
                step=step,
                seed_type="异熟种",
                subtype="焦虑决策",
                align=0.2,  # 更低：这是带着不确定性的强迫行为
                tag="习气_焦虑决策",
                note=f"高不确定({unc:.0%}) + 重复动作{self._action_streak}次 → 强迫倾向",
                triggered=True,
            )

        # 模式4：陷阱命中率 > 30% 且步数 > 20 → 坏习惯模式
        if self._total_steps > 20 and self._total_traps > 0:
            hit_ratio = self._total_traps / self._total_steps
            if hit_ratio > 0.30:
                return SeedClassification(
                    step=step,
                    seed_type="异熟种",
                    subtype="高陷阱命中率",
                    align=0.35,
                    tag="习气_高陷阱率",
                    note=f"陷阱命中率 {hit_ratio:.0%}（{self._total_traps}/{self._total_steps}步），需调整策略",
                    triggered=True,
                )

        return None  # 无新模式

    def reset(self):
        """重置 session 内数据（保留跨 session 的异熟信息）。"""
        self._consecutive_failures.clear()
        self._action_streak = 0
        self._last_action = None

    @property
    def stats(self) -> dict[str, Any]:
        """返回当前跨 session 统计。"""
        hit_ratio = self._total_traps / max(1, self._total_steps)
        return {
            "total_steps": self._total_steps,
            "total_resources": self._total_resources,
            "total_traps": self._total_traps,
            "trap_hit_ratio": f"{hit_ratio:.1%}",
            "current_action_streak": self._action_streak,
            "consecutive_failures": len(self._consecutive_failures),
        }


class SeedClassifier:
    """
    种子分类器——基于内省记录决定种子的 tag 和 align。

    对应唯识学的三类种子：
      - 名言种：认知标签，观察到的是"标签"而非"实际"
      - 业  种：行为结果配对，观察到的是"行动-反馈"
      - 异熟种：跨步隐藏模式，不是单步因果而是累积习气
    """

    def __init__(self):
        self.vipaka = VipakaAccumulator()
        self._seed_count = 0

    def classify(
        self,
        action: str,
        reward: float,
        unc: float,
        nature: str,
        ego_markers: list[str],
        step: int,
        manas_intercepted: bool = False,
    ) -> SeedClassification:
        """
        对当前决策做种子分类，返回 SeedClassification。

        分类优先级：
          1. 异熟种（有隐藏模式时优先）
          2. 名言种（高不确定性+我执时）
          3. 业  种（正常行动-结果配对）
        """
        # 先检查异熟模式
        vipaka_result = self.vipaka.record_step(action, reward, unc, step)
        if vipaka_result:
            return vipaka_result

        self._seed_count += 1

        # ── 名言种：标签过度，认知失真 ──
        if unc > 0.7 and ego_markers:
            return SeedClassification(
                step=step,
                seed_type="名言种",
                subtype="失真标签",
                align=0.4,  # 低 align：带偏见，不值得复用
                tag=f"名言_{nature}",
                note=f"高不确定性({unc:.0%}) + 我执标记{ego_markers}，标签失真",
                triggered=False,
            )

        if "遍计所执" in ego_markers:
            return SeedClassification(
                step=step,
                seed_type="名言种",
                subtype="分别戏论",
                align=0.5,
                tag="名言_遍计",
                note="检测到分别戏论，标签超出观察范围",
                triggered=False,
            )

        # ── 业种：正常行动-结果配对 ──
        if reward == 5.0:
            return SeedClassification(
                step=step,
                seed_type="业种",
                subtype="正反馈",
                align=0.85,
                tag=f"业_正反馈_{action}",
                note=f"行动{action}→+5，清晰正反馈，align 0.85",
                triggered=False,
            )
        if reward == -3.0:
            return SeedClassification(
                step=step,
                seed_type="业种",
                subtype="负反馈",
                align=0.65,
                tag=f"业_负反馈_{action}",
                note=f"行动{action}→-3，陷阱惩罚，align 0.65",
                triggered=False,
            )

        # ── 末那识拦截的决策（被修正的行动）──
        if manas_intercepted:
            return SeedClassification(
                step=step,
                seed_type="业种",
                subtype="修正决策",
                align=0.6,
                tag="业_修正决策",
                note=f"末那识拦截原行动{action}，调整为安全行动",
                triggered=False,
            )

        # ── 默认：依他起，中性业种 ──
        base_align = 0.7 if not ego_markers else 0.55
        return SeedClassification(
            step=step,
            seed_type="业种",
            subtype="中性经验",
            align=base_align,
            tag=f"业_中性_{action}",
            note=f"行动{action}→{reward:.1f}，中性经验，align {base_align}",
            triggered=False,
        )

    def recent_classification_summary(self, n: int = 20) -> dict[str, Any]:
        """返回最近 N 次分类的统计摘要。"""
        # vipaka.stats 是跨 session 累积的
        return self.vipaka.stats

    def reset_session(self):
        """新 episode 开始时调用，重置 session 内数据。"""
        self.vipaka.reset()
