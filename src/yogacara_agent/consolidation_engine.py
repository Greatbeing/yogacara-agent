"""
consolidation_engine.py — 压缩整理引擎（记忆巩固）
===================================================
模拟生物睡眠期的记忆整理过程：

1. 质量评分（基于 align + imp + vipaka_history）
2. 分类处理：高(keep+) / 中(keep) / 低(keep- or prune)
3. 操作：删除低质量 + 合并高度相似种子
4. 生成压缩报告

触发条件（默认）：
  - 每 N 步执行一次（N=50，由调用方控制）
  - 或 episode 结束时主动调用
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── 配置常量 ──────────────────────────────────────────────────────────
KEEP_PLUS_THRESHOLD = 0.70  # 高质量：align >= 0.70，直接保留
KEEP_THRESHOLD = 0.30  # 中质量：align >= 0.30，保留
PRUNE_THRESHOLD = 0.20  # 低质量：align < 0.20，删除
MERGE_SIMILARITY = 0.90  # 超过此相似度 → 合并

# 估算每个种子的 token 数
SEED_TOKEN_ESTIMATE = 200


@dataclass
class ConsolidationReport:
    """整理报告。"""

    total_before: int
    total_after: int
    pruned_count: int
    merged_count: int
    kept_plus_count: int
    kept_count: int
    estimated_tokens_saved: int
    quality_distribution: dict[str, int]
    message: str


class ConsolidationEngine:
    """
    压缩整理引擎。

    使用方式：
        consolidator = ConsolidationEngine()
        # 每 50 步调用一次
        report = consolidator.run(alaya.seeds, step=current_step)
        if report.pruned_count > 0:
            alaya.batch_update(alaya.seeds)  # 持久化删除
    """

    def __init__(
        self,
        keep_plus_thresh: float = KEEP_PLUS_THRESHOLD,
        keep_thresh: float = KEEP_THRESHOLD,
        prune_thresh: float = PRUNE_THRESHOLD,
        merge_similarity: float = MERGE_SIMILARITY,
    ):
        self.keep_plus_thresh = keep_plus_thresh
        self.keep_thresh = keep_thresh
        self.prune_thresh = prune_thresh
        self.merge_similarity = merge_similarity

    def run(
        self,
        seeds: list[dict],
        step: int | None = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> ConsolidationReport:
        """
        执行记忆整理。

        Args:
            seeds: Alaya 种子列表（in-place 修改）
            step: 当前步数（日志用）
            dry_run: True = 只分析不修改
            verbose: 打印详细日志
        """
        total_before = len(seeds)
        if not seeds:
            return ConsolidationReport(
                total_before=0,
                total_after=0,
                pruned_count=0,
                merged_count=0,
                kept_plus_count=0,
                kept_count=0,
                estimated_tokens_saved=0,
                quality_distribution={"keep_plus": 0, "keep": 0, "prune": 0},
                message="无种子，跳过整理",
            )

        # ── Step 1: 质量分类 ──────────────────────────────────────────
        categorized = self._categorize(seeds)
        keep_plus, keep, prune = categorized

        # ── Step 2: 合并高度相似的种子 ───────────────────────────────
        merged_from_keep_plus = self._merge_similar(keep_plus, seeds)
        merged_count = len(keep_plus) - len(merged_from_keep_plus)
        keep_plus = merged_from_keep_plus

        # ── Step 3: 决定删除 ──────────────────────────────────────────
        # 规则：只删除 align < PRUNE_THRESHOLD 的种子
        prune_final = [s for s in prune if s.get("align", 0.5) < self.prune_thresh]
        # 低于 keep_thresh 但高于 prune_thresh 的种子：警告但不删除
        borderline = [s for s in prune if self.prune_thresh <= s.get("align", 0.5) < self.keep_thresh]

        if not dry_run:
            # 实际删除
            seeds_to_remove = {id(s) for s in prune_final}
            seeds[:] = [s for s in seeds if id(s) not in seeds_to_remove]

        # ── Step 4: 生成报告 ──────────────────────────────────────────
        total_after = len(seeds)
        pruned_count = len(prune_final)
        estimated_tokens_saved = pruned_count * SEED_TOKEN_ESTIMATE

        dist = {
            "keep_plus": len(keep_plus),
            "keep": len(keep) + len(borderline),
            "borderline": len(borderline),
            "prune": len(prune_final),
        }

        msg_parts = []
        msg_parts.append(f"整理 {'(模拟)' if dry_run else ''}step={step}: {total_before}→{total_after} seeds")
        msg_parts.append(f"删{pruned_count} 合并{merged_count} 保留{len(keep_plus)}+{len(keep)}")
        if borderline and verbose:
            msg_parts.append(f"borderline={len(borderline)} (0.20~0.30) - monitor")
        msg_parts.append(f"节省~{estimated_tokens_saved} tokens")

        report = ConsolidationReport(
            total_before=total_before,
            total_after=total_after,
            pruned_count=pruned_count,
            merged_count=merged_count,
            kept_plus_count=len(keep_plus),
            kept_count=len(keep),
            estimated_tokens_saved=estimated_tokens_saved,
            quality_distribution=dist,
            message=" | ".join(msg_parts),
        )

        if verbose:
            logger.info(f"[Consolidation] {report.message}")

        return report

    # ── 内部方法 ────────────────────────────────────────────────────────
    def _categorize(self, seeds: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """按 align 分类：keep_plus / keep / prune。"""
        keep_plus: list[dict] = []
        keep: list[dict] = []
        prune: list[dict] = []

        for s in seeds:
            align = s.get("align", 0.5)
            if align >= self.keep_plus_thresh:
                keep_plus.append(s)
            elif align >= self.keep_thresh:
                keep.append(s)
            else:
                prune.append(s)

        return keep_plus, keep, prune

    def _merge_similar(
        self,
        high_quality: list[dict],
        all_seeds: list[dict],
    ) -> list[dict]:
        """
        合并高度相似的种子（仅对 keep_plus 组操作）。

        合并策略：同 tag 的多个种子 → 合并为 1 个
        - 新 align = 加权平均
        - 新 imp = max
        - 保留最新 vipaka_step
        """
        if len(high_quality) <= 1:
            return high_quality

        # 按 tag 分组
        by_tag: dict[str, list[dict]] = {}
        for s in high_quality:
            tag = s.get("tag", "")
            by_tag.setdefault(tag, []).append(s)

        merged: list[dict] = []
        for tag, group in by_tag.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 多于1个 → 合并
                merged.append(self._merge_group(group))
                logger.debug(f"[Consolidation] 合并同 tag 种子: {tag} ({len(group)} → 1)")

        return merged

    def _merge_group(self, group: list[dict]) -> dict:
        """合并一组同 tag 种子。"""
        n = len(group)
        avg_align = sum(s.get("align", 0.5) for s in group) / n
        max_imp = max(s.get("imp", 0.8) for s in group)
        latest_vipaka = max(
            (s.get("vipaka_step", 0) for s in group),
            default=0,
        )
        latest_vipaka_value = None
        for s in group:
            vp = s.get("vipaka_last")
            if vp is not None and s.get("vipaka_step") == latest_vipaka:
                latest_vipaka_value = vp

        merged = dict(group[0])
        merged["align"] = avg_align
        merged["imp"] = max_imp
        merged["vipaka_step"] = latest_vipaka
        if latest_vipaka_value is not None:
            merged["vipaka_last"] = latest_vipaka_value
        merged["merged_from"] = n
        return merged
