"""
compression_metrics.py — 压缩质量指标体系
==========================================
整合四智指标 + 种子库状态为统一评分（Compression Quality Score, CQS）。

公式：
    cqs = (avg_align * 四智综合) / log(seed_count + 1)

四智综合 = (大圆镜智 * 0.4 + (1-平等性智) * 0.3 + (1-妙观察智) * 0.2 + 成所作智 * 0.1)

四智指标来源（由调用方注入）：
    - mirror_ratio: 大圆镜智（圆成实占比）[0, 1]
    - ego_score: 平等性智（我执强度，越低越好）[0, ~]
    - misapprehension_ratio: 妙观察智（遍计所执比例，越低越好）[0, 1]
    - execution_rate: 成所作智（执行成功率）[0, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """压缩质量指标快照。"""
    # 种子库状态
    seed_count: int
    total_tokens: int  # 估算

    # 质量分布
    avg_align: float
    high_quality_ratio: float  # align > 0.7 的种子占比
    low_quality_ratio: float   # align < 0.3 的种子占比

    # 压缩率
    compression_ratio: float   # (初始 - 当前) / 初始（需外部注入初始值）
    seed_decay_rate: float     # imp 衰减率（时间维度压缩）

    # 四智指标（从 ego_monitor 注入）
    mirror_ratio: float        # 大圆镜智
    ego_score: float           # 平等性智（我执强度）
    misapprehension_ratio: float  # 妙观察智（遍计比例）
    execution_rate: float      # 成所作智

    # 综合评分
    cqs: float                 # Compression Quality Score

    def summary(self) -> dict[str, Any]:
        return {
            "seed_count": self.seed_count,
            "avg_align": round(self.avg_align, 3),
            "high_q_ratio": round(self.high_quality_ratio, 3),
            "cqs": round(self.cqs, 3),
            "mirror": round(self.mirror_ratio, 3),
            "ego_score": round(self.ego_score, 3),
            "misapp": round(self.misapprehension_ratio, 3),
            "execution": round(self.execution_rate, 3),
            "compression": round(self.compression_ratio, 3),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"CQS={s['cqs']} | seeds={s['seed_count']} | "
            f"avg_align={s['avg_align']} | high_q={s['high_q_ratio']} | "
            f"大圆镜={s['mirror']} | 平等性智={s['ego_score']} | "
            f"妙观察={s['misapp']} | 成所作={s['execution']}"
        )


class CompressionMetricsCalculator:
    """
    计算压缩质量指标。

    用法：
        calc = CompressionMetricsCalculator(initial_context_tokens=50000)
        metrics = calc.compute(
            seeds=alaya.seeds,
            initial_tokens=50000,
            mirror_ratio=0.75,
            ego_score=0.12,
            misapprehension_ratio=0.0,
            execution_rate=0.99,
        )
    """

    HIGH_ALIGN_THRESHOLD = 0.7
    LOW_ALIGN_THRESHOLD = 0.3

    def __init__(self, initial_context_tokens: int = 50_000):
        self.initial_context_tokens = initial_context_tokens

    def compute(
        self,
        seeds: list[dict],
        initial_tokens: int | None = None,
        mirror_ratio: float = 0.0,
        ego_score: float = 0.0,
        misapprehension_ratio: float = 0.0,
        execution_rate: float = 0.0,
        verbose: bool = False,
    ) -> CompressionMetrics:
        """
        计算当前压缩质量指标。

        Args:
            seeds: Alaya 种子列表
            initial_tokens: 初始上下文 token 数（用于计算压缩率）
            mirror_ratio: 大圆镜智（圆成实占比）
            ego_score: 平等性智（我执分数，越低越好）
            misapprehension_ratio: 妙观察智（遍计所执比例，越低越好）
            execution_rate: 成所作智（执行成功率）
        """
        seed_count = len(seeds)

        # 质量分布
        if seeds:
            aligns = [s.get("align", 0.5) for s in seeds]
            avg_align = sum(aligns) / len(aligns)
            high_q = sum(1 for a in aligns if a > self.HIGH_ALIGN_THRESHOLD) / seed_count
            low_q = sum(1 for a in aligns if a < self.LOW_ALIGN_THRESHOLD) / seed_count
            imps = [s.get("imp", 0.8) for s in seeds]
            avg_imp = sum(imps) / len(imps)
        else:
            avg_align = 0.0
            high_q = 0.0
            low_q = 0.0
            avg_imp = 0.0

        # 压缩率（当前上下文估算 vs 初始）
        init_tokens = initial_tokens or self.initial_context_tokens
        current_tokens = seed_count * 200  # 估算：每个种子约 200 tokens
        compression_ratio = max(0.0, (init_tokens - current_tokens) / init_tokens)

        # 种子衰减率（imp 分布的熵，衡量时间维度压缩程度）
        seed_decay_rate = self._compute_decay_rate(seeds)

        # 四智综合评分
        wisdom_score = self._compute_wisdom_score(
            mirror_ratio, ego_score, misapprehension_ratio, execution_rate
        )

        # CQS = 质量 * 效率 / 规模惩罚
        # log(seed_count+1) 控制规模：种子越多，惩罚越大
        cqs = (avg_align * wisdom_score) / (0.1 + 0.5 * (seed_count / 100) ** 0.5)

        metrics = CompressionMetrics(
            seed_count=seed_count,
            total_tokens=current_tokens,
            avg_align=avg_align,
            high_quality_ratio=high_q,
            low_quality_ratio=low_q,
            compression_ratio=compression_ratio,
            seed_decay_rate=seed_decay_rate,
            mirror_ratio=mirror_ratio,
            ego_score=ego_score,
            misapprehension_ratio=misapprehension_ratio,
            execution_rate=execution_rate,
            cqs=cqs,
        )

        if verbose:
            logger.info(f"[CQS] {metrics}")

        return metrics

    @staticmethod
    def _compute_wisdom_score(
        mirror: float,
        ego: float,
        misapp: float,
        execution: float,
    ) -> float:
        """
        四智综合评分（加权几何平均）。

        设计原则：
        - 大圆镜智（镜像真实）权重最高：40%
        - 平等性智（减少我执）次高：30%（用 1-ego_score，归一化）
        - 成所作智（有效执行）：20%
        - 妙观察智（减少遍计）：10%
        """
        # 平等性智归一化（ego_score 典型范围 0~1，超过1按1算）
        ego_normalized = min(ego, 1.0)
        ego_score_norm = max(0.0, 1.0 - ego_normalized)

        # 妙观察智归一化（本身就是比例，直接用）
        misapp_score_norm = max(0.0, 1.0 - misapp)

        score = (
            mirror * 0.40 +
            ego_score_norm * 0.30 +
            execution * 0.20 +
            misapp_score_norm * 0.10
        )
        return score

    @staticmethod
    def _compute_decay_rate(seeds: list[dict]) -> float:
        """
        计算种子衰减率（衡量时间维度压缩）。

        方法：imp 的变异系数（CV = std/mean）。CV 越高，说明种子质量分布越分散，
        压缩越有效（好的保留、差的衰减）。
        """
        if not seeds:
            return 0.0
        imps = [s.get("imp", 0.8) for s in seeds]
        mean = sum(imps) / len(imps)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in imps) / len(imps)
        std = variance ** 0.5
        return std / mean  # CV
