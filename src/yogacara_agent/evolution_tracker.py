"""种子演化快照记录器。"""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import time
from typing import Any


def _seed_value(seed: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = seed.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


class EvolutionTracker:
    def __init__(self, max_snapshots: int = 200):
        self._max_snapshots = max_snapshots
        self._snapshots: deque[dict[str, Any]] = deque(maxlen=max_snapshots)
        self._lock = Lock()

    def snapshot(self, alaya) -> dict[str, Any]:
        seeds = list(getattr(alaya, "seeds", []))
        type_counts: dict[str, int] = {}
        total_importance = 0.0
        total_alignment = 0.0

        for seed in seeds:
            seed_type = seed.get("seed_type", "未知") if isinstance(seed, dict) else getattr(seed, "seed_type", "未知")
            type_counts[seed_type] = type_counts.get(seed_type, 0) + 1
            if isinstance(seed, dict):
                total_importance += _seed_value(seed, "imp", 0.0)
                total_alignment += _seed_value(seed, "align", 0.0)

        count = len(seeds)
        snap = {
            "ts": time(),
            "total_seeds": count,
            "seed_types": type_counts,
            "avg_importance": round(total_importance / count, 4) if count else 0.0,
            "avg_alignment": round(total_alignment / count, 4) if count else 0.0,
        }

        with self._lock:
            self._snapshots.append(snap)
        return snap

    def get_snapshots(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._snapshots)
        if limit is not None:
            return items[-limit:]
        return items

    def reset(self) -> None:
        with self._lock:
            self._snapshots.clear()
