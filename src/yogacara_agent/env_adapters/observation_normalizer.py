"""观测规范化工具。"""

from __future__ import annotations

from typing import Any

import numpy as np


class ObservationNormalizer:
    def normalize_grid(self, grid_view) -> np.ndarray:
        return np.asarray(grid_view, dtype=np.float32).reshape(-1)

    def normalize_continuous(
        self, values, mode: str = "zscore", bounds: list[tuple[float, float]] | None = None
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if mode == "minmax" and bounds:
            scaled = []
            for value, (low, high) in zip(arr, bounds):
                span = max(high - low, 1e-6)
                scaled.append((float(value) - low) / span)
            return np.asarray(scaled, dtype=np.float32)
        if mode == "zscore":
            mean = float(arr.mean()) if arr.size else 0.0
            std = float(arr.std()) if arr.size else 1.0
            return ((arr - mean) / max(std, 1e-6)).astype(np.float32)
        return arr.astype(np.float32)

    def metadata(self, values: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(values)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "low": arr.min().tolist() if arr.size else [],
            "high": arr.max().tolist() if arr.size else [],
        }
