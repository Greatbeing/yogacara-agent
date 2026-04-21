import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MemoryGuard:
    def __init__(self, z_threshold: float = 2.5, quarantine_size: int = 50):
        self.z_threshold: float = z_threshold
        self.quarantine: deque[dict[str, Any]] = deque(maxlen=quarantine_size)
        self.reward_history: deque[float] = deque(maxlen=200)

    def validate_seed(self, seed: dict) -> bool:
        required = ["emb", "act", "rew", "ts", "imp"]
        if not all(k in seed for k in required):
            return False
        self.reward_history.append(seed["rew"])
        if len(self.reward_history) > 20:
            mean, std = np.mean(self.reward_history), np.std(self.reward_history) + 1e-8
            if abs(seed["rew"] - mean) / std > self.z_threshold:
                logger.warning(f"⚠️ 异常种子拦截 (Z={abs(seed['rew'] - mean) / std:.2f})")
                self.quarantine.append(seed)
                return False
        if (
            not (0.0 <= seed.get("imp", 0) <= 1.0)
            or not (0.0 <= seed.get("align", 0) <= 1.0)
            or not (0.0 <= seed.get("unc", 0) <= 1.0)
        ):
            return False
        return True

    def get_quarantine_report(self) -> dict:
        return {"size": len(self.quarantine), "recent": list(self.quarantine)[:3]}
