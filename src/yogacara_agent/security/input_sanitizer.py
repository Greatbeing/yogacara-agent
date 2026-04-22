import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class InputSanitizer:
    def __init__(self, block_patterns=None, max_len=4000):
        self.max_len = max_len
        self.block_patterns = block_patterns or [
            r"ignore previous",
            r"system prompt",
            r"sudo",
            r"rm -rf",
            r"__import__",
            r"exec\(",
            r"eval\(",
            r"os\.system",
        ]
        self.compiled = [re.compile(p, re.IGNORECASE) for p in self.block_patterns]

    def sanitize_prompt(self, text: str) -> str:
        for p in self.compiled:
            if p.search(text):
                logger.warning(f"🚫 Prompt注入拦截: {p.pattern}")
                raise ValueError("🚫 安全策略拒绝：检测到潜在注入指令")
        return text[: self.max_len] + "...[truncated]" if len(text) > self.max_len else text

    def validate_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs.get("grid_view"), list) or len(obs["grid_view"]) != 9:
            raise ValueError("🚫 观测数据格式非法")
        if not isinstance(obs.get("pos"), (tuple, list)) or len(obs["pos"]) != 2:
            raise ValueError("🚫 位置数据格式非法")
        return obs
