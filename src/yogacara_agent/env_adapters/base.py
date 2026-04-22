from abc import ABC, abstractmethod
from typing import Any


class BaseSimEnv(ABC):
    @abstractmethod
    def reset(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict]:
        pass

    @abstractmethod
    def _observe(self) -> dict[str, Any]:
        pass

    def close(self):
        pass
