"""环境适配器基础契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np


class EnvAdapter(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str | int) -> tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> dict[str, Any]:
        raise NotImplementedError


class AdapterRegistry:
    _factories: dict[str, Callable[..., EnvAdapter]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., EnvAdapter]) -> None:
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> EnvAdapter:
        if name not in cls._factories:
            raise KeyError(f"Unknown adapter: {name}. Available: {sorted(cls._factories)}")
        return cls._factories[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._factories)


class BaseSimEnv(EnvAdapter):
    def get_observation_space(self) -> dict[str, Any]:
        return {"shape": [0], "dtype": "object", "low": [], "high": []}

    def get_action_space(self) -> dict[str, Any]:
        return {"type": "discrete", "size": 5, "actions": ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]}

    def close(self):
        pass
