"""CartPole 示例适配器。"""

from __future__ import annotations

import numpy as np

from .action_mapper import ActionMapper
from .base import BaseSimEnv
from .observation_normalizer import ObservationNormalizer


class CartPoleAdapter(BaseSimEnv):
    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.actions = ActionMapper()
        self.normalizer = ObservationNormalizer()
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            try:
                import gymnasium as gym
            except ImportError:
                try:
                    import gym
                except ImportError as exc:
                    raise ImportError("CartPoleAdapter requires gym or gymnasium") from exc
            self._gym = gym
            self._env = gym.make(self.env_name)
        return self._env

    def reset(self, seed: int | None = None) -> np.ndarray:
        env = self._ensure_env()
        obs = env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self.normalizer.normalize_continuous(obs, mode="minmax", bounds=[(-4.8, 4.8), (-5, 5), (-0.418, 0.418), (-5, 5)])

    def step(self, action: str | int):
        env = self._ensure_env()
        if isinstance(action, int):
            action = self.actions.to_action(action)
        gym_action = 1 if action in {"UP", "RIGHT"} else 0
        obs = env.step(gym_action)
        if len(obs) == 5:
            observation, reward, terminated, truncated, info = obs
            done = terminated or truncated
        else:
            observation, reward, done, info = obs
        return self.normalizer.normalize_continuous(observation, mode="minmax", bounds=[(-4.8, 4.8), (-5, 5), (-0.418, 0.418), (-5, 5)]), float(reward), bool(done), info

    def get_observation_space(self) -> dict:
        return {"shape": [4], "dtype": "float32", "low": [-1.0] * 4, "high": [1.0] * 4}

    def get_action_space(self) -> dict:
        return {"type": "discrete", "size": 5, "actions": self.actions.actions}
