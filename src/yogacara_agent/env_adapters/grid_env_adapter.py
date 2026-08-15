"""栅格世界适配器。"""

from __future__ import annotations

import numpy as np

from yogacara_agent.yogacara_test import GridSimEnv

from .action_mapper import ActionMapper
from .base import BaseSimEnv
from .observation_normalizer import ObservationNormalizer


class GridEnvAdapter(BaseSimEnv):
    def __init__(self):
        self.env = GridSimEnv()
        self.actions = ActionMapper()
        self.normalizer = ObservationNormalizer()

    def _format_obs(self, obs: dict) -> np.ndarray:
        pos = obs.get("pos", (0, 0))
        grid = self.normalizer.normalize_grid(obs.get("grid_view", [0.0] * 9))
        extras = np.asarray([pos[0] / 10.0, pos[1] / 10.0], dtype=np.float32)
        return np.concatenate([grid, extras]).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = self.env.reset()
        return self._format_obs(obs)

    def step(self, action: str | int):
        if isinstance(action, int):
            action = self.actions.to_action(action)
        obs, reward, done = self.env.step(action)
        return self._format_obs(obs), float(reward), bool(done), {"raw_obs": obs, "action": action}

    def get_observation_space(self) -> dict:
        return {"shape": [11], "dtype": "float32", "low": [-1.0] * 11, "high": [1.0] * 11}

    def get_action_space(self) -> dict:
        return {"type": "discrete", "size": 5, "actions": self.actions.actions}
