import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from .base import BaseSimEnv

class UnityEnv(BaseSimEnv):
    def __init__(self, file_path=None, worker_id=0):
        self.channel = EngineConfigurationChannel(); self.env = UnityEnvironment(file_name=file_path, worker_id=worker_id, side_channels=[self.channel])
        self.env.reset(); self.behavior_name = list(self.env.behavior_specs)[0]; self.step_count = 0
    def reset(self): self.env.reset(); self.step_count = 0; return self._observe()
    def step(self, action: str):
        act_idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "STAY": 4}.get(action, 4)
        self.env.set_actions(self.behavior_name, np.array([[act_idx]])); self.env.step(); self.step_count += 1; return self._observe(), -0.1, False, {}
    def _observe(self):
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs_vec = decision_steps.obs[0][0] if len(decision_steps) > 0 else np.zeros(11)
        return {"grid_view": obs_vec[:9].tolist(), "pos": (float(obs_vec[9]), float(obs_vec[10])), "step": self.step_count}
