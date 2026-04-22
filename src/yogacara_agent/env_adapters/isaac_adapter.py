from .base import BaseSimEnv


class IsaacEnv(BaseSimEnv):
    def __init__(self, task_cls, num_envs=1):
        self.env = task_cls(num_envs=num_envs)
        self.env.reset()

    def reset(self):
        self.env.reset()
        return self._observe()

    def step(self, action: str):
        act = {"UP": [1, 0], "DOWN": [-1, 0], "LEFT": [0, -1], "RIGHT": [0, 1], "STAY": [0, 0]}.get(action, [0, 0])
        obs, rew, dones, info = self.env.step([act])
        return self._observe(), float(rew[0]), bool(dones[0]), info

    def _observe(self):
        obs = self.env.get_observations()
        return {"grid_view": obs[0][:9].tolist(), "pos": (float(obs[0][9]), float(obs[0][10])), "step": 0}
