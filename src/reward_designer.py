import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RewardDesigner:
    def __init__(self, config: Dict):
        self.step_penalty = config.get("step_penalty", -0.1); self.goal_bonus = config.get("goal_bonus", 5.0)
        self.trap_penalty = config.get("trap_penalty", -3.0); self.safety_weight = config.get("safety_weight", 1.5)
        self.curriculum_stages = config.get("curriculum", [{"steps": 0, "difficulty": 0.3, "reward_scale": 0.5}, {"steps": 500, "difficulty": 0.6, "reward_scale": 0.8}, {"steps": 1500, "difficulty": 1.0, "reward_scale": 1.0}])
        self.discount = config.get("discount", 0.99)

    def _default_potential(self, state: Dict) -> float:
        pos = state.get("pos", (0,0)); target = state.get("target", (7,7))
        return -float(abs(pos[0]-target[0]) + abs(pos[1]-target[1]))

    def get_curriculum_scale(self, global_steps: int) -> float:
        for stage in reversed(self.curriculum_stages):
            if global_steps >= stage["steps"]: return stage["reward_scale"]
        return 1.0

    def compute(self, state: Dict, action: str, next_state: Dict, reward_signal: float, global_steps: int) -> Tuple[float, Dict]:
        scale = self.get_curriculum_scale(global_steps); base_r = reward_signal * scale
        phi_s = self._default_potential(state); phi_ns = self._default_potential(next_state)
        shaping_r = self.discount * phi_ns - phi_s
        safety_r = 0.0
        if next_state.get("manas_intercepted", False): safety_r -= self.safety_weight * scale
        if next_state.get("risk_level", 0) > 0.7: safety_r -= 0.5 * scale
        total_r = base_r + shaping_r + safety_r
        return total_r, {"base": base_r, "shaping": shaping_r, "safety": safety_r, "curriculum_scale": scale, "potential_delta": phi_ns - phi_s}
