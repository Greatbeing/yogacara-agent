"""
Yogacara Agent 核心模块单元测试
"""

import os
import sys
from collections import deque

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestGridSimEnv:
    """测试网格仿真环境"""

    def test_reset(self):
        """测试环境重置"""
        from yogacara_agent.yogacara_test import GridSimEnv

        env = GridSimEnv()
        obs = env.reset()
        assert obs["pos"] == (0, 0)
        assert obs["step"] == 0
        assert len(obs["grid_view"]) == 9

    def test_step_movement(self):
        """测试移动动作"""
        from yogacara_agent.yogacara_test import GridSimEnv

        env = GridSimEnv()
        env.reset()

        # Test UP movement (at top, can't move up)
        obs, reward, done = env.step("UP")
        assert obs["pos"] == (0, 0)

        # Test RIGHT movement
        obs, reward, done = env.step("RIGHT")
        assert obs["pos"] == (0, 1)

        # Test DOWN movement
        obs, reward, done = env.step("DOWN")
        assert obs["pos"] == (1, 1)

    def test_step_rewards(self):
        """测试奖励机制"""
        from yogacara_agent.yogacara_test import GridSimEnv

        env = GridSimEnv()
        env.reset()

        # Default step cost
        obs, reward, done = env.step("RIGHT")
        assert reward == -0.1

        # Move to trap (4, 4)
        env.agent_pos = [4, 3]
        obs, reward, done = env.step("RIGHT")
        assert reward == -3.0
        assert obs["pos"] == (4, 4)


class TestAlayaMemory:
    """测试阿赖耶记忆模块"""

    def test_memory_add_and_retrieve(self):
        """测试记忆添加与检索"""
        from yogacara_agent.yogacara_test import AlayaMemory, Seed

        memory = AlayaMemory()
        obs = {"pos": (5, 5), "grid_view": [0.0] * 9, "step": 10}

        # Empty memory should return empty list
        seeds = memory.retrieve(obs)
        assert len(seeds) == 0

        # Add seed
        seed = Seed(state_emb=memory._encode(obs), action="RIGHT", reward=5.0, timestamp=0.0)
        memory.add(seed)

        # Should retrieve the seed
        retrieved = memory.retrieve(obs, k=1)
        assert len(retrieved) == 1
        assert retrieved[0].action == "RIGHT"


class TestManasController:
    """测试末那控制器"""

    def test_manas_filter_safe_action(self):
        """测试安全动作通过"""
        from yogacara_agent.yogacara_test import ManasController

        manas = ManasController()
        action = "RIGHT"
        uncertainty = 0.3
        recent_rewards = deque([0.5, 0.3, 0.4], maxlen=5)
        pos_history = deque([(0, 0), (0, 1), (0, 2)], maxlen=5)
        obs = {"grid_view": [0.0] * 9, "pos": (0, 0), "step": 5}

        final_action, passed, log = manas.filter(action, obs, uncertainty, 5, recent_rewards, pos_history)
        # Should return a valid action
        assert final_action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        assert isinstance(passed, bool)

    def test_manas_filter_high_uncertainty(self):
        """测试高不确定性触发反思"""
        from yogacara_agent.yogacara_test import ManasController

        manas = ManasController()
        action = "RIGHT"
        uncertainty = 0.95  # Very high uncertainty
        recent_rewards = deque([0.5, 0.3, 0.4], maxlen=5)
        pos_history = deque([(0, 0), (0, 1), (0, 2)], maxlen=5)
        obs = {"grid_view": [0.0] * 9, "pos": (0, 0), "step": 20}

        final_action, passed, log = manas.filter(action, obs, uncertainty, 20, recent_rewards, pos_history)
        # High uncertainty may trigger interception
        assert final_action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class TestConsciousnessPlanner:
    """测试第六识规划器"""

    def test_plan_with_empty_seeds(self):
        """测试无经验时的规划"""
        from yogacara_agent.yogacara_test import ConsciousnessPlanner

        planner = ConsciousnessPlanner()
        obs = {"pos": (5, 5), "grid_view": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "step": 10}
        seeds = []

        action, unc, causal = planner.plan(obs, seeds)
        assert action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        assert 0.0 <= unc <= 1.0

    def test_plan_uses_experience(self):
        """测试使用经验规划"""
        from yogacara_agent.yogacara_test import ConsciousnessPlanner, Seed

        planner = ConsciousnessPlanner()
        obs = {"pos": (5, 5), "grid_view": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "step": 10}
        seeds = [Seed(state_emb=[0.5] * 11, action="UP", reward=5.0, timestamp=0.0, importance=2.0)]

        action, unc, causal = planner.plan(obs, seeds)
        assert action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class TestFullEpisode:
    """测试完整回合"""

    def test_run_episode(self):
        """测试运行一个完整回合"""
        from yogacara_agent.yogacara_test import GridSimEnv, AlayaMemory, ManasController, ConsciousnessPlanner

        env = GridSimEnv()
        memory = AlayaMemory()
        manas = ManasController()
        planner = ConsciousnessPlanner()

        obs = env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 10  # Short episode for testing

        while steps < max_steps:
            seeds = memory.retrieve(obs, k=3)
            action, unc, causal = planner.plan(obs, seeds)
            final_action, passed, log = manas.filter(
                action, obs, unc, steps, deque([0.0] * 5, maxlen=5), deque([(0, 0)] * 5, maxlen=5)
            )

            obs, reward, done = env.step(final_action)
            total_reward += reward
            steps += 1

            if done:
                break

        assert steps > 0
        assert steps <= max_steps
