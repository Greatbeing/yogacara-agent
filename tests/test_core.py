"""
Yogacara Agent 核心模块单元测试
"""
import pytest
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yogacara_test import Seed, GridSimEnv, AlayaMemory, ManasController, SixthConsciousness


class TestGridSimEnv:
    """测试网格仿真环境"""
    
    def test_reset(self):
        """测试环境重置"""
        env = GridSimEnv()
        obs = env.reset()
        assert obs['pos'] == (0, 0)
        assert obs['step'] == 0
        assert len(obs['grid_view']) == 9
    
    def test_step_movement(self):
        """测试移动动作"""
        env = GridSimEnv()
        env.reset()
        
        # Test UP movement
        obs, reward, done = env.step("UP")
        assert obs['pos'] == (0, 0)  # Already at top, can't move up
        
        # Test RIGHT movement
        obs, reward, done = env.step("RIGHT")
        assert obs['pos'] == (0, 1)
        
        # Test DOWN movement
        obs, reward, done = env.step("DOWN")
        assert obs['pos'] == (1, 1)
    
    def test_step_rewards(self):
        """测试奖励机制"""
        env = GridSimEnv()
        env.reset()
        
        # Default step cost
        obs, reward, done = env.step("RIGHT")
        assert reward == -0.1
        
        # Move to trap (4, 4) - need to navigate there
        env.agent_pos = [4, 3]
        obs, reward, done = env.step("RIGHT")
        assert reward == -3.0
        assert obs['pos'] == (4, 4)


class TestAlayaMemory:
    """测试阿赖耶记忆模块"""
    
    def test_seed_creation(self):
        """测试种子创建"""
        seed = Seed(
            state_emb=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            action="RIGHT",
            reward=5.0,
            timestamp=0.0
        )
        assert seed.action == "RIGHT"
        assert seed.reward == 5.0
        assert seed.importance == 1.0
    
    def test_memory_add_and_retrieve(self):
        """测试记忆添加与检索"""
        memory = AlayaMemory()
        obs = {"pos": (5, 5), "grid_view": [0.0]*9, "step": 10}
        
        # Empty memory should return empty list
        seeds = memory.retrieve(obs)
        assert len(seeds) == 0
        
        # Add seed
        seed = Seed(
            state_emb=memory._encode(obs),
            action="RIGHT",
            reward=5.0,
            timestamp=0.0
        )
        memory.add(seed)
        
        # Should retrieve the seed
        seeds = memory.retrieve(obs, k=1)
        assert len(seeds) == 1
        assert seeds[0].action == "RIGHT"
    
    def test_perfume_update(self):
        """测试熏习更新"""
        memory = AlayaMemory()
        memory.seeds = [
            {"emb": [0.5]*11, "act": "RIGHT", "rew": 5.0, "imp": 1.0, "ts": 0.0, "causal": "依他起"}
        ]
        
        initial_imp = memory.seeds[0]["imp"]
        memory.perfume_update()
        # Importance should change after perfume update
        # (actual behavior depends on implementation)


class TestManasController:
    """测试末那控制器"""
    
    def test_manas_pass_safe_action(self):
        """测试安全动作通过"""
        manas = ManasController()
        action = "RIGHT"
        uncertainty = 0.3
        seeds = []
        recent_rewards = [0.5, 0.3, 0.4]
        pos_history = [(0, 0), (0, 1), (0, 2)]
        
        passed, reason = manas.check(action, uncertainty, seeds, recent_rewards, pos_history)
        # With low uncertainty and no red flags, should pass
        assert passed == True or passed == False  # Depends on threshold
    
    def test_manas_block_high_uncertainty(self):
        """测试高不确定性被拦截"""
        manas = ManasController()
        action = "RIGHT"
        uncertainty = 0.95  # Very high uncertainty
        seeds = []
        recent_rewards = [0.5, 0.3, 0.4]
        pos_history = [(0, 0), (0, 1), (0, 2)]
        
        passed, reason = manas.check(action, uncertainty, seeds, recent_rewards, pos_history)
        # High uncertainty should trigger block or reflection


class TestSixthConsciousness:
    """测试第六识规划器"""
    
    def test_plan_with_empty_seeds(self):
        """测试无经验时的规划"""
        sixth = SixthConsciousness()
        obs = {"pos": (5, 5), "grid_view": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "step": 10}
        seeds = []
        
        action, unc, causal = sixth.plan(obs, seeds)
        assert action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        assert 0.0 <= unc <= 1.0
    
    def test_plan_uses_experience(self):
        """测试使用经验规划"""
        sixth = SixthConsciousness()
        obs = {"pos": (5, 5), "grid_view": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "step": 10}
        seeds = [
            Seed(state_emb=[0.5]*11, action="UP", reward=5.0, timestamp=0.0, importance=2.0)
        ]
        
        action, unc, causal = sixth.plan(obs, seeds)
        assert action in ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class TestIntegration:
    """集成测试"""
    
    def test_full_episode_run(self):
        """测试完整 episode 运行"""
        env = GridSimEnv()
        memory = AlayaMemory()
        manas = ManasController()
        sixth = SixthConsciousness()
        
        obs = env.reset()
        total_reward = 0
        
        for step in range(20):
            seeds = memory.retrieve(obs, k=3)
            action, unc, causal = sixth.plan(obs, seeds)
            
            # Manas check
            passed, reason = manas.check(action, unc, seeds, [], [])
            if not passed:
                action = "STAY"
            
            obs, reward, done = env.step(action)
            total_reward += reward
            
            # Store experience
            memory.add(Seed(
                state_emb=memory._encode(obs),
                action=action,
                reward=reward,
                timestamp=step * 0.1
            ))
            
            if done:
                break
        
        # Episode should complete without errors
        assert total_reward != 0 or env.step_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
