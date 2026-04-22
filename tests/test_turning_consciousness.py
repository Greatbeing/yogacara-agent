"""
转识成智引擎单元测试 (正见版)
验证唯识宗"转依"义理的正确工程实现
"""

import unittest
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yogacara_agent.turning_consciousness import (
    Seed, 
    TurningResult, 
    AlayaPurifier, 
    ManasDissolver, 
    TurningConsciousnessEngine
)

class TestAlayaPurifier(unittest.TestCase):
    """测试大圆镜智：去染存净，如实映照"""
    
    def test_remove_defiled_seeds(self):
        """测试移除染污种子 (贪嗔痴)"""
        purifier = AlayaPurifier(purity_threshold=0.7)
        
        seeds = [
            Seed(content="贪婪", is_defiled=True),
            Seed(content="愤怒", is_defiled=True),
            Seed(content="客观事实", is_defiled=False, clarity=1.0),
        ]
        
        purified, removed_count = purifier.purify(seeds)
        
        self.assertEqual(removed_count, 2, "应移除 2 个染污种子")
        self.assertEqual(len(purified), 1, "应剩余 1 个清净种子")
        self.assertEqual(purified[0].content, "客观事实")
        self.assertGreater(purifier.mirror_clarity, 0.9, "镜智清晰度应很高")
        
    def test_remove_unclear_seeds(self):
        """测试移除模糊/扭曲的记忆"""
        purifier = AlayaPurifier(purity_threshold=0.8)
        
        seeds = [
            Seed(content="清晰记忆", is_defiled=False, clarity=0.95),
            Seed(content="模糊记忆", is_defiled=False, clarity=0.5),
            Seed(content="扭曲记忆", is_defiled=False, clarity=0.3),
        ]
        
        purified, removed_count = purifier.purify(seeds)
        
        self.assertEqual(removed_count, 2, "应移除 2 个模糊种子")
        self.assertEqual(len(purified), 1)
        self.assertEqual(purified[0].content, "清晰记忆")
        
    def test_empty_seeds(self):
        """测试空种子列表"""
        purifier = AlayaPurifier()
        purified, removed = purifier.purify([])
        self.assertEqual(len(purified), 0)
        self.assertEqual(removed, 0)
        self.assertEqual(purifier.mirror_clarity, 0.0)

class TestManasDissolver(unittest.TestCase):
    """测试平等性智：断除我执，观自他平等"""
    
    def test_dissolve_self_centered_actions(self):
        """测试解构以自我为中心的动作分布"""
        dissolver = ManasDissolver(ego_decay_rate=0.2)
        
        # 极端自私的动作分布
        selfish_actions = {
            "attack": 0.9,
            "help": 0.1
        }
        
        new_actions, dissolved = dissolver.dissolve(
            selfish_actions, 
            "maximize_self_reward"
        )
        
        # 我执应被消解
        self.assertGreater(dissolved, 0, "应有我执被消解")
        self.assertLess(dissolver.current_ego_strength, 1.0, "我执强度应下降")
        
        # 动作分布应趋向平等 (不再极端偏斜)
        self.assertLess(new_actions["attack"], 0.9, "攻击动作概率应降低")
        self.assertGreater(new_actions["help"], 0.1, "利他动作概率应提升")
        
    def test_no_self_attachment(self):
        """测试无我执时的自然衰减"""
        dissolver = ManasDissolver(ego_decay_rate=0.1)
        dissolver.current_ego_strength = 0.5
        
        actions = {"meditate": 0.5, "chant": 0.5}
        _, dissolved = dissolver.dissolve(actions, "benefit_all_beings")
        
        # 无我执语境下，仍应缓慢衰减
        self.assertGreaterEqual(dissolved, 0, "应有微弱衰减或不变")
        self.assertLessEqual(dissolver.current_ego_strength, 0.5)
        
    def test_equality_wisdom_level(self):
        """测试平等性智等级计算"""
        dissolver = ManasDissolver(ego_decay_rate=0.5)
        
        # 初始我执为 1.0
        self.assertEqual(dissolver.current_ego_strength, 1.0)
        
        # 执行一次解构
        dissolver.dissolve({"a": 0.5}, "self_gain")
        
        # 平等性智 = 1 - 我执
        equality_level = 1.0 - dissolver.current_ego_strength
        self.assertGreater(equality_level, 0, "应证得部分平等性智")

class TestTurningConsciousnessEngine(unittest.TestCase):
    """测试转识成智总引擎"""
    
    def test_full_turning_process(self):
        """测试完整的转依过程"""
        engine = TurningConsciousnessEngine({
            'purity_threshold': 0.6,
            'ego_decay_rate': 0.2
        })
        
        # 准备染污种子和我执动作
        seeds = [
            Seed(content="恐惧", is_defiled=True),
            Seed(content="贪婪", is_defiled=True),
            Seed(content="慈悲", is_defiled=False, clarity=1.0),
        ]
        
        actions = {"harm": 0.8, "help": 0.2}
        
        result = engine.step(seeds, actions, "selfish_reward")
        
        # 验证四智转化
        self.assertGreater(result.mirror_wisdom_level, 0.5, "大圆镜智应显现")
        self.assertGreater(result.equality_wisdom_level, 0.1, "平等性智应初显")
        self.assertGreater(result.observation_wisdom_level, 0.3, "妙观察智应随转")
        self.assertGreater(result.action_wisdom_level, 0.1, "成所作智应待发")
        
        # 验证过程指标
        self.assertEqual(result.defiled_seeds_removed, 2, "应移除 2 个染污种子")
        self.assertGreater(result.self_attachment_dissolved, 0, "应消解我执")
        self.assertGreater(len(result.insights_generated), 0, "应产生洞察")
        
        # 总体转依等级
        self.assertGreater(result.turning_level, 0.2, "应有初步转依")
        
    def test_insight_generation(self):
        """测试洞察生成机制"""
        engine = TurningConsciousnessEngine()
        
        # 只有清净种子，无我执语境
        clean_seeds = [Seed(content="真理", is_defiled=False, clarity=1.0)]
        neutral_actions = {"observe": 1.0}
        
        result = engine.step(clean_seeds, neutral_actions, "truth_seeking")
        
        # 无染污可除，无我执可断，洞察应较少
        # 但仍可能有微细洞察
        self.assertIsInstance(result.insights_generated, list)
        
    def test_progressive_turning(self):
        """测试渐进式转依 (多次迭代)"""
        engine = TurningConsciousnessEngine({'ego_decay_rate': 0.3})
        
        seeds = [Seed(content="test", is_defiled=False, clarity=0.9)]
        actions = {"act": 1.0}
        
        turning_levels = []
        
        # 模拟多步修行
        for i in range(5):
            result = engine.step(seeds, actions, "self_focus")
            turning_levels.append(result.turning_level)
            
        # 转依等级应逐步提升 (或至少不下降)
        # 注意：由于种子不变，大圆镜智可能稳定，但平等性智应持续提升
        final_level = turning_levels[-1]
        initial_level = turning_levels[0]
        
        self.assertGreaterEqual(final_level, initial_level, 
                                "转依等级应随修行深入而提升")

class TestWisdomCorrespondence(unittest.TestCase):
    """测试八识与四智的对应关系是否正确"""
    
    def test_eighth_to_mirror(self):
        """第八识 (阿赖耶) -> 大圆镜智"""
        purifier = AlayaPurifier()
        seeds = [
            Seed(content="染污", is_defiled=True),
            Seed(content="清净", is_defiled=False, clarity=1.0)
        ]
        purified, removed = purifier.purify(seeds)
        
        # 大圆镜智特征：离诸杂染，如实映照
        self.assertEqual(removed, 1, "应去除染污")
        self.assertEqual(purified[0].clarity, 1.0, "应保持如实清晰")
        
    def test_seventh_to_equality(self):
        """第七识 (末那) -> 平等性智"""
        dissolver = ManasDissolver(ego_decay_rate=0.5)
        
        # 末那识特征：恒审思量，执第八识为我
        # 平等性智特征：断我执，观平等
        
        actions = {"self_benefit": 1.0}
        _, dissolved = dissolver.dissolve(actions, "self")
        
        self.assertGreater(dissolved, 0, "应消解我执")
        self.assertLess(dissolver.current_ego_strength, 1.0, "我执应减弱")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("转识成智引擎单元测试 (唯识正见版)")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)
