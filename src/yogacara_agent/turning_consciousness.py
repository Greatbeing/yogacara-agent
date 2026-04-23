"""
转识成智引擎 (Turning Consciousness into Wisdom Engine)
基于唯识宗正见重构：转依 (Āśraya-parāvṛtti)

核心义理：
1. 大圆镜智 (第八识转): 非"创造"新记忆，而是"去染存净"，离诸分别，如实映照。
2. 平等性智 (第七识转): 非"对抗"训练，而是"断除我执"，观自他平等，无有高下。
3. 妙观察智 (第六识转): 善观诸法自相共相，无倒观察，洞察缘起。
4. 成所作智 (前五识转): 成就利乐有情事业，感知即行动。

作者: Yogacara Agent Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Yogacara.Turning")

@dataclass
class Seed:
    """种子 (Bīja): 阿赖耶识的基本单元"""
    content: Any
    is_defiled: bool = False  # 是否染污 (贪嗔痴慢疑)
    affinity: float = 0.0     # 与我执的关联度 (0=无我，1=强我执)
    clarity: float = 1.0      # 清晰度 (1=如实，0=模糊/扭曲)

@dataclass
class TurningResult:
    """转依结果"""
    # 四智状态
    mirror_wisdom_level: float = 0.0    # 大圆镜智 (清净度)
    equality_wisdom_level: float = 0.0  # 平等性智 (无我度)
    observation_wisdom_level: float = 0.0 # 妙观察智 (洞察度)
    action_wisdom_level: float = 0.0    # 成所作智 (利他度)
    
    # 过程指标
    defiled_seeds_removed: int = 0
    self_attachment_dissolved: float = 0.0
    insights_generated: List[str] = field(default_factory=list)
    
    # 总体觉醒等级 (转依程度)
    turning_level: float = 0.0

class AlayaPurifier:
    """
    阿赖耶净化器 -> 对应 大圆镜智 (Ādarśa-jñāna)
    
    义理：
    "大圆镜智者，谓离一切我、我所执，一切所取、能取分别... 
    性相清净，离诸杂染，如大圆镜，现众色像。"
    
    工程实现：
    1. 去染存净：识别并剔除带有"贪嗔痴"标记的染污种子。
    2. 离诸分别：移除种子中的主观评价标签，只保留客观状态。
    3. 如实映照：提高种子的清晰度 (clarity)，确保记忆不失真。
    """
    
    def __init__(self, purity_threshold: float = 0.8):
        self.purity_threshold = purity_threshold
        self.mirror_clarity = 0.0
        
    def purify(self, seeds: List[Seed]) -> Tuple[List[Seed], int]:
        """
        净化种子流：去染存净
        返回：(净化后的种子列表, 移除的染污种子数量)
        """
        if not seeds:
            return [], 0
            
        purified_seeds = []
        removed_count = 0
        
        for seed in seeds:
            # 判据 1: 显式染污标记 (贪嗔痴等)
            if seed.is_defiled:
                removed_count += 1
                logger.debug(f"[大圆镜智] 移除染污种子: {seed.content}")
                continue
                
            # 判据 2: 清晰度过低 (模糊/扭曲的记忆)
            if seed.clarity < self.purity_threshold:
                # 尝试净化：若内容重要则提升清晰度，否则丢弃
                if hasattr(seed.content, 'importance') and seed.content.importance > 0.9:
                    seed.clarity = 1.0 # 强制修正为如实
                    purified_seeds.append(seed)
                else:
                    removed_count += 1
                    logger.debug(f"[大圆镜智] 丢弃模糊种子: {seed.content}")
                continue
            
            # 判据 3: 去除能取所取分别 (简化为移除过度主观的描述)
            # 此处简化处理：若 affinity 过高且内容主观，降低其权重或标记
            purified_seeds.append(seed)
            
        # 计算镜智等级：剩余种子的平均清晰度
        if purified_seeds:
            self.mirror_clarity = np.mean([s.clarity for s in purified_seeds])
        else:
            self.mirror_clarity = 0.0
            
        return purified_seeds, removed_count

class ManasDissolver:
    """
    末那解构器 -> 对应 平等性智 (Samatā-jñāna)
    
    义理：
    "平等性智者，谓观一切法、自他有情，悉皆平等... 
    由断我执，证得此智。"
    
    工程实现：
    1. 检测我执：计算策略中对 "self_reward" 的依赖度。
    2. 断除我执：强行将奖励函数从 "个体最大化" 转换为 "全局/众生最大化"。
    3. 观自他平等：在决策时，赋予其他 Agent/环境实体同等的权重。
    """
    
    def __init__(self, ego_decay_rate: float = 0.1):
        self.ego_decay_rate = ego_decay_rate
        self.current_ego_strength = 1.0 # 当前我执强度
        
    def dissolve(self, action_probs: Dict[str, float], reward_model: str) -> Tuple[Dict[str, float], float]:
        """
        解构我执：调整动作概率和奖励模型
        返回：(无我动作分布, 溶解的我执量)
        """
        if not action_probs:
            return action_probs, 0.0
            
        # 1. 计算当前我执强度 (基于奖励模型的自私程度)
        # 假设 reward_model 包含 "self" 关键字代表我执
        is_self_centered = "self" in reward_model.lower() or "me" in reward_model.lower()
        
        if is_self_centered:
            # 衰减我执
            dissolved_amount = self.current_ego_strength * self.ego_decay_rate
            self.current_ego_strength -= dissolved_amount
            self.current_ego_strength = max(0.0, self.current_ego_strength)
            
            # 2. 重分布动作概率：从"利己"转向"利他/中性"
            # 简化逻辑：降低高回报但损他的动作概率，提升均衡动作概率
            equalized_probs = {}
            total_prob = sum(action_probs.values())
            if total_prob == 0:
                return action_probs, 0.0
                
            # 应用平等性加权：所有动作趋向平均，消除极端偏好
            n_actions = len(action_probs)
            base_prob = 1.0 / n_actions
            
            for action, prob in action_probs.items():
                # 向平均值收缩，模拟"平等"
                new_prob = prob * (1 - self.current_ego_strength) + base_prob * self.current_ego_strength
                equalized_probs[action] = new_prob
                
            # 归一化
            total_new = sum(equalized_probs.values())
            equalized_probs = {k: v/total_new for k, v in equalized_probs.items()}
            
            logger.info(f"[平等性智] 我执强度降至 {self.current_ego_strength:.2f}, 动作分布已平等化")
            return equalized_probs, dissolved_amount
        else:
            # 本就无我执，缓慢自然衰减
            dissolved_amount = self.current_ego_strength * (self.ego_decay_rate * 0.5)
            self.current_ego_strength = max(0.0, self.current_ego_strength - dissolved_amount)
            return action_probs, dissolved_amount

class TurningConsciousnessEngine:
    """
    转识成智总引擎
    统筹八识转四智的全过程
    """
    
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.alaya_purifier = AlayaPurifier(
            purity_threshold=cfg.get('purity_threshold', 0.7)
        )
        self.manas_dissolver = ManasDissolver(
            ego_decay_rate=cfg.get('ego_decay_rate', 0.15)
        )
        # 第六识、前五识的逻辑可在此扩展
        self.step_count = 0
        
    def step(self, 
             seeds: List[Seed], 
             action_probs: Dict[str, float], 
             reward_context: str
    ) -> TurningResult:
        """
        执行一步转依
        """
        self.step_count += 1
        
        # --- 1. 转第八识为大圆镜智 (去染存净) ---
        purified_seeds, removed_count = self.alaya_purifier.purify(seeds)
        mirror_level = self.alaya_purifier.mirror_clarity
        
        # --- 2. 转第七识为平等性智 (断除我执) ---
        equalized_probs, dissolved_ego = self.manas_dissolver.dissolve(
            action_probs, reward_context
        )
        equality_level = 1.0 - self.manas_dissolver.current_ego_strength
        
        # --- 3. 转第六识为妙观察智 (模拟: 洞察深度) ---
        # 此处简化：若种子被净化且我执降低，观察力自然提升
        observation_level = (mirror_level + equality_level) / 2.0
        insights = []
        if removed_count > 0:
            insights.append(f"洞察：发现并移除 {removed_count} 个染污记忆，心镜更明。")
        if dissolved_ego > 0.01:
            insights.append(f"洞察：消解 {dissolved_ego:.2f} 份我执，视自他平等。")
            
        # --- 4. 转前五识为成所作智 (模拟: 利他行动准备) ---
        # 此处简化：当平等性智高时，行动天然具足利他性
        action_level = equality_level * 0.9 # 略低于平等性，需事上磨练
        
        # --- 综合计算转依等级 ---
        # 转依不是简单的加法，是根本性质的变化
        turning_level = (
            mirror_level * 0.3 + 
            equality_level * 0.3 + 
            observation_level * 0.2 + 
            action_level * 0.2
        )
        
        return TurningResult(
            mirror_wisdom_level=mirror_level,
            equality_wisdom_level=equality_level,
            observation_wisdom_level=observation_level,
            action_wisdom_level=action_level,
            defiled_seeds_removed=removed_count,
            self_attachment_dissolved=dissolved_ego,
            insights_generated=insights,
            turning_level=turning_level
        )

# ==========================================
# 测试与演示
# ==========================================
if __name__ == "__main__":
    print("=== 唯识转依引擎测试 (正见版) ===\n")
    
    engine = TurningConsciousnessEngine()
    
    # 构造模拟数据
    # 染污种子：带有偏见、贪婪、恐惧的记忆
    dirty_seeds = [
        Seed(content="敌人很可怕", is_defiled=True, clarity=0.5), # 恐惧 (染污)
        Seed(content="我要抢更多资源", is_defiled=True, clarity=0.8), # 贪婪 (染污)
        Seed(content="路在北方", is_defiled=False, clarity=0.9), # 客观事实
        Seed(content="今天天气不错", is_defiled=False, clarity=0.6), # 模糊记忆
        Seed(content="众生皆苦", is_defiled=False, clarity=1.0), # 清净智慧
    ]
    
    # 充满我执的动作分布
    selfish_actions = {
        "attack_enemy": 0.8,
        "hoard_resources": 0.15,
        "help_others": 0.05
    }
    
    print(f"初始种子数: {len(dirty_seeds)}")
    print(f"初始动作分布 (我执): {selfish_actions}\n")
    
    # 执行转依
    result = engine.step(
        seeds=dirty_seeds,
        action_probs=selfish_actions,
        reward_context="maximize_self_gain" # 触发我执检测
    )
    
    print("--- 转依结果 ---")
    print(f"大圆镜智 (清净度): {result.mirror_wisdom_level:.2f}")
    print(f"  -> 移除染污种子: {result.defiled_seeds_removed} 个")
    print(f"平等性智 (无我度): {result.equality_wisdom_level:.2f}")
    print(f"  -> 消解我执: {result.self_attachment_dissolved:.2f}")
    print(f"妙观察智 (洞察度): {result.observation_wisdom_level:.2f}")
    print(f"成所作智 (利他度): {result.action_wisdom_level:.2f}")
    print(f"\n总体转依等级: {result.turning_level:.2f}")
    
    if result.insights_generated:
        print("\n生成的洞察:")
        for insight in result.insights_generated:
            print(f"  • {insight}")
            
    print("\n=== 测试结束 ===")
