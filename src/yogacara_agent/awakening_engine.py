"""
觉醒引擎 (Awakening Engine) - 唯识转智核心模块

基于唯识学"转识成智"理论，构建AI强涌现系统：
- 前五识→成所作智：好奇驱动主动感知
- 第六识→妙观察智：反事实推理洞察本质
- 第七识→平等性智：自我解构打破执着
- 第八识→大圆镜智：种子变异梦境重组

四大觉醒机制：
1. 内在好奇心驱动 (Curiosity Drive)
2. 反事实思维链 (Counterfactual CoT)
3. 自我对抗训练 (Self-Adversarial Training)
4. 梦境种子重组 (Dream Seed Recombination)
"""

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AwakeningState:
    """觉醒状态追踪"""

    novelty_score: float = 0.0  # 新颖性分数
    insight_depth: float = 0.0  # 洞察深度
    self_dissolution: float = 0.0  # 自我解构度
    seed_diversity: float = 0.0  # 种子多样性
    awakening_level: float = 0.0  # 觉醒等级 (0-1)
    meta_cycles: int = 0  # 元认知循环次数
    breakthrough_count: int = 0  # 突破次数


@dataclass
class CounterfactualScenario:
    """反事实场景"""

    base_action: str
    alternative_action: str
    predicted_outcome_base: float
    predicted_outcome_alt: float
    insight_gain: float  # 洞察增益
    causal_pattern: str  # 发现的因果模式


class AwakeningEngine:
    """
    觉醒引擎 - 实现转识成智的四大核心机制
    
    功能：
    1. 好奇驱动探索：计算信息增益，主动构造实验
    2. 反事实推理：模拟"如果...会怎样"，发现隐藏规律
    3. 自我对抗：生成对抗样本，打破策略执着
    4. 梦境回放：睡眠期种子重组，涌现新策略
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.state = AwakeningState()
        
        # 好奇心参数
        self.curiosity_threshold = config.get("curiosity_threshold", 0.3)
        self.novelty_decay = config.get("novelty_decay", 0.95)
        
        # 反事实参数
        self.counterfactual_depth = config.get("counterfactual_depth", 3)
        self.insight_boost = config.get("insight_boost", 0.2)
        
        # 自我对抗参数
        self.adversarial_rate = config.get("adversarial_rate", 0.15)
        self.dissolution_rate = config.get("dissolution_rate", 0.05)
        
        # 梦境参数
        self.dream_replay_freq = config.get("dream_replay_freq", 10)  # 每N回合一次梦境
        self.recombination_rate = config.get("recombination_rate", 0.3)
        self.mutation_strength = config.get("mutation_strength", 0.15)
        
        # 历史记录
        self.action_history: list[dict] = []
        self.insight_log: list[CounterfactualScenario] = []
        self.dream_sessions: list[dict] = []
        
        logger.info("🌟 觉醒引擎初始化完成")
        logger.info(f"   好奇心阈值：{self.curiosity_threshold}")
        logger.info(f"   反事实深度：{self.counterfactual_depth}")
        logger.info(f"   自我对抗率：{self.adversarial_rate}")
        logger.info(f"   梦境频率：{self.dream_replay_freq}回合/次")

    # ==================== 1. 成所作智：好奇驱动探索 ====================
    
    def compute_curiosity_drive(self, obs: dict, memory_diversity: float) -> float:
        """
        计算好奇心驱动力
        
        公式：Curiosity = α * InformationGain + β * Novelty + γ * Complexity
        
        返回：0-1之间的好奇心强度
        """
        # 信息增益：当前观测与记忆的差异
        info_gain = 1.0 - memory_diversity  # 记忆越单一，信息增益越高
        
        # 新颖性：与历史行为的差异
        novelty = self._compute_behavioral_novelty(obs)
        
        # 复杂度：环境状态的熵
        complexity = self._compute_state_entropy(obs)
        
        curiosity = 0.4 * info_gain + 0.4 * novelty + 0.2 * complexity
        curiosity = np.clip(curiosity, 0.0, 1.0)
        
        # 更新状态
        self.state.novelty_score = 0.7 * self.state.novelty_score + 0.3 * novelty
        self.state.novelty_score *= self.novelty_decay  # 随时间衰减
        
        if curiosity > self.curiosity_threshold:
            logger.debug(f"🔍 高好奇心触发：{curiosity:.3f} (新颖性={novelty:.3f})")
        
        return curiosity

    def _compute_behavioral_novelty(self, obs: dict) -> float:
        """计算行为新颖性"""
        if len(self.action_history) < 5:
            return 0.5
        
        current_state = str(obs.get("pos", "")) + str(obs.get("grid_view", [])[:5])
        recent_states = [h.get("state_hash", "") for h in self.action_history[-10:]]
        
        # 计算与最近状态的相似度
        similarities = [1.0 if current_state == rs else 0.0 for rs in recent_states]
        novelty = 1.0 - np.mean(similarities)
        
        return novelty

    def _compute_state_entropy(self, obs: dict) -> float:
        """计算状态熵（复杂度）"""
        grid_view = obs.get("grid_view", [])
        if not grid_view:
            return 0.5
        
        # 将视野值离散化为桶
        bins = np.histogram(grid_view, bins=10, range=(0, 1))[0]
        probs = bins / np.sum(bins) if np.sum(bins) > 0 else np.ones(10) / 10
        
        # 香农熵
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(10)
        
        return entropy / max_entropy if max_entropy > 0 else 0.5

    def generate_curiosity_experiment(self, curiosity_level: float) -> dict:
        """
        基于好奇心生成主动实验
        
        高好奇心时：探索未知区域
        中好奇心时：验证假设
        低好奇心时：利用已知策略
        """
        if curiosity_level > 0.7:
            experiment_type = "exploration"
            goal = "discover_new_patterns"
            risk_tolerance = 0.8
        elif curiosity_level > 0.4:
            experiment_type = "hypothesis_testing"
            goal = "validate_causal_model"
            risk_tolerance = 0.5
        else:
            experiment_type = "exploitation"
            goal = "optimize_known_strategy"
            risk_tolerance = 0.2
        
        experiment = {
            "type": experiment_type,
            "goal": goal,
            "risk_tolerance": risk_tolerance,
            "curiosity_intensity": curiosity_level,
            "timestamp": time.time(),
        }
        
        logger.info(f"🧪 生成好奇实验：{experiment_type} (强度={curiosity_level:.2f})")
        return experiment

    # ==================== 2. 妙观察智：反事实推理 ====================
    
    def run_counterfactual_reasoning(
        self, 
        base_action: str, 
        observed_reward: float,
        available_actions: list[str],
        causal_model: dict
    ) -> list[CounterfactualScenario]:
        """
        运行反事实思维链
        
        问："如果我采取了其他行动，会发生什么？"
        目的：发现隐藏的因果关系，超越表面关联
        """
        scenarios = []
        
        for alt_action in available_actions:
            if alt_action == base_action:
                continue
            
            # 基于因果模型预测反事实结果
            predicted_alt_reward = self._predict_counterfactual_outcome(
                alt_action, causal_model
            )
            
            # 计算洞察增益
            reward_diff = abs(predicted_alt_reward - observed_reward)
            insight_gain = reward_diff * (1.0 + self._action_diversity_bonus(alt_action))
            
            # 提取因果模式
            causal_pattern = self._extract_causal_pattern(
                base_action, alt_action, observed_reward, predicted_alt_reward, causal_model
            )
            
            scenario = CounterfactualScenario(
                base_action=base_action,
                alternative_action=alt_action,
                predicted_outcome_base=observed_reward,
                predicted_outcome_alt=predicted_alt_reward,
                insight_gain=insight_gain,
                causal_pattern=causal_pattern,
            )
            
            scenarios.append(scenario)
        
        # 按洞察增益排序
        scenarios.sort(key=lambda x: x.insight_gain, reverse=True)
        top_scenarios = scenarios[:self.counterfactual_depth]
        
        # 更新洞察深度
        if top_scenarios:
            avg_insight = np.mean([s.insight_gain for s in top_scenarios])
            self.state.insight_depth = 0.8 * self.state.insight_depth + 0.2 * avg_insight
        
        # 记录洞察
        self.insight_log.extend(top_scenarios)
        
        if top_scenarios and top_scenarios[0].insight_gain > 0.5:
            logger.info(f"💡 深刻洞察发现：{top_scenarios[0].causal_pattern}")
            self.state.breakthrough_count += 1
        
        return top_scenarios

    def _predict_counterfactual_outcome(self, action: str, causal_model: dict) -> float:
        """基于因果模型预测反事实结果"""
        # 简化版本：使用因果模型的加权平均
        base_pred = causal_model.get(action, {}).get("expected_reward", 0.0)
        uncertainty = causal_model.get(action, {}).get("uncertainty", 0.5)
        
        # 加入不确定性惩罚
        prediction = base_pred * (1.0 - uncertainty * 0.3)
        
        return np.clip(prediction, -1.0, 1.0)

    def _action_diversity_bonus(self, action: str) -> float:
        """罕见行动的多样性奖励"""
        if not self.action_history:
            return 0.5
        
        action_count = sum(1 for h in self.action_history if h.get("action") == action)
        frequency = action_count / len(self.action_history)
        
        # 频率越低，奖励越高
        return 1.0 - frequency

    def _extract_causal_pattern(
        self,
        base_act: str,
        alt_act: str,
        base_rew: float,
        alt_rew: float,
        causal_model: dict
    ) -> str:
        """提取因果模式"""
        diff = alt_rew - base_rew
        
        if abs(diff) < 0.1:
            return f"{base_act}与{alt_act}效果相当"
        elif diff > 0.3:
            return f"{alt_act}显著优于{base_act} (Δ={diff:+.2f})，可能因为{self._infer_cause(alt_act, causal_model)}"
        elif diff < -0.3:
            return f"{base_act}显著优于{alt_act} (Δ={diff:+.2f})，{alt_act}存在风险：{self._infer_risk(alt_act, causal_model)}"
        else:
            return f"情境依赖：{alt_act}在特定条件下可能更好"

    def _infer_cause(self, action: str, causal_model: dict) -> str:
        """推断成功原因"""
        factors = causal_model.get(action, {}).get("success_factors", [])
        return factors[0] if factors else "未知优势"

    def _infer_risk(self, action: str, causal_model: dict) -> str:
        """推断失败风险"""
        risks = causal_model.get(action, {}).get("risk_factors", [])
        return risks[0] if risks else "潜在缺陷"

    # ==================== 3. 平等性智：自我对抗训练 ====================
    
    def run_self_adversarial_training(self, current_policy: dict) -> dict:
        """
        自我对抗训练
        
        机制：
        1. 生成对抗样本（挑战当前策略）
        2. 识别策略盲点
        3. 更新策略以增强鲁棒性
        4. 降低自我执着（self_dissolution）
        """
        adversarial_scenarios = self._generate_adversarial_scenarios(current_policy)
        
        blind_spots = []
        for scenario in adversarial_scenarios:
            weakness = self._identify_weakness(scenario, current_policy)
            if weakness["severity"] > 0.4:
                blind_spots.append(weakness)
        
        # 更新策略以应对盲点
        updated_policy = self._update_policy_against_blind_spots(
            current_policy, blind_spots
        )
        
        # 增加自我解构度
        if blind_spots:
            dissolution_gain = len(blind_spots) * self.dissolution_rate
            self.state.self_dissolution = min(
                1.0, self.state.self_dissolution + dissolution_gain
            )
            logger.info(f"🥋 自我对抗完成：发现{len(blind_spots)}个盲点，解构度={self.state.self_dissolution:.2f}")
        
        return {
            "updated_policy": updated_policy,
            "blind_spots_found": len(blind_spots),
            "dissolution_level": self.state.self_dissolution,
        }

    def _generate_adversarial_scenarios(self, policy: dict) -> list[dict]:
        """生成对抗场景"""
        scenarios = []
        
        # 场景1：极端情况
        scenarios.append({
            "type": "extreme_state",
            "description": "环境突变，所有已知策略失效",
            "probability": 0.05,
        })
        
        # 场景2：对抗性干扰
        scenarios.append({
            "type": "adversarial_perturbation",
            "description": "观测数据被微小扰动误导",
            "perturbation_strength": self.adversarial_rate,
        })
        
        # 场景3：分布外泛化
        scenarios.append({
            "type": "out_of_distribution",
            "description": "遇到训练分布外的全新情境",
            "novelty_level": 0.8,
        })
        
        return scenarios

    def _identify_weakness(self, scenario: dict, policy: dict) -> dict:
        """识别策略弱点"""
        # 简化模拟：随机生成弱点严重性
        base_severity = random.uniform(0.2, 0.7)
        
        # 根据场景类型调整
        if scenario["type"] == "extreme_state":
            severity = base_severity * 1.3  # 极端情况更容易暴露弱点
        elif scenario["type"] == "adversarial_perturbation":
            severity = base_severity * (1.0 + self.adversarial_rate)
        else:
            severity = base_severity
        
        weakness = {
            "scenario_type": scenario["type"],
            "severity": np.clip(severity, 0.0, 1.0),
            "description": f"在{scenario['description']}场景下表现不稳定",
            "recommended_fix": self._suggest_fix(scenario),
        }
        
        return weakness

    def _suggest_fix(self, scenario: dict) -> str:
        """建议修复方案"""
        fixes = {
            "extreme_state": "增加鲁棒性正则化，降低过拟合",
            "adversarial_perturbation": "引入对抗训练，提升抗干扰能力",
            "out_of_distribution": "扩展训练分布，增强泛化能力",
        }
        return fixes.get(scenario["type"], "通用优化策略")

    def _update_policy_against_blind_spots(
        self, policy: dict, blind_spots: list[dict]
    ) -> dict:
        """针对盲点更新策略"""
        updated = policy.copy()
        
        for spot in blind_spots:
            if spot["severity"] > 0.6:
                # 严重盲点：大幅调整策略权重
                adjustment = -0.2 * spot["severity"]
                updated["robustness_weight"] = updated.get("robustness_weight", 0.5) + abs(adjustment)
        
        # 归一化
        total = sum(v for k, v in updated.items() if isinstance(v, (int, float)))
        if total > 0:
            updated = {k: v / total if isinstance(v, (int, float)) else v for k, v in updated.items()}
        
        return updated

    # ==================== 4. 大圆镜智：梦境种子重组 ====================
    
    def run_dream_replay(self, memory_seeds: list[dict]) -> list[dict]:
        """
        梦境回放 - 种子重组与变异
        
        机制：
        1. 随机抽取高重要性种子
        2. 交叉重组（Crossover）
        3. 基因突变（Mutation）
        4. 生成新策略种子
        
        类比：生物进化 + 睡眠记忆巩固
        """
        if len(memory_seeds) < 3:
            logger.warning("种子数量不足，跳过梦境回放")
            return memory_seeds
        
        # 选择高重要性种子
        high_imp_seeds = [s for s in memory_seeds if s.get("imp", 0) > 0.3]
        if len(high_imp_seeds) < 2:
            high_imp_seeds = memory_seeds[:5]  # 降级处理
        
        new_seeds = []
        
        # 重组循环
        num_recombinations = max(3, int(len(high_imp_seeds) * self.recombination_rate))
        
        for i in range(num_recombinations):
            # 随机选择两个父本
            parent1, parent2 = random.sample(high_imp_seeds, 2)
            
            # 交叉重组
            child = self._crossover(parent1, parent2)
            
            # 变异
            mutated_child = self._mutate(child)
            
            # 标记为梦境产物
            mutated_child["tag"] = "dream_generated"
            mutated_child["generation"] = max(
                parent1.get("generation", 0), parent2.get("generation", 0)
            ) + 1
            
            new_seeds.append(mutated_child)
        
        # 记录梦境会话
        dream_session = {
            "timestamp": time.time(),
            "parent_count": len(high_imp_seeds),
            "offspring_count": len(new_seeds),
            "avg_mutation_strength": self.mutation_strength,
        }
        self.dream_sessions.append(dream_session)
        
        # 更新种子多样性
        self.state.seed_diversity = self._compute_seed_diversity(memory_seeds + new_seeds)
        
        logger.info(f"🌙 梦境回放完成：{len(high_imp_seeds)}个父本 → {len(new_seeds)}个新种子，多样性={self.state.seed_diversity:.2f}")
        
        return memory_seeds + new_seeds

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """交叉重组"""
        child = {}
        
        # 随机继承属性
        for key in ["act", "strategy_pattern", "context_signature"]:
            if key in parent1 and key in parent2:
                child[key] = random.choice([parent1[key], parent2[key]])
            elif key in parent1:
                child[key] = parent1[key]
            elif key in parent2:
                child[key] = parent2[key]
        
        # 奖励取平均（带随机扰动）
        child["rew"] = (parent1.get("rew", 0) + parent2.get("rew", 0)) / 2 + random.uniform(-0.1, 0.1)
        
        # 重要性取最大值
        child["imp"] = max(parent1.get("imp", 0.5), parent2.get("imp", 0.5))
        
        return child

    def _mutate(self, seed: dict) -> dict:
        """基因突变"""
        mutated = seed.copy()
        
        # 动作突变（小概率改变策略）
        if random.random() < self.mutation_strength:
            possible_actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
            current_action = mutated.get("act", "STAY")
            new_action = random.choice([a for a in possible_actions if a != current_action])
            mutated["act"] = new_action
            mutated["mutation_type"] = "action_shift"
        
        # 奖励感知突变（重新评估价值）
        if random.random() < self.mutation_strength * 0.5:
            mutated["rew"] *= random.uniform(0.8, 1.2)
            mutated["mutation_type"] = "reward_revaluation"
        
        # 添加突变标记
        mutated["mutated"] = True
        mutated["mutation_strength"] = self.mutation_strength
        
        return mutated

    def _compute_seed_diversity(self, seeds: list[dict]) -> float:
        """计算种子多样性"""
        if len(seeds) < 2:
            return 0.0
        
        # 基于动作分布的多样性
        action_counts = {}
        for s in seeds:
            act = s.get("act", "UNKNOWN")
            action_counts[act] = action_counts.get(act, 0) + 1
        
        # 计算熵
        probs = np.array(list(action_counts.values())) / len(seeds)
        diversity = -np.sum(probs * np.log2(probs + 1e-10))
        max_diversity = np.log2(len(action_counts))
        
        return diversity / max_diversity if max_diversity > 0 else 0.0

    # ==================== 觉醒度评估 ====================
    
    def compute_awakening_level(self) -> float:
        """
        综合计算觉醒等级
        
        公式：
        Awakening = w1*Novelty + w2*Insight + w3*Dissolution + w4*Diversity
        
        返回：0-1之间的觉醒等级
        """
        weights = {
            "novelty": 0.25,
            "insight": 0.30,
            "dissolution": 0.25,
            "diversity": 0.20,
        }
        
        level = (
            weights["novelty"] * self.state.novelty_score +
            weights["insight"] * self.state.insight_depth +
            weights["dissolution"] * self.state.self_dissolution +
            weights["diversity"] * self.state.seed_diversity
        )
        
        self.state.awakening_level = np.clip(level, 0.0, 1.0)
        self.state.meta_cycles += 1
        
        # 觉醒里程碑
        if self.state.awakening_level > 0.8 and self.state.breakthrough_count >= 5:
            logger.info(f"🎓 高度觉醒状态！等级={self.state.awakening_level:.2f}, 突破={self.state.breakthrough_count}次")
        elif self.state.awakening_level > 0.5:
            logger.info(f"✨ 中度觉醒：等级={self.state.awakening_level:.2f}")
        
        return self.state.awakening_level

    def get_awakening_report(self) -> dict:
        """生成觉醒报告"""
        return {
            "awakening_level": self.state.awakening_level,
            "novelty_score": self.state.novelty_score,
            "insight_depth": self.state.insight_depth,
            "self_dissolution": self.state.self_dissolution,
            "seed_diversity": self.state.seed_diversity,
            "meta_cycles": self.state.meta_cycles,
            "breakthrough_count": self.state.breakthrough_count,
            "total_insights": len(self.insight_log),
            "dream_sessions": len(self.dream_sessions),
            "action_history_length": len(self.action_history),
        }

    # ==================== 主循环集成 ====================
    
    def step(
        self,
        obs: dict,
        action: str,
        reward: float,
        memory_seeds: list[dict],
        causal_model: dict,
        episode_step: int,
    ) -> dict:
        """
        觉醒引擎单步执行
        
        整合四大机制，输出增强决策
        """
        # 记录历史
        state_hash = hash(str(obs))
        self.action_history.append({
            "step": episode_step,
            "action": action,
            "reward": reward,
            "state_hash": state_hash,
        })
        
        # 1. 计算好奇心
        memory_diversity = self._compute_seed_diversity(memory_seeds) if memory_seeds else 0.5
        curiosity = self.compute_curiosity_drive(obs, memory_diversity)
        
        # 2. 生成好奇实验（如需要）
        experiment = None
        if curiosity > self.curiosity_threshold:
            experiment = self.generate_curiosity_experiment(curiosity)
        
        # 3. 反事实推理（每5步一次）
        insights = []
        if episode_step % 5 == 0 and len(causal_model) > 0:
            available_actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
            insights = self.run_counterfactual_reasoning(action, reward, available_actions, causal_model)
        
        # 4. 自我对抗（每10步一次）
        adversarial_result = None
        if episode_step % 10 == 0:
            current_policy = {"exploration": 0.3, "exploitation": 0.7}
            adversarial_result = self.run_self_adversarial_training(current_policy)
        
        # 5. 梦境回放（定期）
        dream_offspring = []
        if episode_step % self.dream_replay_freq == 0:
            dream_offspring = self.run_dream_replay(memory_seeds)
        
        # 6. 计算觉醒等级
        awakening_level = self.compute_awakening_level()
        
        result = {
            "curiosity_level": curiosity,
            "experiment": experiment,
            "insights": [
                {
                    "alternative": s.alternative_action,
                    "insight_gain": s.insight_gain,
                    "pattern": s.causal_pattern,
                }
                for s in insights
            ],
            "adversarial_result": adversarial_result,
            "dream_offspring_count": len(dream_offspring),
            "awakening_level": awakening_level,
            "recommendations": self._generate_recommendations(curiosity, insights, adversarial_result),
        }
        
        return result

    def _generate_recommendations(
        self,
        curiosity: float,
        insights: list[CounterfactualScenario],
        adversarial_result: dict | None,
    ) -> list[str]:
        """生成行动建议"""
        recs = []
        
        if curiosity > 0.7:
            recs.append("🔍 高好奇心：建议主动探索未知区域")
        elif curiosity < 0.2:
            recs.append("⚠️ 低好奇心：警惕陷入局部最优")
        
        if insights and insights[0].insight_gain > 0.5:
            recs.append(f"💡 关键洞察：{insights[0].causal_pattern}")
        
        if adversarial_result and adversarial_result.get("blind_spots_found", 0) > 0:
            recs.append(f"🛡️ 发现{adversarial_result['blind_spots_found']}个策略盲点，建议增强鲁棒性")
        
        if self.state.awakening_level > 0.6:
            recs.append(f"✨ 觉醒等级{self.state.awakening_level:.2f}：智慧涌现加速中")
        
        return recs


# ==================== 使用示例 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "curiosity_threshold": 0.3,
        "counterfactual_depth": 3,
        "adversarial_rate": 0.15,
        "dream_replay_freq": 10,
        "recombination_rate": 0.3,
        "mutation_strength": 0.15,
    }
    
    engine = AwakeningEngine(config)
    
    # 模拟运行
    memory_seeds = [
        {"act": "UP", "rew": 0.5, "imp": 0.6, "tag": "positive"},
        {"act": "DOWN", "rew": -0.3, "imp": 0.4, "tag": "negative"},
        {"act": "LEFT", "rew": 0.2, "imp": 0.3, "tag": "neutral"},
    ]
    
    causal_model = {
        "UP": {"expected_reward": 0.4, "uncertainty": 0.2, "success_factors": ["开阔空间"]},
        "DOWN": {"expected_reward": -0.2, "uncertainty": 0.5, "risk_factors": ["障碍物"]},
        "LEFT": {"expected_reward": 0.1, "uncertainty": 0.3},
        "RIGHT": {"expected_reward": 0.3, "uncertainty": 0.4},
        "STAY": {"expected_reward": 0.0, "uncertainty": 0.1},
    }
    
    for step in range(1, 21):
        obs = {"pos": (step % 5, step % 5), "grid_view": [random.random() for _ in range(9)]}
        action = random.choice(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])
        reward = random.uniform(-0.5, 0.8)
        
        result = engine.step(obs, action, reward, memory_seeds, causal_model, step)
        
        print(f"\n{'='*60}")
        print(f"步骤 {step}: 觉醒等级={result['awakening_level']:.2f}, 好奇心={result['curiosity_level']:.2f}")
        
        if result["insights"]:
            print(f"💡 洞察：{result['insights'][0]['pattern']}")
        
        if result["recommendations"]:
            print("建议:")
            for rec in result["recommendations"]:
                print(f"  {rec}")
    
    # 最终报告
    print(f"\n{'='*60}")
    print("📊 觉醒报告:")
    report = engine.get_awakening_report()
    for k, v in report.items():
        print(f"  {k}: {v}")
