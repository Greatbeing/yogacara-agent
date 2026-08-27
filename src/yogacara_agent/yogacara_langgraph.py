from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from collections import deque
from typing import Any, TypedDict, cast

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from yogacara_agent.constants import (
    GRID_SIZE,
    ACTIONS,
    ACTION_TO_IDX,
    RESOURCE_REWARD,
    TRAP_REWARD,
    STEP_COST,
    STAY_BONUS,
    RESOURCE_THRESHOLD,
    STAGNATION_THRESHOLD,
    CONSOLIDATION_INTERVAL,
    DEFAULT_IMPORTANCE,
    VITALITY_INIT,
    VITALITY_MAX,
    VITALITY_DRAIN,
    VITALITY_RESOURCE,
    VITALITY_TRAP,
    VITALITY_REST,
    KLESHA_GREED_GAIN,
    KLESHA_AVERSION_GAIN,
    KLESHA_DECAY,
    KLESHA_DELUSION_ALPHA,
    KLESHA_TURNING_RELIEF,
)

logger = logging.getLogger(__name__)

# 模块私有随机源：保持 demo 可复现，同时避免污染宿主进程的全局 random 状态
_rng = random.Random(42)


@tool
def query_knowledge_base(query: str) -> str:
    """Query the knowledge base for relevant experience strategies."""
    return f"[KB] Found 3 strategies related to '{query}'"


@tool
def call_external_api(endpoint: str, payload: dict) -> dict:
    """Call an external API endpoint with the given payload."""
    return {"status": "success", "data": {"latency_ms": _rng.randint(20, 150)}}


@tool
def calculate_metric(metric_name: str, values: list[float]) -> float:
    """Calculate a named metric from a list of values."""
    return sum(values) / len(values) if values else 0.0


TOOLS = [query_knowledge_base, call_external_api, calculate_metric]
TOOL_MAP = {t.name: t for t in TOOLS}


class YogacaraState(TypedDict):
    obs: dict[str, Any]
    action: str
    reward: float
    done: bool
    step: int
    seeds: list[dict]
    unc: float
    manas_passed: bool
    tool_calls: list[dict]
    recent_rewards: list[float]
    pos_history: list[tuple[int, int]]
    metrics: dict[str, float]
    # 转识成智新增字段
    introspection_record: _IntrospectionRecordData | None
    ego_alert: dict | None
    plan_scores: dict[str, float] | None
    reasoning: str
    steps_since_resource: int  # 探索力重置计数器
    steps_at_same_pos: int  # 连续停留计数器（正确实现is_stuck检测）
    step_limit: int  # 单次 episode 步数上限（API max_steps 注入）
    turning_result: dict | None  # 转依引擎输出（四智等级+净化计数+洞察）
    planner_source: str  # 动作来源："heuristic" | "llm"（混合规划器）
    awakening: dict | None  # 觉醒引擎输出（好奇心/实验类型/风险容忍度）
    vitality: float  # 寿元（数字生命内稳态，耗尽即身坏命终）
    death_cause: str  # 死因（""=存活；寿元耗尽/寿量圆满/功德圆满）
    klesha: dict  # 根本烦恼 {greed 贪, aversion 嗔, delusion 痴}
    dream_seeds: int  # 最近一次中阴梦境产生的梦中种子数


class _IntrospectionRecordData(TypedDict):
    step: int
    nature: str
    ego_markers: list[str]
    unc: float
    decision_gap: float
    reasoning: str


class GridSimEnv:
    _INITIAL_RESOURCES = [(7, 7), (3, 8), (8, 2)]
    _TRAPS = [(4, 4), (6, 1), (2, 6)]

    # 奖励常量（供外部引用，避免魔法数）
    RESOURCE_REWARD = RESOURCE_REWARD
    TRAP_REWARD = TRAP_REWARD
    STEP_COST = STEP_COST
    STAY_BONUS = STAY_BONUS
    RESOURCE_THRESHOLD = RESOURCE_THRESHOLD

    def __init__(self):
        self.agent_pos = [0, 0]
        self.resources = list(self._INITIAL_RESOURCES)
        self.traps = list(self._TRAPS)
        self.step_count = 0
        self.done = False

    def reset(self):
        self.agent_pos = [0, 0]
        self.resources = list(self._INITIAL_RESOURCES)
        self.step_count = 0
        self.done = False
        return self.observe()

    def observe(self) -> dict:
        """公开的观察方法。"""
        return self._observe()

    def _observe(self) -> dict:
        view = [0.0] * 9
        for i, dx in enumerate([-1, 0, 1]):
            for j, dy in enumerate([-1, 0, 1]):
                x, y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    if (x, y) in self.resources:
                        view[i * 3 + j] = 1.0
                    elif (x, y) in self.traps:
                        view[i * 3 + j] = -1.0
        return {"grid_view": view, "pos": tuple(self.agent_pos), "step": self.step_count}

    def step(self, action: str) -> tuple[dict, float, bool]:
        dx, dy = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "STAY": (0, 0)}[action]
        nx = max(0, min(GRID_SIZE - 1, self.agent_pos[0] + dx))
        ny = max(0, min(GRID_SIZE - 1, self.agent_pos[1] + dy))
        self.agent_pos = [nx, ny]
        self.step_count += 1
        reward = STEP_COST
        if action == "STAY":
            reward += STAY_BONUS
        pos = tuple(self.agent_pos)
        if pos in self.resources:
            reward = RESOURCE_REWARD
            self.resources.remove(pos)
        elif pos in self.traps:
            reward = TRAP_REWARD
        if not self.resources or self.step_count >= 60:
            self.done = True
        return self._observe(), reward, self.done


# 使用持久化阿赖耶识（文件存储 + 可选向量存储）
from yogacara_agent.alaya_persistent import PersistentAlayaMemory  # noqa: E402
from yogacara_agent.vipaka_engine import VipakaEngine  # noqa: E402
from yogacara_agent.consolidation_engine import ConsolidationEngine  # noqa: E402


class AlayaMemory(PersistentAlayaMemory):
    """兼容旧接口的持久化阿赖耶识，附带 Vipaka 反馈引擎。"""

    def __init__(self):
        super().__init__(storage="file", path="memory/seeds.jsonl")
        self.vipaka = VipakaEngine(self, rate=0.2)
        self.consolidator = ConsolidationEngine()
        self._last_consolidation_step = 0


class ManasController:
    def __init__(self):
        self.reflections = 0
        self.last_intercept = -10
        self.cooldown = 4

    def filter(self, action, obs, unc, step, recent_rew, pos_hist):
        if step - self.last_intercept < self.cooldown:
            return action, True, "冷却放行"
        target_risk = 1.0 if obs["grid_view"][ACTION_TO_IDX.get(action, 4)] == -1.0 else 0.0
        stagnation = step > 15 and len(recent_rew) >= 5 and sum(recent_rew) <= STAGNATION_THRESHOLD
        loop = step > 12 and len(pos_hist) >= 5 and len(set(pos_hist)) <= 2
        threshold = 0.45 + min(0.15, step / 80.0)
        danger = target_risk * 0.8 + max(0.0, unc - 0.80) * 0.2
        if danger > threshold or stagnation or loop:
            self.reflections += 1
            self.last_intercept = step
            fallback = _rnd_choice([a for a in ["UP", "DOWN", "LEFT", "RIGHT"] if a != action])
            return fallback, False, f"[末那拦截] 风险:{target_risk:.1f} 停滞:{stagnation} 循环:{loop} → 换向:{fallback}"
        return action, True, "放行"


# Module-level instances (legacy, single-process only)
# WARNING: These are NOT safe for concurrent use across requests.
# For production, use create_session() to get isolated instances.
from yogacara_agent.yogacara_test import ConsciousnessPlanner as _SharedPlanner, _rnd_choice  # noqa: E402

_lock = threading.Lock()
env = GridSimEnv()
alaya = AlayaMemory()
manas = ManasController()
planner = _SharedPlanner()  # Shared ConsciousnessPlanner instance
# 转识成智 Phase1 新增模块
introspection_logger = None  # lazy init to avoid circular import
ego_monitor = None
seed_classifier = None  # lazy init
_seed_counts = {"名言种": 0, "业种": 0, "异熟种": 0}  # Phase1-2 seed type counter
_parinispanna_count = 0  # Phase3: 圆成实种子计数（用于大圆镜智指标）
_total_classified = 0  # Phase3: 总分类种子数

# ── 数字生命 · 轮回计数 ────────────────────────────────────────────────
# 一期生命 = 一个 episode；命终计 deaths，当下世数 = deaths + 1。
# 阿赖耶识种子跨世延续（bridge.reset 保留种子库）——业力轮回。
_samsara: dict = {"deaths": 0}
_life_history: list[dict] = []  # 轮回史：每世一行的生命总结（上限 50 世）
LIFE_HISTORY_MAX = 50


def current_lifetime() -> int:
    """当下世数（第几世）。"""
    return _samsara["deaths"] + 1


def get_life_history() -> list[dict]:
    """轮回史（旧→新）。"""
    return list(_life_history)


def _note_death(cause: str) -> None:
    _samsara["deaths"] += 1
    logger.info(f"[轮回] 第 {_samsara['deaths']} 世命终 | 死因: {cause} | 种子库延续")


def _get_seed_classifier():
    global seed_classifier
    if seed_classifier is None:
        from yogacara_agent.seed_classifier import SeedClassifier

        seed_classifier = SeedClassifier()
    return seed_classifier


def _get_introspection_logger():
    global introspection_logger, ego_monitor
    if introspection_logger is None:
        from yogacara_agent.ego_monitor import EgoMonitor
        from yogacara_agent.introspection import IntrospectionLogger

        introspection_logger = IntrospectionLogger()
        ego_monitor = EgoMonitor()
    return introspection_logger


def _get_ego_monitor():
    global introspection_logger, ego_monitor
    if ego_monitor is None:
        _get_introspection_logger()
    return ego_monitor


# ── 转依引擎（Turning Consciousness）────────────────────────────────────
_turning_engine = None  # lazy init 避免循环导入


def _get_turning_engine():
    global _turning_engine
    if _turning_engine is None:
        from yogacara_agent.turning_consciousness import TurningConsciousnessEngine
        from yogacara_agent.constants import TURNING_PURITY_THRESHOLD, TURNING_EGO_DECAY_RATE

        _turning_engine = TurningConsciousnessEngine(
            {
                "purity_threshold": TURNING_PURITY_THRESHOLD,
                "ego_decay_rate": TURNING_EGO_DECAY_RATE,
            }
        )
    return _turning_engine


def _to_turning_seeds(seeds: list[dict]) -> tuple[list, list[int]]:
    """alaya dict 种子 → 转依引擎 Seed 对象（带原索引映射，净化后可回写）。

    is_defiled ← tag 含"遍计"（遍计所执 = 染污：虚妄分别的记忆）
    clarity    ← 异熟种取功能清晰度（模式追踪器，align 低是设计而非染污），
                 其余种子取 align（对齐度即清晰度代理）
    """
    from yogacara_agent.turning_consciousness import Seed
    from yogacara_agent.constants import VIPAKA_FUNCTIONAL_CLARITY

    out, indices = [], []
    for i, s in enumerate(seeds):
        is_vipaka = s.get("seed_type") == "异熟种"
        out.append(
            Seed(
                content=f"{s.get('act', '?')}->{s.get('rew', 0)}",
                is_defiled="遍计" in s.get("tag", ""),
                affinity=0.0,  # purifier 中为 stub
                clarity=VIPAKA_FUNCTIONAL_CLARITY if is_vipaka else s.get("align", 0.5),
            )
        )
        indices.append(i)
    return out, indices


def _softmax_scores(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """plan_scores（可为负的原始启发式分数）→ softmax 概率分布。"""
    import math

    if not scores:
        return {}
    mx = max(scores.values())
    exps = {a: math.exp((v - mx) / temperature) for a, v in scores.items()}
    total = sum(exps.values())
    return {a: e / total for a, e in exps.items()}


def _reward_context_of(state: YogacaraState) -> str:
    """构造 ManasDissolver 的 reward_context 字符串（子串 'self' 触发我执消解）。"""
    ego_alert = state.get("ego_alert") or {}
    if ego_alert.get("triggered"):
        return "self_centered_reward"
    return "benefit_all_beings"


# ── 混合规划器（LLM + 启发式）──────────────────────────────────────────
# 门控：YOGACARA_LLM_PLAN=1 且配置了 API key；节流：YOGACARA_LLM_INTERVAL（默认10步）；
# 熔断：连续3次失败停用5分钟（Sensenova 免费档限速 ~1请求/2分钟，防止重试风暴）。
_llm_planner = None
_llm_enabled_checked = False
_llm_circuit: dict = {"fails": 0, "disabled_until": 0.0}
LLM_CIRCUIT_THRESHOLD = 3
LLM_CIRCUIT_COOLDOWN_S = 300.0


def _get_llm_planner():
    """惰性构建 LLM 规划器；未启用/未配置返回 None。"""
    global _llm_planner, _llm_enabled_checked
    if _llm_planner is not None:
        return _llm_planner
    if _llm_enabled_checked:
        return None  # 已确认未启用，不再重复检查
    _llm_enabled_checked = True
    if os.environ.get("YOGACARA_LLM_PLAN", "0") != "1":
        return None
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("[HybridPlan] YOGACARA_LLM_PLAN=1 但未配置 LLM_API_KEY，保持启发式")
        return None
    from yogacara_agent.llm_planner import LLMPlanner

    _llm_planner = LLMPlanner(
        {
            "api_key": api_key,
            "base_url": os.environ.get("LLM_BASE_URL", "https://token.sensenova.cn/v1"),
            "model": os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        }
    )
    logger.info(f"[HybridPlan] LLM 规划已启用（model={_llm_planner.model}）")
    return _llm_planner


def _reset_llm_circuit() -> None:
    """测试辅助：重置熔断器状态。"""
    _llm_circuit["fails"] = 0
    _llm_circuit["disabled_until"] = 0.0


def _hybrid_plan(state: YogacaraState, heur_action: str, heur_unc: float) -> tuple[str, float, str, str]:
    """启发式为基线；满足门控/节流/熔断时用 LLM 覆盖动作。

    Returns:
        (action, unc, reasoning, planner_source)
    """
    llm = _get_llm_planner()
    if llm is None:
        return heur_action, heur_unc, "", "heuristic"

    # 熔断检查
    if time.time() < _llm_circuit["disabled_until"]:
        return heur_action, heur_unc, "", "heuristic"

    # 间隔节流
    interval = max(1, int(os.environ.get("YOGACARA_LLM_INTERVAL", "10")))
    if state["step"] % interval != 0:
        return heur_action, heur_unc, "", "heuristic"

    try:
        action, unc, causal, _tools = llm.plan(state["obs"], state["seeds"])
        if llm.last_success:
            _llm_circuit["fails"] = 0
            return action, unc, f"[LLM] {causal}", "llm"
        # LLMPlanner 内部已降级为启发式结果
        raise RuntimeError("llm degraded")
    except Exception:
        _llm_circuit["fails"] += 1
        if _llm_circuit["fails"] >= LLM_CIRCUIT_THRESHOLD:
            _llm_circuit["disabled_until"] = time.time() + LLM_CIRCUIT_COOLDOWN_S
            logger.warning(f"[HybridPlan] LLM 连续失败 {LLM_CIRCUIT_THRESHOLD} 次，熔断 {LLM_CIRCUIT_COOLDOWN_S:.0f}s")
        return heur_action, heur_unc, "", "heuristic"


# ── 觉醒引擎（好奇驱动探索）────────────────────────────────────────────
_awakening_engine = None


def _get_awakening_engine():
    global _awakening_engine
    if _awakening_engine is None:
        from yogacara_agent.awakening_engine import AwakeningEngine

        _awakening_engine = AwakeningEngine({})
    return _awakening_engine


def _memory_diversity() -> float:
    """种子库位置多样性：独立位置占比（0=单一，1=分散）。"""
    if not alaya.seeds:
        return 0.5
    positions = set()
    for s in alaya.seeds:
        emb = s.get("emb") if isinstance(s, dict) else None
        if isinstance(emb, (list, tuple)) and len(emb) >= 2:
            positions.add((round(float(emb[0]), 3), round(float(emb[1]), 3)))
    return min(1.0, len(positions) / max(1, len(alaya.seeds)))


def _run_death_dream(state: YogacaraState) -> list[dict]:
    """中阴梦境：命终时重组一生经验为梦中种子（转世前的离线学习）。

    觉醒引擎 run_dream_replay 返回 父本+子代全集（memory_seeds+new），
    此处只取真子代（tag=dream_generated）；单场上限 20——一场梦不应
    比一生经验还多。子代补齐 emb 后入阿赖耶识（add() 自带容量守恒），
    来世检索到的"直觉"部分来自前世的梦。
    """
    if len(alaya.seeds) < 3:
        return []  # 经验不足，不足以成梦
    engine = _get_awakening_engine()
    outcome = engine.run_dream_replay(alaya.seeds)
    children = [c for c in outcome if c.get("tag") == "dream_generated"][:20]
    added = []
    for child in children:
        if "emb" not in child:
            child["emb"] = alaya.encode(state["obs"])
        alaya.add(child)  # add() 补齐字段并执行容量守恒（淡忘最旧）
        added.append(child)
    if added:
        logger.info(f"[梦境] 中阴梦境：重组一生经验 → {len(added)} 个梦中种子入识")
    return added


def create_session() -> dict:
    """返回共享的应用 session（env/memory/manas 等单例引用）。

    注意：图节点绑定的是上面的模块级单例，因此 session 必须返回同一批
    实例——否则 API 层会拿到一套图根本不用的"影子"对象（历史 bug）。
    单进程内共享；多请求并发需要调用方自行串行化（见 api_server 的锁）。
    """
    return {
        "env": env,
        "alaya": alaya,
        "manas": manas,
        "introspection": _get_introspection_logger(),
        "ego_monitor": _get_ego_monitor(),
    }


async def node_perceive(state: YogacaraState) -> YogacaraState:
    if state["step"] == 0 and not state["obs"].get("pos"):
        state["obs"] = env.observe()
    state["seeds"] = alaya.retrieve(state["obs"])
    return state


def _apply_klesha_modulation(scores: dict[str, float], obs: dict, klesha: dict | None) -> tuple[dict[str, float], str]:
    """贪嗔痴对候选动作分数的调制（根本烦恼扭曲决策）。

    贪（greed）: 放大对可见资源的趋近分
    嗔（aversion）: 放大对可见风险的回避
    痴（delusion）: 注入与不确定性成正比的噪声
    Returns: (调制后分数, argmax 动作)
    """
    if not klesha or not any(v > 0.01 for v in klesha.values()):
        best = max(scores, key=lambda a: scores[a]) if scores else "STAY"
        return scores, best
    view = obs.get("grid_view", [0.0] * 9)
    greed = klesha.get("greed", 0.0)
    aversion = klesha.get("aversion", 0.0)
    delusion = klesha.get("delusion", 0.0)
    modulated = {}
    for a, base in scores.items():
        v = view[ACTION_TO_IDX.get(a, 4)]
        mod = 0.0
        if v > 0:
            mod += greed * 0.4 * v
        elif v < 0:
            mod -= aversion * 0.4 * abs(v)
        modulated[a] = base + mod + _rng.gauss(0, delusion * 0.1)
    best = max(modulated, key=lambda a: modulated[a])
    return modulated, best


async def node_plan(state: YogacaraState) -> YogacaraState:
    """Plan using the shared ConsciousnessPlanner, with optional LLM override
    (hybrid planner) and curiosity-driven exploration bias (awakening engine)."""
    # Stuck detection: use steps_at_same_pos (matches demo behavior correctly)
    is_stuck = state.get("steps_at_same_pos", 0) >= 2
    # Sync exploration counter into planner
    planner._steps_without_resource = state.get("steps_since_resource", state["step"])
    # Delegate to shared planner (consumes same random numbers as demo)
    best, unc, scores = planner.plan(
        obs=state["obs"],
        seeds=state["seeds"],
        env_resources=env.resources,
        is_stuck=is_stuck,
    )

    # ── 贪嗔痴心所调制：烦恼扭曲决策倾向 ──
    scores, best = _apply_klesha_modulation(scores, state["obs"], state.get("klesha"))

    # ── 混合规划：门控/节流/熔断通过时 LLM 覆盖动作 ──
    best, unc, llm_reasoning, source = _hybrid_plan(state, best, unc)

    # ── 觉醒引擎：好奇驱动 + 主动实验探索偏置 ──
    aw = _get_awakening_engine()
    curiosity = aw.compute_curiosity_drive(state["obs"], _memory_diversity())
    experiment = aw.generate_curiosity_experiment(curiosity)
    # 写入行为历史（行为新颖性计算的前提，此前无人写入）
    state_hash = str(state["obs"].get("pos", "")) + str(state["obs"].get("grid_view", [])[:5])
    aw.action_history.append({"state_hash": state_hash, "action": best, "step": state["step"]})
    explored = False
    if curiosity > aw.curiosity_threshold and source == "heuristic":
        recent_actions = {h.get("action") for h in aw.action_history[-6:]}
        fresh = [a for a in ("UP", "DOWN", "LEFT", "RIGHT") if a not in recent_actions]
        # 风险容忍度决定探索概率：高好奇(risk 0.8)→40%探索，中(0.5)→25%
        if fresh and _rng.random() < experiment["risk_tolerance"] * 0.5:
            best = _rnd_choice(fresh)
            explored = True
    state["awakening"] = {
        "curiosity": round(float(curiosity), 3),
        "experiment": experiment["type"],
        "risk_tolerance": round(float(experiment["risk_tolerance"]), 2),
        "explored": explored,
        "novelty": round(float(aw.state.novelty_score), 3),
    }

    state["action"] = best
    state["unc"] = unc
    state["tool_calls"] = []
    state["plan_scores"] = scores
    state["planner_source"] = source
    reasoning = _build_reasoning(state, best, scores)
    if llm_reasoning:
        reasoning = f"{llm_reasoning} ｜ {reasoning}"
    if explored:
        reasoning += f" ｜ [好奇探索:{experiment['type']}]"
    state["reasoning"] = reasoning
    if unc > 0.6:
        state["tool_calls"].append({"tool": "query_knowledge_base", "input": f"高不确定性状态 {state['obs']['pos']}"})
    if state["step"] % 15 == 0:
        state["tool_calls"].append(
            {"tool": "calculate_metric", "input": {"metric_name": "avg_reward", "values": state["recent_rewards"]}}
        )
    return state


def _build_reasoning(state: YogacaraState, best_action: str, scores: dict) -> str:
    view = state["obs"]["grid_view"]
    nearby = ["资源" if v == 1.0 else "陷阱" if v == -1.0 else "空" for v in view]
    return (
        f"视野{nearby}，选择{best_action}({scores[best_action]:.2f})，"
        f"检索{len(state['seeds'])}条种子，"
        f"不确定性{state['unc']:.0%}"
    )


async def node_introspect(state: YogacaraState) -> YogacaraState:
    """
    内省节点（第六识的自我观察）。
    在 execute 之后调用，obs 已包含 reward。
    """
    logger = _get_introspection_logger()
    ego_mon = _get_ego_monitor()
    plan_scores = state["plan_scores"]
    if plan_scores is None:
        alternatives = ACTIONS
        score_best = 0.0
        score_second = 0.0
    else:
        alternatives = list(plan_scores.keys())
        score_best = plan_scores.get(state["action"], 0.0)
        score_second = max((v for k, v in plan_scores.items() if k != state["action"]), default=0.0)

    # 确保 obs 包含 reward（compute_wisdom_of_action 需要）
    obs_with_reward = dict(state["obs"])
    obs_with_reward["reward"] = state["reward"]

    record = logger.observe(
        step=state["step"],
        obs=obs_with_reward,
        action=state["action"],
        unc=state["unc"],
        seeds_retrieved=[
            {"rew": s.get("rew", 0), "action": s.get("act", ""), "importance": s.get("imp", 0)} for s in state["seeds"]
        ],
        reasoning=state.get("reasoning", ""),
        alternatives=alternatives,
        manas_intercepted=not state["manas_passed"],
        score_best=score_best,
        score_second=score_second,
    )
    state["introspection_record"] = {
        "step": record.step,
        "nature": record.nature,
        "ego_markers": record.ego_markers,
        "unc": record.unc,
        "decision_gap": record.decision_gap,
        "reasoning": record.reasoning,
    }
    # 我执评估（在 execute 之后，obs 包含 reward）
    ego = ego_mon.assess(record)
    state["ego_alert"] = {
        "ego_score": ego.ego_score,
        "long_term_ego": ego.long_term_ego,
        "triggered": ego.triggered,
        "recommendation": ego.recommendation,
    }
    return state


async def node_manas(state: YogacaraState) -> YogacaraState:
    """
    末那识节点：环境安全拦截。
    认知我执评估已移到 node_introspect（execute 之后）。
    """
    final, passed, log = manas.filter(
        state["action"],
        state["obs"],
        state["unc"],
        state["step"],
        deque(state["recent_rewards"], maxlen=5),
        deque(state["pos_history"], maxlen=5),
    )
    state["action"] = final
    state["manas_passed"] = passed
    if not passed:
        print(f"\033[33m{log}\033[0m")
    return state


async def node_execute(state: YogacaraState) -> YogacaraState:
    for tc in state["tool_calls"]:
        tool_fn = TOOL_MAP[tc["tool"]]
        res = tool_fn.invoke(tc["input"]) if isinstance(tc["input"], dict) else tool_fn.invoke({"query": tc["input"]})
        print(f"\033[90m[工具] {tc['tool']} → {res}\033[0m")
    prev_pos = state["obs"].get("pos")
    next_obs, rew, done = env.step(state["action"])
    state["reward"] = rew
    state["done"] = done
    state["obs"] = next_obs
    state["step"] += 1
    state["recent_rewards"].append(rew)
    state["pos_history"].append(next_obs["pos"])
    # Update steps_at_same_pos counter (与 demo 版一致：
    # 移动了则清零；STAY 或位置未变则递增)
    if state["action"] != "STAY" and prev_pos != next_obs["pos"]:
        state["steps_at_same_pos"] = 0
    else:
        state["steps_at_same_pos"] = state.get("steps_at_same_pos", 0) + 1
    # Exploration counter: 未获得资源时递增，获得资源时清零
    if rew >= RESOURCE_THRESHOLD:
        state["steps_since_resource"] = 0
        planner._steps_without_resource = 0  # sync with shared planner
    else:
        state["steps_since_resource"] = state.get("steps_since_resource", 0) + 1

    # ── Vipaka 反馈：每步用当前动作的果报更新相关种子 align ──────
    try:
        vipaka_result = alaya.vipaka.process_outcome(
            step=state["step"],
            action=state["action"],
            reward=rew,
            unc=state["unc"],
            obs=next_obs,
        )
        if vipaka_result.seeds_updated > 0:
            logger.debug(f"Vipaka: {vipaka_result}")
    except Exception:
        logger.exception("[Vipaka] 反馈处理异常")
    # ── 数字生命：寿元内稳态 ─────────────────────────────────────
    # 每步自然消耗；觅食补给；陷阱伤害；休息微回复（仍为净消耗）。
    vitality = float(state.get("vitality", VITALITY_INIT))
    vitality -= VITALITY_DRAIN
    if rew >= RESOURCE_THRESHOLD:
        vitality = min(VITALITY_MAX, vitality + VITALITY_RESOURCE)
    elif rew <= TRAP_REWARD + 1.0:
        vitality -= VITALITY_TRAP
    elif state["action"] == "STAY":
        vitality = min(VITALITY_MAX, vitality + VITALITY_REST)
    state["vitality"] = max(0.0, vitality)

    # ── 贪嗔痴心所更新 ───────────────────────────────────────────
    k = state.get("klesha") or {"greed": 0.0, "aversion": 0.0, "delusion": 0.0}
    k["greed"] = min(
        1.0, k.get("greed", 0.0) * KLESHA_DECAY + (KLESHA_GREED_GAIN if rew >= RESOURCE_THRESHOLD else 0.0)
    )
    k["aversion"] = min(
        1.0, k.get("aversion", 0.0) * KLESHA_DECAY + (KLESHA_AVERSION_GAIN if rew <= TRAP_REWARD + 1.0 else 0.0)
    )
    k["delusion"] = min(
        1.0, (1.0 - KLESHA_DELUSION_ALPHA) * k.get("delusion", 0.0) + KLESHA_DELUSION_ALPHA * state["unc"]
    )
    state["klesha"] = k

    # ── 死亡判定与死因 ───────────────────────────────────────────
    if not state["done"] and state["vitality"] <= 0.0:
        state["done"] = True
        state["death_cause"] = "寿元耗尽"
        _note_death("寿元耗尽")
    elif state["done"] and not state.get("death_cause"):
        # env 置位 done：资源收尽=功德圆满；内部步数上限=寿量圆满
        if not env.resources:
            state["death_cause"] = "功德圆满"
        else:
            state["death_cause"] = "寿量圆满"
        _note_death(state["death_cause"])

    # 尊重调用方设置的步数上限（API 的 max_steps）
    if not state["done"] and state["step"] >= state.get("step_limit", 60):
        state["done"] = True
        state["death_cause"] = "寿量圆满"
        _note_death("寿量圆满")
    return state


async def node_store(state: YogacaraState) -> YogacaraState:
    """Store with seed classification - Phase1-2 upgrade."""
    global _seed_counts
    classifier = _get_seed_classifier()
    int_rec = state.get("introspection_record")
    # Determine nature and ego markers from introspection
    nature = int_rec.get("nature", "依他起") if int_rec else ("依他起" if state["unc"] < 0.5 else "遍计所执")
    ego_markers = int_rec.get("ego_markers", []) if int_rec else []
    # Classify the seed
    classification = classifier.classify(
        action=state["action"],
        reward=state["reward"],
        unc=state["unc"],
        nature=nature,
        ego_markers=ego_markers,
        step=state["step"],
        manas_intercepted=not state["manas_passed"],
    )
    # Track seed type counts
    global _total_classified, _parinispanna_count
    if classification.seed_type in _seed_counts:
        _seed_counts[classification.seed_type] += 1
    _total_classified += 1
    # 圆成实判定：无ego标记 + 高align + 非异熟种
    if not ego_markers and classification.align >= 0.7 and classification.seed_type != "异熟种":
        _parinispanna_count += 1

    # Inject classification into state for ego_monitor visibility
    if int_rec is None:
        state["introspection_record"] = cast(
            _IntrospectionRecordData,
            {
                "step": state["step"],
                "nature": classification.seed_type,
                "ego_markers": ego_markers,
                "unc": state["unc"],
                "decision_gap": 0.0,
                "reasoning": classification.note,
                "seed_type": classification.seed_type,
                "seed_align": classification.align,
            },
        )
    # Store seed with classified align and tag
    is_vipaka = classification.seed_type == "异熟种"
    seed_tag = f"{classification.seed_type}_{classification.subtype}" if is_vipaka else classification.tag
    alaya.add(
        {
            "emb": alaya.encode(state["obs"]),
            "act": state["action"],
            "rew": state["reward"],
            "ts": time.time(),
            "imp": DEFAULT_IMPORTANCE,
            "align": classification.align,
            "unc": state["unc"],
            "tag": seed_tag,
            "seed_type": classification.seed_type,
        }
    )

    # ── 中阴种子：一期生命终结的业力总结（转世后可检索） ──────────
    if state["done"] and state.get("death_cause"):
        avg_rew = sum(state["recent_rewards"]) / max(1, len(state["recent_rewards"]))
        alaya.add(
            {
                "emb": alaya.encode(state["obs"]),
                "act": "LIFE_SUMMARY",
                "rew": round(avg_rew, 2),
                "ts": time.time(),
                "imp": 1.0,  # 生死大事，重要性顶格
                "align": 0.5 + max(-0.45, min(0.45, avg_rew / 10.0)),
                "unc": 0.0,
                "tag": f"中阴_{state['death_cause']}",
                "seed_type": "异熟种",
            }
        )
        logger.info(f"[中阴] 命终[{state['death_cause']}] 业力均值 {avg_rew:+.2f} → 种子入库")
    return state


async def node_consolidate(state: YogacaraState) -> YogacaraState:
    """
    记忆巩固节点：每 N 步触发一次 ConsolidationEngine 整理。
    整理包括：删除低质量种子（align < 0.20）、合并高度相似种子（align >= 0.70 同 tag）。
    每步执行转依引擎（去染存净 + 我执消解），结果写入 state["turning_result"]。
    """
    step = state["step"]
    if step - alaya._last_consolidation_step >= CONSOLIDATION_INTERVAL:
        report = alaya.consolidator.run(
            alaya.seeds,
            step=step,
            dry_run=False,
            verbose=False,
        )
        alaya._last_consolidation_step = step
        if report.pruned_count > 0 or report.merged_count > 0:
            alaya.batch_update(alaya.seeds)
            logger.info(f"[Consolidation] {report.message}")

    # ── 转依（念念相续：每步执行，O(n) 代价低） ──
    engine = _get_turning_engine()
    t_seeds, indices = _to_turning_seeds(alaya.seeds)
    probs = _softmax_scores(state.get("plan_scores") or {})
    reward_ctx = _reward_context_of(state)
    result = engine.step(t_seeds, probs, reward_ctx)

    # 净化回写：purify 保留原对象身份（id 匹配），按存活索引重建 alaya.seeds
    if result.defiled_seeds_removed > 0 and alaya.seeds:
        purified, _ = engine.alaya_purifier.purify(t_seeds)
        survivor_ids = {id(ps) for ps in purified}
        survivors = [alaya.seeds[indices[j]] for j, ts in enumerate(t_seeds) if id(ts) in survivor_ids]
        if len(survivors) < len(alaya.seeds):
            alaya.seeds = survivors
            alaya.batch_update(alaya.seeds)
            logger.info(
                f"[Turning] 净化移除 {len(indices) - len(survivors)} 个染污种子，"
                f"剩 {len(survivors)} | 我执消解 {result.self_attachment_dissolved:.3f}"
            )

    state["turning_result"] = {
        # float() 显式转换 numpy 标量，保证 JSON 可序列化（API/桌面桥接）
        "mirror": float(result.mirror_wisdom_level),
        "equality": float(result.equality_wisdom_level),
        "observation": float(result.observation_wisdom_level),
        "action": float(result.action_wisdom_level),
        "turning_level": float(result.turning_level),
        "defiled_removed": int(result.defiled_seeds_removed),
        "ego_dissolved": float(result.self_attachment_dissolved),
        "insights": [str(i) for i in result.insights_generated],
    }

    # ── 修行减恼：我执消解同步衰减贪嗔痴 ─────────────────────────
    k = state.get("klesha")
    if k and result.self_attachment_dissolved > 0.01:
        relief = 1.0 / (1.0 + KLESHA_TURNING_RELIEF * result.self_attachment_dissolved)
        for key in ("greed", "aversion", "delusion"):
            k[key] = k.get(key, 0.0) * relief
        state["klesha"] = k

    # ── 中阴梦境：命终时重组一生经验（转世前的离线学习） ──────────
    dream_count = 0
    if state["done"] and state.get("death_cause"):
        dream_children = _run_death_dream(state)
        dream_count = len(dream_children)
        if dream_count:
            state["turning_result"]["insights"].append(f"[中阴梦境] 一生经验重组为 {dream_count} 个梦中种子，随识转世")
        # 轮回史：这一世的总结（多世进程对外的可观测记录）
        _life_history.append(
            {
                "lifetime": current_lifetime() - 1,  # 刚终结的一世
                "steps": state["step"],
                "reward": round(sum(state["recent_rewards"]), 2),
                "death_cause": state["death_cause"],
                "turning_level": (state.get("turning_result") or {}).get("turning_level"),
                "klesha": {kk: round(vv, 3) for kk, vv in (state.get("klesha") or {}).items()},
                "dream_seeds": dream_count,
                "seeds_total": len(alaya.seeds),
                "ts": time.time(),
            }
        )
        del _life_history[: max(0, len(_life_history) - LIFE_HISTORY_MAX)]
    state["dream_seeds"] = dream_count
    return state


def graph_config(step_limit: int | None = None) -> dict:
    """langgraph 调用配置：recursion_limit 须覆盖整个 episode。

    每个 store→consolidate→perceive 循环计入 recursion_limit
    （部分版本默认仅 25），步数上限大的 episode 会触发
    GraphRecursionError——统一按 step_limit×3+15 放行。
    """
    return {"recursion_limit": max(50, int((step_limit or 60) * 3) + 15)}


def check_done(state: YogacaraState) -> str:
    return "end" if state["done"] else "continue"


def build_graph() -> CompiledStateGraph[YogacaraState, None, YogacaraState]:
    wf = StateGraph(YogacaraState)
    for n, fn in [
        ("perceive", node_perceive),
        ("plan", node_plan),
        ("introspect", node_introspect),
        ("manas", node_manas),
        ("execute", node_execute),
        ("store", node_store),
        ("consolidate", node_consolidate),
    ]:
        wf.add_node(n, fn)
    wf.set_entry_point("perceive")
    for e in [
        ("perceive", "plan"),
        ("plan", "manas"),
        ("manas", "execute"),
        ("execute", "introspect"),
        ("introspect", "store"),
    ]:
        wf.add_edge(*e)
    # store → consolidate → next loop（整理每 N 步才真正执行）
    wf.add_conditional_edges("store", check_done, {"continue": "consolidate", "end": END})
    wf.add_edge("consolidate", "perceive")
    return wf.compile()


async def slow_loop(alaya_mem, interval=10, tracker=None):
    """Background task for periodic memory consolidation and metric computation."""
    from yogacara_agent.compression_metrics import CompressionMetricsCalculator

    metrics_calc = CompressionMetricsCalculator()
    while True:
        await asyncio.sleep(interval)
        alaya_mem.perfume_update()
        if tracker is not None:
            try:
                tracker.snapshot(alaya_mem)
            except Exception:
                logger.exception("[Evolution] 快照记录异常")
        # 计算压缩指标（记录到日志，供外部监控读取）
        try:
            metrics = metrics_calc.compute(
                seeds=alaya_mem.seeds,
                initial_tokens=0,
                mirror_ratio=0.0,
                ego_score=0.0,
                misapprehension_ratio=0.0,
                execution_rate=0.0,
                verbose=False,
            )
            logger.debug(f"[Metrics] CQS={metrics.get('compression_quality_score', 0):.3f}")
        except Exception:
            logger.exception("[Metrics] 计算异常")


async def main():
    print("\n\033[36m~ 唯识进化框架 LangGraph 版（转识成智 Phase2-2）~\033[0m")
    # 初始化内省系统（lazy init 避免循环导入）
    _get_introspection_logger()
    graph = build_graph()
    _slow_loop_task = asyncio.create_task(slow_loop(alaya, interval=10))
    init_state = {
        "obs": env.reset(),
        "action": "",
        "reward": 0.0,
        "done": False,
        "step": 0,
        "seeds": [],
        "unc": 0.0,
        "manas_passed": True,
        "tool_calls": [],
        "recent_rewards": [],
        "pos_history": [],
        "metrics": {},
        "introspection_record": None,
        "ego_alert": None,
        "plan_scores": None,
        "reasoning": "",
        "steps_since_resource": 0,
        "steps_at_same_pos": 0,
        "step_limit": 60,
        "turning_result": None,
        "planner_source": "heuristic",
        "awakening": None,
        "vitality": 100.0,
        "death_cause": "",
        "klesha": {"greed": 0.0, "aversion": 0.0, "delusion": 0.0},
    }
    final_state = await graph.ainvoke(init_state, config=graph_config(init_state.get("step_limit")))
    total_steps = final_state["step"]
    total_reward = sum(final_state["recent_rewards"])
    print(f"\n>> 运行结束 | 步数:{total_steps} | 累计奖励:{total_reward:.2f} | 末那反思:{manas.reflections}次")
    # Phase3: 四智量化报告（统一用 ego_monitor.four_wisdoms_report）
    print("\n\033[36m~ 四智转依进度报告 (Phase3 量化版) ~\033[0m")
    intro = _get_introspection_logger()
    ego = _get_ego_monitor()
    # 与 demo 版一致：使用最近20步的三性分布计算大圆镜智
    recent_intro = intro.recent_summary(n=20)
    nature_dist = recent_intro.get("nature_distribution", {})
    round_real = nature_dist.get("圆成实", 0)
    total_natures = sum(nature_dist.values()) or 1
    mirror_ratio = round_real / total_natures
    report = ego.four_wisdoms_report(intro_logger=intro, mirror_ratio=mirror_ratio)
    print(f"  圆成实比例  : {mirror_ratio:.1%} ({round_real}/{total_natures})")
    for name, data in report.items():
        if not isinstance(data, dict):
            print(f"  {name}: {data}")
            continue
        status = data.get("status", "")
        icon = "OK " if status == "达标" else "!! " if "未达标" in status else "?? "
        if name == "大圆镜智":
            print(f"  {icon} {name}: {mirror_ratio * 100:.1f}% (target >60%) | {status}")
        elif name == "平等性智":
            print(f"  {icon} {name}: {data.get('raw_long_term_ego', '?')} (target <0.3) | {status}")
        elif name == "妙观察智":
            print(f"  {icon} {name}: {data.get('raw_prajna_ratio', '?')} (target <15%) | {status}")
        elif name == "成所作智":
            score = data.get("score", "?")
            res = data.get("resources_found", "?")
            steps = data.get("total_steps", "?")
            print(f"  {icon} {name}: score={score} | {status}")
            if isinstance(res, int) and isinstance(steps, int):
                print(f"       资源发现: {res}/3 ({steps}步中)")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    asyncio.run(main())
