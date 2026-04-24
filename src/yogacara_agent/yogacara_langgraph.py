import asyncio
import logging
import math
import random
import threading
import time
from collections import deque
from typing import Any, TypedDict, cast

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

GRID_SIZE = 10
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
ACTION_TO_IDX = {"UP": 1, "DOWN": 7, "LEFT": 3, "RIGHT": 5, "STAY": 4}
random.seed(42)


@tool
def query_knowledge_base(query: str) -> str:
    """Query the knowledge base for relevant experience strategies."""
    return f"[KB] Found 3 strategies related to '{query}'"


@tool
def call_external_api(endpoint: str, payload: dict) -> dict:
    """Call an external API endpoint with the given payload."""
    return {"status": "success", "data": {"latency_ms": random.randint(20, 150)}}


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
    introspection_record: "_IntrospectionRecordData | None"
    ego_alert: dict | None
    plan_scores: "dict[str, float] | None"
    reasoning: str


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
        return self._observe()

    def step(self, action: str) -> tuple[dict, float, bool]:
        dx, dy = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "STAY": (0, 0)}[action]
        nx = max(0, min(GRID_SIZE - 1, self.agent_pos[0] + dx))
        ny = max(0, min(GRID_SIZE - 1, self.agent_pos[1] + dy))
        self.agent_pos = [nx, ny]
        self.step_count += 1
        reward = -0.1
        pos = tuple(self.agent_pos)
        if pos in self.resources:
            reward = 5.0
            self.resources.remove(pos)
        elif pos in self.traps:
            reward = -3.0
        if not self.resources or self.step_count >= 60:
            self.done = True
        return self._observe(), reward, self.done

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


class AlayaMemory:
    def __init__(self):
        self.seeds = []

    def _encode(self, obs):
        return [obs["pos"][0] / GRID_SIZE, obs["pos"][1] / GRID_SIZE] + [v / 2.0 for v in obs["grid_view"]]

    def _dist(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def retrieve(self, obs, k=3):
        if not self.seeds:
            return []
        emb = self._encode(obs)
        scored = sorted([(self._dist(emb, s["emb"]), s) for s in self.seeds], key=lambda x: x[0])
        return [s for _, s in scored[:k]]

    def add(self, seed):
        self.seeds.append(seed)

    def perfume_update(self):
        now = time.time()
        for s in self.seeds:
            dt = now - s["ts"]
            if dt <= 0 or dt > 86400 * 365:
                continue
            s["imp"] *= math.exp(-0.12 * dt)
            s["imp"] = min(1.0, s["imp"] + 0.3 * max(0, s["rew"]))


class ManasController:
    def __init__(self):
        self.reflections = 0
        self.last_intercept = -10
        self.cooldown = 4

    def filter(self, action, obs, unc, step, recent_rew, pos_hist):
        if step - self.last_intercept < self.cooldown:
            return action, True, "冷却放行"
        target_risk = 1.0 if obs["grid_view"][ACTION_TO_IDX.get(action, 4)] == -1.0 else 0.0
        stagnation = step > 15 and len(recent_rew) >= 5 and sum(recent_rew) <= -0.48
        loop = step > 12 and len(pos_hist) >= 5 and len(set(pos_hist)) <= 2
        threshold = 0.45 + min(0.15, step / 80.0)
        danger = target_risk * 0.8 + max(0.0, unc - 0.80) * 0.2
        if danger > threshold or stagnation or loop:
            self.reflections += 1
            self.last_intercept = step
            fallback = random.choice([a for a in ["UP", "DOWN", "LEFT", "RIGHT"] if a != action])
            return fallback, False, f"[末那拦截] 风险:{target_risk:.1f} 停滞:{stagnation} 循环:{loop} → 换向:{fallback}"
        return action, True, "放行"


# Module-level instances (legacy, single-process only)
# WARNING: These are NOT safe for concurrent use across requests.
# For production, use create_session() to get isolated instances.
_lock = threading.Lock()
env = GridSimEnv()
alaya = AlayaMemory()
manas = ManasController()
# 转识成智 Phase1 新增模块
introspection_logger = None  # lazy init to avoid circular import
ego_monitor = None
seed_classifier = None  # lazy init
_seed_counts = {"名言种": 0, "业种": 0, "异熟种": 0}  # Phase1-2 seed type counter
_parinispanna_count = 0  # Phase3: 圆成实种子计数（用于大圆镜智指标）
_total_classified = 0  # Phase3: 总分类种子数


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


def create_session() -> dict:
    """Create an isolated session with fresh env/memory/manas instances."""
    from yogacara_agent.ego_monitor import EgoMonitor
    from yogacara_agent.introspection import IntrospectionLogger

    return {
        "env": GridSimEnv(),
        "alaya": AlayaMemory(),
        "manas": ManasController(),
        "introspection": IntrospectionLogger(),
        "ego_monitor": EgoMonitor(),
    }


async def node_perceive(state: YogacaraState) -> YogacaraState:
    if state["step"] == 0 and not state["obs"].get("pos"):
        state["obs"] = env._observe()
    state["seeds"] = alaya.retrieve(state["obs"])
    return state


async def node_plan(state: YogacaraState) -> YogacaraState:
    view = state["obs"]["grid_view"]
    pos = state["obs"].get("pos", (0, 0))
    best_dir_r: str | None = None
    best_dir_c: str | None = None
    dist_bonus = 0.0
    if not any(v == 1.0 for v in view) and env.resources:
        nearest = min(env.resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))
        best_dir_r = "DOWN" if nearest[0] > pos[0] else "UP" if nearest[0] < pos[0] else "STAY"
        best_dir_c = "RIGHT" if nearest[1] > pos[1] else "LEFT" if nearest[1] < pos[1] else "STAY"
        dist_bonus = 0.4

    scores = {}
    for a in ACTIONS:
        idx = ACTION_TO_IDX[a]
        base = view[idx] if 0 <= idx < 9 else -0.5
        pos_b = sum(s["rew"] * s["imp"] for s in state["seeds"] if s["act"] == a and s["rew"] > 0) * 0.8
        neg_p = sum(abs(s["rew"]) * s["imp"] for s in state["seeds"] if s["act"] == a and s["rew"] < 0) * 0.5
        approach = dist_bonus if best_dir_r is not None and a in (best_dir_r, best_dir_c) else 0.0
        scores[a] = base + pos_b - neg_p + approach + (0.25 if a != "STAY" else -0.8) + random.uniform(-0.03, 0.03)
    best = max(scores, key=lambda k: scores[k])  # type: ignore[arg-type]
    unc = max(0.0, min(1.0, 1.0 - (scores[best] - min(scores.values())) / 2.0))
    state["action"] = best
    state["unc"] = unc
    state["tool_calls"] = []
    state["plan_scores"] = scores
    state["reasoning"] = _build_reasoning(state, best, scores)
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

    在每次决策后调用，对认知过程做结构化记录。
    这是"自指环"的核心——Agent 观察自己的决策过程，
    积累数据后才能谈"转依"。
    """
    logger = _get_introspection_logger()
    plan_scores = state["plan_scores"]
    if plan_scores is None:
        alternatives = ACTIONS
        score_best = 0.0
        score_second = 0.0
    else:
        alternatives = list(plan_scores.keys())
        score_best = plan_scores.get(state["action"], 0.0)
        score_second = max((v for k, v in plan_scores.items() if k != state["action"]), default=0.0)

    record = logger.observe(
        step=state["step"],
        obs=state["obs"],
        action=state["action"],
        unc=state["unc"],
        seeds_retrieved=state["seeds"],
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
    return state


async def node_manas(state: YogacaraState) -> YogacaraState:
    """
    增强的末那识节点：同时处理环境安全 + 认知我执。

    原功能：环境安全拦截（陷阱、停滞、循环）
    新增功能：认知我执评估 → 生成转依提醒（不强制拦截）
    """
    # 原功能：环境安全过滤
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

    # 新增：我执评估（仅提醒，不拦截）
    _int_rec = state.get("introspection_record")
    if _int_rec is not None:
        from yogacara_agent.introspection import IntrospectionRecord

        rec_data: _IntrospectionRecordData = _int_rec  # type: ignore[assignment, misc]
        rec = IntrospectionRecord(
            step=rec_data["step"],
            timestamp=0.0,
            obs=state["obs"],
            action=final,
            unc=rec_data["unc"],
            seeds_retrieved=state["seeds"],
            reasoning=rec_data.get("reasoning", "") or "",
            alternatives=[],
            ego_markers=rec_data.get("ego_markers", []) or [],
            nature=rec_data.get("nature", "") or "",
            nature_confidence=0.5,
            score_best=0.0,
            score_second=0.0,
            decision_gap=rec_data.get("decision_gap", 0.0),
            manas_intercepted=not passed,
        )
        ego = _get_ego_monitor().assess(rec)
        state["ego_alert"] = {
            "ego_score": ego.ego_score,
            "long_term_ego": ego.long_term_ego,
            "triggered": ego.triggered,
            "recommendation": ego.recommendation,
            "equanimity_wisdom": ego.equanimity_wisdom,
            "prajna_wisdom": ego.prajna_wisdom,
        }
        if ego.triggered:
            nature_tag = rec_data.get("nature", "")
            print(
                f"\033[35m[末那识提醒 step {state['step']}] {ego.recommendation} | 三性:{nature_tag} | 平等性智:{ego.equanimity_wisdom:.0%} 妙观察智:{ego.prajna_wisdom:.0%}\033[0m"
            )

    return state


async def node_execute(state: YogacaraState) -> YogacaraState:
    for tc in state["tool_calls"]:
        tool_fn = TOOL_MAP[tc["tool"]]
        res = tool_fn.invoke(tc["input"]) if isinstance(tc["input"], dict) else tool_fn.invoke({"query": tc["input"]})
        print(f"\033[90m[工具] {tc['tool']} → {res}\033[0m")
    next_obs, rew, done = env.step(state["action"])
    state["reward"] = rew
    state["done"] = done
    state["obs"] = next_obs
    state["step"] += 1
    state["recent_rewards"].append(rew)
    state["pos_history"].append(next_obs["pos"])
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
            "emb": alaya._encode(state["obs"]),
            "act": state["action"],
            "rew": state["reward"],
            "ts": time.time(),
            "imp": 0.8 if is_vipaka else classification.align,
            "align": classification.align,
            "unc": state["unc"],
            "tag": seed_tag,
        }
    )
    return state


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
    ]:
        wf.add_node(n, fn)
    wf.set_entry_point("perceive")
    for e in [
        ("perceive", "plan"),
        ("plan", "introspect"),
        ("introspect", "manas"),
        ("manas", "execute"),
        ("execute", "store"),
    ]:
        wf.add_edge(*e)
    wf.add_conditional_edges("store", check_done, {"continue": "perceive", "end": END})
    return wf.compile()


async def slow_loop(alaya_mem, interval=10):
    """Background task for periodic memory consolidation."""
    while True:
        await asyncio.sleep(interval)
        alaya_mem.perfume_update()


async def main():
    print("\n\033[36m~ 唯识进化框架 LangGraph 版（转识成智 Phase2-2）~\033[0m")
    # 初始化内省系统（lazy init 避免循环导入）
    _get_introspection_logger()
    graph = build_graph()
    asyncio.create_task(slow_loop(alaya, interval=10))
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
    }
    final_state = await graph.ainvoke(init_state)
    total_steps = final_state["step"]
    total_reward = sum(final_state["recent_rewards"])
    print(f"\n>> 运行结束 | 步数:{total_steps} | 累计奖励:{total_reward:.2f} | 末那反思:{manas.reflections}次")
    # Phase3: 四智量化报告（基于种子分类统计）
    print("\n\033[36m~ 四智转依进度报告 (Phase3 量化版) ~\033[0m")
    # 1. 大圆镜智 = 圆成实种子占比
    mirror_ratio = _parinispanna_count / _total_classified if _total_classified > 0 else 0
    mirror_status = "达标" if mirror_ratio >= 0.6 else "未达标"
    mirror_sc = "\033[32m" if mirror_status == "达标" else "\033[33m"
    print(f"  大圆镜智: {mirror_sc}{mirror_status}\033[0m  圆成实占比:{mirror_ratio:.1%} (目标:>60%)")
    # 2. 平等性智 = 我执分数均值（来自ego_monitor）
    report = _get_ego_monitor().four_wisdoms_report()
    equality_data = report.get("平等性智", {})
    if isinstance(equality_data, dict):
        eq_raw = equality_data.get("raw_long_term_ego", "")
        eq_status = equality_data.get("status", "")
        eq_sc = "\033[32m" if "达标" in eq_status else "\033[33m"
        print(f"  平等性智: {eq_sc}{eq_status}\033[0m  我执均值:{eq_raw} (目标:<0.3)")
    else:
        print(f"  平等性智: {equality_data}")
    # 3. 妙观察智 = 遍计所执比例（来自内省摘要）
    summary = _get_introspection_logger().recent_summary()
    total_nature = sum(summary["nature_distribution"].values())
    parikalpita_ratio = summary["nature_distribution"]["遍计所执"] / total_nature if total_nature > 0 else 0
    prajna_status = "达标" if parikalpita_ratio <= 0.15 else "未达标"
    prajna_sc = "\033[32m" if prajna_status == "达标" else "\033[33m"
    print(f"  妙观察智: {prajna_sc}{prajna_status}\033[0m  遍计所执比例:{parikalpita_ratio:.1%} (目标:<15%)")
    # 4. 成所作智 = 感知-行动-反馈闭环完成率
    # 计算：高奖励决策占比（奖励>0且非异熟种=环境奖励与预期一致）
    total_seeds = sum(_seed_counts.values())
    vipaka_count = _seed_counts["异熟种"]
    non_vipaka = total_seeds - vipaka_count
    # 成所作智：非异熟种中，奖励为正的占比 = 前五识如实反映环境
    # 简化计算：用 recent_rewards 中正奖励比例
    positive_rewards = sum(1 for r in final_state["recent_rewards"] if r > 0)
    action_ratio = positive_rewards / len(final_state["recent_rewards"]) if final_state["recent_rewards"] else 0
    action_status = "达标" if action_ratio >= 0.9 else "未达标"
    action_sc = "\033[32m" if action_status == "达标" else "\033[33m"
    print(f"  成所作智: {action_sc}{action_status}\033[0m  正反馈率:{action_ratio:.1%} (目标:>90%)")
    # 内省数据摘要
    print("\n\033[36m~ 内省数据摘要（最近20步）~\033[0m")
    print(
        f"  三性: 圆成实{summary['nature_distribution']['圆成实']} | 依他起{summary['nature_distribution']['依他起']} | 遍计所执{summary['nature_distribution']['遍计所执']}"
    )
    print(f"  我执模式: {summary['ego_patterns']}  末那拦截率:{summary['intercept_rate']:.0%}")
    # 种子分类统计
    print("\n\033[36m~ 种子分类统计（全程）~\033[0m")
    print(f"  名言种: {_seed_counts['名言种']} | 业种: {_seed_counts['业种']} | 异熟种: {_seed_counts['异熟种']}")
    if total_seeds > 0:
        print(
            f"  占比: 名言{_seed_counts['名言种'] * 100 // total_seeds}% | 业{_seed_counts['业种'] * 100 // total_seeds}% | 异熟{_seed_counts['异熟种'] * 100 // total_seeds}%"
        )
        print(f"  圆成实种子: {_parinispanna_count}/{_total_classified} ({mirror_ratio:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
