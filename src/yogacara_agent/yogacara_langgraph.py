import asyncio
import logging
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
    steps_since_resource: int  # 探索力重置计数器
    steps_at_same_pos: int  # 连续停留计数器（正确实现is_stuck检测）


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
        # GridSimV2: STAY has positive reward (existence bonus)
        if action == "STAY":
            reward += 0.5
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


# 使用持久化阿赖耶识（文件存储 + 可选向量存储）
from yogacara_agent.alaya_persistent import PersistentAlayaMemory  # noqa: E402


class AlayaMemory(PersistentAlayaMemory):
    """兼容旧接口的持久化阿赖耶识。"""

    def __init__(self):
        super().__init__(storage="file", path="memory/seeds.jsonl")


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
    """Plan using the shared ConsciousnessPlanner (same as demo)."""
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
    next_obs, rew, done = env.step(state["action"])
    state["reward"] = rew
    state["done"] = done
    state["obs"] = next_obs
    state["step"] += 1
    state["recent_rewards"].append(rew)
    state["pos_history"].append(next_obs["pos"])
    # Update steps_at_same_pos counter (correct is_stuck detection)
    if next_obs["pos"] == state["obs"].get("pos"):
        state["steps_at_same_pos"] = state.get("steps_at_same_pos", 0) + 1
    else:
        state["steps_at_same_pos"] = 0
    # Sync exploration counter: demo uses counter-as-trigger (never increments
    # between resources, resets to 0 when resource found). Match that behavior.
    if rew >= 4.0:
        state["steps_since_resource"] = 0
        planner._steps_without_resource = 0  # sync with shared planner
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
            "imp": 0.8,  # Fixed importance (matches demo behavior)
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
        ("plan", "manas"),
        ("manas", "execute"),
        ("execute", "introspect"),
        ("introspect", "store"),
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
        "steps_since_resource": 0,
        "steps_at_same_pos": 0,
    }
    final_state = await graph.ainvoke(init_state)
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
