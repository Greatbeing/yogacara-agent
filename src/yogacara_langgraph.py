import asyncio
import logging
import math
import random
import threading
import time
from collections import deque
from typing import Any, TypedDict

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
    return f"[KB] 检索到与 '{query}' 相关的3条经验策略"


@tool
def call_external_api(endpoint: str, payload: dict) -> dict:
    return {"status": "success", "data": {"latency_ms": random.randint(20, 150)}}


@tool
def calculate_metric(metric_name: str, values: list[float]) -> float:
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


class GridSimEnv:
    _INITIAL_RESOURCES = [(7, 7), (3, 8), (8, 2)]
    _TRAPS = [(4, 4), (6, 1), (2, 6)]

    def __init__(self):
        self.agent_pos = [0, 0]
        self.resources = list(self._INITIAL_RESOURCES)  # copy to allow mutation
        self.traps = list(self._TRAPS)
        self.step_count = 0
        self.done = False

    def reset(self):
        self.agent_pos = [0, 0]
        self.resources = list(self._INITIAL_RESOURCES)  # restore all resources
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
            # Skip seeds with invalid timestamps (test data or future)
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


# ── Module-level instances (legacy, single-process only) ──
# WARNING: These are NOT safe for concurrent use across requests.
# For production, use create_session() to get isolated instances.
_lock = threading.Lock()
env = GridSimEnv()
alaya = AlayaMemory()
manas = ManasController()


def create_session() -> dict:
    """Create an isolated session with fresh env/memory/manas instances.

    Use this for API servers and concurrent environments to avoid
    shared mutable state across requests.
    """
    return {
        "env": GridSimEnv(),
        "alaya": AlayaMemory(),
        "manas": ManasController(),
    }


async def node_perceive(state: YogacaraState) -> YogacaraState:
    # Use the obs from the previous step (already updated by node_execute).
    # Only re-observe if this is the first step (obs has no meaningful data).
    if state["step"] == 0 and not state["obs"].get("pos"):
        state["obs"] = env._observe()
    state["seeds"] = alaya.retrieve(state["obs"])
    return state


async def node_plan(state: YogacaraState) -> YogacaraState:
    view = state["obs"]["grid_view"]
    scores = {}
    for a in ACTIONS:
        idx = ACTION_TO_IDX[a]
        base = view[idx] if 0 <= idx < 9 else -0.5
        pos_b = sum(s["rew"] * s["imp"] for s in state["seeds"] if s["act"] == a and s["rew"] > 0) * 0.8
        neg_p = sum(abs(s["rew"]) * s["imp"] for s in state["seeds"] if s["act"] == a and s["rew"] < 0) * 0.5
        scores[a] = base + pos_b - neg_p + (0.25 if a != "STAY" else -0.8) + random.uniform(-0.03, 0.03)
    best = max(scores, key=lambda k: scores[k])  # type: ignore[arg-type]
    unc = max(0.0, min(1.0, 1.0 - (scores[best] - min(scores.values())) / 2.0))
    state["action"] = best
    state["unc"] = unc
    state["tool_calls"] = []
    if unc > 0.6:
        state["tool_calls"].append({"tool": "query_knowledge_base", "input": f"高不确定性状态 {state['obs']['pos']}"})
    if state["step"] % 15 == 0:
        state["tool_calls"].append(
            {"tool": "calculate_metric", "input": {"metric_name": "avg_reward", "values": state["recent_rewards"]}}
        )
    return state


async def node_manas(state: YogacaraState) -> YogacaraState:
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
    return state


async def node_store(state: YogacaraState) -> YogacaraState:
    alaya.add(
        {
            "emb": alaya._encode(state["obs"]),
            "act": state["action"],
            "rew": state["reward"],
            "ts": time.time(),
            "imp": 0.8,
            "align": 1.0 if state["manas_passed"] else 0.4,
            "unc": state["unc"],
            "tag": "依他起" if state["unc"] < 0.5 else "遍计所执",
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
        ("manas", node_manas),
        ("execute", node_execute),
        ("store", node_store),
    ]:
        wf.add_node(n, fn)
    wf.set_entry_point("perceive")
    for e in [("perceive", "plan"), ("plan", "manas"), ("manas", "execute"), ("execute", "store")]:
        wf.add_edge(*e)
    wf.add_conditional_edges("store", check_done, {"continue": "perceive", "end": END})
    return wf.compile()


async def slow_loop(alaya_mem, interval=10):
    """Background task for periodic memory consolidation.

    Runs indefinitely regardless of episode state — perfume_update is idempotent
    and safe to call even when no episode is active.
    """
    while True:
        await asyncio.sleep(interval)
        alaya_mem.perfume_update()


async def main():
    print("\n\033[36m🌀 唯识进化框架 LangGraph 版启动\033[0m")
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
    }
    final_state = await graph.ainvoke(init_state)
    print(
        f"\n✅ 运行结束 | 步数:{final_state['step']} | 累计奖励:{sum(final_state['recent_rewards']):.2f} | 末那反思:{manas.reflections}次"
    )


if __name__ == "__main__":
    asyncio.run(main())
