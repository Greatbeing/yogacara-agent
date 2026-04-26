"""Check if LangGraph's ainvoke consumes random numbers."""
import sys, asyncio, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)
from yogacara_agent.yogacara_langgraph import (
    env, alaya, manas, planner, build_graph, _get_introspection_logger
)
_get_introspection_logger()

# Reset instances
random.seed(42)
import yogacara_agent.yogacara_langgraph as yl
yl.env = yl.GridSimEnv()
yl.alaya = yl.AlayaMemory()
yl.manas = yl.ManasController()
yl.planner = type(yl.planner)()  # Fresh planner

random.seed(42)

# Monkey-patch node_plan to print the random state
_orig_plan = yl.node_plan

async def _tracked_plan(state):
    # Check random state right before planner.plan()
    r_test = random.random()
    random.seed(42)  # Reset to undo the consumption
    # Actually, we need to get state without consuming
    state_tuple = random.getstate()
    # Compute what the "next" random would be
    random.seed(42)
    expected_first = random.random()
    # Now restore
    random.setstate(state_tuple)
    next_r = random.random()
    print(f"  [TRACE] step={state['step']} next_random={next_r} expected_first={expected_first}")
    result = await _orig_plan(state)
    print(f"  [TRACE] step={state['step']} action={result['action']} unc={result['unc']:.3f}")
    return result

yl.node_plan = _tracked_plan
graph = yl.build_graph()

init_state = {
    "obs": yl.env.reset(), "action": "", "reward": 0.0, "done": False,
    "step": 0, "seeds": [], "unc": 0.0, "manas_passed": True,
    "tool_calls": [], "recent_rewards": [], "pos_history": [],
    "metrics": {}, "introspection_record": None, "ego_alert": None,
    "plan_scores": None, "reasoning": "", "steps_since_resource": 0,
}

async def run():
    final = await graph.ainvoke(init_state)
    return final

# Reset seed one more time before ainvoke
random.seed(42)
final = asyncio.run(run())
