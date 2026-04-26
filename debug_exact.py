"""Check random state right before first node_plan call in production."""
import sys, asyncio, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Simulate exact production startup sequence
random.seed(42)
from yogacara_agent.yogacara_langgraph import (
    env, alaya, manas, planner, build_graph,
    _get_introspection_logger, YogacaraState
)
_get_introspection_logger()

# Now re-seed to simulate what happens in main() (no re-seed!)
# In the real main(), there's NO random.seed(42) call after module-level seed
# The module-level seed sets the state, and then main() just runs

# Check random state
r1 = random.random()
random.seed(42)  # Reset because we just consumed one
print(f"Random state after module init (first random): {r1}")

# Now run production
random.seed(42)
import yogacara_agent.yogacara_langgraph as yl

# Monkey-patch node_plan to capture random state
_orig_plan = yl.node_plan
async def _debug_plan(state):
    r_before = random.random()
    random.seed(42)  # Oops, that consumed one. Let me not do that.
    # Actually, just print the state
    result = await _orig_plan(state)
    return result

# Instead, let's just check the sequence of random calls
# by running with identical seed and printing positions
random.seed(42)
yl.env = yl.GridSimEnv()
yl.alaya = yl.AlayaMemory()
yl.manas = yl.ManasController()
yl.planner.random_state_check = True
random.seed(42)

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

final = asyncio.run(run())
print(f"\nSteps: {final['step']}")
print(f"Resources: {3 - len(yl.env.resources)}/3")

# Compare with demo step 0
random.seed(42)
from yogacara_agent.yogacara_test import GridSimEnv as DE, ConsciousnessPlanner as DP
de = DE()
dp = DP()
obs = de.reset()
best, unc, scores = dp.plan(obs, [], env_resources=de.resources, is_stuck=False)
print(f"\nDemo step 0: {best} (unc={unc:.3f})")
