"""Minimal reproduction: does langgraph consume random numbers between seed and first plan?"""
import sys, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Step 1: Set seed
random.seed(42)

# Step 2: Import production module (should not consume random, but let's verify)
from yogacara_agent.yogacara_langgraph import (
    env, alaya, manas, planner, build_graph,
    _get_introspection_logger
)
_get_introspection_logger()

# Step 3: Check random state
# If no random numbers were consumed, the next random should be 0.6394267984578837
# But we can't check without consuming one, so let's use getstate()
random.seed(42)
state_after_seed = random.getstate()

# Now reload the module and check
import importlib
import yogacara_agent.yogacara_langgraph as yl
importlib.reload(yl)

random.seed(42)
# Simulate what main() does: just call graph.ainvoke with seed=42
yl._get_introspection_logger()

# Check state right before graph.ainvoke
state_before_invoke = random.getstate()
random.seed(42)
expected_state = random.getstate()

if state_before_invoke == expected_state:
    print("PASS: Random state is correct before graph.ainvoke")
else:
    print("FAIL: Random state was modified between seed and invoke!")
    # Find which random call broke it
    random.seed(42)
    print(f"Expected next random: {random.random()}")
    random.seed(42)
    # Now replicate the setup
    r = random.random()
    print(f"Actual next random: {r}")

# Direct test: run planner.plan() with seed 42
random.seed(42)
from yogacara_agent.yogacara_test import GridSimEnv, ConsciousnessPlanner
test_env = GridSimEnv()
test_planner = ConsciousnessPlanner()
test_obs = test_env.reset()
best, unc, scores = test_planner.plan(test_obs, [], env_resources=test_env.resources, is_stuck=False)
print(f"\nDirect planner.plan() with seed 42: best={best} unc={unc:.3f}")

# Now test through production
random.seed(42)
yl.env = yl.GridSimEnv()
yl.alaya = yl.AlayaMemory()
yl.manas = yl.ManasController()
yl.planner = ConsciousnessPlanner()
random.seed(42)

# Check what node_plan does
import asyncio
state0 = {
    "obs": yl.env.reset(), "action": "", "reward": 0.0, "done": False,
    "step": 0, "seeds": [], "unc": 0.0, "manas_passed": True,
    "tool_calls": [], "recent_rewards": [], "pos_history": [],
    "metrics": {}, "introspection_record": None, "ego_alert": None,
    "plan_scores": None, "reasoning": "", "steps_since_resource": 0,
}

# Run just node_plan directly
async def test_plan():
    result = await yl.node_plan(state0)
    return result

result0 = asyncio.run(test_plan())
print(f"Production node_plan step 0: best={result0['action']} unc={result0['unc']:.3f}")
