"""详细追踪 Step 8 的决策差异."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from collections import deque
from yogacara_agent.yogacara_test import (
    GridSimEnv, AlayaMemory, ManasController, ConsciousnessPlanner,
    _planner_rng, _manas_rng,
)

_planner_rng.seed(42)
_manas_rng.seed(42)

env = GridSimEnv()
planner = ConsciousnessPlanner()
planner._steps_without_resource = 0

obs = env.reset()

print("=== Replicating Demo Path (no exploration counter) ===")
for step in range(9):  # Run to step 8
    # Demo 的 plan
    action, unc, scores = planner.plan(obs, [], env.resources, is_stuck=False)
    
    print(f"Step {step}: pos={obs['pos']} plan={action} unc={unc:.3f} "
          f"_steps_without_resource={planner._steps_without_resource} "
          f"exploration_force={planner._steps_without_resource >= 15}")
    
    # Demo execute
    obs, rew, done = env.step(action)
    
    # Demo 的同步逻辑（只在发现资源时重置）
    if rew > 2.0:
        planner._steps_without_resource = 0

print("\n=== Replicating Production Path (with exploration counter) ===")
_planner_rng.seed(42)
env = GridSimEnv()
planner = ConsciousnessPlanner()

state = {
    "obs": env.reset(),
    "step": 0,
    "steps_since_resource": 0,
}

for step in range(9):  # Run to step 8
    # Production node_plan 的同步
    planner._steps_without_resource = state.get("steps_since_resource", state["step"])
    
    # Production plan
    action, unc, scores = planner.plan(state["obs"], [], env.resources, is_stuck=False)
    
    print(f"Step {step}: pos={state['obs']['pos']} plan={action} unc={unc:.3f} "
          f"_steps_without_resource={planner._steps_without_resource} "
          f"exploration_force={planner._steps_without_resource >= 15}")
    
    # Production execute
    obs, rew, done = env.step(action)
    state["obs"] = obs
    state["step"] += 1
    
    # Production node_execute 的同步
    if rew >= 4.0:
        state["steps_since_resource"] = 0
        planner._steps_without_resource = 0
    else:
        state["steps_since_resource"] = state.get("steps_since_resource", state["step"]) + 1
        planner._steps_without_resource = state["steps_since_resource"]
