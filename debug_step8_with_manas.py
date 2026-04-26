"""完整模拟 Demo 路径（包含 manas 拦截）."""
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
alaya = AlayaMemory()
manas = ManasController()
planner = ConsciousnessPlanner()
planner._steps_without_resource = 0

recent_rewards = deque(maxlen=5)
pos_history = deque(maxlen=5)

obs = env.reset()

print("=== Demo Path (with manas) ===")
for step in range(12):
    pos_history.append(obs["pos"])
    
    # Demo plan
    action, unc, scores = planner.plan(obs, [], env.resources, is_stuck=False)
    
    # Demo manas filter
    final, passed, log = manas.filter(action, obs, unc, step, recent_rewards, pos_history)
    
    print(f"Step {step}: pos={obs['pos']} plan={action:6s} final={final:6s} "
          f"unc={unc:.3f} passed={passed} _steps_without_resource={planner._steps_without_resource}")
    
    if not passed:
        print(f"  [Manas intercept]: {log}")
    
    # Demo execute
    obs, rew, done = env.step(final)
    recent_rewards.append(rew)
    
    # Demo 的同步逻辑
    if rew > 2.0:
        planner._steps_without_resource = 0

print("\n=== Production Path (with manas + exploration counter) ===")
_planner_rng.seed(42)
_manas_rng.seed(42)

env = GridSimEnv()
manas = ManasController()
planner = ConsciousnessPlanner()

state = {
    "obs": env.reset(),
    "step": 0,
    "steps_since_resource": 0,
    "recent_rewards": [],
    "pos_history": [],
}

for step in range(12):
    # Production node_plan 的同步
    planner._steps_without_resource = state.get("steps_since_resource", state["step"])
    
    # Production plan
    action, unc, scores = planner.plan(state["obs"], [], env.resources, is_stuck=False)
    
    # Production manas
    final, passed, log = manas.filter(
        action, state["obs"], unc, step,
        deque(state["recent_rewards"], maxlen=5),
        deque(state["pos_history"], maxlen=5),
    )
    
    print(f"Step {step}: pos={state['obs']['pos']} plan={action:6s} final={final:6s} "
          f"unc={unc:.3f} passed={passed} _steps_without_resource={planner._steps_without_resource}")
    
    if not passed:
        print(f"  [Manas intercept]: {log}")
    
    # Production execute
    obs, rew, done = env.step(final)
    state["obs"] = obs
    state["step"] += 1
    state["recent_rewards"].append(rew)
    state["pos_history"].append(obs["pos"])
    
    # Production node_execute 的同步
    if rew >= 4.0:
        state["steps_since_resource"] = 0
        planner._steps_without_resource = 0
    else:
        state["steps_since_resource"] = state.get("steps_since_resource", state["step"]) + 1
        planner._steps_without_resource = state["steps_since_resource"]
