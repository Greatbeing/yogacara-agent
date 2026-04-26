"""验证 _steps_without_resource 同步逻辑."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from collections import deque
from yogacara_agent.yogacara_test import (
    GridSimEnv, AlayaMemory, ManasController, ConsciousnessPlanner,
    _planner_rng, _manas_rng,
)

# Reset RNG
_planner_rng.seed(42)
_manas_rng.seed(42)

# ===== DEMO PATH =====
print("=== DEMO PATH ===")
env = GridSimEnv()
planner = ConsciousnessPlanner()
planner._steps_without_resource = 0  # 初始化
obs = env._observe()

steps_without_resource_log = []
for step in range(10):
    steps_without_resource_log.append(planner._steps_without_resource)
    # Demo 的 plan
    action, unc, scores = planner.plan(obs, [], env.resources, is_stuck=False)
    # Demo 的 execute
    next_obs, rew, done = env.step(action)
    # Demo 的同步逻辑（在 run 方法里）
    if rew > 2.0:
        planner._steps_without_resource = 0
    # Demo 不在 else 里累加！这是关键差异！
    obs = next_obs
    if done:
        break

print(f"Demo _steps_without_resource: {steps_without_resource_log}")
print(f"Demo final: step={step}, resources_left={len(env.resources)}")

# ===== PRODUCTION PATH =====
print("\n=== PRODUCTION PATH ===")
_planner_rng.seed(42)
_manas_rng.seed(42)
env = GridSimEnv()
planner = ConsciousnessPlanner()

state = {
    "obs": env._observe(),
    "step": 0,
    "steps_since_resource": 0,
    "recent_rewards": [],
    "pos_history": [],
}

steps_without_resource_log_prod = []
for step in range(10):
    # Production node_plan 的同步
    planner._steps_without_resource = state.get("steps_since_resource", state["step"])
    steps_without_resource_log_prod.append(planner._steps_without_resource)
    
    # Production plan
    action, unc, scores = planner.plan(state["obs"], [], env.resources, is_stuck=False)
    
    # Production execute
    next_obs, rew, done = env.step(action)
    state["obs"] = next_obs
    state["step"] += 1
    state["recent_rewards"].append(rew)
    
    # Production node_execute 的同步
    if rew >= 4.0:
        state["steps_since_resource"] = 0
        planner._steps_without_resource = 0
    else:
        state["steps_since_resource"] = state.get("steps_since_resource", state["step"]) + 1
        planner._steps_without_resource = state["steps_since_resource"]
    
    if done:
        break

print(f"Prod _steps_without_resource: {steps_without_resource_log_prod}")
print(f"Prod final: step={step}, resources_left={len(env.resources)}")

# 对比
print("\n=== DIFF ===")
for i in range(min(len(steps_without_resource_log), len(steps_without_resource_log_prod))):
    d = steps_without_resource_log[i]
    p = steps_without_resource_log_prod[i]
    marker = "  " if d == p else "!!"
    print(f"{marker} Step {i}: Demo={d} Prod={p}")
