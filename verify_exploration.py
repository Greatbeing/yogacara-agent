"""检查 demo 是否使用了 exploration_force."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from yogacara_agent.yogacara_test import (
    GridSimEnv, AlayaMemory, ManasController, ConsciousnessPlanner, Seed,
    _planner_rng, _manas_rng,
)

_planner_rng.seed(42)
_manas_rng.seed(42)

env = GridSimEnv()
alaya = AlayaMemory()
manas = ManasController()
planner = ConsciousnessPlanner()
planner._steps_without_resource = 0

obs = env.reset()
print(f"Initial: resources={env.resources}")

exploration_triggered = []
steps_log = []

for step in range(50):
    seeds = alaya.retrieve(obs)
    
    # 检查 exploration_force 是否会触发
    exploration_force = planner._steps_without_resource >= 15
    exploration_triggered.append(exploration_force)
    
    action, unc, scores = planner.plan(obs, seeds, env.resources, is_stuck=False)
    final, passed, log = manas.filter(action, obs, unc, step, [], [])
    
    next_obs, rew, done = env.step(final)
    
    steps_log.append({
        "step": step,
        "pos": obs["pos"],
        "action": final,
        "rew": rew,
        "steps_without_resource": planner._steps_without_resource,
        "exploration_force": exploration_force,
    })
    
    # Demo 的同步逻辑
    if rew > 2.0:
        planner._steps_without_resource = 0
    
    seed = Seed(alaya._encode(obs), final, rew, 1.0, 0.8, 1.0 if passed else 0.4, unc, "依他起")
    alaya.add(seed)
    
    obs = next_obs
    if done:
        print(f"Done at step {step}")
        break

print(f"\nTotal steps: {len(steps_log)}")
print(f"Resources found: {sum(1 for s in steps_log if s['rew'] > 2.0)}")
print(f"Exploration triggered: {sum(exploration_triggered)} times")

# 打印 steps_without_resource 变化
print("\n_steps_without_resource evolution:")
for s in steps_log:
    if s["rew"] > 2.0:
        print(f"  Step {s['step']}: {s['steps_without_resource']} -> RESET (found resource)")
    elif s["exploration_force"]:
        print(f"  Step {s['step']}: {s['steps_without_resource']} -> EXPLORATION!")
