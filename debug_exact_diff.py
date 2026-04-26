"""Replicate production node logic EXACTLY and compare with demo."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import deque
from yogacara_agent.yogacara_test import (
    GridSimEnv, AlayaMemory, ManasController, ConsciousnessPlanner,
    _planner_rng, _manas_rng,
)

_planner_rng.seed(42)
_manas_rng.seed(42)

# ===== PRODUCTION PATH (exact node_plan / node_manas / node_execute replication) =====
env_p = GridSimEnv()
alaya_p = AlayaMemory()
manas_p = ManasController()
planner_p = ConsciousnessPlanner()

# Production initial state
state = {
    "obs": env_p._observe(),
    "step": 0,
    "seeds": [],
    "recent_rewards": [],
    "pos_history": [],
    "steps_since_resource": 0,
    "action": None,
    "unc": 0.5,
    "plan_scores": None,
    "manas_passed": True,
    "reward": 0.0,
    "done": False,
}
print("=== PRODUCTION path (exact node replication) ===")
for step in range(3):
    state["seeds"] = alaya_p.retrieve(state["obs"])
    # is_stuck detection (from node_plan)
    pos_hist_deque = deque(state["pos_history"], maxlen=5)
    is_stuck_prod = len(pos_hist_deque) >= 3 and len(set(pos_hist_deque[-3:])) == 1
    # Sync steps_since_resource
    planner_p._steps_without_resource = state.get("steps_since_resource", state["step"])
    # node_plan
    best, unc, scores = planner_p.plan(state["obs"], state["seeds"], env_p.resources, is_stuck=is_stuck_prod)
    state["action"] = best
    state["unc"] = unc
    state["plan_scores"] = scores
    # node_manas
    final, passed, log = manas_p.filter(
        state["action"], state["obs"], state["unc"], state["step"],
        deque(state["recent_rewards"], maxlen=5),
        deque(state["pos_history"], maxlen=5),
    )
    state["action"] = final
    state["manas_passed"] = passed
    # node_execute
    obs2, rew, done = env_p.step(state["action"])
    state["obs"] = obs2
    state["reward"] = rew
    state["done"] = done
    state["step"] += 1
    state["recent_rewards"].append(rew)
    state["pos_history"].append(obs2["pos"])
    if rew >= 4.0:
        state["steps_since_resource"] = 0
        planner_p._steps_without_resource = 0
    else:
        state["steps_since_resource"] = state.get("steps_since_resource", state["step"] - 1) + 1
    print(f"  Step {step}: plan={best} final={state['action']} "
          f"pos={obs2['pos']} unc={unc:.3f} "
          f"_steps_without_resource={planner_p._steps_without_resource} "
          f"steps_since_resource={state['steps_since_resource']} "
          f"is_stuck={is_stuck_prod} scores={dict(sorted(scores.items()))}")

print()
print("=== DEMO path (from run_demo.py source) ===")
_planner_rng.seed(42)
_manas_rng.seed(42)
env_d = GridSimEnv()
alaya_d = AlayaMemory()
manas_d = ManasController()
planner_d = ConsciousnessPlanner()
_last_pos = None
_steps_stuck = 0
planner_d._steps_without_resource = 0  # Demo initializes to 0

for step in range(3):
    obs = env_d._observe()
    seeds = alaya_d.retrieve(obs)
    is_stuck = (_last_pos == obs["pos"] and _steps_stuck >= 2)
    action, unc, scores = planner_d.plan(obs, seeds, env_resources=env_d.resources, is_stuck=is_stuck)
    final, passed, log = manas_d.filter(action, obs, unc, step, deque(maxlen=5), deque(maxlen=5))
    next_obs, rew, done = env_d.step(final)
    print(f"  Step {step}: plan={action} final={final} "
          f"pos={next_obs['pos']} unc={unc:.3f} "
          f"_steps_without_resource={planner_d._steps_without_resource} "
          f"is_stuck={is_stuck} scores={dict(sorted(scores.items()))}")
    # Demo's stuck tracking
    if final != "STAY" and _last_pos != obs["pos"]:
        _steps_stuck = 0
    else:
        _steps_stuck += 1
    _last_pos = obs["pos"]
    if rew > 2.0:
        planner_d._steps_without_resource = 0
    if done:
        break
