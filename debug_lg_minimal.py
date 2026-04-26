"""Minimal production graph run - print step-by-step."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import deque
from yogacara_agent.yogacara_test import (
    GridSimEnv, AlayaMemory, ManasController, ConsciousnessPlanner,
    _planner_rng, _manas_rng,
)

# Reset RNG so this is deterministic
_planner_rng.seed(42)
_manas_rng.seed(42)

env = GridSimEnv()
alaya = AlayaMemory()
manas = ManasController()
planner = ConsciousnessPlanner()

obs = env._observe()
for step in range(5):
    seeds = alaya.retrieve(obs)
    # is_stuck detection (same as demo YogacaraAgent.__init__ / run loop)
    _last_pos = (env.agent_pos if hasattr(env, 'agent_pos') else list(obs["pos"]))
    # Replicate demo's _steps_stuck logic (not in production node_plan)
    is_stuck = False  # production node_plan doesn't track _steps_stuck
    # Sync exploration counter
    # production: steps_since_resource from state (starts at 0)
    steps_since_resource = 0
    planner._steps_without_resource = steps_since_resource
    # Call plan
    action, unc, scores = planner.plan(obs, seeds, env.resources, is_stuck=is_stuck)
    # Call manas filter
    action2, passed, note = manas.filter(
        action, obs, unc, step,
        deque(maxlen=5),  # empty recent_rewards (matches first few steps)
        deque(maxlen=5),  # empty pos_history
    )
    # Execute
    obs, rew, done = env.step(action2)
    print(f"Step {step}: plan={action} manas={'pass' if passed else 'intercept->'+action2} "
          f"pos={obs['pos']} rew={rew} resources_left={len(env.resources)} "
          f"planner_next={_planner_rng.random():.6f}")
    if done:
        print(f"Done at step {step}")
        break
