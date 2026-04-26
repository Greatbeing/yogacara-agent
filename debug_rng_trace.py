"""Trace RNG state at each plan() call in both execution paths."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from yogacara_agent import yogacara_test as yt
from yogacara_agent.yogacara_test import _planner_rng, _manas_rng

STEPS = 10  # Just first 10 steps to find the divergence point

def trace_planner(obs, seeds, resources, manas_filter, step_count, rng_state_list):
    # Record RNG state BEFORE this plan call
    rng_state_list.append({
        "step": step_count,
        "planner_state": _planner_rng.getstate(),
        "planner_next": _planner_rng.random(),
        "manas_state": _manas_rng.getstate(),
    })

    action, unc, scores = yt.ConsciousnessPlanner().plan(obs, seeds, resources)
    action2, passed, note = manas_filter(action, obs, unc, step_count, [], [])
    return action, action2, passed, note, scores

def run_demo_trace():
    agent = yt.YogacaraAgent()
    rng_log = []
    for step in range(STEPS):
        obs = agent.env._observe()
        seeds = agent.alaya.retrieve(obs)
        a1, a2, passed, note, scores = trace_planner(
            obs, seeds, agent.env.resources,
            lambda a, o, u, s, h: agent.manas.filter(a, o, u, s, h),
            agent.env.step_count, rng_log
        )
        print(f"  Step {step}: plan={a1} scores={dict(sorted(scores.items()))} "
              f"manas={a2 if a2!=a1 else 'pass'} note={note}")
        obs2, _, done = agent.env.step(a2)
        if done:
            print(f"  Done at step {step}")
            break
    return rng_log

# Reset RNG before each run
_planner_rng.seed(42)
_manas_rng.seed(42)
print("=== DEMO path ===")
log_demo = run_demo_trace()

print()
print("RNG states BEFORE each plan call (planner_next is the random() value):")
for entry in log_demo[:5]:
    print(f"  Step {entry['step']}: next_random={entry['planner_next']:.10f}")
