"""Track all random consumption in both paths to find where they diverge."""
import sys, random as _sys_random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from yogacara_agent import yogacara_test as yt
from yogacara_agent.yogacara_test import _planner_rng, _manas_rng

_planner_rng.seed(42)
_manas_rng.seed(42)

STEPS = 5

# Hook: monkey-patch _rnd_uniform and _rnd_choice to log calls
_log = []

_orig_uniform = yt._rnd_uniform
_orig_choice  = yt._rnd_choice

def logged_uniform(a, b):
    result = _orig_uniform(a, b)
    _log.append(("planner", "uniform", f"{a},{b}", result))
    return result

def logged_choice(seq):
    result = _orig_choice(seq)
    _log.append(("manas", "choice", str(seq), result))
    return result

yt._rnd_uniform = logged_uniform
yt._rnd_uniform.__name__ = "_rnd_uniform"
yt._rnd_choice = logged_choice
yt._rnd_choice.__name__ = "_rnd_choice"

# Also track global random consumption
_orig_global = _sys_random.random
_global_count = []
def logged_global():
    result = _orig_global()
    _global_count.append(result)
    return result
_sys_random.random = logged_global

agent = yt.YogacaraAgent()
for step in range(STEPS):
    obs = agent.env._observe()
    seeds = agent.alaya.retrieve(obs)
    action, unc, scores = agent.planner.plan(obs, seeds, agent.env.resources)
    action2, _, note = agent.manas.filter(action, obs, unc, agent.env.step_count, [], [])
    _log.append(("env", "step", action2, None))
    obs2, _, done = agent.env.step(action2)
    print(f"Step {step}: action={action}→{action2}  "
          f"planner_rng_next={_planner_rng.random():.10f}  "
          f"manas_rng_next={_manas_rng.random():.10f}  "
          f"global_random={_global_count[-1]:.10f}" if _global_count else "  global=init")
    if done:
        break

print()
print("=== All random calls ===")
for entry in _log:
    print(f"  {entry}")

print(f"\nGlobal random() called {len(_global_count)} times during execution")
print(f"First 5 global randoms: {_global_count[:5]}")
