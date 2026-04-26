"""Debug trajectory comparison: demo vs production."""
import sys
import asyncio
import random

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Demo trajectory ──
print("=== DEMO ===")
random.seed(42)
from yogacara_agent.yogacara_test import GridSimEnv as DemoEnv, AlayaMemory as DemoAlaya
from yogacara_agent.yogacara_test import ConsciousnessPlanner as DemoPlanner, ManasController as DemoManas
from yogacara_agent.yogacara_test import ACTIONS, ACTION_TO_IDX

demo_env = DemoEnv()
demo_alaya = DemoAlaya()
demo_manas = DemoManas()
demo_planner = DemoPlanner()
obs = demo_env.reset()
demo_traj = []
for step in range(60):
    seeds = demo_alaya.retrieve(obs)
    is_stuck = False  # simplified
    action, unc, scores = demo_planner.plan(obs, seeds, env_resources=demo_env.resources, is_stuck=is_stuck)
    final, passed, log = demo_manas.filter(action, obs, unc, step, [], [])
    next_obs, rew, done = demo_env.step(final)
    demo_traj.append((step, obs["pos"], final, rew, unc, passed))
    obs = next_obs
    if done:
        break

for s, pos, act, rew, unc, passed in demo_traj:
    flag = "" if passed else "[INTERCEPTED]"
    print(f"  Step {s:2d} | Pos:{pos} | Act:{act:5s} | R:{rew:+.1f} | Unc:{unc:.3f} {flag}")

# ── Production trajectory ──
print("\n=== PRODUCTION ===")
# Reset everything
import yogacara_agent.yogacara_langgraph as yl
yl.env = yl.GridSimEnv()
yl.alaya = yl.AlayaMemory()
yl.manas = yl.ManasController()
yl.planner = DemoPlanner()  # Use shared planner
# Reset introspection
from yogacara_agent.introspection import IntrospectionLogger
from yogacara_agent.ego_monitor import EgoMonitor
from yogacara_agent.seed_classifier import SeedClassifier
yl.introspection_logger = IntrospectionLogger()
yl.ego_monitor = EgoMonitor()
yl.seed_classifier = SeedClassifier()
yl._seed_counts = {"名言种": 0, "业种": 0, "异熟种": 0}
yl._parinispanna_count = 0
yl._total_classified = 0

random.seed(42)
graph = yl.build_graph()
init_state = {
    "obs": yl.env.reset(), "action": "", "reward": 0.0, "done": False,
    "step": 0, "seeds": [], "unc": 0.0, "manas_passed": True,
    "tool_calls": [], "recent_rewards": [], "pos_history": [],
    "metrics": {}, "introspection_record": None, "ego_alert": None,
    "plan_scores": None, "reasoning": "", "steps_since_resource": 0,
}

prod_traj = []

# Monkey-patch node_plan and node_manas to log
_orig_plan = yl.node_plan
async def _log_plan(state):
    result = await _orig_plan(state)
    prod_traj.append((state["step"], state["obs"]["pos"], result["action"], state.get("unc", 0), True))
    return result
yl.node_plan = _log_plan

_orig_manas = yl.node_manas
async def _log_manas(state):
    result = await _orig_manas(state)
    if not result["manas_passed"]:
        # Update last entry
        if prod_traj:
            s, p, a, u, _ = prod_traj[-1]
            prod_traj[-1] = (s, p, result["action"], u, False)
    return result
yl.node_manas = _log_manas

graph = yl.build_graph()

async def run():
    final = await graph.ainvoke(init_state)
    return final

final = asyncio.run(run())

for s, pos, act, unc, passed in prod_traj:
    flag = "" if passed else "[INTERCEPTED]"
    print(f"  Step {s:2d} | Pos:{pos} | Act:{act:5s} | Unc:{unc:.3f} {flag}")

# ── Diff ──
print("\n=== DIVERGENCE POINT ===")
min_len = min(len(demo_traj), len(prod_traj))
for i in range(min_len):
    ds, dpos, dact, drew, dunc, dpassed = demo_traj[i]
    ps, ppos, pact, punc, ppassed = prod_traj[i]
    if dact != pact or dpos != ppos:
        print(f"  FIRST DIVERGENCE at step {i}:")
        print(f"    Demo: pos={dpos} act={dact} unc={dunc:.3f}")
        print(f"    Prod: pos={ppos} act={pact} unc={punc:.3f}")
        break
else:
    if len(demo_traj) != len(prod_traj):
        print(f"  Different lengths: demo={len(demo_traj)}, prod={len(prod_traj)}")
    else:
        print("  IDENTICAL trajectories!")
