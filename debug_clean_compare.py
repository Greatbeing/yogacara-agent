"""Clean test: production vs demo with no debug interference."""
import sys, asyncio, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Demo
random.seed(42)
from yogacara_agent.yogacara_test import GridSimEnv, AlayaMemory, ConsciousnessPlanner, ManasController, ACTIONS, Seed
import time as _time

demo_env = GridSimEnv()
demo_alaya = AlayaMemory()
demo_manas = DemoManas = ManasController()
demo_planner = ConsciousnessPlanner()
demo_obs = demo_env.reset()
demo_positions = []

for step in range(60):
    seeds = demo_alaya.retrieve(demo_obs)
    is_stuck = False
    action, unc, scores = demo_planner.plan(demo_obs, seeds, env_resources=demo_env.resources, is_stuck=is_stuck)
    final, passed, log = demo_manas.filter(action, demo_obs, unc, step, [], [])
    next_obs, rew, done = demo_env.step(final)
    demo_positions.append((step, demo_obs["pos"], final, rew, unc))
    demo_alaya.add(Seed(demo_alaya._encode(demo_obs), final, rew, _time.time(), 0.8, 1.0 if passed else 0.4, unc, "依他起" if unc < 0.5 else "遍计所执"))
    demo_obs = next_obs
    if done:
        break

# Production - clean run
random.seed(42)
import yogacara_agent.yogacara_langgraph as yl
yl.env = yl.GridSimEnv()
yl.alaya = yl.AlayaMemory()
yl.manas = yl.ManasController()
yl.planner = ConsciousnessPlanner()
yl.introspection_logger = None
yl.ego_monitor = None
yl.seed_classifier = None
yl._seed_counts = {"名言种": 0, "业种": 0, "异熟种": 0}
yl._parinispanna_count = 0
yl._total_classified = 0

# Patch node_plan to track positions
_orig_plan = yl.node_plan
prod_positions = []
async def _track_plan(state):
    result = await _orig_plan(state)
    prod_positions.append((state["step"], state["obs"]["pos"], result["action"], 0, result["unc"]))
    return result
yl.node_plan = _track_plan

# Patch node_execute to add reward
_orig_exec = yl.node_execute
async def _track_exec(state):
    result = await _orig_exec(state)
    if prod_positions:
        s, p, a, _, u = prod_positions[-1]
        prod_positions[-1] = (s, p, a, result["reward"], u)
    return result
yl.node_execute = _track_exec

graph = yl.build_graph()
init_state = {
    "obs": yl.env.reset(), "action": "", "reward": 0.0, "done": False,
    "step": 0, "seeds": [], "unc": 0.0, "manas_passed": True,
    "tool_calls": [], "recent_rewards": [], "pos_history": [],
    "metrics": {}, "introspection_record": None, "ego_alert": None,
    "plan_scores": None, "reasoning": "", "steps_since_resource": 0,
}

# CRITICAL: Re-seed right before ainvoke, same as demo
random.seed(42)

async def run():
    final = await graph.ainvoke(init_state)
    return final

final = asyncio.run(run())

# Compare
print("Step | Demo Pos    | Demo Act | Prod Pos    | Prod Act | Match")
print("-" * 70)
min_len = min(len(demo_positions), len(prod_positions))
for i in range(min_len):
    ds, dp, da, dr, du = demo_positions[i]
    ps, pp, pa, pr, pu = prod_positions[i]
    match = "OK" if da == pa else "DIFF"
    if da != pa:
        print(f" {ds:4d} | {str(dp):10s} | {da:8s} | {str(pp):10s} | {pa:8s} | {match}")
        # Only show first few differences
        if sum(1 for j in range(i+1) if demo_positions[j][2] != prod_positions[j][2]) >= 5:
            print("  ... (more differences)")
            break

same_count = sum(1 for i in range(min_len) if demo_positions[i][2] == prod_positions[i][2])
print(f"\nMatching actions: {same_count}/{min_len}")
print(f"Demo steps: {len(demo_positions)}, Prod steps: {len(prod_positions)}")
