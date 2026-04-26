"""Verify production LangGraph nodes vs demo share identical planner/manas (after RNG fix)."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Production LangGraph nodes import directly from yogacara_test
from yogacara_agent.yogacara_test import (
    GridSimEnv as DemoEnv, AlayaMemory as DemoMem,
    ManasController as DemoManas, ConsciousnessPlanner as DemoPlanner,
    _planner_rng, _manas_rng,
)
from yogacara_agent.yogacara_langgraph import (
    GridSimEnv as ProdEnv, AlayaMemory as ProdMem,
    ManasController as ProdManas,
    planner as prod_planner,  # imported as _SharedPlanner
)

STEPS = 60

def run_via_demo_api():
    """Standalone: use yogacara_test.py classes directly (same as run_demo.py)."""
    env = DemoEnv()
    mem = DemoMem()
    manas = DemoManas()
    planner = DemoPlanner()
    traj = []
    obs = env._observe()
    for _ in range(STEPS):
        seeds = mem.retrieve(obs)
        action, _, _ = planner.plan(obs, seeds, env.resources)
        action2, _, _ = manas.filter(action, obs, 0.5, env.step_count, [], [])
        obs, _, done = env.step(action2)
        traj.append(action2)
        if done:
            break
    return traj

def run_via_langgraph_nodes():
    """Production path: use same classes imported from yogacara_langgraph."""
    env = ProdEnv()
    mem = ProdMem()
    manas = ProdManas()
    planner = prod_planner  # same _SharedPlanner instance
    traj = []
    obs = env._observe()
    for _ in range(STEPS):
        seeds = mem.retrieve(obs)
        action, _, _ = planner.plan(obs, seeds, env.resources)
        action2, _, _ = manas.filter(action, obs, 0.5, env.step_count, [], [])
        obs, _, done = env.step(action2)
        traj.append(action2)
        if done:
            break
    return traj

# Same seed → same isolated RNG → should be identical
traj_demo = run_via_demo_api()
traj_lg   = run_via_langgraph_nodes()

print(f"Demo API:      {len(traj_demo)} steps")
print(f"LangGraph path: {len(traj_lg)} steps")
print(f"Match: {traj_demo == traj_lg}")

if traj_demo == traj_lg:
    print("\nRandom isolation WORKING — both paths use identical planner/manas RNG.")
    # Quick four-wisdoms sanity check
    from yogacara_agent.yogacara_test import YogacaraAgent
    agent = YogacaraAgent()
    for _ in range(STEPS):
        obs = agent.env._observe()
        seeds = agent.alaya.retrieve(obs)
        action, _, _ = agent.planner.plan(obs, seeds, agent.env.resources)
        action2, _, _ = agent.manas.filter(action, obs, 0.5, agent.env.step_count, [], [])
        obs, _, done = agent.env.step(action2)
        if done:
            break
    print(f"\nQuick run: {agent.env.step_count} steps, reward={agent.metrics['reward']:.2f}")
else:
    for i, (a, b) in enumerate(zip(traj_demo, traj_lg)):
        if a != b:
            print(f"  First diff @ step {i}: demo={a}, langgraph={b}")
            break
