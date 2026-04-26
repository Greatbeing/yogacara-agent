"""Direct comparison: step 0 with identical setup."""
import sys, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# === Test 1: Demo-style ===
print("=== Demo-style (direct import) ===")
random.seed(42)
from yogacara_agent.yogacara_test import GridSimEnv, AlayaMemory, ConsciousnessPlanner, ACTIONS, ACTION_TO_IDX
env1 = GridSimEnv()
alaya1 = AlayaMemory()
planner1 = ConsciousnessPlanner()
obs1 = env1.reset()
seeds1 = alaya1.retrieve(obs1)
best1, unc1, scores1 = planner1.plan(obs1, seeds1, env_resources=env1.resources, is_stuck=False)
print(f"  Best: {best1}, Unc: {unc1:.3f}")
print(f"  Scores: { {a: f'{scores1[a]:.6f}' for a in ACTIONS} }")

# === Test 2: Production-style (through langgraph import) ===
print("\n=== Production-style (langgraph import) ===")
random.seed(42)
import yogacara_agent.yogacara_langgraph as yl
# Reset state
yl.env = yl.GridSimEnv()
yl.alaya = yl.AlayaMemory()
yl.planner = ConsciousnessPlanner()
random.seed(42)  # Re-seed to match demo

obs2 = yl.env.reset()
seeds2 = yl.alaya.retrieve(obs2)
best2, unc2, scores2 = yl.planner.plan(obs2, seeds2, env_resources=yl.env.resources, is_stuck=False)
print(f"  Best: {best2}, Unc: {unc2:.3f}")
print(f"  Scores: { {a: f'{scores2[a]:.6f}' for a in ACTIONS} }")

# === Compare ===
print(f"\n=== Comparison ===")
print(f"  Same best? {best1 == best2}")
print(f"  Same scores? {all(abs(scores1[a] - scores2[a]) < 1e-10 for a in ACTIONS)}")
for a in ACTIONS:
    diff = scores1[a] - scores2[a]
    if abs(diff) > 1e-10:
        print(f"  DIFF: {a}: demo={scores1[a]:.6f} prod={scores2[a]:.6f} diff={diff:.10f}")
