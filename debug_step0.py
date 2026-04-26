"""Debug step 0: why different actions?"""
import sys, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Simulate step 0 with same seed
random.seed(42)
from yogacara_agent.yogacara_test import ACTIONS, ACTION_TO_IDX, ConsciousnessPlanner
from yogacara_agent.yogacara_test import GridSimEnv

env = GridSimEnv()
obs = env.reset()
planner = ConsciousnessPlanner()
seeds = []  # no seeds at step 0

# Call planner.plan()
best, unc, scores = planner.plan(obs, seeds, env_resources=env.resources, is_stuck=False)
print(f"Best: {best}, Unc: {unc:.3f}")
print("Scores:")
for a in ACTIONS:
    print(f"  {a}: {scores[a]:.6f}")

# Now check which action max() would pick
max_score = max(scores.values())
tied = [a for a, v in scores.items() if abs(v - max_score) < 0.0001]
print(f"\nMax score: {max_score:.6f}")
print(f"Tied actions: {tied}")
print(f"Dict key order: {list(scores.keys())}")

# The max() picks the FIRST key with max value in dict iteration order
first_max = max(scores, key=scores.get)
print(f"max() picks: {first_max}")
