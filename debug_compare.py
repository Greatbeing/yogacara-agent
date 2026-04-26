#!/usr/bin/env python3
"""Debug: compare demo vs production step by step."""

import sys, os
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

from yogacara_agent.yogacara_test import ConsciousnessPlanner, ManasController, GridSimEnv, AlayaMemory
from yogacara_agent.introspection import IntrospectionLogger
from yogacara_agent.ego_monitor import EgoMonitor
from yogacara_agent.seed_classifier import SeedClassifier
import random

random.seed(42)

# Create shared instances (same as production)
env = GridSimEnv()
alaya = AlayaMemory()
manas = ManasController()
planner = ConsciousnessPlanner()
introspection = IntrospectionLogger()

obs = env.reset()
recent_rewards = []
pos_history = []

print("Step | Pos      | Action | Unc  | Nature     | Markers")
print("-" * 70)

for step in range(60):
    pos_history.append(obs["pos"])
    seeds = alaya.retrieve(obs)
    
    # Stuck detection (same as demo)
    is_stuck = len(pos_history) >= 3 and len(set(pos_history[-3:])) == 1
    
    # Plan
    action, unc, scores = planner.plan(obs, seeds, env_resources=env.resources, is_stuck=is_stuck)
    
    # Manas filter
    final, passed, log = manas.filter(action, obs, unc, step, recent_rewards, pos_history)
    
    # Execute
    next_obs, rew, done = env.step(final)
    recent_rewards.append(rew)
    
    # Introspection
    seeds_data = [{"rew": s.reward, "action": s.action, "importance": s.importance} for s in seeds]
    record = introspection.observe(
        step=step,
        obs={**obs, "reward": rew},
        action=final,
        unc=unc,
        seeds_retrieved=seeds_data,
        reasoning="test",
        alternatives=list(scores.keys()),
        manas_intercepted=not passed,
        score_best=max(scores.values()) if scores else 0.0,
        score_second=sorted(set(scores.values()), reverse=True)[1] if len(scores) > 1 else 0.0,
    )
    
    # Store seed
    tag = "依他起" if unc < 0.5 else "遍计所执"
    from yogacara_agent.yogacara_test import Seed
    seed = Seed(alaya._encode(obs), final, rew, 0, 0.8, 1.0 if passed else 0.4, unc, tag)
    alaya.add(seed)
    
    print(f"{step:4d} | {obs['pos']} | {final:6s} | {unc:.2f} | {record.nature:10s} | {record.ego_markers}")
    
    if done:
        print(f"\nDone at step {step}")
        break
    obs = next_obs

print(f"\nTotal steps: {step + 1}")
print(f"Nature distribution (last 20): {introspection.recent_summary(20)['nature_distribution']}")
print(f"Resources found: {3 - len(env.resources)}/3")
