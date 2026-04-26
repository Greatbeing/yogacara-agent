#!/usr/bin/env python3
"""Compare demo vs production step by step."""

import sys, os
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

from yogacara_agent.yogacara_test import ConsciousnessPlanner, ManasController, GridSimEnv, AlayaMemory
from yogacara_agent.introspection import IntrospectionLogger
from yogacara_agent.ego_monitor import EgoMonitor
from yogacara_agent.seed_classifier import SeedClassifier

# Run demo-style loop
random_seed = 42

# Demo version
print("=== DEMO VERSION ===")
import random
random.seed(random_seed)

env = GridSimEnv()
planner = ConsciousnessPlanner()
manas = ManasController()
alaya = AlayaMemory()
introspection = IntrospectionLogger()

obs = env.reset()
for step in range(60):
    seeds = alaya.retrieve(obs)
    is_stuck = False  # simplified
    action, unc, scores = planner.plan(obs, seeds, env_resources=env.resources, is_stuck=is_stuck)
    final, passed, log = manas.filter(action, obs, unc, step, [], [])
    next_obs, rew, done = env.step(final)
    
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
    
    print(f"Step {step}: pos={obs['pos']} action={final} unc={unc:.2f} nature={record.nature} markers={record.ego_markers}")
    
    if done:
        print(f"Done at step {step}")
        break
    obs = next_obs

print(f"\nTotal steps: {step + 1}")
print(f"Nature distribution (last 20): {introspection.recent_summary(20)['nature_distribution']}")
