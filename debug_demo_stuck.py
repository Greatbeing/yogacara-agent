#!/usr/bin/env python3
"""Debug: use demo's is_stuck logic."""

import sys, os
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

from yogacara_agent.yogacara_test import ConsciousnessPlanner, ManasController, GridSimEnv, AlayaMemory, Seed
from yogacara_agent.introspection import IntrospectionLogger
import random

random.seed(42)

env = GridSimEnv()
planner = ConsciousnessPlanner()
manas = ManasController()
alaya = AlayaMemory()
introspection = IntrospectionLogger()

obs = env.reset()
recent_rewards = []
pos_history = []

# Demo's stuck tracking
_last_pos = None
_steps_stuck = 0

for step in range(60):
    pos_history.append(obs["pos"])
    seeds = alaya.retrieve(obs)
    
    # Demo's is_stuck logic
    is_stuck = _last_pos == obs["pos"] and _steps_stuck >= 2
    
    action, unc, scores = planner.plan(obs, seeds, env_resources=env.resources, is_stuck=is_stuck)
    final, passed, log = manas.filter(action, obs, unc, step, recent_rewards, pos_history)
    next_obs, rew, done = env.step(final)
    recent_rewards.append(rew)
    
    # Demo's stuck update logic
    if final != "STAY" and _last_pos != obs["pos"]:
        _steps_stuck = 0
    else:
        _steps_stuck += 1
    _last_pos = obs["pos"]
    
    seeds_data = [{"rew": s.reward, "action": s.action, "importance": s.importance} for s in seeds]
    record = introspection.observe(
        step=step, obs={**obs, "reward": rew}, action=final, unc=unc,
        seeds_retrieved=seeds_data, reasoning="test", alternatives=list(scores.keys()),
        manas_intercepted=not passed,
        score_best=max(scores.values()) if scores else 0.0,
        score_second=sorted(set(scores.values()), reverse=True)[1] if len(scores) > 1 else 0.0,
    )
    
    tag = "依他起" if unc < 0.5 else "遍计所执"
    seed = Seed(alaya._encode(obs), final, rew, 0, 0.8, 1.0 if passed else 0.4, unc, tag)
    alaya.add(seed)
    
    print(f"{step:4d} | {obs['pos']} | {final:6s} | {rew:+6.1f} | {unc:.2f} | {record.nature:10s} | stuck={is_stuck}")
    
    if done:
        break
    obs = next_obs

print(f"\nTotal steps: {step + 1}")
print(f"Resources found: {3 - len(env.resources)}/3")
