#!/usr/bin/env python3
"""Debug planner decisions step by step."""

import sys, os
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

from yogacara_agent.yogacara_test import ConsciousnessPlanner, ManasController, GridSimEnv, AlayaMemory, Seed, ACTIONS, ACTION_TO_IDX
import random

random.seed(42)

env = GridSimEnv()
planner = ConsciousnessPlanner()
manas = ManasController()
alaya = AlayaMemory()

obs = env.reset()
recent_rewards = []
pos_history = []

for step in range(20):
    pos_history.append(obs["pos"])
    seeds = alaya.retrieve(obs)
    
    # Stuck detection (same as demo)
    is_stuck = len(pos_history) >= 3 and len(set(pos_history[-3:])) == 1
    
    # Plan with debug output
    action, unc, scores = planner.plan(obs, seeds, env_resources=env.resources, is_stuck=is_stuck)
    
    print(f"\n=== Step {step} | pos={obs['pos']} | is_stuck={is_stuck} ===")
    print(f"Seeds in memory: {len(alaya.seeds)}")
    for i, s in enumerate(seeds):
        print(f"  Seed {i}: pos=({s.state_emb[0]*10:.0f},{s.state_emb[1]*10:.0f}) action={s.action} rew={s.reward:.1f} imp={s.importance:.2f}")
    
    # Calculate base_scores manually
    view = obs["grid_view"]
    pos = obs["pos"]
    dist_bonus = 0.0
    best_dir_r = best_dir_c = None
    if not any(v == 1.0 for v in view) and env.resources:
        nearest = min(env.resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))
        best_dir_r = "DOWN" if nearest[0] > pos[0] else "UP" if nearest[0] < pos[0] else "STAY"
        best_dir_c = "RIGHT" if nearest[1] > pos[1] else "LEFT" if nearest[1] < pos[1] else "STAY"
        dist_bonus = 0.4
    
    print(f"Nearest resource: {nearest if env.resources else None}")
    print(f"best_dir_r={best_dir_r}, best_dir_c={best_dir_c}, dist_bonus={dist_bonus}")
    
    base_scores = {}
    for a in ACTIONS:
        idx = ACTION_TO_IDX[a]
        base = view[idx] if 0 <= idx < 9 else -0.5
        pos_b = sum(s.reward * s.importance for s in seeds if s.action == a and s.reward > 0) * 0.8
        neg_p = sum(abs(s.reward) * s.importance for s in seeds if s.action == a and s.reward < 0) * 0.5
        approach = dist_bonus if best_dir_r and a in (best_dir_r, best_dir_c) else 0.0
        base_scores[a] = base + pos_b - neg_p + approach
    
    print("Base scores (no random):")
    for a in sorted(ACTIONS, key=lambda x: base_scores[x], reverse=True):
        print(f"  {a}: {base_scores[a]:+.3f}")
    
    best_base = max(base_scores, key=base_scores.get)
    unc_base = max(0.0, min(1.0, 1.0 - (base_scores[best_base] - min(base_scores.values())) / 2.0))
    print(f"best_base={best_base}, unc_base={unc_base:.2f}")
    
    # Bias
    exploration_force = planner._steps_without_resource >= 15
    print(f"exploration_force={exploration_force}")
    
    biases = {}
    for a in ACTIONS:
        has_bonus = best_dir_r and a in (best_dir_r, best_dir_c)
        if is_stuck:
            bias = -0.8 if a == "STAY" else 0.35
        elif exploration_force:
            bias = -0.8 if a == "STAY" else 0.2
        elif unc_base >= 0.5 and not has_bonus:
            bias = 0.30 if a == "STAY" else -0.35
        elif unc_base < 0.3:
            bias = -0.20 if a == "STAY" else 0.15
        else:
            bias = 0.0
        biases[a] = bias
    
    print("Biases:")
    for a in ACTIONS:
        print(f"  {a}: {biases[a]:+.2f}")
    
    final_scores = {a: base_scores[a] + biases[a] for a in ACTIONS}
    print("Final scores (no random):")
    for a in sorted(ACTIONS, key=lambda x: final_scores[x], reverse=True):
        print(f"  {a}: {final_scores[a]:+.3f} (base={base_scores[a]:+.3f} + bias={biases[a]:+.2f})")
    
    print(f"Planner chose: {action} (unc={unc:.2f})")
    
    # Execute
    final, passed, log = manas.filter(action, obs, unc, step, recent_rewards, pos_history)
    next_obs, rew, done = env.step(final)
    recent_rewards.append(rew)
    
    # Store seed
    tag = "依他起" if unc < 0.5 else "遍计所执"
    seed = Seed(alaya._encode(obs), final, rew, 0, 0.8, 1.0 if passed else 0.4, unc, tag)
    alaya.add(seed)
    
    if done:
        break
    obs = next_obs
