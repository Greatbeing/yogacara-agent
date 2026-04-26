"""运行 run_demo.py YogacaraAgent."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

# 导入 run_demo.py 中的 YogacaraAgent
from run_demo import YogacaraAgent, _planner_rng, _manas_rng

_planner_rng.seed(42)
_manas_rng.seed(42)

agent = YogacaraAgent()
# 只运行不显示网格
obs = agent.env.reset()
print(f"Initial resources: {agent.env.resources}")

for step in range(50):
    agent.pos_history.append(obs["pos"])
    seeds = agent.alaya.retrieve(obs)
    is_stuck = (agent._last_pos == obs["pos"] and agent._steps_stuck >= 2)
    action, unc, scores = agent.planner.plan(obs, seeds, env_resources=agent.env.resources, is_stuck=is_stuck)
    final, passed, log = agent.manas.filter(action, obs, unc, step, agent.recent_rewards, agent.pos_history)
    next_obs, rew, done = agent.env.step(final)
    agent.recent_rewards.append(rew)
    agent.metrics["steps"] += 1
    agent.metrics["reward"] += rew
    
    if rew > 2.0:
        agent.metrics["resources_found"] += 1
        agent._steps_stuck = 0
        print(f"Step {step}: Found resource! pos={obs['pos']} rew={rew}")
    
    if final != "STAY" and agent._last_pos != obs["pos"]:
        agent._steps_stuck = 0
    else:
        agent._steps_stuck += 1
    agent._last_pos = obs["pos"]
    
    seed = type('Seed', (), {})()  # dummy
    agent.alaya.add({"emb": agent.alaya._encode(obs), "action": final, "reward": rew, "importance": 0.8})
    
    obs = next_obs
    if done:
        print(f"Done at step {step}")
        break

print(f"\nTotal steps: {agent.metrics['steps']}")
print(f"Resources found: {agent.metrics['resources_found']}")
print(f"Total reward: {agent.metrics['reward']:.2f}")
