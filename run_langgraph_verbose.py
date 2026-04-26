"""直接运行 production 版本并打印详细输出."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import asyncio
from yogacara_agent.yogacara_langgraph import main, env, planner, manas

# 添加详细日志
original_step = env.step
def verbose_step(action):
    obs, rew, done = original_step(action)
    print(f"  Step {env.step_count}: action={action} pos={obs['pos']} rew={rew} resources_left={len(env.resources)}")
    return obs, rew, done

env.step = verbose_step

asyncio.run(main())
