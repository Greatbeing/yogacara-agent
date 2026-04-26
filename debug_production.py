#!/usr/bin/env python3
"""Debug production version with detailed output."""

import sys, os, asyncio
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

from yogacara_agent.yogacara_langgraph import main

# Monkey-patch to add step-by-step output
import yogacara_agent.yogacara_langgraph as yl

original_node_execute = yl.node_execute

async def patched_node_execute(state):
    result = await original_node_execute(state)
    print(f"  Step {state['step']:2d} | pos={state['obs']['pos']} | action={state['action']:5s} | reward={state['reward']:+.1f} | unc={state['unc']:.2f}")
    return result

yl.node_execute = patched_node_execute

asyncio.run(main())
