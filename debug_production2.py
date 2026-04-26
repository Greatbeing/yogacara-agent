#!/usr/bin/env python3
"""Debug production version with detailed output including final action."""

import sys, os, asyncio
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

from yogacara_agent.yogacara_langgraph import main

# Monkey-patch to add step-by-step output
import yogacara_agent.yogacara_langgraph as yl

original_node_manas = yl.node_manas

async def patched_node_manas(state):
    result = await original_node_manas(state)
    print(f"  Step {state['step']:2d} | pos={state['obs']['pos']} | plan={state['action']:5s} | final={result['action']:5s} | reward={state.get('reward', 0):+.1f} | unc={state['unc']:.2f} | passed={result.get('manas_passed', True)}")
    return result

yl.node_manas = patched_node_manas

asyncio.run(main())
