#!/usr/bin/env python3
"""Debug seeds type in production."""

import sys, os, asyncio
os.chdir('C:\\Users\\Administrator\\.openclaw\\workspace\\yogacara-agent')
sys.path.insert(0, 'src')

import yogacara_agent.yogacara_langgraph as yl

original_node_introspect = yl.node_introspect

async def patched_node_introspect(state):
    print(f"Step {state['step']}: seeds type = {type(state['seeds'])}")
    if state['seeds']:
        print(f"  First seed type = {type(state['seeds'][0])}")
        print(f"  First seed = {state['seeds'][0]}")
    return await original_node_introspect(state)

yl.node_introspect = patched_node_introspect

from yogacara_agent.yogacara_langgraph import main
asyncio.run(main())
