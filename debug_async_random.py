"""Minimal reproduction: does asyncio or langgraph consume random numbers?"""
import sys, random, asyncio
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Step 1: Seed and check state
random.seed(42)
state_0 = random.getstate()

# Step 2: Create simple async function
async def dummy():
    await asyncio.sleep(0)
    return random.random()

# Step 3: Run
random.seed(42)
r1 = asyncio.run(dummy())
print(f"After async sleep(0): {r1}")

random.seed(42)
r2 = random.random()
print(f"Direct random:       {r2}")

print(f"Same? {r1 == r2}")

# Now check with langgraph
random.seed(42)
from langgraph.graph import StateGraph, END

class SimpleState(dict):
    pass

async def node_a(state):
    state["r"] = random.random()
    return state

wf = StateGraph(dict)
wf.add_node("a", node_a)
wf.set_entry_point("a")
wf.add_edge("a", END)
graph = wf.compile()

random.seed(42)
result = asyncio.run(graph.ainvoke({"r": 0}))
print(f"\nLangGraph node random: {result['r']}")

random.seed(42)
expected = random.random()
print(f"Expected:             {expected}")
print(f"Same? {result['r'] == expected}")
