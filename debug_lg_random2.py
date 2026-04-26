"""Narrow down: does compile() or ainvoke() consume random?"""
import sys, random, asyncio
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, END

async def node_a(state):
    state["r"] = random.random()
    return state

wf = StateGraph(dict)
wf.add_node("a", node_a)
wf.set_entry_point("a")
wf.add_edge("a", END)

# Test 1: Does compile() consume random?
random.seed(42)
r_before_compile = random.random()  # consume one
random.seed(42)
graph = wf.compile()
r_after_compile = random.random()
print(f"Before compile consumed: {r_before_compile}")
print(f"After compile, next random: {r_after_compile}")
print(f"Compile consumed random? {r_before_compile != r_after_compile}")
print()

# Test 2: Does ainvoke() consume random?
random.seed(42)
result1 = asyncio.run(graph.ainvoke({"r": 0}))
print(f"ainvoke result random: {result1['r']}")

random.seed(42)
# Check what random we EXPECT to get (without langgraph)
expected = random.random()
print(f"Expected (no langgraph): {expected}")
print(f"LangGraph ainvoke consumed random? {result1['r'] != expected}")
print()

# Test 3: How many random numbers does ainvoke consume before node_a runs?
# We can't easily intercept, but we can check the difference
random.seed(42)
# Run once to "use up" whatever langgraph consumes + node_a's random
asyncio.run(graph.ainvoke({"r": 0}))
# Now random state is whatever results after 1 invocation
r_after_one = random.random()  # get current state
random.seed(42)
# Now run step by step manually - invoke twice and see the delta
result_a = asyncio.run(graph.ainvoke({"r": 0}))
random.seed(42)
result_b = asyncio.run(graph.ainvoke({"r": 0}))
print(f"Two ainvoke calls, same seed:")
print(f"  Call 1 result: {result_a['r']}")
print(f"  Call 2 result: {result_b['r']}")
print(f"  Same? {result_a['r'] == result_b['r']}")
print()
print("If same, langgraph resets something. If different, langgraph consumes random internally.")
