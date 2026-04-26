"""Figure out HOW MANY random numbers langgraph consumes before node_a runs."""
import sys, random, asyncio
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, END

# We know:
# seed(42) -> random() = 0.6394267984578837
# but ainvoke -> node_a gets 0.22321073814882275
# So langgraph consumes N random numbers before node_a runs

# Let's figure out N by checking random.getstate() before and after
# We can't easily intercept, but we can use a SEPARATE random generator

import random as _r

# The idea: use a separate RNG to avoid contaminating the main one
# Actually, let's just count: call random.random() N times until we get 0.2232...

random.seed(42)
target = 0.22321073814882275

for i in range(1, 50):
    val = random.random()
    if abs(val - target) < 1e-10:
        print(f"Found! After {i} random.random() calls, we get {val}")
        print(f"LangGraph consumes {i} random numbers before running nodes")
        break
else:
    print(f"Not found in 50 steps. Target={target}")
    # Let's just print the sequence
    random.seed(42)
    for i in range(20):
        print(f"  {i}: {random.random()}")
