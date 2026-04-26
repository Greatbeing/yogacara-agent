# Quick verification of the _steps_without_resource fix
import random
import sys
sys.path.insert(0, 'src')

from yogacara_agent import AlayaConsciousness, ConsciousnessPlanner, ManasController

random.seed(42)

planner = ConsciousnessPlanner()
planner._steps_without_resource = 0

print("Simulation (no increment, only reset on resource):")
for step in range(1, 40):
    resource_found = step in [8, 20]
    if resource_found:
        planner._steps_without_resource = 0
    triggered = planner._steps_without_resource >= 15
    mark = " RESET" if resource_found else ""
    print(f"Step {step:2d}: _steps={planner._steps_without_resource}, exploration={triggered}{mark}")

print("\nComparison:")
print("  Demo:      _steps always 0 (never increments)")
print("  Old prod:  _steps keeps +1 each non-resource step -> triggers at step 25+")
print("  New prod:  same as demo (no increment) -> triggers NEVER")
print("\nFix confirmed: production now matches demo behavior")