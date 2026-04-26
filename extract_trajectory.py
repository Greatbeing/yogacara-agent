"""提取 demo 和 production 的轨迹对比."""
import subprocess
import sys

# Demo
demo_result = subprocess.run(
    [sys.executable, "run_demo.py", "--episodes", "1", "--max-steps", "60"],
    cwd=r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=30,
)

demo_actions = []
for line in demo_result.stdout.split("\n"):
    if "| Act:" in line:
        # 提取动作
        parts = line.split("| Act:")
        if len(parts) > 1:
            action = parts[1].split("|")[0].strip()
            demo_actions.append(action)

# Production
prod_result = subprocess.run(
    [sys.executable, "run_langgraph_verbose.py"],
    cwd=r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=30,
)

prod_actions = []
for line in prod_result.stdout.split("\n"):
    if "action=" in line:
        parts = line.split("action=")
        if len(parts) > 1:
            action = parts[1].split()[0]
            prod_actions.append(action)

print(f"Demo actions ({len(demo_actions)} steps): {demo_actions[:20]}")
print(f"\nProd actions ({len(prod_actions)} steps): {prod_actions[:20]}")

print("\n=== Step-by-step comparison ===")
for i in range(min(len(demo_actions), len(prod_actions))):
    d = demo_actions[i]
    p = prod_actions[i]
    marker = "  " if d == p else "!!"
    print(f"{marker} Step {i}: Demo={d:6s} Prod={p:6s}")
    if d != p and i < 10:
        break
