"""对比 demo vs production 运行轨迹."""
import subprocess
import sys

print("=== Running demo (run_demo.py) ===")
demo_result = subprocess.run(
    [sys.executable, "run_demo.py", "--episodes", "1", "--max-steps", "50"],
    cwd=r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=30,
)

# 提取关键信息
demo_lines = demo_result.stdout.split("\n")
for line in demo_lines:
    if "Step" in line and ("R:" in line or "resources" in line.lower()):
        print(line)

print("\n=== Running production (yogacara_langgraph.py) ===")
prod_result = subprocess.run(
    [sys.executable, "-c", "from yogacara_agent.yogacara_langgraph import main; import asyncio; asyncio.run(main())"],
    cwd=r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=30,
)

# 提取关键信息
prod_lines = prod_result.stdout.split("\n")
for line in prod_lines:
    if "Step" in line or "步数" in line or "奖励" in line:
        print(line)

print("\n=== Comparison ===")
print(f"Demo return code: {demo_result.returncode}")
print(f"Prod return code: {prod_result.returncode}")
