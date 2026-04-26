"""提取 Demo Step 8 的详细信息."""
import subprocess
import sys

demo_result = subprocess.run(
    [sys.executable, "run_demo.py", "--episodes", "1", "--max-steps", "12"],
    cwd=r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=30,
)

print("=== Demo Steps 0-11 ===")
for i, line in enumerate(demo_result.stdout.split("\n")):
    if "Step" in line and "Act:" in line:
        print(line)
