"""直接运行 run_demo.py."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "run_demo.py", "--episodes", "1", "--max-steps", "50"],
    cwd=r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=30,
)

print("STDOUT:")
print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
print("\nSTDERR:")
print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
print(f"\nReturn code: {result.returncode}")
