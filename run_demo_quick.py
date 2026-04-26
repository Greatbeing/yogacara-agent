import subprocess, sys, os
os.chdir(r"C:\Users\Administrator\.openclaw\workspace\yogacara-agent")
result = subprocess.run([sys.executable, "run_demo.py", "-n", "1", "-s", "60", "--seed", "42"], capture_output=True, text=True, encoding="utf-8")
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)