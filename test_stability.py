#!/usr/bin/env python3
"""Multi-run stability test using subprocess."""
import subprocess, sys

for i in range(3):
    result = subprocess.run(
        [sys.executable, '-c', """
import sys, os
os.chdir('C:\\\\Users\\\\Administrator\\\\.openclaw\\\\workspace\\\\yogacara-agent')
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from yogacara_agent.yogacara_langgraph import main
import asyncio
asyncio.run(main())
"""],
        capture_output=True, text=False, encoding='utf-8', errors='replace'
    )
    output = result.stdout
    for line in output.split('\n'):
        if any(k in line for k in ['步数', '圆成实', '大圆镜', '平等性', '妙观察', '成所']):
            print(f'Run {i+1}: {line}')
    print()