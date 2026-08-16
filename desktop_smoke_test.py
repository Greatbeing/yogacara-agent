"""桌面版真实窗口冒烟测试：启动 4 秒后自动关闭，验证 WebView2/js_api/图标/干净退出。"""

import os
import sys
import threading

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)

import webview

from desktop.agent_bridge import AgentBridge

bridge = AgentBridge(max_steps=30, speed_ms=100)
HTML_PATH = os.path.join(_ROOT, "desktop", "index.html")
ICON_PATH = os.path.join(_ROOT, "assets", "yogacara.ico")

window = webview.create_window(
    title="冒烟测试 · 唯识进化框架",
    url=f"file:///{HTML_PATH.replace(os.sep, '/')}",
    js_api=bridge,
    width=1180,
    height=820,
    min_size=(980, 700),
    text_select=False,
    background_color="#0f1420",
)


def worker():
    """窗口内让 Agent 跑几步，验证桥接在窗口线程外工作。"""
    import time

    time.sleep(1.0)
    for _ in range(8):
        if bridge.state["done"]:
            break
        bridge.step_once()
        time.sleep(0.1)
    snap = bridge.get_snapshot()
    print(f"[冒烟] 窗口运行中 step={snap['step']} seeds={snap['seeds_total']} "
          f"turning={snap['turning'].get('turning_level')} planner={snap['planner_source']}")


def closer():
    import time

    time.sleep(4.0)
    print("[冒烟] 定时关闭窗口")
    window.destroy()


threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=closer, daemon=True).start()

icon_arg = {"icon": ICON_PATH} if os.path.isfile(ICON_PATH) else {}
webview.start(debug=False, **icon_arg)
bridge.stop()
snap = bridge.get_snapshot()
print(f"[冒烟] 退出成功 | step={snap['step']} seeds_total={snap['seeds_total']} | 窗口生命周期 OK")
