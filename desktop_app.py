"""
唯识进化框架 · Windows 桌面版
==============================
基于 pywebview 的原生窗口桌面应用：
网格世界可视化 + 运行控制 + 四智/记忆仪表盘 + 内省日志流。

启动方式:
    cd yogacara-agent
    python desktop_app.py

依赖:
    pip install pywebview   (Windows 使用 WebView2/EdgeChromium)
"""

import logging
import os
import sys

# 使 src/ 包可导入（无论从哪个 cwd 启动）
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)  # 记忆文件 memory/seeds.jsonl 相对仓库根

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
logging.getLogger("yogacara_agent").setLevel(logging.INFO)

# .env 的 LLM_API_KEY / YOGACARA_LLM_PLAN 等对混合规划器生效
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HTML_PATH = os.path.join(_ROOT, "desktop", "index.html")
ICON_PATH = os.path.join(_ROOT, "assets", "yogacara.ico")


def main() -> int:
    try:
        import webview
    except ImportError:
        print("❌ 缺少依赖 pywebview，请先安装：")
        print("   pip install pywebview")
        print("   （Windows 需系统自带 WebView2 运行时，Win10/11 一般已内置）")
        return 1

    from desktop.agent_bridge import AgentBridge

    bridge = AgentBridge(max_steps=60, speed_ms=300)

    webview.create_window(
        title="唯识进化框架 · Yogacara Agent Desktop",
        url=f"file:///{HTML_PATH.replace(os.sep, '/')}",
        js_api=bridge,
        width=1180,
        height=820,
        min_size=(980, 700),
        text_select=False,
        background_color="#0f1420",
    )
    # Windows 平台使用 .ico（pywebview 6.2+ 支持）
    # 若 ICO 不存在则降级为无图标启动
    if os.path.isfile(ICON_PATH):
        webview.start(debug=False, icon=ICON_PATH)
    else:
        webview.start(debug=False)
    # 窗口关闭后收尾（seeds_total 为阿赖耶识种子库总数，state["seeds"] 只是单步检索结果）
    bridge.stop()
    print(f"[Desktop] 已退出 | 种子库保留 {bridge.get_snapshot()['seeds_total']} 条记忆")
    return 0


if __name__ == "__main__":
    sys.exit(main())
