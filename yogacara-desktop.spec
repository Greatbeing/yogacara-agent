# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：唯识进化框架 · 桌面版

构建： pyinstaller yogacara-desktop.spec --noconfirm
产物： dist/YogacaraDesktop/YogacaraDesktop.exe（onedir，exe 旁持久化 memory/）
"""
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    "desktop.agent_bridge",
    # langgraph/langchain 运行时动态导入较多，显式收集防漏
    *collect_submodules("langgraph"),
    *collect_submodules("langchain_core"),
]

a = Analysis(
    ["desktop_app.py"],
    pathex=[".", "src"],
    binaries=[],
    datas=[
        ("desktop/index.html", "desktop"),
        ("assets/yogacara.ico", "assets"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "pandas", "torch", "transformers", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YogacaraDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/yogacara.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="YogacaraDesktop",
)
