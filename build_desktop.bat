@echo off
rem 唯识进化框架 · 桌面版一键打包
rem 产物: dist\YogacaraDesktop\YogacaraDesktop.exe
rem 依赖: pip install pyinstaller  （首次）

cd /d "%~dp0"
echo [1/2] 运行测试门禁...
python -m pytest tests/ -q || (echo 测试未通过，终止打包 & exit /b 1)

echo [2/2] PyInstaller 构建...
python -m PyInstaller yogacara-desktop.spec --noconfirm || (echo 构建失败 & exit /b 1)

echo.
echo ✅ 完成: dist\YogacaraDesktop\YogacaraDesktop.exe
echo    记忆数据将持久化在 exe 同级 memory\ 目录
