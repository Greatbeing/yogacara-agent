"""
一次性脚本：从 yogacara.svg 生成 .ico 和 .png 图标文件。
运行：python assets/generate_icon.py
"""

import os
import subprocess
import sys

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
SVG_PATH = os.path.join(ASSETS_DIR, "yogacara.svg")
ICO_PATH = os.path.join(ASSETS_DIR, "yogacara.ico")
PNG_PATH = os.path.join(ASSETS_DIR, "yogacara.png")
TARGET_SIZE = 256


def generate():
    # 尝试 cairosvg（推荐，渲染质量高）
    try:
        import cairosvg

        cairosvg.svg2png(url=SVG_PATH, write_to=PNG_PATH, output_width=TARGET_SIZE, output_height=TARGET_SIZE)
        print(f"[cairosvg] {PNG_PATH} OK")
    except (ImportError, OSError, Exception):
        pass  # Windows 缺少原生 cairo DLL，走 fallback

    # 尝试 Pillow + svglib 渲染
    if not os.path.exists(PNG_PATH):
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM

            drawing = svg2rlg(SVG_PATH)
            scale = TARGET_SIZE / max(drawing.width, drawing.height)
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
            renderPM.drawToFile(drawing, PNG_PATH, fmt="PNG")
            print(f"[svglib] {PNG_PATH} OK")
        except ImportError:
            pass

    # Pillow SVG 加载（Pillow 10.1+ 支持）
    if not os.path.exists(PNG_PATH):
        try:
            from PIL import Image

            img = Image.open(SVG_PATH)
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            img.save(PNG_PATH)
            print(f"[Pillow SVG] {PNG_PATH} OK")
        except Exception:
            pass

    # 最终 fallback：用 Inkscape CLI
    if not os.path.exists(PNG_PATH):
        try:
            subprocess.run(
                ["inkscape", SVG_PATH, "-w", str(TARGET_SIZE), "-h", str(TARGET_SIZE), "-o", PNG_PATH],
                check=True, capture_output=True,
            )
            print(f"[Inkscape] {PNG_PATH} OK")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  无法渲染 SVG，尝试内嵌 base64 PNG 方式...")
            _generate_png_fallback()

    # PNG → ICO
    if os.path.exists(PNG_PATH):
        from PIL import Image

        img = Image.open(PNG_PATH).convert("RGBA")
        img.save(ICO_PATH, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        print(f"[Pillow] {ICO_PATH} OK (multi-size)")
    else:
        print("❌ PNG 生成失败，无法制作 ICO")
        sys.exit(1)

    # 同时生成 32px favicon PNG
    from PIL import Image

    img = Image.open(PNG_PATH).convert("RGBA")
    favicon = img.resize((32, 32), Image.LANCZOS)
    favicon_path = os.path.join(ASSETS_DIR, "favicon.png")
    favicon.save(favicon_path)
    print(f"[Pillow] {favicon_path} OK (32×32 favicon)")
    print("✅ 图标生成完成")


def _generate_png_fallback():
    """用纯 Pillow 绘制八瓣莲花图标（旋转贴图法，质量接近矢量渲染）。"""
    import math

    from PIL import Image, ImageDraw, ImageFilter

    size = TARGET_SIZE
    cx = cy = size // 2

    # ── 1. 绘制单片花瓣（透明底，竖直向上） ──
    petal_w, petal_h = 44, 96
    petal = Image.new("RGBA", (petal_w, petal_h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(petal)
    # 花瓣形状：顶部尖、底部圆的叶形
    petal_pts = [
        (petal_w // 2, 0),           # 顶尖
        (petal_w, 24),               # 右上
        (petal_w - 4, petal_h - 14), # 右下
        (petal_w // 2, petal_h),     # 底中
        (4, petal_h - 14),           # 左下
        (0, 24),                     # 左上
    ]
    pd.polygon(petal_pts, fill=(77, 163, 255, 235))
    # 花瓣高光（内层更亮的小椭圆）
    hl = Image.new("RGBA", (petal_w, petal_h), (0, 0, 0, 0))
    hld = ImageDraw.Draw(hl)
    hld.ellipse([8, 18, petal_w - 8, petal_h - 30], fill=(190, 220, 255, 90))
    hl = hl.filter(ImageFilter.GaussianBlur(3))
    petal = Image.alpha_composite(petal, hl)
    # 花瓣边缘柔化
    petal = petal.filter(ImageFilter.GaussianBlur(0.6))

    # ── 2. 主画布 ──
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 背景圆（深色渐变模拟：多层同心圆）
    for r, alpha in [(120, 255), (110, 250), (95, 240)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15 + (120 - r) // 4, 20 + (120 - r) // 4, 32, alpha))
    draw.ellipse([cx - 120, cy - 120, cx + 120, cy + 120], outline=(42, 52, 80, 200), width=2)

    # ── 3. 贴 8 片旋转花瓣 ──
    petal_distance = 48  # 花瓣中心距圆心的距离
    for i in range(8):
        angle = i * 45
        # 旋转花瓣（绕自身底中心旋转）
        rotated = petal.rotate(angle, resample=Image.BICUBIC, expand=True)
        rw, rh = rotated.size
        # 计算花瓣顶尖在圆周上的位置
        rad = math.radians(angle - 90)  # -90 让 angle=0 时朝上
        px = cx + int(petal_distance * math.cos(rad)) - rw // 2
        py = cy + int(petal_distance * math.sin(rad)) - rh // 2
        img.alpha_composite(rotated, (px, py))

    # ── 4. 花蕊（金色中心 + 高光） ──
    draw = ImageDraw.Draw(img)
    # 外圈光晕
    for r, a in [(26, 40), (22, 80), (18, 160)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 200, 60, a))
    # 实心金蕊
    draw.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=(255, 215, 0, 255))
    # 内层高光
    draw.ellipse([cx - 10, cy - 12, cx + 6, cy + 4], fill=(255, 248, 225, 180))
    # 中心白点
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255, 220))

    # ── 5. 外圈禅圆（虚线 ensō） ──
    enso = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ed = ImageDraw.Draw(enso)
    ed.ellipse([cx - 88, cy - 88, cx + 88, cy + 88], outline=(77, 163, 255, 70), width=2)
    enso = enso.filter(ImageFilter.GaussianBlur(1.5))
    img.alpha_composite(enso)

    img.save(PNG_PATH)
    print(f"[fallback] {PNG_PATH} OK (rotated-petal lotus)")


if __name__ == "__main__":
    generate()
