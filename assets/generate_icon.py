"""
一次性脚本：程序化渲染 Yogacara 桌面图标（纯 Pillow，零原生依赖）
运行：python assets/generate_icon.py

设计（与 assets/yogacara.svg 同一几何）：
  - Win11 圆角方形深底 + 径向渐变
  - 外层八瓣莲花（八识）：泪滴贝塞尔花瓣，45° 均分，轴向渐变
  - 内层高光花瓣（22.5° 偏移，0.66 缩放）
  - 断口禅圆 ensō（336° 弧，修行无尽）
  - 金蕊（转依之智）+ 高光
小尺寸（16/32/48）用简化变体（粗瓣大蕊无禅圆），缩放后轮廓清晰。
矢量源文件 yogacara.svg / yogacara_small.svg 供 Web 与文档使用。
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFilter

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
ICO_PATH = os.path.join(ASSETS_DIR, "yogacara.ico")
PNG_PATH = os.path.join(ASSETS_DIR, "yogacara.png")
PNG_SMALL_PATH = os.path.join(ASSETS_DIR, "yogacara_small.png")
FAVICON_PATH = os.path.join(ASSETS_DIR, "favicon.png")

RENDER_N = 1024  # 超采样渲染尺寸
FINAL_N = 512


# ── 基础工具 ────────────────────────────────────────────────────────────
def _hex(c: str) -> tuple:
    c = c.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def _grad_color(stops: list, t: float):
    """多段线性渐变取色。stops: [(pos, (r,g,b)), ...]"""
    t = max(0.0, min(1.0, t))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= t <= p1:
            return _lerp(c0, c1, (t - p0) / max(1e-6, p1 - p0))
    return stops[-1][1]


def _bezier(p0, p1, p2, p3, n=24):
    """三次贝塞尔采样点。"""
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _petal_polygon(scale: float = 1.0) -> list:
    """泪滴花瓣轮廓（SVG 路径的贝塞尔展平）。坐标以花心为原点，尖朝上。"""
    segs = [
        ((0, -168), (30, -138), (40, -96), (40, -58)),
        ((40, -58), (40, -18), (22, 8), (0, 24)),
        ((0, 24), (-22, 8), (-40, -18), (-40, -58)),
        ((-40, -58), (-40, -96), (-30, -138), (0, -168)),
    ]
    pts = []
    for p0, p1, p2, p3 in segs:
        pts.extend(_bezier(p0, p1, p2, p3))
    return [(x * scale, y * scale) for x, y in pts]


# ── 组件渲染 ────────────────────────────────────────────────────────────
def _bg_rounded_rect(img: Image.Image):
    """圆角方形深底：径向渐变 + 边框 + 内侧高光线。"""
    N = img.size[0]
    d = ImageDraw.Draw(img)
    # 径向渐变（分层椭圆近似，圆角矩形裁剪）
    stops = [(0.0, _hex("#232e47")), (0.55, _hex("#171e2e")), (1.0, _hex("#0d111c"))]
    grad = Image.new("RGB", (N, N))
    gd = ImageDraw.Draw(grad)
    cx, cy, R = int(N * 0.5), int(N * 0.42), int(N * 0.78)
    for r in range(R, 0, -2):
        t = 1 - r / R
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_grad_color(stops, t))
    mask = Image.new("L", (N, N), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([16 / 512 * N, 16 / 512 * N, 496 / 512 * N, 496 / 512 * N],
                         radius=112 / 512 * N, fill=255)
    img.paste(grad, (0, 0), mask)
    d.rounded_rectangle([16 / 512 * N, 16 / 512 * N, 496 / 512 * N, 496 / 512 * N],
                        radius=112 / 512 * N, outline=_hex("#2a3450"), width=max(2, int(2.5 / 512 * N)))
    d.rounded_rectangle([22 / 512 * N, 22 / 512 * N, 490 / 512 * N, 490 / 512 * N],
                        radius=106 / 512 * N, outline=_hex("#3d4a6b"), width=max(1, int(1.2 / 512 * N)))


def _enso(img: Image.Image):
    """断口禅圆：336° 弧，圆头端帽。"""
    N = img.size[0]
    s = N / 512
    layer = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = 159 * s
    cx = cy = N / 2
    # SVG: M 158,-14 A 159 159 0 1 1 96,127（以花心为原点）→ 起止角
    import math

    a0 = math.degrees(math.atan2(-14 * s, 158 * s))
    a1 = math.degrees(math.atan2(127 * s, 96 * s))
    d.arc([cx - r, cy - r, cx + r, cy + r], start=a0, end=a1 + 360 if a1 < a0 else a1,
          fill=(77, 163, 255, 87), width=max(3, int(5 * s)))
    # 圆头端帽
    for ang in (a0, a1 if a1 > a0 else a1 + 360):
        px = cx + r * math.cos(math.radians(ang))
        py = cy + r * math.sin(math.radians(ang))
        w = max(3, int(5 * s))
        d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2], fill=(77, 163, 255, 87))
    img.alpha_composite(layer)


def _paste_petal(img: Image.Image, angle: float, scale: float, stops: list, alpha: int = 255):
    """单片花瓣：局部渐变 + 形状遮罩 → 旋转 → 贴合到花心。"""
    poly = _petal_polygon(scale)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 4
    W, H = int(x1 - x0) + pad * 2, int(y1 - y0) + pad * 2

    # 轴向渐变（局部坐标 y: 顶部=尖 → 底部=根）
    tile = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    for y in range(H):
        t = y / max(1, H - 1)
        td.line([(0, y), (W, y)], fill=_grad_color(stops, t) + (alpha,))

    # 形状遮罩（多边形 + 轻微模糊抗锯齿）
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(
        [(x - x0 + pad, y - y0 + pad) for x, y in poly], fill=255
    )
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))
    tile.putalpha(Image.composite(mask, Image.new("L", (W, H), 0), tile.getchannel("A")))

    rot = tile.rotate(angle, resample=Image.BICUBIC, expand=True)
    # 贴合：花瓣局部中心（花心原点 (0,0) 在 (−x0+pad, −y0+pad)）
    ox, oy = -x0 + pad, -y0 + pad
    px = int(img.size[0] / 2 - ox)
    py = int(img.size[1] / 2 - oy)
    img.alpha_composite(rot, (px, py))


def _gold_core(img: Image.Image, r_unit: float):
    """金蕊：光晕 + 主体径向渐变 + 描边 + 高光。"""
    N = img.size[0]
    s = N / 512
    cx = cy = N / 2
    # 光晕
    glow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r in range(int(86 * s), 0, -2):
        t = 1 - r / (86 * s)
        a = int(140 * (1 - t) ** 2)
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 200, 60, a))
    glow = glow.filter(ImageFilter.GaussianBlur(6 * s))
    img.alpha_composite(glow)
    # 主体
    stops = [(0.0, _hex("#fff3c4")), (0.45, _hex("#ffd700")), (1.0, _hex("#f08c00"))]
    core = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    R = r_unit * s
    for r in range(int(R), 0, -1):
        t = 1 - r / R
        cd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_grad_color(stops, t) + (255,))
    cd.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(184, 134, 11, 153),
               width=max(1, int(1.6 * s)))
    # 高光
    hl = Image.new("RGBA", (int(40 * s), int(30 * s)), (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse([0, 0, hl.size[0] - 1, hl.size[1] - 1], fill=(255, 255, 255, 140))
    hl = hl.rotate(32, expand=True).filter(ImageFilter.GaussianBlur(2 * s))
    img.alpha_composite(core)
    img.alpha_composite(hl, (int(cx - 18 * s), int(cy - 22 * s)))


# ── 主图 & 小尺寸变体 ──────────────────────────────────────────────────
def render_master(n: int = RENDER_N) -> Image.Image:
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    _bg_rounded_rect(img)
    _enso(img)
    outer = [(0.0, _hex("#8fd0ff")), (0.55, _hex("#4da3ff")), (1.0, _hex("#2b6fd4"))]
    inner = [(0.0, _hex("#d8efff")), (1.0, _hex("#7ec3ff"))]
    for k in range(8):
        _paste_petal(img, k * 45, n / 512, outer)
    for k in range(8):
        _paste_petal(img, 22.5 + k * 45, 0.66 * n / 512, inner, alpha=230)
    _gold_core(img, 40)
    return img.resize((FINAL_N, FINAL_N), Image.LANCZOS)


def render_small(n: int = RENDER_N) -> Image.Image:
    """小尺寸变体：粗长八瓣 + 大金蕊，无禅圆/内层（16px 下高可见度）。"""
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    # 收窄底边距，让莲花占满画布
    s = n / 512
    d = ImageDraw.Draw(img)
    stops_bg = [(0.0, _hex("#232e47")), (0.55, _hex("#171e2e")), (1.0, _hex("#0d111c"))]
    grad = Image.new("RGB", (n, n))
    gd = ImageDraw.Draw(grad)
    cx, cy, R = int(n * 0.5), int(n * 0.42), int(n * 0.78)
    for r in range(R, 0, -2):
        t = 1 - r / R
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_grad_color(stops_bg, t))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([6 * s, 6 * s, 506 * s, 506 * s], radius=116 * s, fill=255)
    img.paste(grad, (0, 0), mask)
    d.rounded_rectangle([6 * s, 6 * s, 506 * s, 506 * s], radius=116 * s,
                        outline=_hex("#2a3450"), width=max(2, int(3 * s)))
    # 粗长花瓣：tip -200（超出主图禅圆区），宽 ±58
    segs = [
        ((0, -200), (46, -150), (58, -84), (58, -34)),
        ((58, -34), (58, 14), (30, 46), (0, 60)),
        ((0, 60), (-30, 46), (-58, 14), (-58, -34)),
        ((-58, -34), (-58, -84), (-46, -150), (0, -200)),
    ]
    pts = []
    for p0, p1, p2, p3 in segs:
        pts.extend(_bezier(p0, p1, p2, p3))
    poly = [(x * s, y * s) for x, y in pts]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W, H = int(x1 - x0) + 8, int(y1 - y0) + 8
    stops = [(0.0, _hex("#8fd0ff")), (0.55, _hex("#4da3ff")), (1.0, _hex("#2b6fd4"))]
    tile = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    for y in range(H):
        td.line([(0, y), (W, y)], fill=_grad_color(stops, y / max(1, H - 1)) + (255,))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(x - x0 + 4, y - y0 + 4) for x, y in poly], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))
    tile.putalpha(mask)
    for k in range(8):
        rot = tile.rotate(k * 45, resample=Image.BICUBIC, expand=True)
        img.alpha_composite(rot, (int(n / 2 - (-x0 + 4)), int(n / 2 - (-y0 + 4))))
    _gold_core(img, 66)
    return img.resize((FINAL_N, FINAL_N), Image.LANCZOS)


def main() -> int:
    master = render_master()
    small = render_small()
    master.save(PNG_PATH)
    small.save(PNG_SMALL_PATH)
    print(f"[render] yogacara.png OK ({FINAL_N}×{FINAL_N}, 程序化矢量渲染)")

    master.save(ICO_PATH, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("[Pillow] yogacara.ico OK（16/32/48 用小尺寸变体）")

    small.resize((32, 32), Image.LANCZOS).save(FAVICON_PATH)
    print("[Pillow] favicon.png OK (32×32)")
    print("✅ 图标生成完成（纯 Pillow，无原生依赖）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
