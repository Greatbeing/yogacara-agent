"""
一次性脚本：程序化渲染 Yogacara「八转四」桌面图标（纯 Pillow）
运行：python assets/generate_icon.py

设计（与 assets/yogacara.svg 同一几何）：
  - Win11 圆角方形深底
  - 8 枚菱形刀刃放射（八识）：正位×4 实心蓝 = 转成的四智；
    斜位×4 描边蓝 = 在转化的现行识
  - 腰线细环 + 中心金点（阿赖耶识自性）为构图锚
小尺寸变体（16/32/48）：粗刃粗描边大金蕊、去腰线。
刀刃顶点以三角函数直接落在最终位置（无旋转贴图位移），1024 超采样→512。
"""

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
ICO_PATH = os.path.join(ASSETS_DIR, "yogacara.ico")
PNG_PATH = os.path.join(ASSETS_DIR, "yogacara.png")
PNG_SMALL_PATH = os.path.join(ASSETS_DIR, "yogacara_small.png")
FAVICON_PATH = os.path.join(ASSETS_DIR, "favicon.png")

RENDER_N = 1024
FINAL_N = 512


def _hex(c: str) -> tuple:
    c = c.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def _rot(px: float, py: float, deg: float) -> tuple:
    """绕原点旋转。SVG rotate() 为顺时针（y 向下坐标系），此处保持一致。"""
    t = math.radians(deg)
    return (px * math.cos(t) - py * math.sin(t), px * math.sin(t) + py * math.cos(t))


def _blade_vertices(small: bool) -> list:
    """未旋转的菱形刀刃局部顶点（尖朝上）。"""
    if small:
        pts = [(0, -64), (34, -126), (0, -196), (-34, -126)]  # 实心用
        pts_s = [(0, -70), (30, -126), (0, -188), (-30, -126)]  # 描边用（略短避免叠线）
        return pts, pts_s
    return [(0, -58), (25, -126), (0, -192), (-25, -126)], [
        (0, -62),
        (23, -126),
        (0, -186),
        (-23, -126),
    ]


def _rounded_bg(img: Image.Image, inset: float, radius: float):
    """圆角方形深底：径向渐变 + 外沿描边。坐标按 512 基准缩放。"""
    n = img.size[0]
    s = n / 512
    stops = [(0.0, _hex("#1a2334")), (0.55, _hex("#121828")), (1.0, _hex("#0b0f1a"))]
    grad = Image.new("RGB", (n, n))
    gd = ImageDraw.Draw(grad)
    cx, cy, rmax = int(n * 0.5), int(n * 0.42), int(n * 0.78)
    for r in range(rmax, 0, -2):
        t = 1 - r / rmax
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_grad(stops, t))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [inset * s, inset * s, (512 - inset) * s, (512 - inset) * s],
        radius=radius * s,
        fill=255,
    )
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [inset * s, inset * s, (512 - inset) * s, (512 - inset) * s],
        radius=radius * s,
        outline=_hex("#2a3450"),
        width=max(2, int(2.5 * s)),
    )


def _grad(stops: list, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= t <= p1:
            f = (t - p0) / max(1e-6, p1 - p0)
            return tuple(int(c0[i] + (c1[i] - c0[i]) * f) for i in range(3))
    return stops[-1][1]


def _gold_core(img: Image.Image, radius_unit: float):
    """中心金点（阿赖耶识自性）：径向渐变圆 + 暗金描边 + 高光。"""
    n = img.size[0]
    s = n / 512
    cx = cy = n / 2
    R = radius_unit * s
    stops = [(0.0, _hex("#fff3c4")), (0.45, _hex("#ffd700")), (1.0, _hex("#e8940a"))]
    core = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    for r in range(int(R), 0, -1):
        t = 1 - r / R
        cd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_grad(stops, t) + (255,))
    cd.ellipse(
        [cx - R, cy - R, cx + R, cy + R],
        outline=(184, 134, 11, 166),
        width=max(1, int(2 * s)),
    )
    hl_w, hl_h = max(8, int(20 * s)), max(6, int(14 * s))
    hl = Image.new("RGBA", (hl_w * 2, hl_h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse([hl_w // 2, hl_h // 2, hl_w * 3 // 2, hl_h * 3 // 2], fill=(255, 255, 255, 140))
    hl = hl.rotate(32, resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(max(1, int(2 * s))))
    img.alpha_composite(core)
    img.alpha_composite(hl, (int(cx - 12 * s), int(cy - 16 * s)))


def render(master=True) -> Image.Image:
    small = not master
    img = Image.new("RGBA", (RENDER_N, RENDER_N), (0, 0, 0, 0))
    _rounded_bg(img, inset=6 if small else 16, radius=116)

    s = RENDER_N / 512
    cx = cy = RENDER_N / 2

    draw_layer = Image.new("RGBA", (RENDER_N, RENDER_N), (0, 0, 0, 0))
    ld = ImageDraw.Draw(draw_layer)

    if master:
        solid_pts, stroke_pts = _blade_vertices(small)
        # 腰线细环
        ring_r = 126 * s
        ring = Image.new("RGBA", (RENDER_N, RENDER_N), (0, 0, 0, 0))
        ImageDraw.Draw(ring).ellipse(
            [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
            outline=_hex("#223047") + (255,),
            width=max(1, int(2 * s)),
        )
        img.alpha_composite(ring)
        # 实心 ×4 正位：四智
        for ang in (0, 90, 180, 270):
            poly = [tuple(v * s for v in _rot(x, y, ang)) for x, y in solid_pts]
            ld.polygon([(cx + x, cy + y) for x, y in poly], fill=_hex("#5aabf5") + (255,))
        # 描边 ×4 斜位：现行识
        width_px = max(3, int(13 * s / 2))
        for ang in (45, 135, 225, 315):
            poly = [tuple(v * s for v in _rot(x, y, ang)) for x, y in stroke_pts]
            ld.polygon(
                [(cx + x, cy + y) for x, y in poly],
                outline=_hex("#39b7ff") + (255,),
                width=width_px,
            )
        draw_layer = draw_layer.filter(ImageFilter.GaussianBlur(max(0.5, 1.0 * s)))
    else:
        # 小变体：全实心、以明度做交替（16px 线条必糊，明度对比稳定）
        wide_pts = [(0, -68), (42, -128), (0, -198), (-42, -128)]
        bright = _hex("#7cc4ff")
        deep = _hex("#2e63a8")
        for ang in range(0, 360, 45):
            poly = [tuple(v * s for v in _rot(x, y, ang)) for x, y in wide_pts]
            ld.polygon([(cx + x, cy + y) for x, y in poly], fill=(bright if ang % 90 == 0 else deep) + (255,))
    img.alpha_composite(draw_layer)

    _gold_core(img, 52 if small else 34)
    return img.resize((FINAL_N, FINAL_N), Image.LANCZOS)


def _write_multi_ico(path: str, images: list) -> None:
    """手写 ICONDIR 合并多源帧（Pillow 的 append_images 不适用于 ICO）。

    每帧以 PNG 载荷嵌入（Vista+ 完整支持，Win10/11 任务栏正常显示）。
    """
    import io
    import struct

    ordered = sorted(images, key=lambda im: im.size[0])
    header = struct.pack("<HHH", 0, 1, len(ordered))
    offset = 6 + 16 * len(ordered)
    directory = b""
    payloads = b""
    for im in ordered:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        png = buf.getvalue()
        w, h = im.size
        directory += struct.pack(
            "<BBBBHHII",
            0 if w >= 256 else w,
            0 if h >= 256 else h,
            0,  # 调色板色数（PNG 帧不用）
            0,  # reserved
            1,  # planes
            32,  # bpp
            len(png),
            offset,
        )
        offset += len(png)
        payloads += png
    with open(path, "wb") as f:
        f.write(header + directory + payloads)


def main() -> int:
    master = render(master=True)
    small = render(master=False)
    master.save(PNG_PATH)
    small.save(PNG_SMALL_PATH)
    print(f"[render] yogacara.png OK ({FINAL_N}×{FINAL_N} 八转四主版)")
    print(f"[render] yogacara_small.png OK ({FINAL_N}×{FINAL_N} 小尺寸变体)")

    frames = [small.resize((n, n), Image.LANCZOS) for n in (16, 24, 32, 48)]
    frames += [master.resize((n, n), Image.LANCZOS) for n in (64, 128, 256)]
    _write_multi_ico(ICO_PATH, frames)
    print("[Pillow] yogacara.ico OK：7 帧（≤48←small 明度变体，≥64←master）")

    small.resize((32, 32), Image.LANCZOS).save(FAVICON_PATH)
    print("[Pillow] favicon.png OK (32×32)")
    print("✅ 图标生成完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
