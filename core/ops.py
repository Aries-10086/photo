"""渲染 op 实现。

每个函数签名统一： (canvas: RGBA Image, op: dict, ctx: Ctx) -> RGBA Image
不修改入参 canvas，返回新图。ctx 提供素材查找与缓存。

设计要点（见 PRD）：
  · 硬边优先 —— 默认不做羽化，羽化必须显式开启
  · 色彩迁移默认关闭 —— 色差是笑点，不是缺陷
  · 降质是独立 op，不混在合成里
  · 所有坐标归一化 0~1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core import fonts, imageio

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 上下文


@dataclass
class Ctx:
    """渲染上下文：提供素材查询。"""

    assets: dict[str, Image.Image] = field(default_factory=dict)
    """asset_id -> RGBA 图（已按需缩放）"""

    base_long: int = 1
    """基底长边像素，用于把 scale 从「相对比例」换算成像素"""

    def asset(self, aid: str) -> Image.Image:
        img = self.assets.get(aid)
        if img is None:
            raise KeyError(f"素材未加载: {aid}")
        return img


# ---------------------------------------------------------------- 几何工具


def _rotate_expand(img: Image.Image, angle: float) -> Image.Image:
    if not angle:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)


def _flip(img: Image.Image, fx: bool, fy: bool) -> Image.Image:
    if fx and fy:
        return img.transpose(Image.ROTATE_180)
    if fx:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if fy:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def _crop(img: Image.Image, box: list[float] | None) -> Image.Image:
    """按归一化 [x, y, w, h] 裁剪素材自身。"""
    if not box:
        return img
    w, h = img.size
    x, y, bw, bh = box
    left = int(round(x * w))
    top = int(round(y * h))
    right = max(left + 1, int(round((x + bw) * w)))
    bottom = max(top + 1, int(round((y + bh) * h)))
    return img.crop((left, top, min(right, w), min(bottom, h)))


def _paste(canvas: Image.Image, piece: Image.Image, center_px: tuple[float, float],
           opacity: float = 1.0, blend: str = "normal") -> Image.Image:
    """把 piece 以中心对齐方式贴到 canvas 上（返回新图）。"""
    cx, cy = center_px
    left = int(round(cx - piece.width / 2))
    top = int(round(cy - piece.height / 2))

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(piece, (left, top))

    alpha = layer.getchannel("A")
    if opacity < 1.0:
        alpha = alpha.point(lambda v: int(v * opacity))
        layer.putalpha(alpha)

    if blend == "normal":
        return Image.alpha_composite(canvas, layer)
    return _blend(canvas, layer, blend)


def _blend(bg: Image.Image, fg: Image.Image, mode: str) -> Image.Image:
    """实现常用混合模式，最后用 fg 的 alpha 做选择性合成。"""
    if mode == "normal":
        return Image.alpha_composite(bg, fg)

    b = np.asarray(bg.convert("RGB"), dtype=np.float32) / 255.0
    f = np.asarray(fg.convert("RGB"), dtype=np.float32) / 255.0

    if mode == "multiply":
        r = b * f
    elif mode == "screen":
        r = 1 - (1 - b) * (1 - f)
    elif mode == "overlay":
        r = np.where(b <= 0.5, 2 * b * f, 1 - 2 * (1 - b) * (1 - f))
    elif mode == "darken":
        r = np.minimum(b, f)
    elif mode == "lighten":
        r = np.maximum(b, f)
    else:
        r = f

    out = Image.fromarray((np.clip(r, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
    out.putalpha(fg.getchannel("A"))
    return Image.alpha_composite(bg, out)


def _adjust_rgb(img: Image.Image, *, hue_shift: float = 0, saturation: float = 1.0,
                brightness: float = 1.0, contrast: float = 1.0,
                tint: str | None = None, tint_strength: float = 0.5) -> Image.Image:
    """对 RGB 做色相/饱和/亮度/对比/染色调整，保留 alpha。"""
    if (hue_shift == 0 and saturation == 1.0 and brightness == 1.0
            and contrast == 1.0 and not tint):
        return img

    rgb = img.convert("RGB")
    a = img.getchannel("A")

    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    if hue_shift:
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 360.0 * 255.0) % 255.0
    if saturation != 1.0:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    if brightness != 1.0:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    out = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")

    if contrast != 1.0:
        arr = np.asarray(out, dtype=np.float32) / 255.0
        arr = np.clip((arr - 0.5) * contrast + 0.5, 0, 1)
        out = Image.fromarray((arr * 255).astype(np.uint8), "RGB")

    if tint:
        tc = np.array([int(tint[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
        arr = np.asarray(out, dtype=np.float32) / 255.0
        arr = arr * (1 - tint_strength) + (arr * tc) * tint_strength
        out = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")

    out = out.convert("RGBA")
    out.putalpha(a)
    return out


def _color_transfer(src: Image.Image, ref: Image.Image, strength: float) -> Image.Image:
    """把 src 的均值/方差往 ref 拉，强度 0~1。用于「半融合」档。"""
    if strength <= 0:
        return src

    sa = np.asarray(src.convert("RGB"), dtype=np.float32)
    ra = np.asarray(ref.convert("RGB"), dtype=np.float32)
    m = (src.getchannel("A").point(lambda v: 255 if v > 8 else 0)) if src.mode == "RGBA" else None

    if m is not None:
        mask = np.asarray(m) > 0
        if mask.sum() < 16:
            return src
        s_mean = sa[mask].mean(axis=0)
        s_std = sa[mask].std(axis=0) + 1e-6
    else:
        s_mean = sa.reshape(-1, 3).mean(axis=0)
        s_std = sa.reshape(-1, 3).std(axis=0) + 1e-6

    r_mean = ra.reshape(-1, 3).mean(axis=0)
    r_std = ra.reshape(-1, 3).std(axis=0) + 1e-6

    target = (sa - s_mean) / s_std * r_std + r_mean
    blended = sa * (1 - strength) + target * strength

    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    out.putalpha(src.getchannel("A"))
    return out


def _feather(img: Image.Image, px: int) -> Image.Image:
    """只对 alpha 做羽化，不改颜色。"""
    if px <= 0:
        return img
    a = img.getchannel("A").filter(ImageFilter.GaussianBlur(px))
    out = img.copy()
    out.putalpha(a)
    return out


# ---------------------------------------------------------------- ops


def op_backdrop(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    """背景替换。mode=cover/contain/tile/solid。"""
    mode = op.get("mode", "cover")
    W, H = canvas.size
    subject_mask = op.get("subject_mask")
    opacity = float(op.get("opacity", 1.0))
    blur = int(op.get("blur_px", 0))

    if mode == "solid":
        color = op.get("color", "#FFFFFF")
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        bg = Image.new("RGBA", (W, H), rgb + (255,))
    else:
        aid = op.get("asset_id")
        if not aid:
            raise ValueError("backdrop: mode != solid 时必须提供 asset_id")
        src = ctx.asset(aid)
        if mode == "cover":
            scale = max(W / src.width, H / src.height)
            nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
            bg = src.resize((nw, nh), Image.LANCZOS)
            left = (nw - W) // 2
            top = (nh - H) // 2
            bg = bg.crop((left, top, left + W, top + H))
        elif mode == "contain":
            scale = min(W / src.width, H / src.height)
            nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
            tile = src.resize((nw, nh), Image.LANCZOS)
            bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            bg.paste(tile, ((W - nw) // 2, (H - nh) // 2), tile)
        elif mode == "tile":
            bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            for y in range(0, H, src.height):
                for x in range(0, W, src.width):
                    bg.paste(src, (x, y), src)
        else:
            raise ValueError(f"backdrop: 未知 mode {mode}")

    if blur:
        bg = bg.filter(ImageFilter.GaussianBlur(blur))

    # 主体区域用原 canvas 覆盖回来
    if subject_mask:
        mp = Path(subject_mask)
        if not mp.is_absolute():
            mp = ROOT / subject_mask
        if not mp.exists():
            raise FileNotFoundError(f"subject_mask 不存在: {subject_mask}")
        m = imageio.read(mp, keep_alpha=False, limit=None).convert("L")
        if m.size != (W, H):
            m = m.resize((W, H), Image.LANCZOS)
        bg = Image.composite(canvas, bg, m)
    else:
        # 用 canvas 的 alpha 当主体掩码。注意：若是照片（alpha 全不透明），
        # 背景不会被替换 —— 这是预期行为，换背景必须显式给 subject_mask。
        a = canvas.getchannel("A")
        if a.getextrema()[0] == 255:
            raise ValueError(
                "backdrop: 基底 alpha 全不透明，无法自动识别主体。"
                "请提供 subject_mask（主体区域的灰度掩码：白色=保留原图）。")
        bg = Image.composite(canvas, bg, a)

    if opacity < 1.0:
        bg = Image.blend(canvas, bg, opacity)
    return bg


def op_sticker(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    src = ctx.asset(op["asset_id"])
    src = _crop(src, op.get("crop_box"))
    src = _flip(src, bool(op.get("flip_x", False)), bool(op.get("flip_y", False)))

    scale = float(op.get("scale", 1.0))
    target = max(1, int(round(ctx.base_long * scale)))
    ratio = target / max(src.width, src.height)
    nw = max(1, int(round(src.width * ratio)))
    nh = max(1, int(round(src.height * ratio)))
    src = src.resize((nw, nh), Image.LANCZOS)

    src = _rotate_expand(src, float(op.get("rotation_deg", 0)))

    cx = float(op["center"][0]) * canvas.width
    cy = float(op["center"][1]) * canvas.height
    return _paste(canvas, src, (cx, cy), float(op.get("opacity", 1.0)),
                  op.get("blend", "normal"))


def op_face_fit(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    src = ctx.asset(op["asset_id"])
    src = _crop(src, op.get("crop_box"))
    src = _flip(src, bool(op.get("flip_x", False)), bool(op.get("flip_y", False)))

    scale = float(op.get("scale", 1.0))
    target = max(1, int(round(ctx.base_long * scale)))
    ratio = target / max(src.width, src.height)
    nw = max(1, int(round(src.width * ratio)))
    nh = max(1, int(round(src.height * ratio)))
    src = src.resize((nw, nh), Image.LANCZOS)

    src = _rotate_expand(src, float(op.get("rotation_deg", 0)))

    # 色调
    src = _adjust_rgb(
        src,
        hue_shift=float(op.get("hue_shift", 0)),
        saturation=float(op.get("saturation", 1.0)),
        brightness=float(op.get("brightness", 1.0)),
        tint=op.get("tint"),
        tint_strength=float(op.get("tint_strength", 0.5)),
    )

    # 色彩迁移（默认 0，保留色差）
    ct = float(op.get("color_transfer", 0))
    if ct > 0:
        src = _color_transfer(src, canvas, ct)

    # 边缘处理
    edge = op.get("edge", "hard")
    if edge == "soft":
        src = _feather(src, int(op.get("feather_px", 3)) or 3)
    # hard / none: 不动

    cx = float(op["center"][0]) * canvas.width
    cy = float(op["center"][1]) * canvas.height
    return _paste(canvas, src, (cx, cy), float(op.get("opacity", 1.0)),
                  op.get("blend", "normal"))


def op_feature_warp(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    """五官形变移植。

    用仿射（缩放+斜切+旋转）实现，再按 target_box 定位。
    TPS 控制点留给 M2 —— 这里先保证「拉伸 + 斜挑」两个最关键的喜剧参数可用。
    """
    src = ctx.asset(op["asset_id"])
    src = _crop(src, op.get("crop_box"))
    src = _flip(src, bool(op.get("flip_x", False)), bool(op.get("flip_y", False)))

    W, H = canvas.size
    bx, by, bw, bh = (float(v) for v in op["target_box"])
    box_w = max(1, int(round(bw * W)))
    box_h = max(1, int(round(bh * H)))

    sx = float(op.get("stretch_x", 1.0))
    sy = float(op.get("stretch_y", 1.0))

    # 先缩放到目标框，再乘拉伸系数
    fit = min(box_w / src.width, box_h / src.height)
    nw = max(1, int(round(src.width * fit * sx)))
    nh = max(1, int(round(src.height * fit * sy)))
    piece = src.resize((nw, nh), Image.LANCZOS)

    # 斜切（skew）：用仿射实现
    skew = float(op.get("skew_deg", 0))
    if skew:
        shear = math.tan(math.radians(skew))
        # 水平方向斜切：x' = x + shear * (y - h/2)
        coeffs = (1, shear, -shear * nh / 2, 0, 1, 0)
        pad = int(abs(shear) * nh)
        wide = Image.new("RGBA", (nw + 2 * pad, nh), (0, 0, 0, 0))
        wide.paste(piece, (pad, 0))
        piece = wide.transform(wide.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
        piece = piece.crop(piece.getbbox() or (0, 0, wide.width, wide.height))

    hs = float(op.get("hue_shift", 0))
    if hs:
        piece = _adjust_rgb(piece, hue_shift=hs)

    # keep_skin_rim：扩张 alpha 边缘，制造一圈「带皮」的色块边界
    rim = float(op.get("keep_skin_rim", 0))
    if rim > 0:
        r = max(1, int(round(min(piece.size) * rim * 0.5)))
        a = piece.getchannel("A").filter(ImageFilter.MaxFilter(min(9, r * 2 + 1)))
        piece.putalpha(a)

    cx = (bx + bw / 2) * W
    cy = (by + bh / 2) * H
    return _paste(canvas, piece, (cx, cy), float(op.get("opacity", 1.0)))


def op_texture_wrap(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    """四角透视贴纹理。"""
    src = ctx.asset(op["asset_id"])
    src = _crop(src, op.get("crop_box"))
    src = _flip(src, bool(op.get("flip_x", False)), bool(op.get("flip_y", False)))

    W, H = canvas.size
    quad = [(float(p[0]) * W, float(p[1]) * H) for p in op["quad"]]

    coeffs = _perspective_coeffs(
        [(0, 0), (src.width, 0), (src.width, src.height), (0, src.height)],
        quad,
    )
    warped = src.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

    # 掩码：只保留四边形内部
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(x, y) for x, y in quad], fill=255)
    warped.putalpha(Image.composite(warped.getchannel("A"), Image.new("L", (W, H), 0), mask))

    return _paste(canvas, warped, (W / 2, H / 2),
                  float(op.get("opacity", 1.0)), op.get("blend", "normal"))


def _perspective_coeffs(src_pts, dst_pts):
    """解 8 个系数：PIL PERSPECTIVE 用 dst->src 映射。"""
    a = []
    b = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    res = np.linalg.solve(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))
    return tuple(res.tolist())


def op_grade(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    """整体/局部调色 + 可选独立加噪。"""
    out = _adjust_rgb(
        canvas,
        hue_shift=float(op.get("hue_shift", 0)),
        saturation=float(op.get("saturation", 1.0)),
        brightness=float(op.get("brightness", 1.0)),
        contrast=float(op.get("contrast", 1.0)),
        tint=op.get("tint"),
        tint_strength=float(op.get("tint_strength", 0.5)),
    )

    noise = float(op.get("noise", 0))
    if noise > 0:
        arr = np.asarray(out.convert("RGB"), dtype=np.float32)
        rng = np.random.default_rng(0)  # 固定种子，保证可复现
        arr = np.clip(arr + rng.normal(0, noise * 255, arr.shape), 0, 255)
        rgb = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
        rgb.putalpha(out.getchannel("A"))
        out = rgb

    mask_path = op.get("mask")
    if mask_path:
        mp = ROOT / mask_path
        if not mp.exists():
            raise FileNotFoundError(f"grade mask 不存在: {mask_path}")
        m = imageio.read(mp, keep_alpha=False, limit=None).convert("L")
        if m.size != canvas.size:
            m = m.resize(canvas.size, Image.LANCZOS)
        out = Image.composite(out, canvas, m)

    return out


def op_degrade(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    """降质：降分辨率 → JPEG 重压 → 加噪 → 锐化过冲。"""
    p = op.get("params", {})
    if not p.get("enabled", True):
        return canvas

    W, H = canvas.size
    rgb = imageio.to_rgb_on_white(canvas) if canvas.mode == "RGBA" else canvas.convert("RGB")

    ds = float(p.get("downscale", 1.0))
    if ds < 1.0:
        nw, nh = max(1, int(W * ds)), max(1, int(H * ds))
        rgb = rgb.resize((nw, nh), Image.LANCZOS)

    q = int(p.get("jpeg_quality", 95))
    if q < 100:
        import io as _io
        buf = _io.BytesIO()
        rgb.save(buf, "JPEG", quality=q, subsampling=0)
        buf.seek(0)
        rgb = Image.open(buf).convert("RGB")
        buf.close()

    if ds < 1.0:
        rgb = rgb.resize((W, H), Image.LANCZOS)

    noise = float(p.get("noise", 0))
    if noise > 0:
        arr = np.asarray(rgb, dtype=np.float32)
        rng = np.random.default_rng(0)
        arr = np.clip(arr + rng.normal(0, noise * 255, arr.shape), 0, 255)
        rgb = Image.fromarray(arr.astype(np.uint8), "RGB")

    sharp = p.get("oversharpen", "off")
    if sharp != "off":
        amount = {"weak": 0.6, "medium": 1.2, "strong": 2.0}.get(sharp, 1.0)
        blurred = rgb.filter(ImageFilter.GaussianBlur(1.2))
        pa = np.asarray(rgb, dtype=np.float32)
        ba = np.asarray(blurred, dtype=np.float32)
        rgb = Image.fromarray(
            np.clip(pa + amount * (pa - ba), 0, 255).astype(np.uint8), "RGB")

    out = rgb.convert("RGBA")
    out.putalpha(canvas.getchannel("A") if canvas.mode == "RGBA" else Image.new("L", (W, H), 255))
    return out


def op_label(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    """文字标签：纯色/渐变 + 描边 + 投影 + 竖排 + 路径文字。"""
    text = op["text"]
    W, H = canvas.size
    size_ratio = float(op.get("size_ratio", 0.035))
    font_px = max(8, int(round(ctx.base_long * size_ratio)))
    font = fonts.load(op.get("font", "msyh"), font_px)

    color = op.get("color", "#FFFFFF")
    color_to = op.get("color_to")
    stroke = int(op.get("stroke", 0))
    stroke_color = op.get("stroke_color", "#000000")
    vertical = bool(op.get("vertical", False))
    arc_deg = float(op.get("arc_deg", 0))
    arc_radius = float(op.get("arc_radius", 0.3))
    spacing = float(op.get("letter_spacing", 0)) * font_px
    rotation = float(op.get("rotation_deg", 0))
    opacity = float(op.get("opacity", 1.0))
    shadow = op.get("shadow")

    # 布局计算：先量尺寸
    if vertical:
        chars = list(text)
        widths, heights = [], []
        for ch in chars:
            bb = font.getbbox(ch)
            widths.append(bb[2] - bb[0])
            heights.append(font_px + spacing)
        bw = max(widths) if widths else font_px
        bh = sum(heights) if heights else font_px
    else:
        bb = font.getbbox(text)
        bw = bb[2] - bb[0] + spacing * max(0, len(text) - 1)
        bh = bb[3] - bb[1]
        if arc_deg:
            # 弧线排布需要更大画布
            arc_r = arc_radius * ctx.base_long
            span = math.radians(abs(arc_deg))
            bw = int(max(bw, 2 * arc_r * math.sin(span / 2) + font_px))
            bh = int(max(bh, arc_r * (1 - math.cos(span / 2)) + font_px * 2))

    pad = stroke + (shadow["blur"] + max(abs(shadow["dx"]), abs(shadow["dy"])) if shadow else 0) + 4
    layer = Image.new("RGBA", (int(bw + 2 * pad), int(bh + 2 * pad)), (0, 0, 0, 0))

    # 渐变填充用 mask + 渐变色块
    def draw_text(target: Image.Image, fill) -> None:
        d = ImageDraw.Draw(target)
        if vertical:
            y = pad
            for ch in list(text):
                bb = font.getbbox(ch)
                cw = bb[2] - bb[0]
                x = pad + (bw - cw) / 2 - bb[0]
                d.text((x, y), ch, font=font, fill=fill,
                       stroke_width=stroke, stroke_fill=stroke_color)
                y += font_px + spacing
        elif arc_deg:
            arc_r = arc_radius * ctx.base_long
            total = len(text)
            span = math.radians(arc_deg)
            cx = layer.width / 2
            cy = pad + bh
            for i, ch in enumerate(text):
                t = (i / max(1, total - 1)) - 0.5 if total > 1 else 0
                ang = t * span
                x = cx + arc_r * math.sin(ang)
                y = cy - arc_r * math.cos(ang)
                bb = font.getbbox(ch)
                cw, chh = bb[2] - bb[0], bb[3] - bb[1]
                tmp = Image.new("RGBA", (int(cw + 2 * stroke + 4), int(chh + 2 * stroke + 4)),
                                (0, 0, 0, 0))
                ImageDraw.Draw(tmp).text((stroke + 2 - bb[0], stroke + 2 - bb[1]), ch,
                                         font=font, fill=fill,
                                         stroke_width=stroke, stroke_fill=stroke_color)
                tmp = tmp.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
                layer.alpha_composite(tmp, (int(x - tmp.width / 2), int(y - tmp.height / 2)))
        else:
            x = pad
            for ch in text:
                d.text((x, pad), ch, font=font, fill=fill,
                       stroke_width=stroke, stroke_fill=stroke_color)
                x += font.getlength(ch) + spacing

    if color_to:
        # 渐变：先画白字当 mask
        mask = Image.new("L", layer.size, 0)
        draw_text(mask, 255)
        grad = _linear_gradient(layer.size, color, color_to, op.get("gradient_dir", "vertical"))
        # 描边部分用纯色补
        if stroke:
            ring = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            draw_text(ring, (0, 0, 0, 0))
            layer.alpha_composite(ring)
        colored = grad.convert("RGBA")
        colored.putalpha(mask)
        layer = Image.alpha_composite(layer, colored)
        # 描边单独补画
        if stroke:
            stroke_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            m2 = Image.new("L", layer.size, 0)
            d2 = ImageDraw.Draw(m2)
            if vertical:
                y = pad
                for ch in list(text):
                    bb = font.getbbox(ch)
                    cw = bb[2] - bb[0]
                    x = pad + (bw - cw) / 2 - bb[0]
                    d2.text((x, y), ch, font=font, fill=255,
                            stroke_width=stroke, stroke_fill=255)
                    y += font_px + spacing
            else:
                x = pad
                for ch in text:
                    d2.text((x, pad), ch, font=font, fill=255,
                            stroke_width=stroke, stroke_fill=255)
                    x += font.getlength(ch) + spacing
            inner = Image.new("L", layer.size, 0)
            di = ImageDraw.Draw(inner)
            if vertical:
                y = pad
                for ch in list(text):
                    bb = font.getbbox(ch)
                    cw = bb[2] - bb[0]
                    x = pad + (bw - cw) / 2 - bb[0]
                    di.text((x, y), ch, font=font, fill=255)
                    y += font_px + spacing
            else:
                x = pad
                for ch in text:
                    di.text((x, pad), ch, font=font, fill=255)
                    x += font.getlength(ch) + spacing
            ring_mask = np.clip(np.asarray(m2, dtype=np.int16) - np.asarray(inner, dtype=np.int16),
                                0, 255).astype(np.uint8)
            stroke_layer.putalpha(Image.fromarray(ring_mask, "L"))
            sc = tuple(int(stroke_color[i:i + 2], 16) for i in (1, 3, 5))
            solid = Image.new("RGBA", layer.size, sc + (255,))
            solid.putalpha(stroke_layer.getchannel("A"))
            layer = Image.alpha_composite(solid, layer)
    else:
        draw_text(layer, color)

    # 投影
    if shadow:
        sil = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        sil.paste(layer, (0, 0), layer)
        a = sil.getchannel("A")
        if shadow.get("blur", 0):
            a = a.filter(ImageFilter.GaussianBlur(shadow["blur"]))
        sc = tuple(int(shadow.get("color", "#000000")[i:i + 2], 16) for i in (1, 3, 5))
        col = Image.new("RGBA", layer.size, sc + (255,))
        col.putalpha(a.point(lambda v: int(v * float(shadow.get("opacity", 0.7)))))
        shifted = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shifted.paste(col, (shadow.get("dx", 3), shadow.get("dy", 3)))
        layer = Image.alpha_composite(shifted, layer)

    if rotation:
        layer = layer.rotate(rotation, resample=Image.BICUBIC, expand=True)

    # 定位
    anchor = op.get("anchor", "bottom_right")
    pos = op.get("pos")
    margin = int(min(W, H) * 0.02)
    if anchor == "custom" and pos:
        cx, cy = pos[0] * W, pos[1] * H
    else:
        cx = {"left": margin + layer.width / 2, "right": W - margin - layer.width / 2,
              "center": W / 2}
        cy = {"top": margin + layer.height / 2, "bottom": H - margin - layer.height / 2,
              "center": H / 2}
        hk, vk = anchor.split("_") if "_" in anchor else ("center", "center")
        cx = cx.get(hk, W / 2)
        cy = cy.get(vk, H / 2)

    return _paste(canvas, layer, (cx, cy), opacity)


def _linear_gradient(size: tuple[int, int], c1: str, c2: str, direction: str) -> Image.Image:
    W, H = size
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)

    if direction == "horizontal":
        t = np.linspace(0, 1, W).reshape(1, W, 1)
        arr = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        arr = np.repeat(arr, H, axis=0)
    elif direction == "diagonal":
        tx = np.linspace(0, 1, W).reshape(1, W)
        ty = np.linspace(0, 1, H).reshape(H, 1)
        t = ((tx + ty) / 2).reshape(H, W, 1)
        arr = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
    else:
        t = np.linspace(0, 1, H).reshape(H, 1, 1)
        arr = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        arr = np.repeat(arr, W, axis=1)

    return Image.fromarray(arr.astype(np.uint8), "RGB")


# ---------------------------------------------------------------- 分派

OPS = {
    "backdrop": op_backdrop,
    "sticker": op_sticker,
    "face_fit": op_face_fit,
    "feature_warp": op_feature_warp,
    "texture_wrap": op_texture_wrap,
    "grade": op_grade,
    "degrade": op_degrade,
    "label": op_label,
}


def apply_op(canvas: Image.Image, op: dict, ctx: Ctx) -> Image.Image:
    name = op.get("op")
    fn = OPS.get(name)
    if fn is None:
        raise ValueError(f"未知 op: {name}（可用: {sorted(OPS)}）")
    return fn(canvas, op, ctx)
