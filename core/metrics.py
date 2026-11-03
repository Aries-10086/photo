"""M2 验收指标：色差保留率 + 边缘梯度。

这是 PRD FR-5 要求的量化验收，替代「看着差不多」。

两个指标：
  color_diff_retention  色差保留率 = 迁移后色差 / 迁移前色差
                        硬贴应 > 70%；半融合 30~50%；全融合 < 15%
  edge_gradient         融合区边缘梯度均值，判断是硬边还是羽化

用法：
    from core.metrics import color_diff_retention, edge_gradient, report
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _alpha_mask(img: Image.Image, thr: int = 8) -> np.ndarray | None:
    """取不透明区域作为掩码。全不透明时返回 None。"""
    if img.mode != "RGBA":
        return None
    a = np.asarray(img.getchannel("A"))
    m = a > thr
    if m.all() or not m.any():
        return None
    return m


def color_stats(region: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """区域的颜色均值与标准差。"""
    if mask is not None and mask.any():
        px = region[mask]
    else:
        px = region.reshape(-1, 3)
    return px.mean(axis=0), px.std(axis=0) + 1e-6


def color_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两个颜色向量的欧氏距离（0~1 尺度）。"""
    return float(np.linalg.norm(a - b))


def color_diff_retention(
    source_piece: Image.Image,
    composed: Image.Image,
    *,
    box: tuple[int, int, int, int],
    base_image: Image.Image | None = None,
    inset: float = 0.15,
    ring: float = 0.12,
) -> float:
    """色差保留率 = 迁移后色差 / 迁移前色差。

    正确的定义（第二次修正，第一次思路错了）：
        色差 = 「贴入内容」与「它所处的背景」之间的颜色差距。

        迁移前：素材平均色 vs 基底在该位置的平均色（素材还没贴上去时）
        迁移后：合成图核心区平均色 vs 合成图外围背景平均色

        两者之比就是保留比例。硬贴 = 1.0（完全没改色），
        全融合 = 接近 0（脸被染成背景色了）。

    为什么必须传 base_image：
        素材本身是纯色圆/纯色块时，它内部没有对比度，
        无法从素材自身「推导」出迁移前的色差。必须有基底作为参照。

    Args:
        source_piece: 被贴入的素材（RGBA）。
        composed: 合成后的完整画布（RGBA）。
        box: 素材在画布上的位置 (l, t, r, b)，像素。
        base_image: 原始基底图（未贴）���**强烈建议传入**。
        inset: 核心区向内收缩比例。
        ring: 背景环宽度比例。

    Returns:
        0~1.5。1.0 = 色差完全保留；接近 0 = 被抹平。
    """
    comp = composed.convert("RGB")
    W, H = comp.size
    l, t, r, b = box
    l, t = max(0, l), max(0, t)
    r, b = min(W, r), min(H, b)
    if r - l < 4 or b - t < 4:
        return 1.0

    bw, bh = r - l, b - t
    ix, iy = int(bw * inset), int(bh * inset)
    core_box = (l + ix, t + iy, r - ix, b - iy)
    if core_box[2] - core_box[0] < 2 or core_box[3] - core_box[1] < 2:
        core_box = (l, t, r, b)

    comp_arr = _to_arr(comp)

    # ---- 迁移后：合成图核心区 vs 外围背景环 ----
    core_mean, _ = color_stats(
        comp_arr[core_box[1]:core_box[3], core_box[0]:core_box[2]])

    bg_mean = _ring_mean(comp_arr, (l, t, r, b), ring)
    if bg_mean is None:
        return 1.0
    d_after = color_distance(core_mean, bg_mean)

    # ---- 迁移前 ----
    src = source_piece.convert("RGBA")
    sa = _to_arr(src)
    alpha = np.asarray(src.getchannel("A"))
    opaque = alpha > 200
    if not opaque.any():
        return 1.0
    src_mean, _ = color_stats(sa, opaque)

    if base_image is not None:
        # 首选：基底在素材位置的平均色
        ba = _to_arr(base_image.convert("RGB"))
        if ba.shape[:2] != comp_arr.shape[:2]:
            base_img2 = base_image.convert("RGB").resize((W, H), Image.LANCZOS)
            ba = _to_arr(base_img2)
        ref_mean, _ = color_stats(ba[core_box[1]:core_box[3], core_box[0]:core_box[2]])
    else:
        # 退化：用合成图的外围背景当作「原本的背景」（近似，但有偏差）
        ref_mean = bg_mean

    d_before = color_distance(src_mean, ref_mean)
    if d_before < 0.02:
        # 素材与背景本来就很接近，色差无意义
        return 1.0

    return float(np.clip(d_after / d_before, 0.0, 1.5))


def _ring_mean(arr: np.ndarray, box: tuple[int, int, int, int],
               ring: float) -> np.ndarray | None:
    """取 box 外扩一圈的环形区域均值（排除 box 内部）。"""
    H, W = arr.shape[:2]
    l, t, r, b = box
    bw, bh = r - l, b - t
    rx, ry = int(bw * ring), int(bh * ring)
    outer = (max(0, l - rx), max(0, t - ry), min(W, r + rx), min(H, b + ry))
    ow, oh = outer[2] - outer[0], outer[3] - outer[1]
    if ow < 2 or oh < 2:
        return None

    mask = np.ones((oh, ow), dtype=bool)
    ix0, iy0 = l - outer[0], t - outer[1]
    ix1, iy1 = r - outer[0], b - outer[1]
    mask[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
    px = arr[outer[1]:outer[3], outer[0]:outer[2]][mask]
    if len(px) < 8:
        return None
    return px.mean(axis=0)


def edge_gradient(
    img: Image.Image,
    *,
    region: tuple[float, float, float, float] | None = None,
) -> float:
    """融合区边缘梯度均值（0~1 尺度，越大越硬）。

    Args:
        img: 结果图。
        region: 归一化区域 (x, y, w, h)。None 则全图。

    Returns:
        梯度均值。参考区间：
            < 0.02  接近背景噪声（全融合/羽化重）
            0.02~0.08 中间带（半融合）
            > 0.08  硬边（贴纸感）
    """
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    if region:
        h, w = arr.shape
        x, y, rw, rh = region
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = int((x + rw) * w), int((y + rh) * h)
        arr = arr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

    if arr.size < 4:
        return 0.0

    gy = np.abs(np.diff(arr, axis=0))[:-1, :]
    gx = np.abs(np.diff(arr, axis=1))[:, :-1]
    hh = min(gx.shape[0], gy.shape[0])
    ww = min(gx.shape[1], gy.shape[1])
    mag = np.sqrt(gx[:hh, :ww] ** 2 + gy[:hh, :ww] ** 2)
    return float(mag.mean())


def laplacian_variance(img: Image.Image) -> float:
    """拉普拉斯方差：判断主体是否糊到不可辨。

    参考：> 100 清晰；50~100 略糊；< 50 严重糊（可能过度降质）。
    """
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4 * arr[1:-1, 1:-1]
        + arr[:-2, 1:-1] + arr[2:, 1:-1]
        + arr[1:-1, :-2] + arr[1:-1, 2:]
    )
    return float(lap.var())


def subject_ratio(img: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    """主体占画面面积比。"""
    W, H = img.size
    l, t, r, b = bbox
    return float((r - l) * (b - t) / max(1, W * H))


def highfreq_energy(img: Image.Image) -> float:
    """高频能量：判断「像不像牢图」。

    参考（来自四张参考图实测）：
        image.png        0.0160  (A 精贴，最清晰)
        image copy 2.png 0.0115  (C 堆叠)
        image copy.png   0.0086  (B 畸变)
        image copy 3.png 0.0070  (D 换装，大面积纯底)
    """
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy = np.abs(np.diff(arr, axis=0))[:-1, :]
    gx = np.abs(np.diff(arr, axis=1))[:, :-1]
    hh = min(gx.shape[0], gy.shape[0])
    ww = min(gx.shape[1], gy.shape[1])
    return float(np.sqrt(gx[:hh, :ww] ** 2 + gy[:hh, :ww] ** 2).mean())


def report(
    result: Image.Image,
    *,
    source_piece: Image.Image | None = None,
    piece_box: tuple[int, int, int, int] | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> dict:
    """一次性算全部指标，返回可直接落盘的 dict。"""
    out: dict = {
        "edge_gradient": round(edge_gradient(result, region=region), 5),
        "laplacian_variance": round(laplacian_variance(result), 2),
        "highfreq_energy": round(highfreq_energy(result), 5),
    }
    if source_piece is not None:
        out["color_diff_retention"] = round(
            color_diff_retention(source_piece, result, box=piece_box), 4)

    # 档位判定
    eg = out["edge_gradient"]
    if eg > 0.08:
        out["edge_kind"] = "hard"
    elif eg > 0.02:
        out["edge_kind"] = "mid"
    else:
        out["edge_kind"] = "soft"

    lv = out["laplacian_variance"]
    out["clarity"] = "clear" if lv > 100 else ("mild" if lv > 50 else "smeared")

    return out
