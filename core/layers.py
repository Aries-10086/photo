"""前景遮挡分层（M3 核心）。

这是 A 类「精贴」成立的根本机制：
    写实脸能塞进二次元头部而不露接缝，**不是因为羽化好**，
    而是因为**头发/手/道具被保留在最上层，压住了人脸的边缘**。

遮挡比任何混合算法都有效，而且零成本。这是 PRD P-2 的落实。

三层结构：
    back   背景（最底）
    content 被替换区（人脸贴上来）
    front  前景遮挡层（头发/手，压在最上）

实现路线（两级）：
    1. 自动：用抠图模型分割主体 → 主 alpha 内部的「上边缘区」当候选前景
    2. 手动：用户涂抹掩码，覆盖自动结果

当前实现自动部分只做「主体轮廓 + 顶部区域」的启发式，
因为精准头发分割需要专门的 matting 模型（网络条件不允许下 167MB 模型）。
手动涂抹是主要路径 —— PRD 里 M3 已明确标为「半自动」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from core import imageio

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Layers:
    """分层结果。"""

    back: Image.Image
    """背景层（RGBA）"""

    content: Image.Image
    """被替换区（RGBA），即主体去掉前景遮挡后剩下的部分"""

    front: Image.Image
    """前景遮挡层（RGBA），压在最上"""

    front_mask: np.ndarray
    """前景掩码 (H, W) uint8，0~255"""

    @property
    def has_front(self) -> bool:
        return self.front_mask.max() > 8

    def summary(self) -> dict:
        fg = float((self.front_mask > 128).mean()) * 100
        return {
            "size": self.back.size,
            "front_coverage_pct": round(fg, 2),
            "has_front": self.has_front,
        }


def _to_np(img: Image.Image) -> np.ndarray:
    """转 numpy 数组（可写副本）。"""
    return np.array(img.convert("RGBA"), dtype=np.uint8, copy=True)


def _from_np(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, "RGBA")


def _mul_alpha(alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """alpha 与掩码相乘（都是 0~255 语义），返回 uint8。

    注意：mask 可能是 bool（0/1），必须先乘 255 转成 0/255 语义。
    这个 bug 踩过一次：bool*255//255 = 1，掩码整体缩小 255 倍，
    数值不为 0 所以不报错，但掩码实际失效。
    """
    a = np.asarray(alpha)
    if a.ndim != 2:
        a = a[:, :, 0] if a.ndim == 3 else a.reshape(a.shape[:2])

    m = np.asarray(mask)
    if m.ndim != 2:
        m = m[:, :, 0] if m.ndim == 3 else m.reshape(m.shape[:2])

    if m.shape != a.shape:
        m = np.asarray(Image.fromarray(m.astype(np.uint8), "L").resize(
            (a.shape[1], a.shape[0]), Image.LANCZOS))

    if m.dtype == np.bool_:
        m16 = m.astype(np.uint16) * 255
    else:
        m16 = np.clip(m.astype(np.uint16), 0, 255)

    a16 = np.clip(a.astype(np.uint16), 0, 255)
    return ((a16 * m16) // 255).astype(np.uint8)


def apply_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """把掩码乘到 alpha 上。mask 形状 (H, W)，0~255。"""
    a = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
    if mask.shape[:2] != a.shape[:2]:
        m = Image.fromarray(mask.astype(np.uint8), "L").resize(
            (a.shape[1], a.shape[0]), Image.LANCZOS)
        mask = np.asarray(m)
    a[:, :, 3] = _mul_alpha(a[:, :, 3], mask)
    return _from_np(a)


def subtract(src: Image.Image, cut: Image.Image) -> Image.Image:
    """src 减去 cut（alpha 相减），得到 src 剩余部分。"""
    a = _to_np(src)
    b = _to_np(cut)
    if a.shape != b.shape:
        cut2 = cut.resize(src.size, Image.LANCZOS)
        b = _to_np(cut2)
    out = a.copy()
    out[:, :, 3] = np.clip(
        a[:, :, 3].astype(np.int16) - b[:, :, 3].astype(np.int16), 0, 255).astype(np.uint8)
    return _from_np(out)


def auto_front_mask(
    base: Image.Image,
    *,
    top_ratio: float = 0.45,
    softness: int = 2,
    shrink: float = 0.02,
) -> np.ndarray:
    """启发式：主体轮廓的内上部分当作前景遮挡候选。

    这不是真正的头发分割，只是一个**起点**。
    逻辑：头发/帽饰通常位于头部上缘，且是主体的一部分。

    Args:
        base: 基底图（应已抠出主体，带 alpha）。
        top_ratio: 从上往下占主体高度的多少比例视为「上缘区」。
        softness: 边界平滑像素。
        shrink: 向内收缩比例，避免把轮廓外缘当成前景。

    Returns:
        (H, W) uint8 掩码。
    """
    W, H = base.size
    a = np.asarray(base.convert("RGBA").getchannel("A"))
    body = a > 128

    if not body.any():
        return np.zeros((H, W), dtype=np.uint8)

    ys, xs = np.where(body)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    bh = max(1, y1 - y0)

    cut_y = int(y0 + bh * top_ratio)

    mask = np.zeros((H, W), dtype=np.uint8)
    top_region = body.copy()
    top_region[cut_y:, :] = False
    mask[top_region] = 255

    # 纵向渐隐，避免硬切
    if cut_y > y0:
        band = np.linspace(255, 0, min(60, cut_y - y0))
        for i, v in enumerate(band):
            yy = cut_y - len(band) + i
            if 0 <= yy < H:
                row = body[yy]
                mask[yy, row] = np.minimum(mask[yy, row], v).astype(np.uint8)

    # 向内收缩，去掉轮廓外缘的抗锯齿边
    if shrink > 0:
        sw = max(1, int(min(x1 - x0 + 1, bh) * shrink))
        m = Image.fromarray(mask, "L").filter(ImageFilter.MinFilter(sw * 2 + 1))
        mask = np.asarray(m)

    if softness > 0:
        m = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(softness))
        mask = np.asarray(m)

    return mask


def split(
    base: Image.Image,
    *,
    front_mask: np.ndarray | Image.Image | None = None,
    auto: bool = True,
    **auto_kw,
) -> Layers:
    """把基底拆成三层。

    Args:
        base: 基底图（RGBA）。
        front_mask: 手工前景掩码。给了就用它，覆盖 auto。
        auto: 是否在用不到手工掩码时自动生成。
        **auto_kw: 传给 auto_front_mask。

    Returns:
        Layers，含三层与掩码。
    """
    img = base.convert("RGBA")
    W, H = img.size
    arr = _to_np(img)

    # 背景：假设输入已抠图（alpha 有透明区）。
    # 若全不透明，背景为空。
    alpha = arr[:, :, 3]
    body_mask = (alpha > 128)

    back = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not body_mask.all():
        # 原图本身有透明区 → back 就是透明，content = 主体
        pass
    # back 留空（透明），content 与 front 分割主体

    if front_mask is None and auto:
        fm = auto_front_mask(img, **auto_kw)
    elif isinstance(front_mask, Image.Image):
        fm = np.asarray(front_mask.convert("L").resize((W, H), Image.LANCZOS))
    elif front_mask is not None:
        arr_m = np.asarray(front_mask)
        if arr_m.shape[:2] != (H, W):
            fm = np.asarray(
                Image.fromarray(arr_m.astype(np.uint8), "L").resize((W, H), Image.LANCZOS))
        else:
            fm = arr_m.astype(np.uint8)
    else:
        fm = np.zeros((H, W), dtype=np.uint8)

    # 前景不能超出主体
    fm = _mul_alpha(fm, body_mask)

    inverse = (255 - fm).astype(np.uint8)

    # content = 原图 alpha 与 inverse 相乘（直接构造，避免 putalpha 的类型坑）
    ca = _to_np(img)
    ca[:, :, 3] = _mul_alpha(ca[:, :, 3], inverse)
    content = _from_np(ca)

    # front = 原图 alpha 与 fm 相乘
    fa = _to_np(img)
    fa[:, :, 3] = fm
    front = _from_np(fa)

    return Layers(back=back, content=content, front=front, front_mask=fm)


# ---------------------------------------------------------------- 文件级接口


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """读掩码文件（PNG 灰度/L/RGBA 均可）。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"掩码不存在: {p}")
    img = imageio.read(p, keep_alpha=False, limit=None).convert("L")
    if size and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    imageio.write(Image.fromarray(mask.astype(np.uint8), "L").convert("RGBA"), p)
    return p


def coverage(mask: np.ndarray) -> float:
    """掩码覆盖率（0~1）。用于 M3 的「遮挡 ≥ 30%」验收。"""
    return float((mask > 128).mean())
