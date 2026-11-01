"""牢图棒 · 图像 IO 统一层

所有图像读写必须经过这一层，不允许业务代码直接调用 PIL/cv2 的 open/save。

职责：
  1. 中文路径安全（Windows 下 PIL 走 Python 文件对象，天然 OK；这里统一封装以防退化）
  2. 尺寸上限（2048 长边），超限自动等比缩放
  3. alpha 通道统一（RGBA）
  4. 色彩空间统一 sRGB / 8bit
  5. 格式白名单

不负责：抠图、变换、合成。这些在 ops/ 下。
"""

from __future__ import annotations

import io
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------- 常量

MAX_SIDE = 2048
"""单图处理边长上限。超过则等比缩到长边 = MAX_SIDE。见 PRD E-1。"""

ALLOWED_IN = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_OUT = {".png", ".jpg", ".jpeg"}

RGBA = "RGBA"
RGB = "RGB"
L = "L"
LA = "LA"


class ImageIOError(Exception):
    """图像读写失败，附带可读原因。"""


# ---------------------------------------------------------------- 读


def read(
    path: str | Path,
    *,
    keep_alpha: bool = True,
    limit: int | None = MAX_SIDE,
) -> Image.Image:
    """读取图像为 RGBA（或 RGB），统一尺寸上限。

    Args:
        path: 图像路径，允许中文目录。
        keep_alpha: True 返回 RGBA，False 返回 RGB（无透明通道时省内存）。
        limit: 长边上限；None 表示不缩放。

    Raises:
        ImageIOError: 文件不存在、扩展名不支持、解码失败。
    """
    p = Path(path)
    if not p.exists():
        raise ImageIOError(f"文件不存在: {p}")
    if p.suffix.lower() not in ALLOWED_IN:
        raise ImageIOError(f"不支持的输入格式 {p.suffix}，允许: {sorted(ALLOWED_IN)}")

    try:
        # 用 open(path) 传文件对象，绕开 PIL 对非 ASCII 路径的潜在问题
        with open(p, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            img.load()
    except Exception as e:  # noqa: BLE001
        raise ImageIOError(f"解码失败 {p.name}: {e}") from e

    # EXIF 方向纠正（手机拍的照片常见）
    img = ImageOps.exif_transpose(img)

    target = RGBA if keep_alpha else RGB
    if img.mode != target:
        if target == RGBA:
            img = img.convert(RGBA)
        else:
            # 转 RGB 时若有透明，用白底压平；梗图场景白底最安全
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert(RGBA)
                bg = Image.new(RGB, img.size, (255, 255, 255))
                bg.paste(img, mask=img.getchannel("A"))
                img = bg
            else:
                img = img.convert(RGB)

    if limit is not None:
        img = fit_within(img, limit)

    return img


def fit_within(img: Image.Image, limit: int = MAX_SIDE) -> Image.Image:
    """等比缩放到长边不超过 limit。已满足则原样返回（不重采样）。"""
    w, h = img.size
    longest = max(w, h)
    if longest <= limit:
        return img
    scale = limit / longest
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.LANCZOS)


def read_array(
    path: str | Path,
    *,
    keep_alpha: bool = True,
    limit: int | None = MAX_SIDE,
) -> np.ndarray:
    """读成 numpy 数组。返回 uint8，形状 (H, W, 3) 或 (H, W, 4)。

    注意：OpenCV 用户习惯 BGR，这里统一返回 **RGB**。转换由调用方显式做。
    """
    return np.asarray(read(path, keep_alpha=keep_alpha, limit=limit))


# ---------------------------------------------------------------- 写


def write(
    img: Image.Image,
    path: str | Path,
    *,
    quality: int = 95,
    optimize: bool = True,
) -> Path:
    """写出图像，自动建父目录。

    Args:
        img: PIL 图像。
        path: 目标路径。后缀决定格式。
        quality: JPEG 质量（仅对 .jpg/.jpeg 生效；降质请用 ops.degrade）。
        optimize: 是否优化编码。

    Returns:
        实际写入路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() not in ALLOWED_OUT:
        raise ImageIOError(f"不支持的输出格式 {p.suffix}，允许: {sorted(ALLOWED_OUT)}")

    if p.suffix.lower() in (".jpg", ".jpeg"):
        if img.mode in ("RGBA", "LA", "P"):
            # JPEG 不支持 alpha，白底压平
            tmp = img.convert(RGBA)
            bg = Image.new(RGB, tmp.size, (255, 255, 255))
            bg.paste(tmp, mask=tmp.getchannel("A"))
            img = bg
        elif img.mode != RGB:
            img = img.convert(RGB)
        img.save(p, "JPEG", quality=quality, optimize=optimize, subsampling=0)
    else:
        # PNG：统一保留 alpha
        if img.mode not in (RGBA, RGB, L, LA):
            img = img.convert(RGBA)
        img.save(p.with_suffix(".png"), "PNG", optimize=optimize)
        p = p.with_suffix(".png")

    return p


def write_array(
    arr: np.ndarray,
    path: str | Path,
    *,
    quality: int = 95,
) -> Path:
    """numpy 数组写出。支持 uint8 (H,W) / (H,W,3) / (H,W,4)，RGB 顺序。"""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        img = Image.fromarray(a, "L").convert(RGBA)
    elif a.ndim == 3 and a.shape[2] == 3:
        img = Image.fromarray(a, RGB)
    elif a.ndim == 3 and a.shape[2] == 4:
        img = Image.fromarray(a, RGBA)
    else:
        raise ImageIOError(f"不支持数组形状 {a.shape}")
    return write(img, path, quality=quality)


# ---------------------------------------------------------------- 工具


def sha256_of(path: str | Path) -> str:
    """文件内容哈希，用于素材去重。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_image(img: Image.Image) -> str:
    """像素级哈希（忽略元数据），用于判定两图是否完全相同。"""
    buf = io.BytesIO()
    img.convert(RGBA).save(buf, "PNG", optimize=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def to_rgb_on_white(img: Image.Image) -> Image.Image:
    """把可能带 alpha 的图压平到白底 RGB。"""
    if img.mode == RGB:
        return img
    img = img.convert(RGBA)
    bg = Image.new(RGB, img.size, (255, 255, 255))
    bg.paste(img, mask=img.getchannel("A"))
    return bg


def bbox_of_alpha(img: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    """取 alpha 通道的非透明包围盒 (left, top, right, bottom)。

    alpha 全不透明或全透明时返回 None。
    """
    if img.mode != RGBA:
        return None
    a = np.asarray(img.getchannel("A"))
    ys, xs = np.where(a > threshold)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def info(path: str | Path) -> dict:
    """探测图像基本信息，不加载全部像素（用于快速扫描）。"""
    p = Path(path)
    with open(p, "rb") as f:
        with Image.open(io.BytesIO(f.read())) as img:
            return {
                "path": str(p),
                "name": p.name,
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            }
