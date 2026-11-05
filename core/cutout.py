"""抠图（背景移除）。

模型：u2netp（轻量 ONNX，4.4 MB）
    选它而不是 u2net(167MB) 的原因：
      - 网络只有 0.02 MB/s，下 167MB 要 2.3 小时，不可接受
      - u2netp 在「主体清晰的图」上质量够用
      - 内存占用小（约 40MB），符合 E-1 单模型驻留约束

接口设计：
    cutout(img) -> RGBA  主体保留，背景透明

    内部会：
      1. 缩放到 320² 推理（u2netp 的输入尺寸）
      2. sigmoid 得 alpha 掩码
      3. 双三次插值回原尺寸
      4. 边缘做轻微收缩，去掉残留背景色

不负责：
    · 发丝级精修（PRD 明确说精细度是负分项）
    · 多主体分离
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

DEFAULT_MODEL = "u2netp.onnx"
INPUT_SIZE = 320

_session = None
_session_name = None


class ModelNotAvailable(Exception):
    """模型文件不存在。附带下载提示。"""


def model_path(name: str = DEFAULT_MODEL) -> Path:
    return MODEL_DIR / name


def available() -> bool:
    return model_path().exists()


def _get_session(name: str = DEFAULT_MODEL):
    """懒加载 ONNX 会话。同一进程内复用，用完可显式释放。"""
    global _session, _session_name

    if _session is not None and _session_name == name:
        return _session

    p = model_path(name)
    if not p.exists():
        raise ModelNotAvailable(
            f"抠图模型不存在: {p}\n"
            f"请运行: .venv\\Scripts\\python.exe tools\\fetch_model.py")

    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = min(8, (__import__("os").cpu_count() or 4))
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _session = ort.InferenceSession(str(p), opt, providers=["CPUExecutionProvider"])
    _session_name = name
    return _session


def release() -> None:
    """释放会话（E-1 单模型驻留约束要求用完后调用）。"""
    global _session, _session_name
    _session = None
    _session_name = None
    import gc
    gc.collect()


def _preprocess(img: Image.Image, size: int) -> np.ndarray:
    """RGB -> (1, 3, size, size) float32，ImageNet 归一化。"""
    im = img.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    a = (a - mean) / std
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def cutout(
    img: Image.Image,
    *,
    model: str = DEFAULT_MODEL,
    erode_px: int = 1,
    feather_px: int = 1,
    threshold: float = 0.5,
) -> Image.Image:
    """抠出主体，返回 RGBA。

    Args:
        img: 输入图。
        model: 模型文件名。
        erode_px: alpha 边缘收缩像素数，去掉残留背景色（0 = 不收缩）。
        feather_px: 最终 alpha 羽化，让边缘不那么锯齿。
        threshold: 二值化阈值。>0.5 抠得更紧（更少残留），<0.5 保留更多。

    Returns:
        RGBA 图，尺寸与输入一致。
    """
    sess = _get_session(model)
    W, H = img.size

    x = _preprocess(img, INPUT_SIZE)
    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: x})
    pred = _sigmoid(outs[0][0, 0])

    # 归一化到 0~1
    lo, hi = float(pred.min()), float(pred.max())
    if hi - lo > 1e-6:
        pred = (pred - lo) / (hi - lo)

    # 阈值调整对比度：让 >threshold 更接近 1，<threshold 更接近 0
    if abs(threshold - 0.5) > 1e-6:
        pred = np.clip((pred - threshold) / max(0.5, 1e-3) + 0.5, 0.0, 1.0)

    alpha = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, "L").resize((W, H), Image.BICUBIC)

    if erode_px > 0:
        # 用 MinFilter 收缩，等价于 alpha 腐蚀
        k = erode_px * 2 + 1
        alpha_img = alpha_img.filter(ImageFilter.MinFilter(min(9, k)))

    if feather_px > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(feather_px))

    out = img.convert("RGBA")
    out.putalpha(alpha_img)
    return out


def cutout_batch(
    imgs: list[Image.Image],
    **kw,
) -> list[Image.Image]:
    """批量抠图。共用一次会话加载。"""
    return [cutout(im, **kw) for im in imgs]


def benchmark(img: Image.Image, model: str = DEFAULT_MODEL) -> dict:
    """实测速度与内存。"""
    import os

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss / 2**20
    except ImportError:
        proc = None
        rss0 = 0.0

    t0 = time.perf_counter()
    _get_session(model)
    load_ms = (time.perf_counter() - t0) * 1000

    rss1 = proc.memory_info().rss / 2**20 if proc else 0.0

    t0 = time.perf_counter()
    out = cutout(img, model=model)
    infer_ms = (time.perf_counter() - t0) * 1000

    rss2 = proc.memory_info().rss / 2**20 if proc else 0.0

    a = np.asarray(out.getchannel("A"))
    coverage = float((a > 128).mean())

    return {
        "model": model,
        "input": f"{img.width}x{img.height}",
        "load_ms": round(load_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "rss_before_mb": round(rss0, 1),
        "rss_after_load_mb": round(rss1, 1),
        "rss_after_infer_mb": round(rss2, 1),
        "rss_delta_mb": round(rss2 - rss0, 1),
        "coverage_pct": round(coverage * 100, 1),
    }
