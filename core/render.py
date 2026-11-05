"""配方执行器。

把一份 recipe 渲染成图，可选逐步落盘（每步中间产物）。
保证确定性：同配方同输入，逐像素一致。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from core import imageio
from core.ops import Ctx, apply_op
from core.validate import validate_recipe

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StepResult:
    index: int
    op: str
    ms: float
    path: Path | None = None

    def as_dict(self) -> dict:
        return {"index": self.index, "op": self.op, "ms": round(self.ms, 1),
                "path": self.path.name if self.path else None}


@dataclass
class RenderResult:
    image: Image.Image
    steps: list[StepResult] = field(default_factory=list)
    total_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"total_ms": round(self.total_ms, 1),
                "steps": [s.as_dict() for s in self.steps],
                "warnings": self.warnings}


# 执行顺序（与 validate.OP_ORDER 一致，但这里是实际排序）
_ORDER = {
    "backdrop": -5,
    "layer_split": 0,
    "face_fit": 10,
    "feature_warp": 10,
    "texture_wrap": 10,
    "sticker": 10,
    "grade": 15,
    "degrade": 20,
    "label": 30,
}


def sort_ops(ops: list[dict]) -> list[dict]:
    """稳定排序：同秩保持声明顺序（z 序在 sticker 内部处理）。"""
    return sorted(ops, key=lambda o: _ORDER.get(o.get("op", ""), 50))


def load_base(base_path: str | Path) -> Image.Image:
    return imageio.read(base_path)


def load_assets(recipe: dict, asset_dir: Path | None = None) -> dict[str, Image.Image]:
    """按配方里引用的 asset_id 加载素材。asset_id 直接在 asset_dir 下找同名文件。

    真正的素材库索引留到 M1；校准器只需要「id 即文件名（去扩展名）」。
    """
    d = asset_dir or (ROOT / "assets" / "cut")
    out: dict[str, Image.Image] = {}
    for op in recipe.get("ops", []):
        aid = op.get("asset_id")
        if not aid or aid in out:
            continue
        for ext in (".png", ".webp", ".jpg", ".jpeg"):
            p = d / f"{aid}{ext}"
            if p.exists():
                out[aid] = imageio.read(p, keep_alpha=True, limit=None)
                break
    return out


def render(
    recipe: dict,
    *,
    base_path: str | Path | None = None,
    base_image: Image.Image | None = None,
    assets: dict[str, Image.Image] | None = None,
    save_steps_dir: Path | None = None,
    strict: bool = True,
) -> RenderResult:
    """执行一份配方。

    Args:
        recipe: 配方 dict。
        base_path: 基底图路径（与 base_image 二选一）。
        base_image: 已加载的基底。
        assets: asset_id -> Image。为 None 时自动从 assets/cut 加载。
        save_steps_dir: 非 None 时每步落盘为 PNG。
        strict: True 时校验失败抛错；False 时只记警告继续。
    """
    v = validate_recipe(recipe, draft=True)
    warnings = list(v.warnings)
    if not v.ok:
        if strict:
            raise ValueError(v.report())
        warnings.extend(v.errors)
    rec = v.data or recipe

    if base_image is None:
        if base_path is None:
            raise ValueError("必须提供 base_path 或 base_image")
        base_image = load_base(base_path)

    base = base_image.convert("RGBA")
    bt = rec.get("base_transform") or {}
    if bt.get("flip_x") or bt.get("flip_y"):
        base = base.transpose(
            Image.ROTATE_180 if (bt.get("flip_x") and bt.get("flip_y"))
            else (Image.FLIP_LEFT_RIGHT if bt.get("flip_x") else Image.FLIP_TOP_BOTTOM))
    if bt.get("rotation_deg"):
        base = base.rotate(float(bt["rotation_deg"]), resample=Image.BICUBIC, expand=True)

    ctx = Ctx(
        assets=assets if assets is not None else load_assets(rec),
        base_long=max(base.size),
    )

    # ---------------------------------------------------------- A 类前景遮挡
    # 关键机制：先把前景遮挡层（头发/手）从基底中抠出来存起来，
    # 等所有内容 op 执行完后，再把它贴回最上层。
    # 这样人脸被贴入后，头发会自然压住它的边缘 —— 接缝消失，零成本。
    front_layer: Image.Image | None = None
    layers_cfg = rec.get("layers") or {}
    if layers_cfg.get("foreground"):
        try:
            from core.layers import _mul_alpha, _from_np, _to_np, load_mask
            fm = load_mask(layers_cfg["foreground"], size=base.size)
            if fm.max() > 8:
                # 前景层 = 基底 alpha ∩ 掩码
                fa = _to_np(base)
                fa[:, :, 3] = _mul_alpha(fa[:, :, 3], fm)
                front_layer = _from_np(fa)
                # 从基底中挖掉前景，避免它在新图里露出两份
                inv = (255 - fm).astype(np.uint8)
                ba = _to_np(base)
                ba[:, :, 3] = _mul_alpha(ba[:, :, 3], inv)
                base = _from_np(ba)
        except FileNotFoundError as e:
            warnings.append(f"前景掩码读取失败，忽略分层: {e}")

    steps: list[StepResult] = []
    canvas = base
    t_all = time.perf_counter()

    for i, op in enumerate(sort_ops(rec.get("ops", []))):
        t0 = time.perf_counter()
        canvas = apply_op(canvas, op, ctx)
        ms = (time.perf_counter() - t0) * 1000
        p = None
        if save_steps_dir is not None:
            save_steps_dir.mkdir(parents=True, exist_ok=True)
            p = save_steps_dir / f"step{i:02d}_{op.get('op','?')}.png"
            imageio.write(canvas, p)
        steps.append(StepResult(i, op.get("op", "?"), ms, p))

    # 前景遮挡层盖在最上
    if front_layer is not None:
        t0 = time.perf_counter()
        canvas = Image.alpha_composite(canvas, front_layer)
        ms = (time.perf_counter() - t0) * 1000
        p = None
        if save_steps_dir is not None:
            p = save_steps_dir / f"step{len(steps):02d}_front_restore.png"
            imageio.write(canvas, p)
        steps.append(StepResult(len(steps), "front_restore", ms, p))

    total = (time.perf_counter() - t_all) * 1000
    return RenderResult(canvas, steps, total, warnings)


def load_recipe(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_recipe(recipe: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
