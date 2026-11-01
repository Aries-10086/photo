"""基线校准器后端。

只做三件事：
  1. 列出可用的基底 / 素材 / 参考图
  2. 接收配方 JSON，渲染并返回 PNG（base64）
  3. 把校准结果保存成基线 JSON

单用户本地服务，不做鉴权、不做会话。
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from core import imageio, store
from core.render import render as render_recipe
from core.validate import validate_recipe

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="牢图棒 · 基线校准器", version="0.1.0")


# ---------------------------------------------------------------- 模型


class RenderReq(BaseModel):
    recipe: dict
    base: str | None = None          # 基底路径（相对项目根）
    max_side: int = 900              # 预览尺寸上限，越小越快
    save_steps: bool = False


class SaveReq(BaseModel):
    recipe: dict
    filename: str                    # 例如 "B_distort.json"
    note: str = ""


# ---------------------------------------------------------------- 静态


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


# ---------------------------------------------------------------- 数据

@app.get("/api/env")
def env() -> dict:
    """一次返回所有下拉框需要的数据。"""
    return {
        "bases": store.scan_bases(),
        "assets": store.scan(store.CUT_DIR),
        "masks": store.scan_masks(),
        "recipes": store.list_recipes(),
        "root": str(ROOT),
    }


@app.get("/api/asset/{aid}/thumb")
def asset_thumb(aid: str) -> JSONResponse:
    img = store.load_asset(aid, limit=160)
    if img is None:
        raise HTTPException(404, f"素材不存在: {aid}")
    return JSONResponse({"png": _b64(img)})


@app.get("/api/file")
def get_file(path: str, max_side: int = 1200) -> JSONResponse:
    """按相对路径读取图片，返回 base64 PNG（用于显示参考图/基底）。"""
    p = (ROOT / path).resolve()
    if ROOT not in p.parents:
        raise HTTPException(400, "路径越界")
    if not p.exists():
        raise HTTPException(404, f"文件不存在: {path}")
    img = imageio.read(p, keep_alpha=True, limit=max_side)
    return JSONResponse({"png": _b64(img), "w": img.width, "h": img.height})


# ---------------------------------------------------------------- 渲染

@app.post("/api/render")
def do_render(req: RenderReq) -> JSONResponse:
    t0 = time.perf_counter()

    # 基底
    if req.base:
        bp = (ROOT / req.base).resolve()
        if ROOT not in bp.parents:
            raise HTTPException(400, "路径越界")
        if not bp.exists():
            raise HTTPException(404, f"基底不存在: {req.base}")
        base_img = imageio.read(bp, keep_alpha=True, limit=req.max_side)
    else:
        base_img = Image.new("RGBA", (req.max_side, req.max_side), (240, 240, 240, 255))

    # 素材：全部加载（数量少，一次性加载更简单）
    assets = {}
    for a in store.scan():
        img = store.load_asset(a["id"], limit=req.max_side)
        if img is not None:
            assets[a["id"]] = img

    try:
        steps_dir = (ROOT / "tmp" / "steps") if req.save_steps else None
        r = render_recipe(req.recipe, base_image=base_img, assets=assets,
                          save_steps_dir=steps_dir, strict=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e

    ms = (time.perf_counter() - t0) * 1000
    # 渲染用草稿模式：编辑中的配方允许空 ops / degrade 未开
    v = validate_recipe(req.recipe, draft=True)

    return JSONResponse({
        "png": _b64(r.image),
        "w": r.image.width,
        "h": r.image.height,
        "ms": round(ms, 0),
        "steps": [s.as_dict() for s in r.steps],
        "warnings": r.warnings,
        "valid": v.ok,
        "errors": v.errors[:6],
    })


@app.post("/api/validate")
def do_validate(req: RenderReq) -> dict:
    """草稿态校验（编辑过程中允许不完整）。"""
    v = validate_recipe(req.recipe, draft=True)
    return {
        "valid": v.ok,
        "errors": v.errors,
        "warnings": v.warnings,
        "clamped": v.data,
    }


# ---------------------------------------------------------------- 保存

@app.post("/api/save")
def save(req: SaveReq) -> dict:
    # 保存用严格模式：成品必须完整
    v = validate_recipe(req.recipe, draft=False)
    if not v.ok:
        raise HTTPException(400, "配方不合法，无法保存:\n" + "\n".join(v.errors[:6]))

    name = req.filename.strip()
    if not name.endswith(".json"):
        name += ".json"
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "文件名不合法")

    rel = f"docs/baselines/{name}"
    p = store.write_recipe(v.data or req.recipe, rel)

    # 同时写一份到 recipes/（供 M1 直接复用）
    store.write_recipe(v.data or req.recipe, f"recipes/{name}")

    return {"ok": True, "path": str(p.relative_to(ROOT)).replace("\\", "/")}


@app.get("/api/recipe")
def get_recipe(path: str) -> dict:
    try:
        return store.read_recipe(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e


# ---------------------------------------------------------------- 工具


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, "PNG", optimize=False, compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "root": str(ROOT)}
