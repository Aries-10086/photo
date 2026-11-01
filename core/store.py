"""素材库：扫描 assets/ 目录，提供 id -> 图片的查询。

校准器阶段只需要「文件名即 id」，不做符号卡索引（那是 M1）。
但接口留好，M1 接符号卡时不用改调用方。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from core import imageio

ROOT = Path(__file__).resolve().parent.parent
CUT_DIR = ROOT / "assets" / "cut"
RAW_DIR = ROOT / "assets" / "raw"
MASK_DIR = ROOT / "assets" / "masks"

_IMG_EXT = {".png", ".webp", ".jpg", ".jpeg"}


def scan(dir_: Path = CUT_DIR) -> list[dict]:
    """列出目录下的素材。返回 [{id, name, path, size, bytes}]"""
    if not dir_.exists():
        return []
    out = []
    for p in sorted(dir_.iterdir()):
        if p.suffix.lower() not in _IMG_EXT or p.name.startswith("."):
            continue
        try:
            with open(p, "rb") as f:
                with Image.open(f) as im:
                    size = im.size
                    mode = im.mode
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "id": p.stem,
            "name": p.name,
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "w": size[0],
            "h": size[1],
            "mode": mode,
            "bytes": p.stat().st_size,
            "has_alpha": mode in ("RGBA", "LA", "P"),
        })
    return out


def resolve_asset(aid: str) -> Path | None:
    for ext in _IMG_EXT:
        p = CUT_DIR / f"{aid}{ext}"
        if p.exists():
            return p
    return None


def load_asset(aid: str, limit: int | None = 1200) -> Image.Image | None:
    p = resolve_asset(aid)
    if p is None:
        return None
    return imageio.read(p, keep_alpha=True, limit=limit)


def scan_bases(dir_: Path = ROOT) -> list[dict]:
    """列出可作为基底的图。校准器阶段：assets/raw + 项目根目录，按文件名去重。"""
    out = []
    seen: set[str] = set()
    for d in (RAW_DIR, ROOT):
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in _IMG_EXT or p.name.startswith("."):
                continue
            if p.name in seen:
                continue
            try:
                with open(p, "rb") as f:
                    with Image.open(f) as im:
                        size = im.size
            except Exception:  # noqa: BLE001
                continue
            seen.add(p.name)
            out.append({
                "id": p.stem,
                "name": p.name,
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "w": size[0],
                "h": size[1],
                "dir": "raw" if d == RAW_DIR else "root",
            })
    return out


def scan_masks() -> list[dict]:
    if not MASK_DIR.exists():
        return []
    return [{"id": p.stem, "path": str(p.relative_to(ROOT)).replace("\\", "/")}
            for p in sorted(MASK_DIR.iterdir()) if p.suffix.lower() == ".png"]


# ---------------------------------------------------------------- 配方库

RECIPE_DIR = ROOT / "recipes"
BASELINE_DIR = ROOT / "docs" / "baselines"


def list_recipes() -> list[dict]:
    out = []
    for d in (RECIPE_DIR, BASELINE_DIR):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("mem_"):
                continue
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            out.append({
                "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                "recipe_id": r.get("recipe_id", p.stem),
                "subtype": r.get("subtype"),
                "title": r.get("title") or p.stem,
            })
    return out


def read_recipe(rel_path: str) -> dict:
    p = (ROOT / rel_path).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise ValueError("路径越界")
    return json.loads(p.read_text(encoding="utf-8"))


def write_recipe(recipe: dict, rel_path: str) -> Path:
    p = (ROOT / rel_path).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise ValueError("路径越界")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
