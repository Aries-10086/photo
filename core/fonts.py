"""字体解析。

配方的 label.font 允许三种写法：
  1. 键名（如 "msyh" / "msyhbd" / "simhei"）→ 从 WINDIR/Fonts 解析
  2. 绝对路径 → 直接使用
  3. 相对路径 → 相对项目根目录

Windows 上中文字体是必需的，缺字时 PIL 会画成豆腐块，所以要显式报错而不是静默降级。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

# 键名 -> 候选文件名（按优先级）
FONT_MAP: dict[str, list[str]] = {
    "msyh": ["msyh.ttc"],          # 微软雅黑
    "msyhbd": ["msyhbd.ttc"],      # 微软雅黑 粗体
    "msyhl": ["msyhl.ttc"],        # 微软雅黑 细体
    "simhei": ["simhei.ttf"],      # 黑体
    "simsun": ["simsun.ttc"],      # 宋体
    "simkai": ["simkai.ttf"],      # 楷体
    "deng": ["Deng.ttf"],          # 等线
    "dengbd": ["Dengb.ttf"],       # 等线 粗体
    "arial": ["arial.ttf"],
    "impact": ["impact.ttf"],
}


class FontError(Exception):
    pass


def resolve(font: str | None) -> Path:
    """把 font 描述解析成实际文件路径。"""
    if not font:
        font = "msyh"

    p = Path(font)
    if p.is_absolute():
        if not p.exists():
            raise FontError(f"字体文件不存在: {p}")
        return p

    # 键名
    if font in FONT_MAP:
        for name in FONT_MAP[font]:
            cand = FONT_DIR / name
            if cand.exists():
                return cand
        raise FontError(
            f"字体键名 {font!r} 未找到（候选: {FONT_MAP[font]}）。"
            f"请检查 {FONT_DIR}，或改用绝对路径。")

    # 相对路径
    cand = ROOT / font
    if cand.exists():
        return cand

    raise FontError(
        f"无法解析字体 {font!r}。可用键名: {sorted(FONT_MAP)}，或传绝对路径。")


@lru_cache(maxsize=64)
def load(font: str | None, size: int) -> ImageFont.FreeTypeFont:
    """加载指定字号的字体对象。结果缓存（同字号复用）。"""
    path = resolve(font)
    try:
        return ImageFont.truetype(str(path), size)
    except OSError as e:
        raise FontError(f"加载字体失败 {path.name} @ {size}px: {e}") from e


def available() -> dict[str, str]:
    """列出本机可用的键名 -> 路径，供前端下拉框使用。"""
    out: dict[str, str] = {}
    for key, names in FONT_MAP.items():
        for n in names:
            cand = FONT_DIR / n
            if cand.exists():
                out[key] = str(cand)
                break
    return out


def has_cjk(font: str | None) -> bool:
    """粗略判断字体是否可能包含中日韩字形（按文件名推断）。"""
    p = resolve(font)
    name = p.name.lower()
    return any(k in name for k in ("msyh", "simhei", "simsun", "simkai", "deng",
                                   "yahei", "hei", "song", "kai", "noto", "sourcehan"))
