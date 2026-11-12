"""口味画像（PRD FR-6.5）。

把记忆渲染成一张人能看懂、能发出去的卡片。
纯本地生成，零网络请求。

设计意图（PRD 第十节）：
    「没人会晒自己的审美画像，但会晒自己的梗图口味。」
    这张卡片是记忆系统的激励——用户为了画像变丰富，会主动多喂素材。

用法：
    from core import portrait
    img = portrait.render()
    portrait.save(img)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core import fonts
from core.memory import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

SUBTYPE_CN = {
    "precise": "A 精贴",
    "distort": "B 畸变",
    "stack": "C 堆叠",
    "rewear": "D 换装",
}

# 深色主题，与校准器一致
BG = (20, 21, 26)
PANEL = (28, 30, 38)
PANEL2 = (35, 38, 49)
LINE = (46, 50, 64)
TEXT = (232, 234, 240)
DIM = (139, 145, 163)
ACCENT = (110, 168, 254)
OK = (94, 201, 138)
WARN = (232, 180, 74)

W, H = 900, 620


def _font(key: str, size: int):
    try:
        return fonts.load(key, size)
    except Exception:  # noqa: BLE001
        return fonts.load("msyh", size)


def _text(d: ImageDraw.ImageDraw, xy, s, font, fill=TEXT):
    d.text(xy, str(s), font=font, fill=fill)


def _panel(d: ImageDraw.ImageDraw, box, title: str, f_title):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=10, fill=PANEL, outline=LINE, width=1)
    if title:
        _text(d, (x0 + 14, y0 + 11), title, f_title, DIM)


def _bar(d: ImageDraw.ImageDraw, box, ratio: float, color):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=3, fill=PANEL2)
    w = int((x1 - x0) * max(0.0, min(1.0, ratio)))
    if w > 0:
        d.rounded_rectangle((x0, y0, x0 + w, y1), radius=3, fill=color)


def render(store: MemoryStore | None = None) -> Image.Image:
    """渲染口味画像卡片。"""
    if store is None:
        from core import memory
        store = memory.store()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title = _font("msyhbd", 24)
    f_sub = _font("msyh", 13)
    f_lbl = _font("msyh", 12)
    f_big = _font("msyhbd", 30)
    f_val = _font("msyhbd", 16)
    f_small = _font("msyh", 11)

    st = store.stats()
    by = st.get("by_type", {})

    # ---------------- 头部 ----------------
    _text(d, (36, 28), "我的梗图口味", f_title)
    _text(d, (36, 62), "牢图棒 · 本地记忆画像 · 全部数据在你机器上",
          f_sub, DIM)

    # ---------------- 四宫格数字 ----------------
    stats = [
        ("活跃记忆", st.get("active", 0), ACCENT),
        ("成功组合", by.get("combo", 0), OK),
        ("手感参数", by.get("hand", 0), WARN),
        ("平均置信度", f"{st.get('avg_confidence', 0) * 100:.0f}%", TEXT),
    ]
    bx, by0, bw, bh = 36, 96, (W - 36 * 2 - 12 * 3) // 4, 84
    for i, (label, val, color) in enumerate(stats):
        x = bx + i * (bw + 12)
        d.rounded_rectangle((x, by0, x + bw, by0 + bh), radius=9,
                            fill=PANEL, outline=LINE)
        _text(d, (x + 14, by0 + 14), val, f_big, color)
        _text(d, (x + 14, by0 + 54), label, f_lbl, DIM)

    # ---------------- 左：偏好分布 ----------------
    y = 200
    _panel(d, (36, y, 500, y + 200), "常用子类型", f_lbl)
    subs = sorted(store.by_type("subtype_pref"),
                  key=lambda m: -float(m.get("confidence", 0)))
    if subs:
        yy = y + 42
        for m in subs[:4]:
            v = m.get("value") or {}
            name = SUBTYPE_CN.get(v.get("subtype"), v.get("subtype", "?"))
            ratio = float(v.get("ratio", 0))
            _text(d, (52, yy), name, f_val)
            _text(d, (150, yy + 3), f"{ratio * 100:.0f}%", f_small, DIM)
            _bar(d, (200, yy + 6, 480, yy + 12), ratio, ACCENT)
            yy += 38
    else:
        _text(d, (52, y + 60), "还没有足够数据", f_sub, DIM)

    # ---------------- 右：可下手基底 ----------------
    y2 = 200
    _panel(d, (516, y2, W - 36, y2 + 200), "常用基底", f_lbl)
    bases = store.by_type("base")
    bases.sort(key=lambda m: -int((m.get("value") or {}).get("use_count", 0)))
    if bases:
        yy = y2 + 42
        maxn = max(1, int((bases[0].get("value") or {}).get("use_count", 1)))
        for m in bases[:4]:
            v = m.get("value") or {}
            bid = str(v.get("base_id", "?"))[:16]
            n = int(v.get("use_count", 0))
            _text(d, (532, yy), bid, f_val)
            _text(d, (532 + 150, yy + 3), f"{n} 次", f_small, DIM)
            _bar(d, (532 + 200, yy + 6, W - 52, yy + 12), n / maxn, OK)
            yy += 38
    else:
        _text(d, (532, y2 + 60), "还没有足够数据", f_sub, DIM)

    # ---------------- 手感签名 ----------------
    y3 = 416
    _panel(d, (36, y3, 500, y3 + 128), "手感签名", f_lbl)
    hands = sorted(store.by_type("hand"),
                   key=lambda m: -float(m.get("confidence", 0)))
    if hands:
        yy = y3 + 42
        for m in hands[:3]:
            v = m.get("value") or {}
            label = f"{v.get('op', '?')}.{v.get('param_path', '?')}"
            _text(d, (52, yy), label, f_small, DIM)
            _text(d, (52 + 200, yy), str(v.get("value", "?")), f_val)
            yy += 28
    else:
        _text(d, (52, y3 + 60), "还没有调参记录", f_sub, DIM)

    # ---------------- 配文习惯 ----------------
    _panel(d, (516, y3, W - 36, y3 + 128), "配文习惯", f_lbl)
    caps = store.by_type("caption")
    if caps:
        v = caps[0].get("value") or {}
        _text(d, (532, y3 + 42), "平均字数", f_small, DIM)
        _text(d, (616, y3 + 38), f"{v.get('avg_len', '?')} 字", f_val)
        _text(d, (532, y3 + 72), "常用位置", f_small, DIM)
        _text(d, (616, y3 + 68), str(v.get("position", "?")), f_val)
        samples = v.get("samples") or []
        if samples:
            _text(d, (532, y3 + 100), "最近配文", f_small, DIM)
            _text(d, (616, y3 + 97), " / ".join(str(s) for s in samples[-2:])[:22],
                  f_small, TEXT)
    else:
        _text(d, (532, y3 + 60), "还没有配文记录", f_sub, DIM)

    # ---------------- 页脚 ----------------
    ev_path = ROOT / "memories" / "events.jsonl"
    n_events = 0
    if ev_path.exists():
        with open(ev_path, "r", encoding="utf-8") as f:
            n_events = sum(1 for _ in f)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    _text(d, (36, H - 34),
          f"已调教 {n_events} 次 · 生成于 {now} · 换工具也不会丢，因为这是你的",
          f_small, DIM)

    return img


def save(img: Image.Image | None = None, name: str | None = None) -> Path:
    """保存画像。"""
    if img is None:
        img = render()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fn = name or f"口味画像_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    p = OUT_DIR / fn
    img.save(p, "PNG")
    return p


def export_pack(store: MemoryStore | None = None) -> dict:
    """导出「口味包」：记忆 + 素材清单，可导入。

    刻意不含图片本体（体积大），只含 id 与清单。
    导入方需要在本地有对应素材。
    """
    if store is None:
        from core import memory
        store = memory.store()

    from core import store as asset_store

    assets = asset_store.scan()
    return {
        "format": "laotubang-flavor-pack",
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "stats": store.stats(),
        "memories": store.all(status=None),
        "asset_manifest": [
            {"id": a["id"], "name": a["name"], "w": a["w"], "h": a["h"],
             "bytes": a["bytes"]}
            for a in assets
        ],
    }
