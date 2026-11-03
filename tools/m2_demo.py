"""生成 M2 效果对比图：验证 TPS 形变真的能做出「斜眼歪嘴」。"""

import sys
import os
sys.path.insert(0, r"C:\Users\吴金昊\Desktop\崩铁牢图棒")
os.chdir(r"C:\Users\吴金昊\Desktop\崩铁牢图棒")

from pathlib import Path
from PIL import Image, ImageDraw
from core.render import render
from core import imageio

ROOT = Path(r"C:\Users\吴金昊\Desktop\崩铁牢图棒")

# 用砂金那张当基底
base = imageio.read(ROOT / "assets" / "raw" / "image copy.png", limit=700)
print(f"base: {base.size}")

# 造一个「嘴」素材（白牙 + 深色唇）
mouth = Image.new("RGBA", (240, 120), (0, 0, 0, 0))
d = ImageDraw.Draw(mouth)
d.ellipse((0, 0, 240, 120), fill=(70, 40, 30, 255))
d.ellipse((15, 30, 225, 95), fill=(250, 250, 248, 255))
d.line((120, 32, 120, 93), fill=(200, 190, 180, 255), width=2)
assets = {"big_mouth": mouth}

recs = {}

# 基准：无变形
recs["00_原图"] = {
    "schema_version": "1.1", "recipe_id": "t0", "target": "stitch",
    "subtype": "distort", "base_id": "b",
    "ops": [], "degrade": {"enabled": False},
    "created_at": "2026-11-03T23:00:00+08:00",
}

# 1. 纯拉伸
recs["01_纯横向拉伸"] = {
    "schema_version": "1.1", "recipe_id": "t1", "target": "stitch",
    "subtype": "distort", "base_id": "b",
    "ops": [{
        "op": "feature_warp", "asset_id": "big_mouth",
        "target_box": [0.32, 0.52, 0.36, 0.14],
        "stretch_x": 2.2, "stretch_y": 1.4,
        "strength": 0.0, "local": 0.0,
    }],
    "degrade": {"enabled": False}, "created_at": "2026-11-03T23:00:00+08:00",
}

# 2. 加斜挑
recs["02_加斜挑"] = {
    "schema_version": "1.1", "recipe_id": "t2", "target": "stitch",
    "subtype": "distort", "base_id": "b",
    "ops": [{
        "op": "feature_warp", "asset_id": "big_mouth",
        "target_box": [0.32, 0.52, 0.36, 0.14],
        "stretch_x": 2.2, "stretch_y": 1.4,
        "skew_deg": 16,
        "strength": 0.0, "local": 0.0,
    }],
    "degrade": {"enabled": False}, "created_at": "2026-11-03T23:00:00+08:00",
}

# 3. TPS 完整弯曲（局部性 1）
recs["03_TPS弯曲"] = {
    "schema_version": "1.1", "recipe_id": "t3", "target": "stitch",
    "subtype": "distort", "base_id": "b",
    "ops": [{
        "op": "feature_warp", "asset_id": "big_mouth",
        "target_box": [0.32, 0.52, 0.36, 0.14],
        "stretch_x": 2.2, "stretch_y": 1.4,
        "skew_deg": 16,
        "strength": 1.0, "local": 1.0,
    }],
    "degrade": {"enabled": False}, "created_at": "2026-11-03T23:00:00+08:00",
}

# 4. 逐角偏移（歪嘴）—— M2 的关键能力
recs["04_逐角歪嘴"] = {
    "schema_version": "1.1", "recipe_id": "t4", "target": "stitch",
    "subtype": "distort", "base_id": "b",
    "ops": [{
        "op": "feature_warp", "asset_id": "big_mouth",
        "target_box": [0.32, 0.52, 0.36, 0.14],
        "stretch_x": 2.2, "stretch_y": 1.4,
        "strength": 1.0, "local": 1.0,
        "corner_offsets": [[-0.02, -0.03], [0.03, 0.01], [0.02, 0.05], [-0.04, 0.02]],
    }],
    "degrade": {"enabled": False}, "created_at": "2026-11-03T23:00:00+08:00",
}

# 5. 加鼓胀 + 带皮圈 + 降质（完整 B 类）
recs["05_完整B类"] = {
    "schema_version": "1.1", "recipe_id": "t5", "target": "stitch",
    "subtype": "distort", "base_id": "b",
    "ops": [{
        "op": "feature_warp", "asset_id": "big_mouth",
        "target_box": [0.32, 0.52, 0.36, 0.14],
        "stretch_x": 2.2, "stretch_y": 1.4,
        "strength": 1.0, "local": 1.0,
        "corner_offsets": [[-0.02, -0.03], [0.03, 0.01], [0.02, 0.05], [-0.04, 0.02]],
        "keep_skin_rim": 0.22,
        "bulge": {"center": [0.5, 0.55], "radius": 0.12, "amount": 0.03},
    }],
    "degrade": {"enabled": True, "downscale": 0.6, "jpeg_quality": 35,
                "noise": 0.03, "oversharpen": "medium"},
    "created_at": "2026-11-03T23:00:00+08:00",
}

out_dir = ROOT / "tmp" / "m2_demo"
out_dir.mkdir(parents=True, exist_ok=True)

tiles = []
for name, rec in recs.items():
    try:
        r = render(rec, base_image=base, assets=assets, strict=False)
        p = out_dir / f"{name}.png"
        imageio.write(r.image, p)
        tiles.append((name, r.image, r.total_ms))
        print(f"  OK  {name:16} {r.total_ms:6.0f}ms  -> {p.name}")
    except Exception as e:
        print(f"  ERR {name:16} {type(e).__name__}: {e}")

# 拼成对比图
if tiles:
    tw, th = 340, 300
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 24)), (24, 26, 32))
    dd = ImageDraw.Draw(sheet)
    for i, (name, img, ms) in enumerate(tiles):
        cx = (i % cols) * tw
        cy = (i // cols) * (th + 24)
        t = img.convert("RGB")
        t.thumbnail((tw - 8, th - 8), Image.LANCZOS)
        sheet.paste(t, (cx + (tw - t.width) // 2, cy + 22))
        dd.text((cx + 8, cy + 5), name, fill=(230, 232, 240))
    sp = out_dir / "_对比.png"
    sheet.save(sp)
    print(f"\n对比图: {sp}")
