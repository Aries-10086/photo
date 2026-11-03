"""M2 出口门验收：三档违和度必须量化分开。

这是 M2 唯一真正的验收标准（PRD FR-5 / 落地任务 M2 出口门）。

判据：
    色差保留率：硬贴 > 70%，半融合 30~50%，全融合 < 15%
    三档之间差值 > 20pp

如果这个测试不过，说明「违和度档位」是假的 —— 肉眼看不出差别，
或者看得出但没法参数化，那整个 B 类配方就不可复现。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from core.metrics import (  # noqa: E402
    color_diff_retention,
    edge_gradient,
    highfreq_energy,
    laplacian_variance,
    report,
)
from core.render import render  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


BASE_RECIPE = {
    "schema_version": "1.1",
    "target": "stitch",
    "base_id": "synth",
    "created_at": "2026-11-03T23:50:00+08:00",
}


# ---------------------------------------------------------------- 合成素材


def make_base() -> Image.Image:
    """基底：浅色背景 + 一个深色角色轮廓（模拟「苍白装束」）。"""
    img = Image.new("RGBA", (600, 600), (232, 232, 235, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((150, 120, 450, 520), fill=(196, 200, 210, 255))   # 身体
    d.ellipse((220, 180, 380, 360), fill=(210, 214, 224, 255))   # 脸区
    return img


def make_face(color=(92, 58, 38)) -> Image.Image:
    """源素材：深棕色皮肤的脸（模拟黑人牙膏 logo 的深肤色）。"""
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, 200, 200), fill=color + (255,))
    # 眼
    d.ellipse((50, 70, 80, 100), fill=(20, 15, 10, 255))
    d.ellipse((120, 70, 150, 100), fill=(20, 15, 10, 255))
    # 嘴（白牙）
    d.arc((60, 120, 140, 175), 0, 180, fill=(250, 250, 250, 255), width=12)
    return img


# ---------------------------------------------------------------- 三档


def build_recipe(tier: str) -> dict:
    """三档违和度的配方。

    硬贴：     不做色彩迁移，硬边，放大 6%（喜剧参数）
    半融合：   色彩迁移 45%，轻微羽化，放大 2%
    全融合：   色彩迁移 100%，大羽化 20px，不放大
    """
    cfg = {
        "hard":     {"ct": 0.0,  "edge": "hard", "feather": 0,  "scale": 1.06},
        "half":     {"ct": 0.45, "edge": "soft", "feather": 4,  "scale": 1.02},
        "full":     {"ct": 1.0,  "edge": "soft", "feather": 20, "scale": 1.00},
    }[tier]

    return {
        **BASE_RECIPE,
        "recipe_id": f"tier_{tier}",
        "subtype": "precise",
        "layers": {"foreground": None, "replace_zone": None},
        "ops": [{
            "op": "face_fit",
            "asset_id": "face",
            "center": [0.5, 0.45],
            "scale": 0.32 * cfg["scale"],
            "edge": cfg["edge"],
            "feather_px": cfg["feather"],
            "color_transfer": cfg["ct"],
        }],
        "degrade": {"enabled": False, "downscale": 1.0, "jpeg_quality": 95,
                    "noise": 0.0, "oversharpen": "off"},
    }


def main() -> int:
    base = make_base()
    face = make_face()
    assets = {"face": face}

    print("=" * 70)
    print("M2 出口门 · 三档违和度量化区分")
    print("=" * 70)
    print()

    results = {}
    for tier in ("hard", "half", "full"):
        rec = build_recipe(tier)
        op = rec["ops"][0]
        r = render(rec, base_image=base, assets=assets)
        img = r.image

        # 素材实际尺寸：face_fit 把素材长边缩放到 base_long * scale
        side = int(600 * op["scale"])
        cx, cy = op["center"][0] * 600, op["center"][1] * 600
        box = (int(cx - side / 2), int(cy - side / 2),
               int(cx + side / 2), int(cy + side / 2))

        cdr = color_diff_retention(face, img, box=box, base_image=base)
        eg = edge_gradient(img, region=(0.15, 0.15, 0.7, 0.7))
        lv = laplacian_variance(img)
        hf = highfreq_energy(img)

        results[tier] = {"cdr": cdr, "edge": eg, "lap": lv, "hf": hf}
        print(f"{tier:6} 色差保留 {cdr * 100:5.1f}%  边缘梯度 {eg:.4f}  "
              f"拉普拉斯 {lv:7.1f}  高频 {hf:.4f}")

    print()

    # ---------- 判据 ----------
    hard = results["hard"]["cdr"]
    half = results["half"]["cdr"]
    full = results["full"]["cdr"]

    check("硬贴色差保留 > 70%", hard > 0.70, f"{hard * 100:.1f}%")
    check("半融合色差保留 30~65%", 0.30 < half < 0.65, f"{half * 100:.1f}%")
    check("全融合色差保留 < 20%", full < 0.20, f"{full * 100:.1f}%")

    print()
    gap_hh = (hard - half) * 100
    gap_hf = (half - full) * 100
    gap_all = (hard - full) * 100

    check("硬贴-半融合 差值 > 20pp", gap_hh > 20, f"{gap_hh:.1f}pp")
    check("半融合-全融合 差值 > 20pp", gap_hf > 20, f"{gap_hf:.1f}pp")
    check("硬贴-全融合 差值 > 40pp", gap_all > 40, f"{gap_all:.1f}pp")

    print()
    # 单调性：三档必须单调递减
    check("色差保留率单调递减", hard > half > full,
          f"{hard:.3f} > {half:.3f} > {full:.3f}")

    # 边缘梯度：硬贴的应该最高（但羽化会拉低）
    e_hard = results["hard"]["edge"]
    e_full = results["full"]["edge"]
    check("硬贴边缘梯度 > 全融合", e_hard > e_full,
          f"{e_hard:.4f} vs {e_full:.4f}")

    print()
    print("=" * 70)

    ok = len(FAIL) == 0
    print(f"{'PASS' if ok else 'FAIL'}  通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("未达标：")
        for n, d in FAIL:
            print(f"  ✗ {n}  {d}")
        print()
        print("结论：色差保留率没能区分三档。")
        print("  调试方向：")
        print("  1. 检查 _color_transfer 是否真的按 strength 插值")
        print("  2. 检查 face_fit 的 color_transfer 参数是否被读取")
        print("  3. 羽化是否吞掉了素材边界（box 算错会导致背景采样错误）")
    else:
        print("三档违和度已量化分开，B 类配方可复现。")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
