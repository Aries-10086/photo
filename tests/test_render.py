"""渲染管线冒烟测试：用合成素材验证每个 op 都能跑通。

不追求效果，只确认「不崩、有变化、可复现」。
运行：.venv\\Scripts\\python.exe tests\\test_render.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from core import imageio  # noqa: E402
from core.render import render  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def make_assets() -> dict[str, Image.Image]:
    """合成素材：一个圆形「脸」、一个方块贴纸、一张纹理。"""
    face = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(face)
    d.ellipse((10, 10, 290, 290), fill=(200, 160, 130, 255))
    d.ellipse((90, 110, 130, 150), fill=(20, 20, 20, 255))
    d.ellipse((170, 110, 210, 150), fill=(20, 20, 20, 255))
    d.arc((90, 170, 210, 240), 0, 180, fill=(80, 40, 40, 255), width=10)

    sticker = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(sticker).rectangle((10, 10, 110, 110), fill=(40, 200, 90, 255))

    tex = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    td = ImageDraw.Draw(tex)
    for y in range(0, 200, 20):
        td.rectangle((0, y, 200, y + 10), fill=(180, 40, 40, 255))

    return {"face": face, "cute_sticker": sticker, "jersey": tex}


def base() -> Image.Image:
    b = Image.new("RGBA", (800, 800), (245, 245, 245, 255))
    d = ImageDraw.Draw(b)
    d.rectangle((100, 100, 700, 700), outline=(30, 30, 30, 255), width=3)
    d.text((340, 60), "BASE", fill=(0, 0, 0, 255))
    return b


RECIPE_BASE = {
    "schema_version": "1.1",
    "recipe_id": "smoke_001",
    "target": "stitch",
    "subtype": "stack",
    "base_id": "synth_base",
    "created_at": "2026-11-02T03:00:00+08:00",
}


def diff_ratio(a: Image.Image, b: Image.Image) -> float:
    """两张图不同像素占比。"""
    if a.size != b.size:
        return 1.0
    aa = np.asarray(a.convert("RGB"), dtype=np.int16)
    bb = np.asarray(b.convert("RGB"), dtype=np.int16)
    return float((np.abs(aa - bb).sum(axis=2) > 8).mean())


def run_op_case(name: str, ops: list, assets: dict, expect_change: bool = True,
                min_diff: float = 0.0005) -> None:
    rec = {**RECIPE_BASE, "ops": ops}
    try:
        r = render(rec, base_image=base(), assets=assets)
        d = diff_ratio(base(), r.image)
        ok = (d >= min_diff) if expect_change else (d < min_diff)
        check(name, ok, f"差异 {d * 100:.3f}% 耗时 {r.total_ms:.0f}ms")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"{type(e).__name__}: {e}")


def main() -> int:
    assets = make_assets()
    print("=== 逐个 op ===")

    run_op_case("sticker 放置", [
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.5, 0.5], "scale": 0.3},
    ], assets)

    run_op_case("sticker + flip_x", [
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.3, 0.5], "scale": 0.25,
         "flip_x": True},
    ], assets)

    run_op_case("sticker + blend multiply", [
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.5, 0.5], "scale": 0.3,
         "blend": "multiply"},
    ], assets)

    run_op_case("face_fit 整脸", [
        {"op": "face_fit", "asset_id": "face", "center": [0.5, 0.4], "scale": 0.5},
    ], assets)

    run_op_case("face_fit + 色相", [
        {"op": "face_fit", "asset_id": "face", "center": [0.5, 0.4], "scale": 0.5,
         "hue_shift": 120},
    ], assets)

    run_op_case("feature_warp 拉伸", [
        {"op": "feature_warp", "asset_id": "face", "target_box": [0.3, 0.3, 0.4, 0.2],
         "stretch_x": 2.0, "stretch_y": 1.2},
    ], assets)

    run_op_case("feature_warp + 斜挑", [
        {"op": "feature_warp", "asset_id": "face", "target_box": [0.3, 0.3, 0.4, 0.2],
         "stretch_x": 1.8, "skew_deg": 15, "keep_skin_rim": 0.2},
    ], assets)

    run_op_case("texture_wrap 四角透视", [
        {"op": "texture_wrap", "asset_id": "jersey",
         "quad": [[0.3, 0.4], [0.7, 0.42], [0.72, 0.78], [0.28, 0.76]]},
    ], assets)

    # 主体掩码：中间方块内保留原图
    mask_path = ROOT / "tmp" / "_smoke_subject_mask.png"
    mp = Image.new("L", (800, 800), 0)
    ImageDraw.Draw(mp).rectangle((200, 200, 600, 600), fill=255)
    imageio.write(mp.convert("RGBA"), mask_path)

    run_op_case("backdrop solid + 主体掩码", [
        {"op": "backdrop", "mode": "solid", "color": "#3366AA",
         "subject_mask": str(mask_path.relative_to(ROOT)).replace("\\", "/")},
    ], assets)

    run_op_case("backdrop cover + 主体掩码", [
        {"op": "backdrop", "asset_id": "jersey", "mode": "cover", "blur_px": 1,
         "subject_mask": str(mask_path.relative_to(ROOT)).replace("\\", "/")},
    ], assets)

    # 无掩码且 alpha 全不透明 → 应明确报错而不是静默失败
    try:
        render({**RECIPE_BASE, "ops": [{"op": "backdrop", "mode": "solid"}]},
               base_image=base(), assets=assets)
        check("backdrop 无掩码时报错", False, "应该报错")
    except ValueError as e:
        check("backdrop 无掩码时报错", "subject_mask" in str(e), str(e)[:60])
    mask_path.unlink(missing_ok=True)

    run_op_case("grade 调色", [
        {"op": "grade", "hue_shift": -20, "saturation": 1.5, "contrast": 1.2},
    ], assets)

    run_op_case("grade 加噪", [
        {"op": "grade", "noise": 0.05},
    ], assets)

    run_op_case("label 中文", [
        {"op": "label", "text": "牢大", "anchor": "bottom_right"},
    ], assets)

    run_op_case("label 竖排", [
        {"op": "label", "text": "牢大别走", "anchor": "top_right", "vertical": True},
    ], assets)

    run_op_case("label 渐变", [
        {"op": "label", "text": "抽象", "anchor": "center",
         "color": "#FF3355", "color_to": "#3355FF"},
    ], assets)

    run_op_case("label 投影", [
        {"op": "label", "text": "立体", "anchor": "center",
         "shadow": {"dx": 4, "dy": 4, "blur": 6}},
    ], assets)

    run_op_case("label 弧线", [
        {"op": "label", "text": "环绕", "anchor": "center", "arc_deg": 120,
         "arc_radius": 0.25},
    ], assets)

    run_op_case("degrade 全套", [
        {"op": "grade", "saturation": 1.2},
        {"op": "degrade", "params": {"enabled": True, "downscale": 0.6, "jpeg_quality": 35,
                                     "noise": 0.02, "oversharpen": "medium"}},
    ], assets)

    print("\n=== 顺序与组合 ===")

    # label 在 degrade 之后
    rec = {**RECIPE_BASE, "ops": [
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.5, 0.5], "scale": 0.3},
        {"op": "degrade", "params": {"enabled": True, "downscale": 0.7, "jpeg_quality": 40,
                                     "noise": 0.0, "oversharpen": "off"}},
        {"op": "label", "text": "测试", "anchor": "bottom_left"},
    ]}
    r = render(rec, base_image=base(), assets=assets)
    check("sticker→degrade→label 顺序执行", len(r.steps) == 3, str([s.op for s in r.steps]))

    # label 在 degrade 之前 —— 校验应拦截（strict 模式抛错）
    rec2 = {**RECIPE_BASE, "ops": [
        {"op": "label", "text": "测试", "anchor": "bottom_left"},
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.5, 0.5], "scale": 0.3},
        {"op": "degrade", "params": {"enabled": True, "downscale": 0.7, "jpeg_quality": 40,
                                     "noise": 0.0, "oversharpen": "off"}},
    ]}
    try:
        render(rec2, base_image=base(), assets=assets)
        check("label 早于 degrade 被拦截", False, "应该报错")
    except ValueError as e:
        check("label 早于 degrade 被拦截", "label" in str(e), str(e)[:60])

    # 正常顺序但声明乱 —— 应被自动排序后成功
    rec4 = {**RECIPE_BASE, "ops": [
        {"op": "grade", "saturation": 1.1},
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.5, 0.5], "scale": 0.3},
    ]}
    r = render(rec4, base_image=base(), assets=assets, strict=False)
    order = [s.op for s in r.steps]
    check("同秩乱序自动归位", order == ["sticker", "grade"], str(order))

    print("\n=== 确定性 ===")

    rec3 = {**RECIPE_BASE, "ops": [
        {"op": "face_fit", "asset_id": "face", "center": [0.5, 0.4], "scale": 0.5},
        {"op": "grade", "noise": 0.03},
        {"op": "degrade", "params": {"enabled": True, "downscale": 0.7, "jpeg_quality": 45,
                                     "noise": 0.02, "oversharpen": "weak"}},
    ]}
    h = []
    for _ in range(3):
        h.append(imageio.sha256_of_image(render(rec3, base_image=base(), assets=assets).image))
    check("同配方 3 次逐像素一致", len(set(h)) == 1, str(len(set(h))))

    print("\n=== 单步落盘 ===")
    steps_dir = ROOT / "tmp" / "_steps"
    r = render(rec3, base_image=base(), assets=assets, save_steps_dir=steps_dir)
    files = sorted(steps_dir.glob("step*.png"))
    check("每步都有中间产物", len(files) == len(r.steps), f"{len(files)} vs {len(r.steps)}")
    for f in files:
        f.unlink(missing_ok=True)
    steps_dir.rmdir()

    print("\n=== 性能 ===")
    big = Image.new("RGBA", (2048, 2048), (240, 240, 240, 255))
    rec_perf = {**RECIPE_BASE, "ops": [
        {"op": "sticker", "asset_id": "cute_sticker", "center": [0.3, 0.4], "scale": 0.2},
        {"op": "face_fit", "asset_id": "face", "center": [0.6, 0.5], "scale": 0.4},
        {"op": "grade", "saturation": 1.2},
        {"op": "degrade", "params": {"enabled": True, "downscale": 0.7, "jpeg_quality": 45,
                                     "noise": 0.02, "oversharpen": "weak"}},
        {"op": "label", "text": "牢图棒", "anchor": "bottom_right"},
    ]}
    t = time.perf_counter()
    r = render(rec_perf, base_image=big, assets=assets)
    ms = (time.perf_counter() - t) * 1000
    check("2048² 全流程 < 1000ms", ms < 1000, f"{ms:.0f}ms")
    print(f"       实际 {ms:.0f} ms，逐步: " +
          ", ".join(f"{s.op}={s.ms:.0f}ms" for s in r.steps))

    total = len(PASS) + len(FAIL)
    print(f"\n{'=' * 52}")
    print(f"通过 {len(PASS)} / {total}")
    if FAIL:
        for f in FAIL:
            print(f"  ✗ {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
