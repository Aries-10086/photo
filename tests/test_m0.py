"""M0 验收测试：图像 IO / Schema 校验 / clamp / op 顺序。

运行：
    .venv\\Scripts\\python.exe -m tests.test_m0
或
    .venv\\Scripts\\python.exe tests\\test_m0.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from core import imageio  # noqa: E402
from core.validate import (  # noqa: E402
    check_op_order,
    validate_asset,
    validate_memory,
    validate_recipe,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    mark = "OK  " if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------- 图像 IO

def test_imageio() -> None:
    section("图像 IO")
    ref = ROOT / "image.png"
    check("参考图存在", ref.exists())

    img = imageio.read(ref)
    check("读取为 RGBA", img.mode == "RGBA", img.mode)
    check("长边 ≤ 2048", max(img.size) <= imageio.MAX_SIDE, str(img.size))

    arr = imageio.read_array(ref)
    check("数组形状 3/4 通道", arr.ndim == 3 and arr.shape[2] in (3, 4), str(arr.shape))

    # 超大图自动缩放
    big = Image.new("RGBA", (4000, 1000), (10, 20, 30, 255))
    tmp = ROOT / "tmp" / "_big.png"
    imageio.write(big, tmp)
    got = imageio.read(tmp)
    check("4000px 自动缩到 2048", max(got.size) == 2048, str(got.size))
    check("等比缩放保持比例", abs(got.size[0] / got.size[1] - 4.0) < 0.01, str(got.size))

    # round-trip 哈希一致
    h1 = imageio.sha256_of_image(img)
    out = ROOT / "tmp" / "_rt.png"
    imageio.write(img, out)
    img2 = imageio.read(out, limit=None)
    check("PNG round-trip 像素一致", imageio.sha256_of_image(img2) == h1)

    # JPEG 写出时 alpha 压平
    jpg = ROOT / "tmp" / "_flat.jpg"
    p = imageio.write(img, jpg)
    check("JPEG 写出成功", p.exists() and p.suffix == ".jpg", str(p))
    check("JPEG 读回为 RGB", Image.open(p).mode == "RGB")

    # bbox
    canvas = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    canvas.paste((255, 0, 0, 255), (10, 20, 60, 70))
    check("alpha bbox 正确", imageio.bbox_of_alpha(canvas) == (10, 20, 60, 70),
          str(imageio.bbox_of_alpha(canvas)))

    # 错误处理
    try:
        imageio.read(ROOT / "tmp" / "_nope.png")
        check("不存在文件应报错", False)
    except imageio.ImageIOError:
        check("不存在文件报错正确", True)

    # 清理
    for f in ("_big.png", "_rt.png", "_flat.jpg"):
        (ROOT / "tmp" / f).unlink(missing_ok=True)


# ---------------------------------------------------------------- 素材卡

GOOD_ASSET = {
    "schema_version": "1.1",
    "id": "kobe_face_a1b2c3",
    "image": "assets/cut/kobe_face_a1b2c3.png",
    "original": "assets/raw/kobe.jpg",
    "slots": ["face"],
    "domain": "real",
    "circle": ["basketball"],
    "mood": "serious",
    "parts": ["full_face"],
    "tags": ["科比"],
    "source": "网络",
    "usable_count": 0,
    "created_at": "2026-11-02T02:40:00+08:00",
}


def test_asset_schema() -> None:
    section("素材卡 Schema")
    r = validate_asset(GOOD_ASSET)
    check("合法素材卡通过", r.ok, r.report())

    bad = {**GOOD_ASSET}
    del bad["slots"]
    r = validate_asset(bad)
    check("缺 slots 被拒", not r.ok)

    bad = {**GOOD_ASSET, "domain": "外星人"}
    r = validate_asset(bad)
    check("非法 domain 被拒", not r.ok)

    bad = {**GOOD_ASSET, "slots": ["face", "face"]}
    r = validate_asset(bad)
    check("slots 重复被拒", not r.ok)

    # 多余字段
    bad = {**GOOD_ASSET, "mystery": 1}
    r = validate_asset(bad)
    check("多余字段被拒", not r.ok)


# ---------------------------------------------------------------- 配方

GOOD_RECIPE_C = {
    "schema_version": "1.1",
    "recipe_id": "r_stack_001",
    "title": "科比配搪瓷盆",
    "target": "stitch",
    "subtype": "stack",
    "base_id": "base_kobe_pose_001",
    "ops": [
        {"op": "sticker", "asset_id": "wubbaboo_a", "center": [0.42, 0.61], "scale": 0.18, "z": 2},
        {"op": "sticker", "asset_id": "bowl_porcelain", "center": [0.47, 0.58], "scale": 0.22, "z": 3},
    ],
    "degrade": {"enabled": True, "downscale": 0.8, "jpeg_quality": 55, "noise": 0.01,
                "oversharpen": "weak"},
    "created_at": "2026-11-02T02:40:00+08:00",
}

GOOD_RECIPE_B = {
    "schema_version": "1.1",
    "recipe_id": "r_distort_001",
    "target": "stitch",
    "subtype": "distort",
    "base_id": "base_aventurine_001",
    "ops": [
        {"op": "feature_warp", "asset_id": "darlie_eyes", "target_box": [0.35, 0.30, 0.30, 0.12],
         "stretch_x": 1.8, "skew_deg": 12, "keep_skin_rim": 0.2},
        {"op": "feature_warp", "asset_id": "darlie_mouth", "target_box": [0.36, 0.52, 0.28, 0.20],
         "stretch_x": 2.4, "stretch_y": 1.9},
    ],
    "degrade": {"enabled": True, "downscale": 0.6, "jpeg_quality": 35, "noise": 0.03,
                "oversharpen": "medium"},
    "created_at": "2026-11-02T02:40:00+08:00",
}

GOOD_RECIPE_A = {
    "schema_version": "1.1",
    "recipe_id": "r_precise_001",
    "target": "stitch",
    "subtype": "precise",
    "base_id": "base_gallagher_001",
    "layers": {
        "foreground": "assets/masks/gallagher_hair.png",
        "replace_zone": "assets/masks/gallagher_face.png",
    },
    "ops": [
        {"op": "face_fit", "asset_id": "midage_face", "center": [0.5, 0.42], "scale": 1.0,
         "edge": "none", "color_transfer": 0.1},
    ],
    "degrade": {"enabled": False, "downscale": 1.0, "jpeg_quality": 92, "noise": 0.0,
                "oversharpen": "off"},
    "created_at": "2026-11-02T02:40:00+08:00",
}

GOOD_RECIPE_D = {
    "schema_version": "1.1",
    "recipe_id": "r_rewear_001",
    "target": "stitch",
    "subtype": "rewear",
    "base_id": "base_chibi_001",
    "ops": [
        {"op": "texture_wrap", "asset_id": "lakers_jersey", "quad": [
            [0.30, 0.40], [0.70, 0.40], [0.72, 0.78], [0.28, 0.78]]},
        {"op": "sticker", "asset_id": "iced_tea", "center": [0.72, 0.66], "scale": 0.14, "z": 5},
        {"op": "label", "text": "牢肘子", "anchor": "bottom_right", "size_ratio": 0.03},
    ],
    "degrade": {"enabled": False, "downscale": 1.0, "jpeg_quality": 90, "noise": 0.0,
                "oversharpen": "off"},
    "created_at": "2026-11-02T02:40:00+08:00",
}


def test_recipe_schema() -> None:
    section("配方 Schema")
    for name, rec in [("C 堆叠", GOOD_RECIPE_C), ("B 畸变", GOOD_RECIPE_B),
                      ("A 精贴", GOOD_RECIPE_A), ("D 换装", GOOD_RECIPE_D)]:
        r = validate_recipe(rec)
        check(f"{name} 配方通过", r.ok, r.report())

    # precise 缺 layers
    bad = {k: v for k, v in GOOD_RECIPE_A.items() if k != "layers"}
    r = validate_recipe(bad)
    check("precise 缺 layers 被拒", not r.ok)

    # target 只允许 stitch
    bad = {**GOOD_RECIPE_C, "target": "generate"}
    r = validate_recipe(bad)
    check("target=generate 被拒（v2 已移除）", not r.ok)

    # 未知 op
    bad = {**GOOD_RECIPE_C, "ops": [{"op": "teleport", "asset_id": "x"}]}
    r = validate_recipe(bad)
    check("未知 op 被拒", not r.ok)

    # label 在 degrade 之前
    bad = {
        **GOOD_RECIPE_C,
        "ops": [
            {"op": "label", "text": "先加字"},
            {"op": "degrade", "params": {"enabled": True}},
            {"op": "sticker", "asset_id": "x", "center": [0.5, 0.5]},
        ],
    }
    r = validate_recipe(bad)
    check("label 早于 degrade 被拒", not r.ok,
          r.report())


def test_clamp() -> None:
    section("数值 clamp")
    bad = {
        **GOOD_RECIPE_C,
        "ops": [
            {"op": "sticker", "asset_id": "w", "center": [1.7, -0.3], "scale": 999, "opacity": 5},
        ],
    }
    r = validate_recipe(bad)
    op = r.data["ops"][0]
    check("center x 钳到 1.0", op["center"][0] == 1.0, str(op["center"]))
    check("center y 钳到 0.0", op["center"][1] == 0.0, str(op["center"]))
    check("scale 钳到 20", op["scale"] == 20, str(op["scale"]))
    check("opacity 钳到 1.0", op["opacity"] == 1.0, str(op["opacity"]))
    check("clamp 产生了警告", len(r.warnings) >= 3, str(r.warnings[:3]))

    bad = {
        **GOOD_RECIPE_C,
        "degrade": {"enabled": True, "downscale": 0.01, "jpeg_quality": 999,
                    "noise": 5, "oversharpen": "weak"},
    }
    r = validate_recipe(bad)
    dg = r.data["degrade"]
    check("downscale 钳到 0.1", dg["downscale"] == 0.1, str(dg["downscale"]))
    check("jpeg_quality 钳到 100 并取整", dg["jpeg_quality"] == 100, str(dg["jpeg_quality"]))
    check("noise 钳到 1.0", dg["noise"] == 1.0, str(dg["noise"]))


def test_op_order() -> None:
    section("op 顺序（独立函数）")
    check("正常顺序通过",
          check_op_order([{"op": "sticker"}, {"op": "degrade"}, {"op": "label"}]) == [])
    check("layer_split 不在首位被拒",
          check_op_order([{"op": "sticker"}, {"op": "layer_split"}]) != [])
    check("两个 degrade 被拒",
          check_op_order([{"op": "degrade"}, {"op": "sticker"}, {"op": "degrade"}]) != [])
    check("label 在 degrade 前被拒",
          check_op_order([{"op": "label"}, {"op": "degrade"}]) != [])
    check("未知 op 被拒", check_op_order([{"op": "vanish"}]) != [])


# ---------------------------------------------------------------- 记忆

GOOD_MEMORY = {
    "id": "mem_combo_001",
    "type": "combo",
    "key": "asset:kobe_face × base:jingyuan × slot:face",
    "value": {"base_id": "base_jingyuan_001", "asset_ids": ["kobe_face_a1b2c3"],
              "slot": "face", "subtype": "precise", "sample_recipe_id": "r_precise_001"},
    "confidence": 0.78,
    "evidence_count": 4,
    "positive_count": 3,
    "negative_count": 1,
    "conditions": ["default"],
    "status": "active",
    "weight": 1.0,
    "source_events": ["evt_001", "evt_002"],
    "hand_edited": False,
    "created_at": "2026-11-02T02:40:00+08:00",
    "last_confirmed": "2026-11-02T02:40:00+08:00",
}


def test_memory_schema() -> None:
    section("记忆 Schema")
    r = validate_memory(GOOD_MEMORY)
    check("合法记忆通过", r.ok, r.report())

    bad = {**GOOD_MEMORY, "type": "telepathy"}
    r = validate_memory(bad)
    check("非法 type 被拒", not r.ok)

    bad = {**GOOD_MEMORY, "confidence": 1.9}
    r = validate_memory(bad)
    check("confidence 超界被钳", r.data["confidence"] == 1.0, str(r.data.get("confidence")))

    bad = {**GOOD_MEMORY, "status": "zombie"}
    r = validate_memory(bad)
    check("非法 status 被拒", not r.ok)


# ---------------------------------------------------------------- 主

def main() -> int:
    try:
        test_imageio()
        test_asset_schema()
        test_recipe_schema()
        test_clamp()
        test_op_order()
        test_memory_schema()
    except Exception:
        traceback.print_exc()
        print("\n测试中断（异常）")
        return 2

    total = len(PASS) + len(FAIL)
    print(f"\n{'=' * 52}")
    print(f"通过 {len(PASS)} / {total}")
    if FAIL:
        print("失败项：")
        for n, d in FAIL:
            print(f"  ✗ {n}  {d}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
