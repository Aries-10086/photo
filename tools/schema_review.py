"""M0-FR8-T04 · 20 用例反向验证

目的：用真实牢图的描述去「压」schema，看哪些情况表达不了。
这不是测试代码，是**设计评审工具** —— 结论写进 docs/schema-review.json。

判定口径：
  完整表达  = 校验通过且语义正确
  语义扭曲  = 能跑通，但要靠「把 A 当 B 用」这种歪门（记为弱项，不算覆盖率）
  表达不了  = 校验失败

运行：
    .venv\\Scripts\\python.exe tools\\schema_review.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.validate import validate_recipe  # noqa: E402

BASE = {
    "schema_version": "1.1",
    "target": "stitch",
    "created_at": "2026-11-02T02:50:00+08:00",
}

NO_DEGRADE = {"enabled": False, "downscale": 1.0, "jpeg_quality": 90,
              "noise": 0.0, "oversharpen": "off"}
DEG_DISTORT = {"enabled": True, "downscale": 0.6, "jpeg_quality": 35,
               "noise": 0.03, "oversharpen": "medium"}


def rec(rid: str, subtype: str, base_id: str, ops: list, **kw) -> dict:
    d = {**BASE, "recipe_id": rid, "subtype": subtype, "base_id": base_id, "ops": ops}
    d.update(kw)
    return d


# (编号, 描述, 配方, 期望结果, 备注)  期望结果: "ok" | "weak" | "fail"
CASES = [
    # ---------------- 核心四类 ----------------
    ("01", "科比脸贴到崩铁 Q 版角色身上（D 换装）", rec(
        "r_01", "rewear", "base_chibi_tingyun",
        [
            {"op": "face_fit", "asset_id": "kobe_face", "center": [0.5, 0.35], "scale": 1.1,
             "edge": "hard"},
            {"op": "texture_wrap", "asset_id": "lakers_jersey",
             "quad": [[0.3, 0.45], [0.7, 0.45], [0.72, 0.8], [0.28, 0.8]]},
            {"op": "sticker", "asset_id": "iced_tea", "center": [0.75, 0.7], "scale": 0.12, "z": 5},
            {"op": "label", "text": "牢大", "anchor": "bottom_right"},
        ],
        degrade=NO_DEGRADE,
    ), "ok", "D 类标准形"),

    ("02", "牙膏 logo 眼+嘴移植到砂金（B 畸变）", rec(
        "r_02", "distort", "base_aventurine",
        [
            {"op": "feature_warp", "asset_id": "darlie_eyes", "target_box": [0.34, 0.3, 0.32, 0.1],
             "stretch_x": 1.9, "skew_deg": 14, "keep_skin_rim": 0.25},
            {"op": "feature_warp", "asset_id": "darlie_mouth", "target_box": [0.35, 0.5, 0.3, 0.22],
             "stretch_x": 2.5, "stretch_y": 1.8},
            {"op": "label", "text": "牢肘子", "anchor": "bottom_right", "size_ratio": 0.028},
        ],
        degrade=DEG_DISTORT,
    ), "ok", "B 类标准形，降质必开"),

    ("03", "写实中年脸塞进加拉赫（A 精贴）", rec(
        "r_03", "precise", "base_gallagher",
        [
            {"op": "face_fit", "asset_id": "midage_face", "center": [0.5, 0.4], "scale": 1.0,
             "edge": "none", "color_transfer": 0.15},
        ],
        layers={"foreground": "masks/gallagher_hair.png",
                "replace_zone": "masks/gallagher_face.png"},
        degrade=NO_DEGRADE,
    ), "ok", "A 类标准形"),

    ("04", "科比 + 呜喵王 ×N + 搪瓷盆（C 堆叠）", rec(
        "r_04", "stack", "base_kobe_pose",
        [
            {"op": "sticker", "asset_id": "wubbaboo", "center": [0.3, 0.5], "scale": 0.12, "z": 2},
            {"op": "sticker", "asset_id": "wubbaboo", "center": [0.36, 0.53], "scale": 0.1, "z": 2},
            {"op": "sticker", "asset_id": "bowl", "center": [0.47, 0.58], "scale": 0.22, "z": 3},
        ],
        degrade={"enabled": True, "downscale": 0.85, "jpeg_quality": 60,
                 "noise": 0.01, "oversharpen": "off"},
    ), "ok", "C 类，同素材多次放置"),

    # ---------------- 之前暴露的缺口，现已补齐 ----------------
    ("05", "只改颜色不改形状（纯调色）", rec(
        "r_05", "stack", "base_x",
        [{"op": "grade", "hue_shift": -20, "saturation": 1.4, "contrast": 1.15,
          "tint": "#2a4b7c", "tint_strength": 0.3}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 grade op"),

    ("06", "只加噪点不降质", rec(
        "r_06", "stack", "base_x",
        [{"op": "grade", "noise": 0.04}],
        degrade=NO_DEGRADE,
    ), "ok", "★ grade.noise 独立于 degrade"),

    ("07", "整个换掉背景", rec(
        "r_07", "rewear", "base_x",
        [{"op": "backdrop", "asset_id": "starfield_bg", "mode": "cover", "blur_px": 2}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 backdrop op"),

    ("08", "纯色背景 + 抠出主体", rec(
        "r_08", "stack", "base_x",
        [{"op": "backdrop", "mode": "solid", "color": "#FFFFFF"}],
        degrade=NO_DEGRADE,
    ), "ok", "backdrop mode=solid"),

    ("09", "整张图镜像翻转后贴", rec(
        "r_09", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.5, 0.5], "flip_x": True}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 flip_x/flip_y"),

    ("10", "同一物件贴两次，第二次水平翻转", rec(
        "r_10", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.3, 0.5]},
         {"op": "sticker", "asset_id": "a", "center": [0.7, 0.5], "flip_x": True}],
        degrade=NO_DEGRADE,
    ), "ok", "flip 逐 op 独立"),

    ("11", "中文竖排文字", rec(
        "r_11", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.5, 0.5]},
         {"op": "label", "text": "牢大别走", "anchor": "top_right", "vertical": True}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 label.vertical"),

    ("12", "文字渐变填充", rec(
        "r_12", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.5, 0.5]},
         {"op": "label", "text": "抽象", "anchor": "center",
          "color": "#FF3355", "color_to": "#3355FF", "gradient_dir": "vertical"}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 color_to 渐变"),

    ("13", "带阴影的立体字", rec(
        "r_13", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.5, 0.5]},
         {"op": "label", "text": "立体", "anchor": "center",
          "shadow": {"dx": 4, "dy": 4, "blur": 6, "color": "#000000", "opacity": 0.7}}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 label.shadow"),

    ("14", "整脸贴合要半透明", rec(
        "r_14", "rewear", "base_x",
        [{"op": "face_fit", "asset_id": "f", "center": [0.5, 0.4], "opacity": 0.5}],
        degrade=NO_DEGRADE,
    ), "ok", "★ face_fit 补齐 opacity（与 sticker 一致）"),

    ("15", "把脸染成青色（阴间效果）", rec(
        "r_15", "rewear", "base_x",
        [{"op": "face_fit", "asset_id": "f", "center": [0.5, 0.4], "hue_shift": 120,
          "saturation": 1.2}],
        degrade=NO_DEGRADE,
    ), "ok", "★ face_fit 补齐 hue_shift/saturation"),

    ("16", "素材只露半张脸（裁剪）", rec(
        "r_16", "rewear", "base_x",
        [{"op": "face_fit", "asset_id": "f", "center": [0.5, 0.25], "scale": 1.6,
          "crop_box": [0.0, 0.0, 1.0, 0.5]}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 crop_box"),

    ("17", "基底本身镜像后再贴", rec(
        "r_17", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.5, 0.5]}],
        base_transform={"flip_x": True, "rotation_deg": 0},
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 base_transform"),

    # ---------------- 已知弱项（可表达但语义扭曲） ----------------
    ("18", "文字沿弧线排列", rec(
        "r_18", "stack", "base_x",
        [{"op": "sticker", "asset_id": "a", "center": [0.5, 0.5]},
         {"op": "label", "text": "圆形环绕文字", "anchor": "center",
          "arc_deg": 180, "arc_radius": 0.35}],
        degrade=NO_DEGRADE,
    ), "ok", "★ v1.1 新增 arc_deg/arc_radius 路径文字"),

    ("19", "多张基底横向拼成一张对比图", rec(
        "r_19", "stack", "base_left",
        [{"op": "sticker", "asset_id": "base_right", "center": [0.75, 0.5], "scale": 0.5}],
        degrade=NO_DEGRADE,
    ), "weak", "把另一张基底当素材贴，可表达但 base/asset 类型边界模糊"),

    # ---------------- 已知缺口（应失败，作为回归保护） ----------------
    ("20", "label 写在 degrade 之前（错误用法）", rec(
        "r_20", "distort", "base_x",
        [{"op": "label", "text": "先加字"},
         {"op": "degrade", "params": {"enabled": True}}],
        degrade=DEG_DISTORT,
    ), "fail", "顺序约束应拦截：先降质再加字，否则字被压糊"),
]


def main() -> int:
    print("=" * 96)
    print("M0-FR8-T04 · 20 用例反向验证（schema v1.1）")
    print("=" * 96)
    print(f"{'#':<4}{'校验':<6}{'期望':<6}{'用例':<38}{'备注'}")
    print("-" * 96)

    ok_n = weak_n = fail_n = mismatch_n = 0
    gaps: list[dict] = []
    weaks: list[str] = []
    mismatches: list[str] = []

    for cid, desc, recipe, expect, note in CASES:
        r = validate_recipe(recipe)
        actual = "ok" if r.ok else "fail"

        # 只有校验失败才算「表达不了」；weak 是人工判定的语义质量
        if actual == "fail":
            fail_n += 1
        elif expect == "weak":
            weak_n += 1
        else:
            ok_n += 1

        # 与预期比对：ok 和 weak 都期望校验通过
        want_pass = expect in ("ok", "weak")
        consistent = (actual == "ok") == want_pass
        if not consistent:
            mismatch_n += 1
            mismatches.append(cid)
            mark = "**偏差**"
        else:
            mark = "通过" if r.ok else "拦截"

        short = desc[:34] + ("…" if len(desc) > 34 else "")
        print(f"{cid:<4}{mark:<6}{expect:<6}{short:<38}{note[:40]}")

        if not r.ok:
            gaps.append({"id": cid, "desc": desc, "expect": expect,
                         "errors": r.errors[:3], "note": note})
        if expect == "weak":
            weaks.append(cid)

    total = len(CASES)
    # 覆盖率分母排除「故意构造的负向用例」（期望 fail 的，用于回归保护）
    negative = sum(1 for c in CASES if c[3] == "fail")
    denominator = total - negative
    coverage = ok_n / denominator

    print("-" * 96)
    print(f"用例总数:       {total}（其中负向回归用例 {negative} 个，不计入分母）")
    print(f"  完整表达:     {ok_n}")
    print(f"  语义弱项:     {weak_n}  ({', '.join(weaks) if weaks else '无'})")
    print(f"  校验失败:     {fail_n}  (应为 {negative}，即故意构造的错误用法被拦截)")
    print(f"  与预期偏差:   {mismatch_n}  ({', '.join(mismatches) if mismatches else '无'})")
    print(f"  覆盖率:       {ok_n}/{denominator} = {coverage * 100:.0f}%  (门槛 90%)")

    verdict = "通过" if coverage >= 0.90 and mismatch_n == 0 else "未达门槛"
    print("\n" + "=" * 96)
    print(f"{ok_n}/{denominator} = {coverage * 100:.0f}%  →  {verdict}")
    print("=" * 96)

    if gaps:
        print("\n【校验失败清单】")
        for g in gaps:
            print(f"  {g['id']} [{g['expect']}] {g['desc'][:34]}")
            for e in g["errors"]:
                print(f"      · {e[:100]}")

    if weaks:
        print("\n【已知语义弱项（不阻塞，但要在 PRD 记录）】")
        for cid in weaks:
            c = next(x for x in CASES if x[0] == cid)
            print(f"  {cid} {c[1][:34]} —— {c[4]}")

    out = ROOT / "docs" / "schema-review.json"
    out.write_text(json.dumps({
        "schema_version": "1.1",
        "total": total,
        "negative_cases": negative,
        "denominator": denominator,
        "fully_covered": ok_n,
        "weak": weak_n,
        "failed": fail_n,
        "mismatch": mismatch_n,
        "coverage_pct": round(coverage * 100, 1),
        "verdict": verdict,
        "weak_cases": weaks,
        "mismatches": mismatches,
        "failures": gaps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结论已写入: {out.relative_to(ROOT)}")

    return 0 if (coverage >= 0.90 and mismatch_n == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
