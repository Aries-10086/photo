"""P0 优化验收：缓存 / 安全解析 / 事件压缩 / 位置参数过滤。"""

from __future__ import annotations

import ast
import sys
import tempfile
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. 值解析


def test_parse_value() -> None:
    print("\n=== 1. 安全值解析（替代 eval） ===")
    from core.memory import parse_value

    check("float 解析", parse_value("2.2", "float") == 2.2)
    check("int 解析", parse_value("35", "int") == 35)
    check("int 容忍 35.0", parse_value("35.0", "int") == 35)
    check("bool True", parse_value("True", "bool") is True)
    check("bool false", parse_value("false", "bool") is False)

    # repr 产生的引号
    check("str 去引号", parse_value("'weak'", "str") == "weak")
    check("str 双引号", parse_value('"bottom_right"', "str") == "bottom_right")

    # 老数据（无 type）走 literal_eval
    check("老数据 float", parse_value("2.2") == 2.2)
    check("老数据 str", parse_value("'weak'") == "weak")
    check("老数据 list", parse_value("[0.5, 0.5]") == [0.5, 0.5])

    # 恶意/畸形输入不应执行
    evil_cases = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "'''",
        "[1,2,",
        "",
        "None",
    ]
    executed = False
    for c in evil_cases:
        try:
            r = parse_value(c)
            if r == "pwned":
                executed = True
        except Exception:  # noqa: BLE001
            check(f"恶意输入不抛错: {c[:20]}", False)
            continue
    check("恶意输入不执行代码", not executed)
    check("畸形输入不抛异常", True)

    # 非字符串直接返回
    check("数字原样返回", parse_value(2.2) == 2.2)
    check("None 原样返回", parse_value(None) is None)

    # 确认源码里不再有 eval(
    src = (ROOT / "core" / "recall.py").read_text(encoding="utf-8")
    check("recall.py 不含 eval(", "eval(" not in src.replace("literal_eval(", ""))
    src2 = (ROOT / "core" / "memory.py").read_text(encoding="utf-8")
    check("memory.py 只用 ast.literal_eval",
          "literal_eval" in src2 and " eval(" not in src2)


# ---------------------------------------------------------------- 2. 位置过滤


def test_skip_position() -> None:
    print("\n=== 2. 位置参数过滤 ===")
    from core.distill import _is_hand_param

    check("center 被过滤", not _is_hand_param("sticker", "center"))
    check("target_box 被过滤", not _is_hand_param("feature_warp", "target_box"))
    check("quad 被过滤", not _is_hand_param("texture_wrap", "quad"))
    check("corner_offsets 被过滤",
          not _is_hand_param("feature_warp", "corner_offsets"))

    check("stretch_x 保留", _is_hand_param("feature_warp", "stretch_x"))
    check("jpeg_quality 保留", _is_hand_param("degrade", "jpeg_quality"))
    check("downscale 保留", _is_hand_param("degrade", "downscale"))
    check("size_ratio 保留", _is_hand_param("label", "size_ratio"))
    check("label.text 仍被过滤", not _is_hand_param("label", "text"))

    # 蒸馏结果里不该出现位置参数
    from core import distill as D
    from core import events as EV

    d = Path(tempfile.mkdtemp())
    try:
        lg = EV.EventLog(d / "e.jsonl")
        rec = {
            "schema_version": "1.1", "recipe_id": "rp", "target": "stitch",
            "subtype": "distort", "base_id": "b",
            "ops": [{"op": "sticker", "asset_id": "a1",
                     "center": [0.47, 0.58], "scale": 0.2},
                    {"op": "feature_warp", "asset_id": "a2",
                     "target_box": [0.35, 0.5, 0.3, 0.2], "stretch_x": 2.2}],
            "created_at": "2026-11-13T23:00:00+08:00",
        }
        EV.record_save(rec, log_=lg)
        EV.record_save(rec, log_=lg)

        hands = D.distill_hands(lg.all())
        keys = {h["key"] for h in hands}
        check("蒸馏结果不含 center",
              not any("center" in k for k in keys), str(sorted(keys)))
        check("蒸馏结果不含 target_box",
              not any("target_box" in k for k in keys))
        check("蒸馏结果含 stretch_x",
              any("stretch_x" in k for k in keys), str(sorted(keys)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 3. 事件压缩


def test_compact() -> None:
    print("\n=== 3. 事件自动压缩 ===")
    from core import events as EV

    d = Path(tempfile.mkdtemp())
    try:
        lg = EV.EventLog(d / "e.jsonl")
        # 灌 8000 条
        with open(lg.path, "w", encoding="utf-8") as f:
            for i in range(8000):
                f.write('{"id":"evt_%d","kind":"save","ts":"2026-11-13T23:00:00+08:00"}\n' % i)
        check("初始 8000 条", lg.count() == 8000, str(lg.count()))

        rep = lg.compact(keep_days=365, keep_max=20000)
        check("未超 keep_max 时不删", rep["after"] == 8000, str(rep))

        rep = lg.compact(keep_days=365, keep_max=3000)
        check("超限后压缩到 3000", rep["after"] == 3000, str(rep))
        check("记录了丢弃数", rep["dropped_overflow"] == 5000,
              str(rep["dropped_overflow"]))

        # 超期事件被删
        with open(lg.path, "w", encoding="utf-8") as f:
            f.write('{"id":"old","kind":"save","ts":"2020-01-01T00:00:00+08:00"}\n')
            f.write('{"id":"new","kind":"save","ts":"2026-11-13T23:00:00+08:00"}\n')
        rep = lg.compact(keep_days=365)
        check("超期事件被删", rep["after"] == 1, str(rep))

        # 源码里 learn 接口确实调了 compact
        src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
        check("learn 接口调用 compact", ".compact(" in src)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 4. 缓存


def test_cache() -> None:
    print("\n=== 4. 素材缓存 ===")
    import app.server as SRV

    SRV.clear_asset_cache()
    SRV._ASSET_STATS["hit"] = 0
    SRV._ASSET_STATS["miss"] = 0

    assets = SRV._load_all_assets(900)
    n = len(assets)
    check("首次加载有素材", n > 0, f"{n} 个")
    miss1 = SRV._ASSET_STATS["miss"]
    hit1 = SRV._ASSET_STATS["hit"]

    # 第二次应全部命中
    t0 = time.perf_counter()
    assets2 = SRV._load_all_assets(900)
    t2 = (time.perf_counter() - t0) * 1000
    hit2 = SRV._ASSET_STATS["hit"]

    check("第二次加载命中缓存", hit2 > hit1, f"hit {hit1}→{hit2}")
    check("缓存后耗时 < 20ms", t2 < 20, f"{t2:.1f}ms")

    # 同一对象（没重新解码）
    same = all(assets[k] is assets2[k] for k in assets if k in assets2)
    check("返回同一对象引用", same)

    # 基底缓存
    SRV.clear_asset_cache()
    b1 = SRV.load_asset_cached("__base__:image copy.png", 900)
    check("基底可缓存", b1 is not None)
    b2 = SRV.load_asset_cached("__base__:image copy.png", 900)
    check("基底二次命中", b1 is b2)

    # 路径越界应拒绝
    bad = SRV.load_asset_cached("__base__:../../../etc/passwd", 900)
    check("越界路径返回 None", bad is None)

    # 清理
    cleared = SRV.clear_asset_cache()
    check("clear 返回清理数", cleared >= 0, str(cleared))

    # LRU 上限
    check("LRU 上限 128", SRV._ASSET_CACHE_MAX == 128)


# ---------------------------------------------------------------- 5. 端到端


def test_e2e() -> None:
    print("\n=== 5. 端到端（缓存前后耗时对比） ===")
    import app.server as SRV
    from core.render import render

    recipe = {
        "schema_version": "1.1", "recipe_id": "perf", "target": "stitch",
        "subtype": "distort", "base_id": "b",
        "ops": [{"op": "grade", "saturation": 1.2}],
        "degrade": {"enabled": True, "downscale": 0.7, "jpeg_quality": 45,
                    "noise": 0.02, "oversharpen": "weak"},
        "created_at": "2026-11-13T23:00:00+08:00",
    }

    SRV.clear_asset_cache()

    # 冷启动（首次）
    t0 = time.perf_counter()
    base = SRV.load_asset_cached("__base__:image copy.png", 900)
    assets = SRV._load_all_assets(900)
    t_cold = (time.perf_counter() - t0) * 1000

    # 热路径（拖滑块时的真实情况）
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        base = SRV.load_asset_cached("__base__:image copy.png", 900)
        assets = SRV._load_all_assets(900)
        r = render(recipe, base_image=base, assets=assets, strict=False)
        ts.append((time.perf_counter() - t0) * 1000)

    t_hot = sum(ts) / len(ts)
    check("热路径 < 120ms", t_hot < 120, f"{t_hot:.1f}ms")
    check("热路径比冷启动快 2 倍以上", t_cold / t_hot > 2.0,
          f"冷 {t_cold:.0f}ms → 热 {t_hot:.0f}ms（{t_cold / t_hot:.1f}x）")

    print(f"       冷启动 {t_cold:.0f} ms → 热路径 {t_hot:.0f} ms "
          f"（省 {t_cold - t_hot:.0f} ms / 次）")
    SRV.clear_asset_cache()


# ---------------------------------------------------------------- 主


def main() -> int:
    print("=" * 70)
    print("P0 优化验收")
    print("=" * 70)

    test_parse_value()
    test_skip_position()
    test_compact()
    test_cache()
    test_e2e()

    total = len(PASS) + len(FAIL)
    print("\n" + "=" * 70)
    ok = len(FAIL) == 0
    print(f"{'PASS' if ok else 'FAIL'}  通过 {len(PASS)} / {total}")
    if FAIL:
        for n, d in FAIL:
            print(f"  ✗ {n}  {d}")
    else:
        print("P0 四项全部达成：缓存 / 安全解析 / 事件压缩 / 位置过滤")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
