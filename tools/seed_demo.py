"""M4 演示数据：灌入一批真实感的事件，跑通学习闭环。

用于验证「越用越懂你」是否成立，以及给前端提供可视内容。

用法：
    python tools/seed_demo.py          # 灌数据 + 学习
    python tools/seed_demo.py --clear  # 先清空
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import distill as D  # noqa: E402
from core import events as EV  # noqa: E402
from core import memory  # noqa: E402


def recipe(rid: str, *, base: str, subtype: str, asset: str,
           label: str, anchor: str = "bottom_right",
           stretch_x: float = 1.0, jpeg: int = 95) -> dict:
    return {
        "schema_version": "1.1",
        "recipe_id": rid,
        "target": "stitch",
        "subtype": subtype,
        "base_id": base,
        "ops": [
            {"op": "feature_warp", "asset_id": asset,
             "target_box": [0.35, 0.5, 0.3, 0.2], "stretch_x": stretch_x},
            {"op": "degrade", "params": {"enabled": True, "downscale": 0.7,
                                         "jpeg_quality": jpeg, "noise": 0.02,
                                         "oversharpen": "weak"}},
            {"op": "label", "text": label, "anchor": anchor},
        ],
        "created_at": "2026-11-13T07:20:00+08:00",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="先清空事件与记忆")
    args = ap.parse_args()

    lg = EV.log()

    if args.clear:
        lg.rewrite([])
        st = memory.store()
        st._data = {}
        st.save()
        print("已清空事件与记忆")

    # ---- 场景：用户反复做 B 类畸变，偏爱砂金 + 牙膏嘴 ----
    # 第 1 次：先随便试，不满意，重生成
    r1 = recipe("r_demo_01", base="base_sands", subtype="distort",
                asset="darlie_mouth", label="牢肘子", stretch_x=1.0)
    EV.record_regenerate(r1, reason="不够歪", log_=lg)

    # 第 2 次：调大 stretch_x，保存
    EV.record_param_change(r1, log_=lg, op_index=0, op="feature_warp",
                           path="stretch_x", old=1.0, new=2.2)
    EV.record_save(r1, log_=lg)

    # 第 3 次：同样配置再来一张，保存
    r2 = recipe("r_demo_02", base="base_sands", subtype="distort",
                asset="darlie_mouth", label="牢肘子", stretch_x=2.2)
    EV.record_save(r2, log_=lg)

    # 第 4 次：换个基底试试（加拉赫），保存
    r3 = recipe("r_demo_03", base="base_gallagher", subtype="distort",
                asset="darlie_mouth", label="牢肘子", stretch_x=2.2)
    EV.record_save(r3, log_=lg)

    # 第 5 次：降质调得更狠，保存
    r4 = recipe("r_demo_04", base="base_sands", subtype="distort",
                asset="darlie_mouth", label="牢肘子", stretch_x=2.2, jpeg=35)
    EV.record_param_change(r4, log_=lg, op_index=1, op="degrade",
                           path="jpeg_quality", old=95, new=35)
    EV.record_save(r4, log_=lg)

    # 第 6 次：配文换位置试一次，不满意
    r5 = recipe("r_demo_05", base="base_sands", subtype="distort",
                asset="darlie_mouth", label="牢肘子", anchor="center",
                stretch_x=2.2)
    EV.record_regenerate(r5, reason="字放中间太挤", log_=lg)

    # 第 7 次：回到右下角，保存
    r6 = recipe("r_demo_06", base="base_sands", subtype="distort",
                asset="darlie_mouth", label="牢肘子", stretch_x=2.2, jpeg=35)
    EV.record_save(r6, log_=lg)

    # 第 8 次：C 类堆叠一次（少数派，验证 subtype_pref 不要被它带偏）
    r7 = recipe("r_demo_07", base="base_kobe", subtype="stack",
                asset="wubbaboo", label="", stretch_x=1.0)
    EV.record_save(r7, log_=lg)

    print(f"灌入 {lg.count()} 条事件")

    # ---- 学习 ----
    rep = D.learn()
    print(f"\n学习报告:")
    print(f"  候选 {rep['candidates']} 条，写入 {rep['written']} 条")
    print(f"  因证据不足跳过 {rep['skipped_low_evidence']} 条")
    print(f"  因置信度不足跳过 {rep['skipped_low_confidence']} 条")
    print(f"  按类型: {rep['by_type']}")

    # ---- 淘汰一次，确保状态一致 ----
    st = memory.store()
    pr = st.prune()
    st.save()
    if pr["archived"]:
        print(f"  归档 {len(pr['archived'])} 条低置信记忆")

    print(f"\n记忆库: {st.stats()}")
    print(f"文件: {st.path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
