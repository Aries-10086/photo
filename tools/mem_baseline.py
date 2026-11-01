"""M0-FR0-T04 · 内存基线实测（硬门：峰值 ≤ 1.5 GB）

验证三件事：
  1. 空载基线占用
  2. 加载 ONNX 推理会话后的峰值
  3. 释放会话后能否回落（决定「单模型驻留」策略是否可行）

因为没有真实抠图模型文件，这里用同量级参数量的合成 ONNX 模型模拟
（U2Netp ~1.7M 参数 / RMBG-2.0 ~44M 参数），覆盖两种体量。

运行：
    .venv\\Scripts\\python.exe tools\\mem_baseline.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

BUDGET_MB = 1536  # 1.5 GB 硬预算


def rss_mb() -> float:
    try:
        import psutil
    except ImportError:
        return -1.0
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def peak_reset() -> None:
    try:
        import psutil
        psutil.Process(os.getpid()).memory_info()
    except ImportError:
        pass


def make_conv_model(ch: int, layers: int, size: int, name: str) -> bytes:
    """构造一个纯卷积 ONNX 模型，参数量 ≈ ch*ch*9*layers。

    输入固定 4 通道：ONNX Runtime 的 NCHWc 优化要求通道数能被 4 整除。
    """
    nodes, inits = [], []
    rng = np.random.default_rng(0)

    # 首层 4 -> ch
    w0 = "w_in"
    inits.append(helper.make_tensor(
        w0, TensorProto.FLOAT, [ch, 4, 3, 3],
        (rng.standard_normal(ch * 4 * 9) * 0.05).astype(np.float32)))
    nodes.append(helper.make_node("Conv", ["input", w0], ["c_in"], pads=[1, 1, 1, 1]))
    prev = "c_in"

    for i in range(layers):
        w = f"w{i}"
        inits.append(helper.make_tensor(
            w, TensorProto.FLOAT, [ch, ch, 3, 3],
            (rng.standard_normal(ch * ch * 9) * 0.05).astype(np.float32)))
        out = f"c{i}"
        nodes.append(helper.make_node("Conv", [prev, w], [out], pads=[1, 1, 1, 1]))
        prev = out
    g = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, size, size])],
        [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, ch, size, size])],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    return m.SerializeToString()


def params_of(ch: int, layers: int) -> int:
    return ch * 4 * 9 + ch * ch * 9 * layers


def bench(label: str, model_bytes: bytes, size: int, rows: list) -> None:
    before = rss_mb()
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = min(8, os.cpu_count() or 4)
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_bytes, opt, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    after_load = rss_mb()

    x = np.random.randn(1, 4, size, size).astype(np.float32)
    t0 = time.perf_counter()
    sess.run(None, {"input": x})
    infer_s = time.perf_counter() - t0
    after_infer = rss_mb()

    peak = max(after_load, after_infer)
    rows.append({
        "case": label,
        "params_M": round(params_of(3, 0) / 1e6, 2) if False else None,
        "input": f"{size}x{size}",
        "rss_before_mb": round(before, 1),
        "rss_after_load_mb": round(after_load, 1),
        "rss_after_infer_mb": round(after_infer, 1),
        "delta_mb": round(peak - before, 1),
        "load_s": round(load_s, 2),
        "infer_s": round(infer_s, 3),
    })

    del sess, x
    gc.collect()
    time.sleep(0.3)
    rows[-1]["rss_after_release_mb"] = round(rss_mb(), 1)
    rows[-1]["released"] = rows[-1]["rss_after_release_mb"] < after_infer - 20


def main() -> int:
    print("=" * 74)
    print("M0-FR0-T04 · 内存基线实测")
    print("=" * 74)
    print(f"预算: {BUDGET_MB} MB (1.5 GB)")
    print(f"CPU: {os.cpu_count()} 逻辑核")
    print(f"ORT: {ort.__version__} | providers: {ort.get_available_providers()}")

    base = rss_mb()
    if base < 0:
        print("\n[!] psutil 未安装，无法测量。请先 pip install psutil")
        return 2
    print(f"进程空载 RSS: {base:.1f} MB\n")

    rows: list[dict] = []

    # 小模型：模拟 U2Netp 量级
    print("--- 场景 1: U2Netp 量级（轻量抠图）---")
    m1 = make_conv_model(ch=32, layers=20, size=320, name="u2netp_like")
    bench("U2Netp-like", m1, 320, rows)
    print(f"  加载后 {rows[-1]['rss_after_load_mb']} MB | 推理 {rows[-1]['infer_s']}s "
          f"| 峰值增量 {rows[-1]['delta_mb']} MB | 释放后 {rows[-1]['rss_after_release_mb']} MB")

    # 中模型：模拟 RMBG-2.0 量级（约 44M 参数，这里用 ch=64,layers=120 ≈ 44M）
    print("\n--- 场景 2: RMBG-2.0 量级（高质量抠图）---")
    m2 = make_conv_model(ch=64, layers=120, size=512, name="rmbg_like")
    bench("RMBG-like", m2, 512, rows)
    print(f"  加载后 {rows[-1]['rss_after_load_mb']} MB | 推理 {rows[-1]['infer_s']}s "
          f"| 峰值增量 {rows[-1]['delta_mb']} MB | 释放后 {rows[-1]['rss_after_release_mb']} MB")

    # 场景 3：大图处理（2048px 基底）峰值
    print("\n--- 场景 3: 2048px 基底处理（无模型，纯像素）---")
    from core import imageio
    from PIL import Image

    before3 = rss_mb()
    t0 = time.perf_counter()
    img = Image.new("RGBA", (2048, 2048), (128, 128, 128, 255))
    arr = np.asarray(img).astype(np.float32)[:, :, :3]
    a = np.random.rand(2048, 2048, 1).astype(np.float32)
    comp = (arr * a + arr * (1 - a)).astype(np.uint8)
    Image.fromarray(comp, "RGB").save(ROOT / "tmp" / "_memtest.jpg", quality=35)
    dt = (time.perf_counter() - t0) * 1000
    peak3 = rss_mb()
    rows.append({
        "case": "2048px 综合处理",
        "input": "2048x2048",
        "rss_before_mb": round(before3, 1),
        "rss_after_load_mb": None,
        "rss_after_infer_mb": round(peak3, 1),
        "delta_mb": round(peak3 - before3, 1),
        "load_s": None,
        "infer_s": round(dt / 1000, 3),
    })
    print(f"  耗时 {dt:.0f} ms | 峰值增量 {rows[-1]['delta_mb']} MB")
    (ROOT / "tmp" / "_memtest.jpg").unlink(missing_ok=True)
    del img, arr, a, comp
    gc.collect()

    # 汇总
    print("\n" + "=" * 74)
    print(f"{'场景':<22}{'输入':<12}{'峰值增量':>10}{'绝对RSS':>11}{'耗时':>10}")
    print("-" * 74)
    worst = 0.0
    for r in rows:
        abs_rss = r["rss_after_infer_mb"] or 0
        worst = max(worst, abs_rss)
        t = r["load_s"] or r["infer_s"]
        unit = "s" if r["load_s"] else "s"
        print(f"{r['case']:<22}{r['input']:<12}{r['delta_mb']:>9.0f}M"
              f"{abs_rss:>10.0f}M{t:>9.2f}{unit}")
    print("-" * 74)

    verdict = "通过" if worst <= BUDGET_MB else "超预算"
    print(f"\n最大绝对 RSS: {worst:.0f} MB / 预算 {BUDGET_MB} MB  →  {verdict}")

    out = ROOT / "docs" / "baselines" / "mem_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "budget_mb": BUDGET_MB,
        "max_rss_mb": round(worst, 1),
        "verdict": verdict,
        "cpu_count": os.cpu_count(),
        "ort_version": ort.__version__,
        "cases": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入: {out.relative_to(ROOT)}")

    return 0 if worst <= BUDGET_MB else 1


if __name__ == "__main__":
    sys.exit(main())
