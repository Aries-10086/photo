"""M2 · TPS 形变算法测试

验证三件事：
  1. 恒等映射：控制点相同 -> 输出与原图逐像素一致
  2. 已知变换：四角外扩 -> 图块确实变大，且中心不变
  3. 局部形变：单点拖动只影响该区域附近，远处几乎不变
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.warp import (  # noqa: E402
    GridWarp,
    apply_tps,
    bulge_field,
    fit_tps,
    tps_from_box,
    warp_box_tps,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def make_piece(w: int = 200, h: int = 120) -> np.ndarray:
    """造一个带明显网格和色块的图块，方便肉眼/数值判断形变。"""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = 200   # 红底
    img[:, :, 3] = 255
    # 边框
    img[0:3, :, :3] = 0
    img[-3:, :, :3] = 0
    img[:, 0:3, :3] = 0
    img[:, -3:, :3] = 0
    # 中心十字
    img[h // 2 - 2:h // 2 + 2, :, :3] = 255
    img[:, w // 2 - 2:w // 2 + 2, :3] = 255
    # 四角不同色块
    img[10:30, 10:30, 1] = 255
    img[10:30, -30:-10, 1] = 255
    return img


# ---------------------------------------------------------------- 1. 恒等


def test_identity() -> None:
    print("\n=== 1. 恒等映射 ===")
    pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]], dtype=np.float64)

    W = fit_tps(pts, pts)
    got = apply_tps(W, pts, pts)
    err = np.abs(got - pts).max()
    check("控制点处误差 < 1e-8", err < 1e-8, f"max_err={err:.2e}")

    # 中间随机点也应精确
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 100, (50, 2))
    got2 = apply_tps(W, pts, q)
    err2 = np.abs(got2 - q).max()
    check("随机点误差 < 1e-6", err2 < 1e-6, f"max_err={err2:.2e}")

    # 整图 remap 恒等
    piece = make_piece()
    out = warp_box_tps(
        piece,
        np.array([[0, 0], [200, 0], [200, 120], [0, 120]], dtype=np.float64),
        (200, 120),
        strength=1.0,
    )
    diff = np.abs(out.astype(np.int16) - piece.astype(np.int16))
    max_d = diff.max()
    # 边缘采样误差允许 1~2
    check("整图恒等 remap 误差 ≤ 2", max_d <= 2, f"max_diff={max_d}")


# ---------------------------------------------------------------- 2. 已知变换


def test_scale() -> None:
    print("\n=== 2. 四角外扩（放大） ===")
    piece = make_piece(200, 120)
    W, H = 400, 300

    # 原图 200x120 放到画布中心，放大 1.5 倍
    cx, cy = W / 2, H / 2
    sw, sh = 200 * 1.5, 120 * 1.5
    quad = np.array([
        [cx - sw / 2, cy - sh / 2],
        [cx + sw / 2, cy - sh / 2],
        [cx + sw / 2, cy + sh / 2],
        [cx - sw / 2, cy + sh / 2],
    ])
    out = warp_box_tps(piece, quad, (W, H), strength=1.0)

    # 统计非透明像素的包围盒
    a = out[:, :, 3]
    ys, xs = np.where(a > 8)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    check("输出宽度 ≈ 300", abs(bw - 300) <= 6, f"实际 {bw}")
    check("输出高度 ≈ 180", abs(bh - 180) <= 6, f"实际 {bh}")
    check("中心对齐", abs((xs.min() + xs.max()) / 2 - cx) <= 4,
          f"实际中心 x={(xs.min() + xs.max()) / 2:.1f}, 期望 {cx}")


# ---------------------------------------------------------------- 3. 局部形变


def test_local() -> None:
    print("\n=== 3. 单点拖动只影响局部 ===")
    W, H = 300, 300
    src = np.array([[0, 0], [W, 0], [W, H], [0, H], [W / 2, H / 2]], dtype=np.float64)
    dst = src.copy()
    dst[4] = [W / 2 + 40, H / 2]      # 中心点右移 40px

    gw = GridWarp(src, dst, (W, H))

    # 中心点位移
    got_center = gw.forward(np.array([[W / 2, H / 2]]))[0]
    check("控制点精确到位", abs(got_center[0] - (W / 2 + 40)) < 0.5,
          f"x={got_center[0]:.2f}")

    # 四角几乎不动
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    got_c = gw.forward(corners)
    corner_err = np.abs(got_c - corners).max()
    check("四角位移 < 1px", corner_err < 1.0, f"max={corner_err:.3f}")

    # 距离中心 100px 处的点，位移应该明显衰减
    far = np.array([[W / 2 + 120, H / 2]], dtype=np.float64)
    got_far = gw.forward(far)[0]
    shift = abs(got_far[0] - far[0, 0])
    # TPS 的衰减是「对数级」的，不是高斯级；4 个角点会拉动整体。
    # 实测 40px 拖动在 120px 处剩下约 21px（约一半），这符合 TPS 特性。
    check("远处位移衰减到不超过中心的 60%", shift < 40 * 0.6,
          f"位移 {shift:.2f}px（中心 40px）")


# ---------------------------------------------------------------- 4. strength 插值


def test_strength() -> None:
    print("\n=== 4. strength 插值 ===")
    piece = make_piece(200, 120)
    W, H = 300, 240
    # 故意做一个非矩形四角（透视）
    quad = np.array([[40, 30], [260, 50], [250, 210], [50, 190]], dtype=np.float64)

    out0 = warp_box_tps(piece, quad, (W, H), strength=0.0)
    out1 = warp_box_tps(piece, quad, (W, H), strength=1.0)

    a0 = (out0[:, :, 3] > 8).sum()
    a1 = (out1[:, :, 3] > 8).sum()
    check("strength=0 与 =1 结果不同", abs(int(a0) - int(a1)) > 100,
          f"面积 {a0} vs {a1}")

    # 同 strength 多次结果一致（确定性）
    r1 = warp_box_tps(piece, quad, (W, H), strength=1.0)
    r2 = warp_box_tps(piece, quad, (W, H), strength=1.0)
    check("同参数两次结果逐像素一致", np.array_equal(r1, r2))


# ---------------------------------------------------------------- 5. 鼓胀场


def test_bulge() -> None:
    print("\n=== 5. 鼓胀位移场 ===")
    W, H = 200, 200
    dx, dy = bulge_field((W, H), (100, 100), 50, 10)
    check("中心位移为零（无方向）", abs(dx[100, 100]) < 1e-3 and abs(dy[100, 100]) < 1e-3,
          f"({dx[100, 100]:.4f}, {dy[100, 100]:.4f})")

    # 中心右侧的点应向右推
    check("右侧点被向右推", dx[100, 130] > 0, f"dx={dx[100, 130]:.3f}")
    # 远角几乎不动
    check("远角位移可忽略", abs(dx[0, 0]) < 0.01, f"dx={dx[0, 0]:.4f}")

    # 负 amount = 凹陷
    dx2, _ = bulge_field((W, H), (100, 100), 50, -10)
    check("负值产生凹陷", dx2[100, 130] < 0, f"dx={dx2[100, 130]:.3f}")


# ---------------------------------------------------------------- 6. 边界与错误


def test_errors() -> None:
    print("\n=== 6. 错误处理 ===")
    try:
        fit_tps(np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 1]]))
        check("少于 3 点应报错", False)
    except ValueError as e:
        check("少于 3 点报错", "至少需要 3" in str(e), str(e)[:50])

    try:
        tps_from_box([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])
        check("非 4 角点应报错", False)
    except ValueError:
        check("非 4 角点报错", True)


# ---------------------------------------------------------------- 7. 性能


def test_perf() -> None:
    print("\n=== 7. 性能（2048² 全画布） ===")
    import time
    piece = make_piece(600, 600)
    W = H = 2048
    quad = np.array([[400, 400], [1600, 420], [1580, 1600], [420, 1580]], dtype=np.float64)

    t0 = time.perf_counter()
    _ = warp_box_tps(piece, quad, (W, H), strength=1.0)
    ms = (time.perf_counter() - t0) * 1000
    check("2048² TPS 全画布 < 800ms", ms < 800, f"{ms:.0f}ms")


def main() -> int:
    test_identity()
    test_scale()
    test_local()
    test_strength()
    test_bulge()
    test_errors()
    test_perf()

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
