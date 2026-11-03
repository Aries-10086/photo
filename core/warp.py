"""形变算法：TPS（薄板样条）+ 网格变形。

为什么需要 TPS：
    仿射变换只能做「整体缩放 + 旋转 + 斜切」，做不出「眼睛中间鼓出来、
    嘴角往下拽」这种局部形变。而 B 类牢图的笑点恰恰在局部畸变。

实现路线：
    TPS 拟合控制点对 -> 在目标网格上求映射 -> cv2.remap 采样

性能：
    2048×2048 的控制点求解是 O(n³)，n=10 时约 1ms；
    remap 是 O(像素数)，实测约 20ms。整体可接受。
"""

from __future__ import annotations

import numpy as np


def _kernel(r2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """TPS 径向基核 U(r) = r² log(r)。数值上对 0 做保护。"""
    return np.where(r2 < eps, 0.0, r2 * np.log(np.maximum(r2, eps)))


def fit_tps(src: np.ndarray, dst: np.ndarray, lam: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """拟合 TPS。

    Args:
        src: (n, 2) 源控制点，归一化或像素坐标均可（但需与 dst 同尺度）。
        dst: (n, 2) 目标控制点。
        lam: 正则化系数。0 = 严格插值（可能过拟合）；>0 = 平滑。
             牢图场景建议 0，因为要「精确到位」。

    Returns:
        (weights, affine) 两部分系数：
            weights: (n+3, 2) 非线性部分系数
            affine:  实际已并入 weights 的前 3 行
        为简化，返回单一矩阵 W (n+3, 2)，见 apply_tps。

    Raises:
        ValueError: 控制点少于 3 个。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 3:
        raise ValueError(f"TPS 至少需要 3 个控制点，当前 {n}")
    if n != len(dst):
        raise ValueError(f"控制点数量不匹配: src={n}, dst={len(dst)}")

    # 核矩阵 K
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    K = _kernel(d2)
    if lam > 0:
        K = K + np.eye(n) * lam

    # 仿射部分 P = [1, x, y]
    P = np.hstack([np.ones((n, 1)), src])

    # 组装 (n+3) × (n+3) 系统
    A = np.zeros((n + 3, n + 3), dtype=np.float64)
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T

    # 右侧
    b = np.zeros((n + 3, 2), dtype=np.float64)
    b[:n] = dst

    # 求解（A 是半正定，用 lstsq 更稳）
    W, *_ = np.linalg.lstsq(A, b, rcond=None)
    return W


def apply_tps(W: np.ndarray, src: np.ndarray, query: np.ndarray) -> np.ndarray:
    """对查询点求 TPS 映射值。

    Args:
        W: fit_tps 返回的 (n+3, 2) 系数。
        src: (n, 2) 拟合时用的源控制点。
        query: (m, 2) 要查询的点。

    Returns:
        (m, 2) 映射后的坐标。
    """
    src = np.asarray(src, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    n = len(src)

    d2 = ((query[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    K = _kernel(d2)                                    # (m, n)
    P = np.hstack([np.ones((len(query), 1)), query])    # (m, 3)
    A = np.hstack([K, P])                              # (m, n+3)
    return A @ W


def tps_from_box(
    src_quad: list | np.ndarray,
    dst_quad: list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用四角点做 TPS（最常见的用法）。

    Returns:
        (src, dst) 控制点数组，供后续 apply_tps 使用。
    """
    src = np.asarray(src_quad, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_quad, dtype=np.float64).reshape(-1, 2)
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("四角点必须各 4 个")

    # 加中心点，避免四角共线导致退化解
    sc = src.mean(axis=0, keepdims=True)
    dc = dst.mean(axis=0, keepdims=True)
    return np.vstack([src, sc]), np.vstack([dst, dc])


class GridWarp:
    """基于 TPS 的整图变形器。

    Usage:
        gw = GridWarp(src_points, dst_points, (W, H))
        out = gw.remap(image_array)
    """

    def __init__(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        size: tuple[int, int],
        *,
        lam: float = 0.0,
        supersample: int = 1,
    ):
        self.size = size
        self.lam = lam
        self.src = np.asarray(src_points, dtype=np.float64)
        self.dst = np.asarray(dst_points, dtype=np.float64)
        self.supersample = max(1, supersample)
        self._W = fit_tps(self.src, self.dst, lam=lam)

    def forward(self, pts: np.ndarray) -> np.ndarray:
        """源坐标 -> 目标坐标。"""
        return apply_tps(self._W, self.src, pts)

    def inverse_map(self) -> tuple[np.ndarray, np.ndarray]:
        """构造 cv2.remap 需要的反向映射表（目标 -> 源）。

        remap 的语义是：对输出图每个像素 (x,y)，去源图取位置 mapx[y,x], mapy[y,x]。
        所以我们要算的是「目标网格点对应回源的哪个位置」。

        做法：对目标网格用 TPS 正向算一遍得到覆盖范围，
              再对目标网格求逆向（用同一套控制点反置）。
        这里用更简单的办法：把 src/dst 互换拟合逆变换。
        """
        h, w = self.size[1], self.size[0]
        xs = np.arange(w, dtype=np.float64)
        ys = np.arange(h, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

        # 逆变换：dst -> src
        W_inv = fit_tps(self.dst, self.src, lam=self.lam)
        mapped = apply_tps(W_inv, self.dst, grid)

        mapx = mapped[:, 0].reshape(h, w).astype(np.float32)
        mapy = mapped[:, 1].reshape(h, w).astype(np.float32)
        return mapx, mapy


def warp_box_tps(
    piece: np.ndarray,
    target_quad: np.ndarray,
    out_size: tuple[int, int],
    *,
    strength: float = 1.0,
    local: float = 1.0,
    lam: float = 0.0,
) -> np.ndarray:
    """把一个矩形图块 TPS 形变到目标四边形区域。

    优化：只在目标包围盒内计算映射，外部直接留空。
    2048² 画布但图块只占 600×400 时，计算量降 90%+。

    Args:
        piece: (h, w, c) 源图块。
        target_quad: (4, 2) 目标四角点（像素坐标，顺序 左上/右上/右下/左下）。
        out_size: 输出画布尺寸 (W, H)。
        strength: 形变强度 0~1。0 = 仿射近似（不弯），1 = 完整 TPS。
        local: 局部性 0~1。0 = 完全仿射；1 = 完整 TPS。
        lam: 平滑正则化。>0 时 TPS 更平滑。

    Returns:
        (H, W, c) 形变后的图块（alpha 保留），超出画布的部分自然裁掉。
    """
    import cv2

    ph, pw = piece.shape[:2]
    W, H = out_size
    ch = piece.shape[2] if piece.ndim == 3 else 1
    out = np.zeros((H, W, ch), dtype=np.uint8)

    src_quad = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float64)

    tq = np.asarray(target_quad, dtype=np.float64)
    center = tq.mean(axis=0, keepdims=True)

    if strength < 1.0:
        aff = _affine_quad_dst(src_quad, tq)
        tq = tq * strength + aff * (1 - strength)

    src_ctrl = np.vstack([src_quad, src_quad.mean(axis=0, keepdims=True)])
    dst_ctrl = np.vstack([tq, center])

    if local < 1.0:
        mids = np.array([[pw / 2, 0], [pw, ph / 2],
                         [pw / 2, ph], [0, ph / 2]], dtype=np.float64)
        aff_all = _affine_quad_dst(np.vstack([src_quad, mids]),
                                   np.vstack([tq, mids]))
        src_ctrl = np.vstack([src_ctrl, mids])
        dst_ctrl = np.vstack([dst_ctrl, aff_all[4:]])
    # 目标包围盒（加边距，因为 TPS 可能让边缘略微外凸）
    pad = 4
    x0 = max(0, int(np.floor(tq[:, 0].min())) - pad)
    x1 = min(W, int(np.ceil(tq[:, 0].max())) + pad)
    y0 = max(0, int(np.floor(tq[:, 1].min())) - pad)
    y1 = min(H, int(np.ceil(tq[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return out

    # 只在 ROI 内构造网格
    roi_w, roi_h = x1 - x0, y1 - y0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    # 逆映射：目标点 -> 源点
    W_inv = fit_tps(dst_ctrl, src_ctrl, lam=lam)
    mapped = apply_tps(W_inv, dst_ctrl, grid)

    mapx = mapped[:, 0].reshape(roi_h, roi_w).astype(np.float32)
    mapy = mapped[:, 1].reshape(roi_h, roi_w).astype(np.float32)

    for c in range(ch):
        roi = cv2.remap(
            piece[:, :, c], mapx, mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[y0:y1, x0:x1, c] = roi

    return out if ch > 1 else out[:, :, 0]


def _affine_quad_dst(src_quad: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    """求最优仿射下所有源点应落在哪（用于 strength 插值）。

    用前三个点定仿射，其余点一并映射。支持任意点数（4 角 or 4 角 + 4 边中点）。
    """
    import cv2
    s = np.asarray(src_quad, dtype=np.float32)
    d = np.asarray(dst_quad, dtype=np.float32)
    M = cv2.getAffineTransform(s[:3], d[:3])
    n = len(s)
    ones = np.ones((n, 1), dtype=np.float32)
    homo = np.hstack([s, ones])
    return (homo @ M.T).astype(np.float64)


def bulge_field(
    size: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    amount: float,
) -> tuple[np.ndarray, np.ndarray]:
    """局部鼓胀/凹陷位移场（可选，用于「眼球凸出」这类效果）。

    与 TPS 独立，可以叠加。

    Args:
        size: (W, H)
        center: 中心（像素）
        radius: 影响半径（像素）
        amount: 位移量（像素）。正=向外鼓，负=向内凹。

    Returns:
        (dx, dy) 位移场，形状各为 (H, W) float32。
    """
    W, H = size
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = center
    dx0 = xs - cx
    dy0 = ys - cy
    dist = np.sqrt(dx0 ** 2 + dy0 ** 2) + 1e-6

    # 高斯衰减
    falloff = np.exp(-(dist / max(radius, 1e-6)) ** 2 * 2.0)
    scale = amount * falloff / dist
    return (dx0 * scale).astype(np.float32), (dy0 * scale).astype(np.float32)
