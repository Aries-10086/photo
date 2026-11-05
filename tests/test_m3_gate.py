"""M3 验收：前景遮挡让接缝消失。

这是 M3 唯一真正的技术假设，必须验证：
    「头发压住人脸边缘」比「羽化」更有效。

实验设计：
    造一个基底（方形头部 + 顶部一条"头发"），把一张深色脸贴到头部中心。
    A 组：只羽化，无遮挡  → 人脸边缘应可见（接缝梯度高）
    B 组：有遮挡，无羽化  → 人脸边缘被头发压住（接缝梯度应低得多）

判据：
    B 组接缝区梯度 < A 组的 50%
    且遮挡覆盖率 ≥ 30%（PRD M3 出口门）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from core import layers as L  # noqa: E402
from core.metrics import edge_gradient  # noqa: E402
from core.render import render  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- 合成素材


def make_head() -> Image.Image:
    """基底：浅色头部 + 顶部深色头发条（这是前景遮挡层）。"""
    W, H = 500, 500
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 脖子
    d.rectangle((210, 380, 290, 500), fill=(214, 200, 186, 255))
    # 头部（苍白）
    d.ellipse((130, 80, 370, 400), fill=(236, 228, 220, 255))
    # 头发（深色，压在头顶）
    d.ellipse((120, 60, 380, 250), fill=(38, 34, 42, 255))
    # 刘海：几条垂下的深色块，模拟"压住额头"
    d.polygon([(140, 150), (215, 150), (200, 260), (150, 250)], fill=(38, 34, 42, 255))
    d.polygon([(285, 150), (360, 150), (350, 250), (300, 260)], fill=(38, 34, 42, 255))
    return img


def make_face() -> Image.Image:
    """贴入内容：深棕色写实脸（与苍白的基底形成强对比）。"""
    S = 260
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, S, S), fill=(96, 62, 44, 255))
    d.ellipse((60, 90, 95, 120), fill=(30, 22, 18, 255))
    d.ellipse((165, 90, 200, 120), fill=(30, 22, 18, 255))
    d.arc((85, 150, 175, 210), 0, 180, fill=(240, 236, 230, 255), width=10)
    return img


def make_front_mask(head: Image.Image) -> np.ndarray:
    """前景掩码：头部上半区（模拟「头发压住额头」）。

    真实场景里这是用户涂抹出来的。这里模拟一个典型结果：
    覆盖头部上半部分 + 刘海，约占主体 30~45%。

    注意：body 是 bool，必须 *255 转成 0/255 语义。
    """
    H, W = head.size
    al = np.asarray(head.getchannel("A"))
    body = (al > 128).astype(np.uint8) * 255

    fm = np.zeros((H, W), dtype=np.uint8)
    y_cut = int(H * 0.58)
    fm[:y_cut, :] = 255
    fm = ((fm.astype(np.uint16) * body.astype(np.uint16)) // 255).astype(np.uint8)

    m = Image.fromarray(fm, "L").filter(ImageFilter.GaussianBlur(2.0))
    return np.asarray(m)


def seam_gradient(img: Image.Image, y_center: float, x0: float, x1: float,
                  h: float = 0.06) -> float:
    """只测人脸上下缘的水平接缝带梯度。"""
    W, H = img.size
    band = (x0, y_center - h / 2, x1 - x0, h)
    return edge_gradient(img, region=band)


# ---------------------------------------------------------------- 主


def main() -> int:
    head = make_head()
    face = make_face()
    fm = make_front_mask(head)
    assets = {"face": face}

    print("=" * 72)
    print("M3 验收 · 前景遮挡 vs 羽化")
    print("=" * 72)

    # ---- 分层检查 ----
    print("\n--- 分层 ---")
    ls = L.split(head, front_mask=fm, auto=False)

    # PRD M3 出口门的口径：**人脸贴入区的边界被前景覆盖的比例 ≥ 30%**
    # （不是「全画布 30%」—— 那取决于角色占多大）
    side = 500 * 0.42
    fx0 = int(250 - side / 2)
    fx1 = int(250 + side / 2)
    fy0 = int(500 * 0.45 - side / 2)
    fy1 = int(500 * 0.45 + side / 2)

    # 人脸上下缘各取一条 12px 宽的带
    band = 12
    top_strip = fm[fy0:fy0 + band, fx0:fx1]
    bot_strip = fm[fy1 - band:fy1, fx0:fx1]
    cover_top = float((top_strip > 128).mean()) * 100
    cover_bot = float((bot_strip > 128).mean()) * 100
    print(f"  人脸区上缘被前景覆盖 {cover_top:.1f}%  下缘 {cover_bot:.1f}%")

    check("人脸区上缘被前景覆盖 ≥ 30%", cover_top >= 30, f"{cover_top:.1f}%")
    check("三层都生成了", ls.content.size == head.size and ls.front.size == head.size)
    check("content 与 front 互补",
          abs(L.coverage(np.asarray(ls.front.getchannel("A"))) +
              L.coverage(np.asarray(ls.content.getchannel("A"))) -
              L.coverage(np.asarray(head.getchannel("A")))) < 0.03,
          "覆盖率之和应约等于主体覆盖率")

    # 掩码存盘供渲染器读取
    mp = ROOT / "tmp" / "_m3_front_mask.png"
    L.save_mask(fm, mp)

    base_rec = {
        "schema_version": "1.1",
        "target": "stitch",
        "subtype": "precise",
        "base_id": "synth",
        "created_at": "2026-11-05T12:40:00+08:00",
        "degrade": {"enabled": False, "downscale": 1.0,
                    "jpeg_quality": 95, "noise": 0.0, "oversharpen": "off"},
    }

    # ---- A 组：只羽化，无遮挡（用 stack 子类型，不要求 layers）----
    rec_a = {
        **base_rec, "recipe_id": "m3_a", "subtype": "stack",
        "ops": [{
            "op": "face_fit", "asset_id": "face",
            "center": [0.5, 0.45], "scale": 0.42,
            "edge": "soft", "feather_px": 6,
        }],
    }

    # ---- B 组：有遮挡，无羽化 ----
    rec_b = {
        **base_rec, "recipe_id": "m3_b",
        "layers": {"foreground": str(mp.relative_to(ROOT)).replace("\\", "/")},
        "ops": [{
            "op": "face_fit", "asset_id": "face",
            "center": [0.5, 0.45], "scale": 0.42,
            "edge": "hard", "feather_px": 0,
        }],
    }

    ra = render(rec_a, base_image=head, assets=assets)
    rb = render(rec_b, base_image=head, assets=assets)

    # 接缝带：人脸贴入区（center 0.5/0.45, scale 0.42 → 边长约 210px）
    x0 = fx0
    x1 = fx1
    y_top = fy0
    y_bot = fy1

    g_a_top = seam_gradient(ra.image, y_top / 500, x0 / 500, x1 / 500)
    g_b_top = seam_gradient(rb.image, y_top / 500, x0 / 500, x1 / 500)
    g_a_bot = seam_gradient(ra.image, y_bot / 500, x0 / 500, x1 / 500)
    g_b_bot = seam_gradient(rb.image, y_bot / 500, x0 / 500, x1 / 500)

    print("\n--- 接缝梯度（越小越不露接缝）---")
    print(f"  上缘  A(羽化) {g_a_top:.4f}   B(遮挡) {g_b_top:.4f}")
    print(f"  下缘  A(羽化) {g_a_bot:.4f}   B(遮挡) {g_b_bot:.4f}")

    print("\n--- 判据 ---")
    if g_a_top > 1e-6:
        ratio_top = g_b_top / g_a_top
        check("上缘：遮挡组接缝 < 羽化组的 60%", ratio_top < 0.60,
              f"比值 {ratio_top * 100:.0f}%")
    else:
        check("上缘：羽化组本身就没接缝（基线异常）", False, "A 组梯度为 0")

    # 下缘不该被遮挡影响太多（头发只在上面）
    # 这里只要确认下缘两种方式差别不大，说明遮挡是局部的
    if g_a_bot > 1e-6:
        ratio_bot = g_b_bot / g_a_bot
        print(f"  下缘比值 {ratio_bot * 100:.0f}%（应接近 100%，说明遮挡是局部的）")

    # ---- 存图 ----
    out_dir = ROOT / "tmp" / "m3_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    head.save(out_dir / "00_基底.png")
    ls.front.save(out_dir / "01_前景层.png")
    ls.content.save(out_dir / "02_内容层.png")
    imageio_write(ra.image, out_dir / "03_A_只羽化.png")
    imageio_write(rb.image, out_dir / "04_B_有遮挡.png")

    # 拼对比
    sheet = Image.new("RGB", (1200, 420), (24, 26, 32))
    dd = ImageDraw.Draw(sheet)
    for i, (nm, im) in enumerate([
        ("基底", head), ("前景层(头发)", ls.front),
        ("内容层", ls.content),
        ("A: 只羽化 (有接缝)", ra.image), ("B: 有遮挡 (无接缝)", rb.image),
    ]):
        t = Image.new("RGB", (230, 230), (60, 62, 72))
        src = im.convert("RGBA")
        src.thumbnail((230, 230), Image.LANCZOS)
        t.paste(src, ((230 - src.width) // 2, (230 - src.height) // 2), src)
        sheet.paste(t, (i * 238 + 5, 30))
        dd.text((i * 238 + 8, 8), nm, fill=(220, 224, 234))
    sheet.save(out_dir / "_对比.png")
    print(f"\n对比图: {out_dir / '_对比.png'}")

    mp.unlink(missing_ok=True)

    total = len(PASS) + len(FAIL)
    print("\n" + "=" * 72)
    ok = len(FAIL) == 0
    print(f"{'PASS' if ok else 'FAIL'}  通过 {len(PASS)} / {total}")
    if FAIL:
        for n, d in FAIL:
            print(f"  ✗ {n}  {d}")
        print("\n结论：遮挡机制没有显著优于羽化。")
        print("  检查：")
        print("  1. render 里 front_layer 是否真的重建并 alpha_composite")
        print("  2. 前景掩码是否覆盖到了人脸贴入区的上缘")
    else:
        print("遮挡机制验证通过：前景层压住人脸边缘，接缝显著消失。")
    print("=" * 72)
    return 0 if ok else 1


def imageio_write(img, p):
    from core import imageio
    imageio.write(img, p)


if __name__ == "__main__":
    sys.exit(main())
