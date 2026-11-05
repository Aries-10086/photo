"""实测 u2netp 抠图：速度、内存、质量。"""

import sys
import os
sys.path.insert(0, r"C:\Users\吴金昊\Desktop\崩铁牢图棒")
os.chdir(r"C:\Users\吴金昊\Desktop\崩铁牢图棒")

import time
from pathlib import Path
from PIL import Image, ImageDraw

from core import cutout, imageio

ROOT = Path(r"C:\Users\吴金昊\Desktop\崩铁牢图棒")
out_dir = ROOT / "tmp" / "cutout_test"
out_dir.mkdir(parents=True, exist_ok=True)

tests = [
    ("image copy.png", "砂金（二次元，白灰装束）"),
    ("image.png", "加拉赫（二次元，暖色）"),
    ("image copy 2.png", "科比（真人照片）"),
]

print("=" * 78)
print("u2netp 抠图实测")
print("=" * 78)
print(f"模型: {cutout.model_path().name} "
      f"({cutout.model_path().stat().st_size / 2**20:.1f} MB)")
print()

rows = []
tiles = []
for fname, desc in tests:
    p = ROOT / "assets" / "raw" / fname
    if not p.exists():
        p = ROOT / fname
    if not p.exists():
        print(f"  跳过（不存在）: {fname}")
        continue

    img = imageio.read(p, keep_alpha=False, limit=900)
    info = cutout.benchmark(img)
    rows.append((fname, desc, info))

    # 二次正式抠图用于看图
    out = cutout.cutout(img)
    outp = out_dir / f"cut_{p.stem}.png"
    imageio.write(out, outp)
    tiles.append((fname, img, out))

    print(f"{desc}")
    print(f"  输入 {info['input']}  加载 {info['load_ms']:.0f}ms  推理 {info['infer_ms']:.0f}ms")
    print(f"  RSS {info['rss_before_mb']:.0f} -> {info['rss_after_infer_mb']:.0f} MB "
          f"(+{info['rss_delta_mb']:.0f})")
    print(f"  主体覆盖 {info['coverage_pct']:.1f}%")
    print()

# 拼图：左原图 / 右抠图(棋盘格底)
if tiles:
    tw = 300
    cols = len(tiles)
    sheet = Image.new("RGB", (cols * tw * 2, tw + 30), (26, 28, 34))
    d = ImageDraw.Draw(sheet)

    def checker(w, h, s=16):
        im = Image.new("RGB", (w, h), (60, 62, 70))
        dd = ImageDraw.Draw(im)
        for y in range(0, h, s):
            for x in range(0, w, s):
                if ((x // s) + (y // s)) % 2 == 0:
                    dd.rectangle((x, y, x + s - 1, y + s - 1), fill=(88, 90, 100))
        return im

    for i, (name, orig, cut) in enumerate(tiles):
        o = orig.copy()
        o.thumbnail((tw - 8, tw - 8), Image.LANCZOS)
        sheet.paste(o, (i * tw * 2 + 4, 26))
        d.text((i * tw * 2 + 6, 7), name, fill=(220, 224, 234))

        bgc = checker(tw - 8, tw - 8)
        c = cut.copy()
        c.thumbnail((tw - 8, tw - 8), Image.LANCZOS)
        bgc.paste(c, ((tw - 8 - c.width) // 2, (tw - 8 - c.height) // 2), c)
        sheet.paste(bgc, (i * tw * 2 + tw + 4, 26))
        d.text((i * tw * 2 + tw + 6, 7), "抠图结果", fill=(140, 220, 160))

    sp = out_dir / "_对比.png"
    sheet.save(sp)
    print(f"对比图: {sp}")

# 释放
cutout.release()
print("会话已释放")
