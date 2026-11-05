"""把图片丢进 assets/cut，供校准器使用。

用法：
  python tools/add_asset.py image.png                    # 整图入库
  python tools/add_asset.py image.png --cut              # 抠背景
  python tools/add_asset.py image.png --name kobe_face   # 指定 id
  python tools/add_asset.py a.png b.png --cut            # 批量抠图

抠图用 u2netp（4.4MB ONNX）。首次使用需先下模型：
  python tools/fetch_model.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import imageio, store  # noqa: E402

ALLOWED = imageio.ALLOWED_IN


def add(src: Path, name: str | None, cut: bool, max_side: int,
        erode: int, feather: int, threshold: float) -> Path | None:
    if not src.exists():
        print(f"  ✗ 不存在: {src}")
        return None

    aid = name or src.stem
    dst = store.CUT_DIR / f"{aid}.png"
    store.CUT_DIR.mkdir(parents=True, exist_ok=True)

    img = imageio.read(src, keep_alpha=True, limit=max_side)

    if cut:
        from core import cutout as C
        if not C.available():
            print(f"  ✗ 抠图模型不存在，先运行: python tools/fetch_model.py")
            return None
        try:
            img = C.cutout(img, erode_px=erode, feather_px=feather, threshold=threshold)
            a = __import__("numpy").asarray(img.getchannel("A"))
            cov = float((a > 128).mean()) * 100
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ 抠图失败: {type(e).__name__}: {e}")
            return None
    else:
        cov = None

    imageio.write(img, dst)

    # 留一份原图在 raw/
    if src.resolve().parent != store.RAW_DIR.resolve():
        store.RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_dst = store.RAW_DIR / src.name
        if not raw_dst.exists():
            shutil.copy2(src, raw_dst)

    tag = f"  主体 {cov:.1f}%" if cov is not None else ""
    print(f"  ✓ {aid}  {img.width}×{img.height}  {img.mode}{tag}  → {dst.relative_to(ROOT)}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="把图片加入素材库")
    ap.add_argument("files", nargs="+", help="图片路径")
    ap.add_argument("--name", help="指定素材 id（只对单文件有效）")
    ap.add_argument("--cut", action="store_true", help="抠背景（用 u2netp）")
    ap.add_argument("--max-side", type=int, default=2048,
                    help="长边上限，默认 2048")
    ap.add_argument("--erode", type=int, default=1,
                    help="alpha 边缘收缩像素，去掉残留背景色，默认 1")
    ap.add_argument("--feather", type=int, default=1,
                    help="alpha 羽化像素，默认 1")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="抠图二值化阈值，>0.5 抠得更紧，默认 0.5")
    args = ap.parse_args()

    if args.name and len(args.files) > 1:
        print("--name 只对单个文件有效")
        return 2

    ok = 0
    for f in args.files:
        p = Path(f)
        if not p.is_absolute():
            p = ROOT / p
        if add(p, args.name, args.cut, args.max_side,
               args.erode, args.feather, args.threshold):
            ok += 1

    if args.cut:
        try:
            from core import cutout as C
            C.release()
        except Exception:  # noqa: BLE001
            pass

    print(f"\n入库 {ok}/{len(args.files)} 个。当前素材总数: {len(store.scan())}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
