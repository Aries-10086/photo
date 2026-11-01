"""把图片丢进 assets/cut，供校准器使用。

用法：
  # 把根目录的参考图里「砂金的眼/嘴」这类素材切出来太麻烦，
  # 这个工具只做最简单的事：整图入素材库，顺便可选抠背景。

  python tools/add_asset.py image.png
  python tools/add_asset.py image.png --cut          # 抠背景（需要模型，暂未接）
  python tools/add_asset.py image.png --name kobe_face
  python tools/add_asset.py a.png b.png c.png        # 批量

当前 --cut 未实现（M1 接 RMBG），会明确提示。
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


def add(src: Path, name: str | None, cut: bool, max_side: int) -> Path | None:
    if not src.exists():
        print(f"  ✗ 不存在: {src}")
        return None

    aid = name or src.stem
    dst = store.CUT_DIR / f"{aid}.png"
    store.CUT_DIR.mkdir(parents=True, exist_ok=True)

    if cut:
        print("  ! --cut 需要抠图模型，M1 才接。本次按原图入库。")

    img = imageio.read(src, keep_alpha=True, limit=max_side)
    imageio.write(img, dst)

    # 同时把原图留一份在 raw/
    if src.resolve().parent != store.RAW_DIR.resolve():
        store.RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_dst = store.RAW_DIR / src.name
        if not raw_dst.exists():
            shutil.copy2(src, raw_dst)

    alpha = "有 alpha" if img.mode == "RGBA" else "无 alpha"
    print(f"  ✓ {aid}  {img.width}×{img.height}  {alpha}  → {dst.relative_to(ROOT)}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="把图片加入素材库")
    ap.add_argument("files", nargs="+", help="图片路径")
    ap.add_argument("--name", help="指定素材 id（只对单文件有效）")
    ap.add_argument("--cut", action="store_true", help="抠背景（暂未实现）")
    ap.add_argument("--max-side", type=int, default=2048)
    args = ap.parse_args()

    if args.name and len(args.files) > 1:
        print("--name 只对单个文件有效")
        return 2

    ok = 0
    for f in args.files:
        p = Path(f)
        if not p.is_absolute():
            p = ROOT / p
        if add(p, args.name, args.cut, args.max_side):
            ok += 1

    print(f"\n入库 {ok}/{len(args.files)} 个。当前素材总数: {len(store.scan())}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
