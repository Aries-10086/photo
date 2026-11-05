"""下载 u2netp (4.4MB) 并实测抠图质量与速度。"""

import sys
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\吴金昊\Desktop\崩铁牢图棒")
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
DST = MODEL_DIR / "u2netp.onnx"


def download():
    if DST.exists():
        print(f"已存在: {DST.name} ({DST.stat().st_size / 2**20:.1f} MB)")
        return True

    print(f"下载中 -> {DST}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(DST, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            last = 0
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and time.time() - last > 10:
                    last = time.time()
                    el = time.time() - t0
                    pct = done / total * 100
                    sp = done / 2**20 / max(el, 0.1)
                    print(f"  {pct:5.1f}%  {done / 2**20:.1f}/{total / 2**20:.1f} MB  "
                          f"{sp:.2f} MB/s  已用 {el:.0f}s")
    except Exception as e:
        print(f"下载失败: {e}")
        if DST.exists():
            DST.unlink()
        return False

    el = time.time() - t0
    print(f"完成: {DST.stat().st_size / 2**20:.1f} MB, 用时 {el:.0f}s")
    return True


if __name__ == "__main__":
    if not download():
        sys.exit(1)
