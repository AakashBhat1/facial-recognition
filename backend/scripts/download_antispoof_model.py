"""Download a pretrained MiniFASNet anti-spoof ONNX model.

Fetches `MiniFASNetV2.onnx` from the public yakhyo/face-anti-spoofing release
and drops it at `backend/models_ml/antispoof.onnx` where `antispoof_detector`
expects to find it.

Run from project root:

    python backend/scripts/download_antispoof_model.py

The model:
- is ~1.7 MB
- takes a 80x80 face crop as NCHW float32 (BGR, no normalization beyond cast)
- outputs logits for 2 classes: [fake, real]  (argmax==1 => real)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_URL = (
    "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/"
    "MiniFASNetV2.onnx"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEST = REPO_ROOT / "backend" / "models_ml" / "antispoof.onnx"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        size = DEST.stat().st_size
        print(f"Model already present at {DEST}  ({size / 1024:.1f} KB)")
        print("Delete it first if you want to re-download.")
        return

    print(f"Downloading MiniFASNetV2 anti-spoof model...\n  from: {MODEL_URL}\n  to:   {DEST}")
    req = Request(MODEL_URL, headers={"User-Agent": "smart-locker-setup/1.0"})
    try:
        with urlopen(req, timeout=60) as resp, open(DEST, "wb") as f:
            data = resp.read()
            f.write(data)
    except Exception as exc:
        print(f"ERROR: download failed: {exc}")
        if DEST.exists():
            DEST.unlink()
        sys.exit(1)

    size_kb = DEST.stat().st_size / 1024
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()[:16]
    print(f"OK  {size_kb:.1f} KB  sha256={digest}...")
    print("\nNext:")
    print("  1. Add to backend/.env :  ANTISPOOF_ENABLED=true")
    print("  2. Restart the backend. You should see 'Anti-spoof model loaded' in the logs.")


if __name__ == "__main__":
    main()
