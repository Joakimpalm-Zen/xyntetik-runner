#!/usr/bin/env python3
"""Generate tiny valid loader/split seeds, without checking model weights into Git."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "fuzz-corpus")
    with tempfile.TemporaryDirectory(prefix="fuzz-seeds-") as td:
        tmp = Path(td)
        model = tmp / "tiny.gguf"
        subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(model)], check=True)
        out = dest / "model_load"
        out.mkdir(parents=True, exist_ok=True)
        (out / "valid-tiny.gguf").write_bytes(model.read_bytes())
        subprocess.run([sys.executable, str(ROOT / "scripts/gguf-split.py"), str(model), str(tmp / "part"), "2"], check=True)
        a, b = [p.read_bytes() for p in sorted(tmp.glob("part-*.gguf"))]
        out = dest / "gguf_split"
        out.mkdir(parents=True, exist_ok=True)
        (out / "valid-pair.bin").write_bytes(struct.pack("<I", len(a)) + a + b)
        (out / "truncated-part.bin").write_bytes(struct.pack("<I", len(a)) + a + b[:24])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
