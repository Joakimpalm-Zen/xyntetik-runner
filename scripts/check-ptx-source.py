#!/usr/bin/env python3
"""Compile CUDA with the committed toolchain pin and compare normalized PTX."""
import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    # embed-ptx.py deliberately down-pins this directive. Comments and spacing
    # are not PTX instructions; register names/order and operands still are.
    text = re.sub(r"(?m)^\.version\s+\S+", ".version 7.8", text)
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    args = ap.parse_args()
    header = (ROOT / "src/kernels_ptx.h").read_text()
    embedded = "".join(ast.literal_eval(line.strip()) for line in header.splitlines()
                       if line.strip().startswith('"'))
    pin = re.search(r"Cuda compilation tools, release [^,]+, (V[\d.]+)", embedded)
    actual = subprocess.check_output([args.nvcc, "--version"], text=True)
    if not pin or pin[1] not in actual:
        raise SystemExit(f"nvcc must match committed PTX compiler {pin[1] if pin else 'UNKNOWN'}; got {actual}")
    with tempfile.TemporaryDirectory(prefix="ptx-check-") as td:
        output = Path(td) / "kernels.ptx"
        cmd = [args.nvcc, "-ptx", "-arch=compute_75", "-O3", "src/kernels.cu", "-o", str(output)]
        if os.environ.get("NVCC_CCBIN"):
            cmd[1:1] = ["-ccbin", os.environ["NVCC_CCBIN"]]
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=300)
        if normalize(embedded) != normalize(output.read_text()):
            out = ROOT / ".build/ci/regenerated.ptx"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(output.read_bytes())
            raise SystemExit(f"CUDA source differs from embedded PTX; review {out}. No files regenerated in src/.")
    print("CUDA source matches embedded PTX with the pinned compiler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
