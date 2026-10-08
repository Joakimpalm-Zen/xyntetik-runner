#!/usr/bin/env python3
"""Every CUDA kernel src/cuda.c names must exist in the embedded PTX.

cuda.c resolves its kernels with cuModuleGetFunction at init, and a name the
embedded src/kernels_ptx.h does not carry fails that init: the model then runs
on the CPU with nothing but a log line to say so (2026-10-08: a branch added
kernels to kernels.cu and cuda.c without regenerating kernels_ptx.h, and the
RTX 3070 gate ran every arm on the CPU). This needs no GPU, so it runs in
every `make test`.
"""
import re
import sys

src = open(sys.argv[1] if len(sys.argv) > 1 else "src/cuda.c").read()
ptx = open(sys.argv[2] if len(sys.argv) > 2 else "src/kernels_ptx.h").read()
names = sorted(set(re.findall(r'"(k_[A-Za-z0-9_]+)"', src)))
entries = set(re.findall(r"\.entry (k_[A-Za-z0-9_]+)\(", ptx))
missing = [n for n in names if n not in entries]
if missing:
    print("FAIL: kernels named in src/cuda.c are missing from src/kernels_ptx.h "
          "(regenerate it with `make ptx` on a CUDA box):")
    for n in missing:
        print("  " + n)
    sys.exit(1)
print("ptx roster: %d kernels named in cuda.c, all present in the embedded PTX (%d entries)"
      % (len(names), len(entries)))
