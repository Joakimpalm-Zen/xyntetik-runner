#!/usr/bin/env python3
"""No raw floating-point class test in a -ffast-math translation unit.

The engine builds with -ffast-math, whose -ffinite-math-only lets the
compiler assume no NaN or infinity exists: clang 22 folds isfinite() to 1
and isnan() to 0, deleting the check. src/fpclass.h holds class tests that
survive it; this refuses isnan/isfinite/isinf/fpclassify/isnormal (and their
__builtin_ forms) anywhere else in src/, outside the units the Makefile
builds without fast math. Comments and string literals do not count.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
# built with -fno-fast-math (QUANTS_CFLAGS), or the helper itself, or the
# portable-math header whose callers compile it without fast math
EXEMPT = {"quants.c", "quantize.c", "pmath.h", "fpclass.h"}
CLASS = re.compile(r"\b(?:__builtin_)?(isnan|isfinite|isinf|fpclassify|isnormal)\s*\(")
STRIP = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)


def blank(m):
    return re.sub(r"[^\n]", " ", m.group(0))


def main():
    bad = []
    files = sorted(p for ext in ("*.c", "*.h", "*.m") for p in (ROOT / "src").glob(ext))
    if not files:
        raise SystemExit("no sources under src/; nothing was checked")
    for p in files:
        if p.name in EXEMPT:
            continue
        code = STRIP.sub(blank, p.read_text(encoding="utf-8", errors="replace"))
        for m in CLASS.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            bad.append(f"{p.relative_to(ROOT)}:{line}: {m.group(1)}() in a -ffast-math unit "
                       f"(use fp_{m.group(1)} from src/fpclass.h)" if m.group(1) in ("isnan", "isfinite")
                       else f"{p.relative_to(ROOT)}:{line}: {m.group(1)}() in a -ffast-math unit "
                            "(add a fast-math-safe form to src/fpclass.h)")
    if bad:
        print("\n".join(bad))
        return 1
    print(f"fpclass: {len(files)} sources, no raw class test in a fast-math unit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
