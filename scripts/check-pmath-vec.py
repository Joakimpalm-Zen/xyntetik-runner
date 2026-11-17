#!/usr/bin/env python3
"""The loops the portable exp sits in are vectorized, by the compiler's own report (R6.86).

Usage: check-pmath-vec.py --cc CC --source src/model.c -- <the compile command's arguments>

It compiles the source once with the compiler's vectorization remarks on:
  - clang: -Rpass=loop-vectorize and -Rpass-missed=loop-vectorize;
  - gcc: -fopt-info-vec-optimized and -fopt-info-vec-missed.
Each REQUIRED loop in LOOPS must have at least one "vectorized" remark at its line; a reported one is printed
with what the compiler said and decides nothing. The lines are found by their text in
the source, so an edit that moves them moves the check along; an edit that changes their text fails it, saying
which one is gone. A loop the compiler copied into several callers is reported once for each; one vectorized copy
is enough, and every remark at those lines is printed for the record.

The arguments must be a portable build of the engine's default flags (-ffast-math and -DRUNNER_PORTABLE_MATH):
make test-pmath-vec gives them. Under strict float (T3=1) the softmax's sum stays in order, which no vector loop
does, so the check says why and passes without reading anything.

Until R6.86 p_expf called pm_exp_d, whose early returns and pm_ldexp2's loops kept the softmax scalar: on Windows
the portable build was 8 to 15% slower on a prompt than the default (R6.85). A compiler's word is not a timing:
make bench-pmath-loops times the same loops.
"""
import re
import subprocess
import sys

# (what it is, the loop's text as it stands on one line of the source, how many times it appears, required)
# The softmax's exp loop is the one the portable exp kept scalar (R6.85, R6.86): required. The SwiGLU loop is
# reported: its body is gated_act's switch, whose GELU branches call tanhf and erff, and Apple clang keeps it scalar
# in the default build and in a portable one alike (it does not unswitch the switch), so it says nothing about the
# exp; on Windows gcc vectorized it in a portable build already on main (R6.85's compiler reports).
LOOPS = [
    ("the attention softmax's exp loop (softmax_sink, softmax)",
     "for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }", 2, True),
    ("the prompt FFN's SwiGLU (lffn_hact)",
     "for (int i = 0; i < J->nff; i++) ht[i] = gated_act(J->act, gt[i], ut[i]);", 1, False),
]


def family(cc):
    out = subprocess.run(cc.split() + ["--version"], capture_output=True, text=True).stdout
    if "clang" in out:
        return "clang"
    if re.search(r"\bgcc\b|GCC|Free Software Foundation", out):
        return "gcc"
    return None


def main(argv):
    if "--" not in argv:
        print(__doc__)
        return 2
    i = argv.index("--")
    opts, args = argv[:i], argv[i + 1:]
    cc = opts[opts.index("--cc") + 1]
    source = opts[opts.index("--source") + 1]
    if "-DRUNNER_PORTABLE_MATH" not in args:
        print("FAIL: not a portable build: the arguments must carry -DRUNNER_PORTABLE_MATH (make test-pmath-vec)")
        return 1
    if "-ffast-math" not in args or "-fno-fast-math" in args:
        print("test-pmath-vec: not run: strict float flags (T3=1) keep the softmax's sum in order, which a vector "
              "loop does not; this check reads the default flags")
        return 0
    fam = family(cc)
    if fam is None:
        print("FAIL: %s is neither clang nor gcc; the check reads only their remarks" % cc)
        return 1
    flags = (["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize"] if fam == "clang"
             else ["-fopt-info-vec-optimized", "-fopt-info-vec-missed"])
    text = open(source, encoding="utf-8").read().split("\n")
    want = {}
    for what, loop, count, required in LOOPS:
        lines = [n for n, ln in enumerate(text, 1) if ln.strip() == loop]
        if len(lines) != count:
            print("FAIL: %s: its loop is on %d line(s) of %s, not %d: %s" % (what, len(lines), source, count, loop))
            return 1
        want[what] = (lines, required)
    r = subprocess.run(cc.split() + args + flags, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        print("FAIL: the compile with remarks failed")
        return 1
    base = source.replace("\\", "/").rsplit("/", 1)[-1]
    at = {}
    for ln in (r.stdout + r.stderr).split("\n"):
        m = re.match(r"^(.*?):(\d+):\d+: (.*)$", ln.replace("\\", "/"))
        if not m or not m.group(1).endswith(base):
            continue
        at.setdefault(int(m.group(2)), []).append(m.group(3).strip())
    vec = (re.compile(r"remark: vectorized loop") if fam == "clang"
           else re.compile(r"optimized: +loop vectorized"))
    bad = []
    for what, (lines, required) in want.items():
        for n in lines:
            said = at.get(n, [])
            ok = any(vec.search(s) for s in said)
            print("pmath-vec: %s, %s:%d (%s): %s%s" % (what, source, n, fam, "vectorized" if ok else "NOT vectorized",
                                                       "" if required else " (reported, not required)"))
            for s in sorted(set(said)):
                print("    " + s)
            if required and not ok:
                bad.append("%s at %s:%d" % (what, source, n))
    if bad:
        print("FAIL: in a portable build these loops are not vectorized, so the portable exp costs a prompt its "
              "vector loop (R6.86):")
        for b in bad:
            print("  " + b)
        return 1
    print("pmath-vec: ok (%s): every required loop is vectorized in a portable build of the default flags" % fam)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
