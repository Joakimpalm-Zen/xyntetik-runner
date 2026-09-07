#!/usr/bin/env python3
"""Attribute a PTX change to the kernels whose source actually moved.

`make ptx` regenerates the whole header, so a two-line kernel edit produces a
several-thousand-line diff: nvcc renumbers virtual registers, basic-block
labels and local depots across the file, and adding a kernel shifts every
counter after it. A raw diff therefore cannot answer the only question that
matters, which is whether anything OTHER than the intended kernels changed.

The rule this serves (docs/cuda-microbatch-identity-2026-08-18.md, learned
after a regeneration that could not be attributed): compare the fixed
regeneration against a BASELINE REGENERATION FROM THE SAME BOX, never against
the committed header, because a different host compiler picks different
address arithmetic and that difference is not yours.

    make ptx && cp src/kernels.ptx /tmp/baseline.ptx && git checkout src/kernels_ptx.h
    <edit kernels.cu>
    make ptx
    python3 scripts/ptx-attribute.py /tmp/baseline.ptx src/kernels.ptx

Exit 0 when the only changes are the kernels named with --expect (plus any
added or removed entry points, which are always listed); exit 1 otherwise.
"""
import argparse
import re
import sys

ENTRY = re.compile(r"\.visible \.entry (\w+)")
# names nvcc numbers per file rather than per kernel: renumbering them is a
# consequence of where a kernel sits in the file, never of what it computes
TOKEN = re.compile(r"%[a-zA-Z]+\d+|\$L__BB\d+_\d+|__local_depot\d+"
                   r"|\$L__func_(?:begin|end)\d+")
KIND = re.compile(r"%[a-zA-Z]+|\$L__BB|__local_depot|\$L__func_begin"
                  r"|\$L__func_end")


def kernels(path):
    """entry name -> body lines, dropping the trailing `// .globl` that names
    the NEXT kernel and would otherwise report every kernel as changed."""
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ENTRY.match(line)
            if m:
                cur = m.group(1)
                out[cur] = []
            elif cur is not None and not line.lstrip().startswith("// .globl"):
                out[cur].append(line)
    return out


def normalize(lines):
    """Renumber file-scoped names by first appearance within the kernel, so
    distinct registers stay distinct and only the numbering is neutralized."""
    seen = {}

    def sub(m):
        s = m.group(0)
        if s not in seen:
            seen[s] = "%s#%d" % (KIND.match(s).group(0), len(seen))
        return seen[s]

    return [TOKEN.sub(sub, ln) for ln in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("regenerated")
    ap.add_argument("--expect", default="",
                    help="comma-separated kernels allowed to change")
    a = ap.parse_args()

    old, new = kernels(a.baseline), kernels(a.regenerated)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(k for k in old if k in new
                     and normalize(old[k]) != normalize(new[k]))

    print("kernels: %d baseline, %d regenerated" % (len(old), len(new)))
    for label, names in (("added", added), ("removed", removed),
                         ("changed", changed)):
        print("%-8s %s" % (label + ":", ", ".join(names) if names else "none"))

    expected = {s.strip() for s in a.expect.split(",") if s.strip()}
    if not expected:
        return 0
    unexpected = [k for k in changed if k not in expected]
    if unexpected:
        print("UNEXPECTED: %s" % ", ".join(unexpected), file=sys.stderr)
        return 1
    print("ok: every changed kernel was expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
