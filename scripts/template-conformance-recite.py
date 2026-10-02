#!/usr/bin/env python3
"""Re-point the allowlist's line citations after the cited file moved.

Every deviation in scripts/template-conformance-allowlist.json cites a file,
a line range and an anchor string, and the gate fails when the anchor is no
longer inside the range. An edit above the cited lines moves all of them, so
any change to src/template.c makes a dozen citations stale at once. This
finds each anchor again and rewrites only the `lines` value, keeping the
range's width; an anchor that is gone from the file is reported and left
alone, because that is the case the gate exists to catch.

    scripts/template-conformance-recite.py [--check]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "scripts", "template-conformance-allowlist.json")


def main():
    check = "--check" in sys.argv[1:]
    raw = open(PATH, encoding="utf-8").read()
    doc = json.loads(raw)
    moved, dead = 0, []
    for d in doc["deviations"]:
        for key in ("cite", "corroborating_cite"):
            c = d.get(key)
            if not c:
                continue
            lines = open(os.path.join(ROOT, c["file"]), encoding="utf-8").read().split("\n")
            lo, hi = (int(x) for x in c["lines"].split("-"))
            if any(c["anchor"] in l for l in lines[lo - 1:hi]):
                continue
            hits = [i + 1 for i, l in enumerate(lines) if c["anchor"] in l]
            if not hits:
                dead.append("%s (%s)" % (d["id"], key))
                continue
            h = min(hits, key=lambda x: abs(x - lo))
            span = hi - lo
            new_lo = max(1, h - span // 2)
            new = "%d-%d" % (new_lo, new_lo + span)
            i = raw.index('"id": ' + json.dumps(d["id"]))
            j = raw.index('"anchor": ' + json.dumps(c["anchor"]), i)
            old = '"lines": ' + json.dumps(c["lines"])
            k = raw.rfind(old, i, j)
            if k < 0:
                dead.append("%s (%s): could not locate the lines field" % (d["id"], key))
                continue
            raw = raw[:k] + '"lines": ' + json.dumps(new) + raw[k + len(old):]
            moved += 1
    for x in dead:
        print("DEAD: " + x, file=sys.stderr)
    if check:
        print("%d citation(s) would move" % moved)
        return 1 if (moved or dead) else 0
    if moved:
        open(PATH, "w", encoding="utf-8").write(raw)
    print("%d citation(s) re-pointed" % moved)
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
