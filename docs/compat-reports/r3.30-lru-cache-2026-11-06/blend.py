"""R3.30 step 3: static hot files at a chosen served share, for reading T(h) with
no uploads. Swaps a base hot file's least used entries, as this trace counts
them, for the trace's most used entries the base lacks; the file keeps its 768
slots. These fit the traced text by construction: an instrument, not a proposal.
Usage: blend.py BASE_HOT TRACE OUTDIR  (writes hot-m1.txt to hot-m4.txt)
"""
import collections
import json
import os
import sys

BASE, TRACE, OUT = sys.argv[1:4]
base = [tuple(map(int, l.split())) for l in open(BASE) if l.strip() and not l.startswith("#")]
assert len(base) == 768
c = collections.Counter()
rec = []
for line in open(TRACE):
    r = json.loads(line)
    rec.append(r)
    for e in r["experts"]:
        c[(r["layer"], e)] += 1
order = [k for k, _ in c.most_common()]
bset = set(base)
add = [k for k in order if k not in bset]
drop = sorted(base, key=lambda k: c[k])


def hit(s):
    s = set(s)
    n = h = 0
    for r in rec:
        for e in r["experts"]:
            n += 1
            h += (r["layer"], e) in s
    return h / n


for tag, sw in (("m1", 60), ("m2", 130), ("m3", 210), ("m4", 330)):
    s = (bset - set(drop[:sw])) | set(add[:sw])
    assert len(s) == 768
    print(tag, "swapped", sw, "share of all routings", round(hit(s), 3))
    with open(os.path.join(OUT, f"hot-{tag}.txt"), "w") as f:
        f.write(f"# hot-all with {sw} swaps toward the traced text\n")
        f.write("".join(f"{l} {e}\n" for l, e in sorted(s)))
