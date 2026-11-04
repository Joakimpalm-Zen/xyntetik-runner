"""R3.30 step 2: a RUNNER_MOE_HOT file from step 1's traces. The B most
frequent (layer, expert) pairs over the given prompts' routings, prefill and
decode, as sim.py's hot set picks them. Usage: hotset.py TRACEDIR B
[--leave-out NAME] > hot.txt"""
import collections
import glob
import gzip
import json
import os
import sys

D, B = sys.argv[1], int(sys.argv[2])
skip = sys.argv[sys.argv.index("--leave-out") + 1] if "--leave-out" in sys.argv else None
freq = collections.Counter()
names = []
for f in sorted(glob.glob(os.path.join(D, "trace-*.jsonl*"))):
    name = os.path.basename(f).split(".")[0][6:]
    if name == skip:
        continue
    names.append(name)
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            for e in r["experts"]:
                freq[(r["layer"], e)] += 1
hot = sorted(k for k, _ in freq.most_common(B))
print(f"# R3.30 hot set: the {B} most frequent (layer, expert) pairs over {','.join(names)}")
for l, e in hot:
    print(l, e)
