"""R3.30 step 3: the charged projection per prompt, from measured constants.

T(h) is the warm decode time at served share h, a least-squares line through the
four static files read at 768 slots with no uploads. C is the critical-path cost
of one admission, the slope through the origin of the time the side load added
against uploads a token. Both come from matrix.log (see README.md).

A policy's projection is T(h_static) / (T(h_policy) + C * uploads_a_token).

Policies at 768 slots, all warm-started from the hot set:
  lru R       every slot evictable (step 1's policy)
  pin P R     the hot set's P most frequent experts never evicted; the other
              768 - P slots run the capped LRU

Usage: project.py TRACEDIR [--held NAME=FILE,PROMPT_TOKENS ...]
"""
import collections
import glob
import gzip
import json
import os
import sys

B = 768
# (decode served share, warm ms a token), medians of three interleaved rounds
CURVE = [(0.738, 1000 / 40.685), (0.824, 1000 / 49.42), (0.879, 1000 / 57.03), (0.916, 1000 / 62.705)]
# (uploads a token, ms a token added by the side load)
SIDE = [(4.0, 1000 / 45.40 - 1000 / 49.42), (7.57, 1000 / 48.975 - 1000 / 57.03),
        (10.45, 1000 / 51.71 - 1000 / 62.705)]

n = len(CURVE)
mx = sum(x for x, _ in CURVE) / n
my = sum(y for _, y in CURVE) / n
SLOPE = sum((x - mx) * (y - my) for x, y in CURVE) / sum((x - mx) ** 2 for x, _ in CURVE)
ICPT = my - SLOPE * mx
C = sum(x * y for x, y in SIDE) / sum(x * x for x, _ in SIDE)


def T(h):
    return ICPT + SLOPE * h


def load(path, pt):
    op = gzip.open if path.endswith(".gz") else open
    tok = collections.defaultdict(dict)
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            tok[r["pos"]][r["layer"]] = r["experts"]
    pos = sorted(tok)
    return [tok[p] for p in pos if p < pt], [tok[p] for p in pos if p >= pt]


D = sys.argv[1]
pts = {}
for line in open(os.path.join(D, "runs.jsonl")):
    r = json.loads(line)
    pts[r["prompt"]] = r["usage"]["prompt_tokens"]
traces = {}
for f in sorted(glob.glob(os.path.join(D, "trace-*.jsonl*"))):
    name = os.path.basename(f).split(".")[0][6:]
    traces[name] = load(f, pts[name])
held = {}
for i, a in enumerate(sys.argv):
    if a == "--held":
        name, rest = sys.argv[i + 1].split("=")
        p, pt = rest.rsplit(",", 1)
        held[name] = load(p, int(pt))


def freq(tr):
    c = collections.Counter()
    for t in tr[0] + tr[1]:
        for l in t:
            for e in t[l]:
                c[(l, e)] += 1
    return c


fr = {k: freq(v) for k, v in traces.items()}


def hot_order(skip):
    c = collections.Counter()
    for k, v in fr.items():
        if k != skip:
            c.update(v)
    return [k for k, _ in c.most_common(B)]


def static(dec, hot):
    s = set(hot)
    n = h = 0
    for t in dec:
        for l in t:
            for e in t[l]:
                n += 1
                h += (l, e) in s
    return h / n


def run(tr, hot, R, pin):
    pinned = set(hot[:pin])
    slots = collections.OrderedDict((k, 1) for k in reversed(hot[pin:]))
    ghost = collections.OrderedDict()
    cap = B - pin
    n = hit = adm = 0
    for dec, toks in ((False, tr[0]), (True, tr[1])):
        for t in toks:
            budget = R
            for l in sorted(t):
                for e in t[l]:
                    k = (l, e)
                    n += dec
                    if k in pinned:
                        hit += dec
                    elif k in slots:
                        slots.move_to_end(k)
                        hit += dec
                    elif k in ghost and budget > 0 and cap > 0:
                        del ghost[k]
                        slots.popitem(last=False)
                        slots[k] = 1
                        budget -= 1
                        adm += dec
                    else:
                        ghost[k] = 1
                        ghost.move_to_end(k)
                        if len(ghost) > B:
                            ghost.popitem(last=False)
    return hit / n, adm / max(1, len(tr[1]))


POL = [("lru R=4", 4, 0), ("lru R=8", 8, 0), ("pin 512 R=4", 4, 512), ("pin 512 R=8", 8, 512),
       ("pin 640 R=4", 4, 640), ("pin 640 R=2", 2, 640)]
print(f"T(h) = {ICPT:.2f} {SLOPE:+.2f} h ms (one routed expert {-SLOPE / 224:.3f} ms); "
      f"one admission {C:.3f} ms on the critical path")
print("prompt   static  T_stat | " + " | ".join(f"{p[0]:>18}" for p in POL))
allhot = hot_order(None)
res = collections.defaultdict(list)
for name, tr in list(traces.items()) + list(held.items()):
    hot = hot_order(name) if name in traces else allhot
    hs = static(tr[1], hot)
    cells = []
    for label, R, pin in POL:
        h, a = run(tr, hot, R, pin)
        g = T(hs) / (T(h) + C * a)
        res[label].append((name, g))
        cells.append(f"{h:.3f} {a:4.1f} {g:5.2f}x")
    print(f"{name:8} {hs:.3f} {T(hs):6.2f} | " + " | ".join(f"{c:>18}" for c in cells))
six = set(traces)
print("mean of six, leave one out: " + "  ".join(
    f"{k} {sum(g for n, g in v if n in six) / len(six):.3f}x" for k, v in res.items()))
print("prompts at 1.15x or more:   " + "  ".join(
    f"{k} {sum(g >= 1.15 for n, g in v if n in six)}/6" for k, v in res.items()))
print("prompts that lose (<1.00x): " + "  ".join(
    f"{k} {sum(g < 1.0 for n, g in v if n in six)}/6" for k, v in res.items()))
