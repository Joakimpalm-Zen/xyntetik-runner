"""R3.30 step 3's kill read: the dynamic cache against the static hot file at
equal slots, with admission charged. Step 1's sim.py made an admitted expert
resident at once, for free, and compared against whole banks. Here the baseline
is RUNNER_MOE_HOT and an admission pays three ways:

  queue:  uploads run one at a time at T_UP ms; an expert serves only from the
          moment its upload lands, so a routing before that is still a miss
  victim: the evicted slot stops serving when its upload starts
  bytes:  each upload costs C_ADM ms of critical path a token (see project())

Scoring is decode routings only, as in step 1. Each prompt's hot set comes from
the other prompts (leave one out), or, for a held-out trace, from all of the
named ones. The cache starts from the hot set (W=hot) or from step 1's whole
banks (W=layers). Prefill admissions are either free as in step 1 (PRE=free)
or absent (PRE=none): the two bracket what a real prefill would allow.

Usage: sim_charged.py TRACEDIR [--held NAME=FILE,PROMPT_TOKENS] [--t-up MS]
       [--token-ms MS] [--slots B] [--rates 2,4,8,16]
"""
import collections
import glob
import gzip
import json
import os
import sys

NL, NE, TOPK = 28, 64, 8


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


D = sys.argv[1]
T_UP = float(arg("--t-up", 0.55))
TOKEN_MS = float(arg("--token-ms", 33.0))
B = int(arg("--slots", 768))
RATES = [int(x) for x in arg("--rates", "2,4,8,16").split(",")]


def load(path, pt):
    op = gzip.open if path.endswith(".gz") else open
    tok = collections.defaultdict(dict)
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            tok[r["pos"]][r["layer"]] = r["experts"]
    pos = sorted(tok)
    return {"pre": [tok[p] for p in pos if p < pt], "dec": [tok[p] for p in pos if p >= pt]}


runs = {}
for line in open(os.path.join(D, "runs.jsonl")):
    r = json.loads(line)
    runs[r["prompt"]] = r["usage"]["prompt_tokens"]
traces = {}
for f in sorted(glob.glob(os.path.join(D, "trace-*.jsonl*"))):
    name = os.path.basename(f).split(".")[0][6:]
    traces[name] = load(f, runs[name])
held = {}
for i, a in enumerate(sys.argv):
    if a == "--held":
        name, rest = sys.argv[i + 1].split("=")
        path, pt = rest.rsplit(",", 1)
        held[name] = load(path, int(pt))


def freq_of(tr):
    c = collections.Counter()
    for t in tr["pre"] + tr["dec"]:
        for l in t:
            for e in t[l]:
                c[(l, e)] += 1
    return c


freqs = {n: freq_of(t) for n, t in traces.items()}


def hot_for(name):
    c = collections.Counter()
    for o, f in freqs.items():
        if o != name:
            c.update(f)
    return [k for k, _ in c.most_common(B)]


def static_share(tr, hot):
    s = set(hot)
    n = h = 0
    for t in tr["dec"]:
        for l in t:
            for e in t[l]:
                n += 1
                h += (l, e) in s
    return h / n


def cache(tr, warm, R, pre, t_up):
    """Returns (served share of decode routings, uploads a decode token,
    routings lost to an upload in flight, a decode token's mean misses)."""
    slots = collections.OrderedDict((k, 0.0) for k in warm[:B])  # key -> usable from (ms)
    ghost = collections.OrderedDict()
    now = 0.0
    q_free = 0.0
    n = hit = adm = late = 0
    per_layer = TOKEN_MS / NL
    phases = (("pre", tr["pre"]), ("dec", tr["dec"])) if pre == "free" else (("dec", tr["dec"]),)
    for phase, toks in phases:
        dec = phase == "dec"
        for t in toks:
            budget = R
            for l in sorted(t):
                now += per_layer
                for e in t[l]:
                    k = (l, e)
                    if dec:
                        n += 1
                    if k in slots:
                        slots.move_to_end(k)
                        if slots[k] <= now:
                            hit += dec
                        else:
                            late += dec
                        continue
                    if k in ghost and budget > 0:
                        del ghost[k]
                        if len(slots) >= B:
                            slots.popitem(last=False)
                        q_free = max(q_free, now) + t_up
                        slots[k] = q_free
                        budget -= 1
                        adm += dec
                    else:
                        ghost[k] = 1
                        ghost.move_to_end(k)
                        if len(ghost) > B:
                            ghost.popitem(last=False)
    nt = max(1, len(tr["dec"]))
    return hit / n, adm / nt, late / nt, (n - hit) / nt


def line(name, tr, hot):
    hs = static_share(tr, hot)
    out = [f"{name:7} static {hs:.3f}"]
    rows = {}
    for R in RATES:
        for warm_name, warm in (("hot", hot), ("layers", [(l, e) for l in range(12) for e in range(NE)])):
            for pre in ("free", "none"):
                h0 = cache(tr, warm, R, pre, 0.0)     # step 1's free admission, for the delta
                h1 = cache(tr, warm, R, pre, T_UP)
                rows[(R, warm_name, pre)] = (h0[0], h1[0], h1[1], h1[2])
    return hs, rows


results = {}
for name, tr in traces.items():
    results[name] = line(name, tr, hot_for(name))
for name, tr in held.items():
    c = collections.Counter()
    for f in freqs.values():
        c.update(f)
    results[name] = line(name, tr, [k for k, _ in c.most_common(B)])

print(f"slots {B}, t_up {T_UP} ms, token {TOKEN_MS} ms; served share of decode routings")
print("prompt   static  | R  warm    prefill  free-adm  charged  uploads/tok  late/tok")
for name, (hs, rows) in results.items():
    for (R, w, p), (h0, h1, a, late) in rows.items():
        print(f"{name:8} {hs:.3f}  | {R:2} {w:7} {p:7} {h0:8.3f} {h1:8.3f} {a:11.2f} {late:9.2f}")

print("\nmean over the six leave-one-out prompts")
six = [n for n in results if n in traces]
ms = sum(results[n][0] for n in six) / len(six)
print(f"static {ms:.3f}")
for key in results[six[0]][1]:
    m = [sum(results[n][1][key][i] for n in six) / len(six) for i in range(4)]
    print(f"R={key[0]:2} warm={key[1]:7} prefill={key[2]:5} free-adm {m[0]:.3f} charged {m[1]:.3f} "
          f"(+{m[1]-ms:.3f} over static) uploads/tok {m[2]:.2f} late/tok {m[3]:.2f}")
json.dump({n: {"static": r[0], "rows": {"/".join(map(str, k)): v for k, v in r[1].items()}}
           for n, r in results.items()}, open("sim-charged.json", "w"), indent=0)
