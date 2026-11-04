"""R3.30 step 1: trace statistics and the three placement policies at equal bytes.

Reads OUTDIR/trace-*.jsonl (RUNNER_MOE_TRACE) and OUTDIR/runs.jsonl (prompt_tokens),
scores DECODE routings only (pos >= prompt_tokens); prefill routings are observed
by the policies (they happen before decode) but never scored.

  (a) static layers: the shallowest K layers' banks resident (today's --cpu-moe auto)
  (b) static hot set: the B = 64 K most frequent (layer, expert) pairs, frequencies
      from the OTHER five prompts (leave one out), so nothing is fitted in sample
  (c) capped LRU: B slots, warm-started with (a)'s experts; a miss runs on the host;
      a missed key is admitted only on its second touch (a ghost list of B keys) and
      at most R admissions a token (the rate cap, uploads off the critical path),
      evicting the least recently used slot

Usage: sim.py OUTDIR K[,K...] [R[,R...]] [--model T_STATIC_MS,T_EXPERT_MS]
With --model the implied decode gain over (a) is projected:
  T(policy) = T_static - (h_policy - h_a) * 224 * t_e
"""
import collections
import glob
import json
import os
import sys

OUT = sys.argv[1]
KS = [int(x) for x in sys.argv[2].split(",")]
RS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else [4, 8, 16]
MODEL = None
if "--model" in sys.argv:
    MODEL = [float(x) for x in sys.argv[sys.argv.index("--model") + 1].split(",")]
NL, NE, TOPK = 28, 64, 8

runs = {json.loads(l)["prompt"]: json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))}
traces = {}
for f in sorted(glob.glob(os.path.join(OUT, "trace-*.jsonl"))):
    name = os.path.basename(f)[6:-6]
    pt = runs[name]["usage"]["prompt_tokens"]
    tok = collections.defaultdict(dict)          # pos -> layer -> experts
    for line in open(f):
        r = json.loads(line)
        tok[r["pos"]][r["layer"]] = r["experts"]
    pos = sorted(tok)
    traces[name] = {"pre": [tok[p] for p in pos if p < pt], "dec": [tok[p] for p in pos if p >= pt]}

def keys(t):
    return [(l, e) for l in sorted(t) for e in t[l]]

# ---- trace statistics (decode) ----
print("== trace statistics (decode tokens)")
print("prompt  prompt_tok  decode_tok  ws1  ws8  ws32  ws128  reuse_p50  reuse_p90  top32share  top16share")
allfreq = {}
for name, tr in traces.items():
    dec = tr["dec"]
    freq = collections.Counter(k for t in dec for k in keys(t))
    allfreq[name] = collections.Counter(k for t in tr["pre"] + dec for k in keys(t))
    # working set: distinct (layer, expert) over a window of W decode tokens, mean over windows
    ws = {}
    for W in (1, 8, 32, 128):
        vals = [len({k for t in dec[i:i + W] for k in keys(t)}) for i in range(0, max(1, len(dec) - W + 1), max(1, W // 2))]
        ws[W] = sum(vals) / len(vals) if vals else 0
    # reuse distance: decode tokens since the same (layer, expert) was last routed
    last, dists = {}, []
    for i, t in enumerate(dec):
        for k in keys(t):
            if k in last:
                dists.append(i - last[k])
            last[k] = i
    fin = sorted(dists)
    p = lambda q: fin[int(q * (len(fin) - 1))] if fin else -1
    # per-layer skew: share of a layer's routings taken by its 32 (16) most used experts, mean over layers
    sk32, sk16 = [], []
    for l in range(NL):
        c = sorted((v for (ll, e), v in freq.items() if ll == l), reverse=True)
        s = sum(c) or 1
        sk32.append(sum(c[:32]) / s)
        sk16.append(sum(c[:16]) / s)
    print(f"{name:6} {runs[name]['usage']['prompt_tokens']:10} {len(dec):11} {ws[1]:5.0f} {ws[8]:4.0f} {ws[32]:5.0f} {ws[128]:6.0f}"
          f" {p(0.5):10} {p(0.9):10} {sum(sk32)/NL:11.3f} {sum(sk16)/NL:11.3f}")
print("(ws W: distinct experts touched in W decode tokens, of 1,792; reuse: decode tokens between two routings"
      " of one expert (uniform routing: mean 8);"
      " topN share: a layer's routings on its N most used of 64, uniform would be 0.500 / 0.250)")

# ---- policies ----
def score(dec, resident):
    n = hit = 0
    for t in dec:
        for k in keys(t):
            n += 1
            hit += k in resident
    return hit / n

def lru(tr, B, R, warm):
    slots = collections.OrderedDict((k, 1) for k in warm[:B])
    ghost = collections.OrderedDict()
    n = hit = adm = 0
    for phase in ("pre", "dec"):
        for t in tr[phase]:
            budget = R
            for k in keys(t):
                if k in slots:
                    slots.move_to_end(k)
                    if phase == "dec":
                        hit += 1
                else:
                    if k in ghost and budget > 0:
                        del ghost[k]
                        if len(slots) >= B:
                            slots.popitem(last=False)
                        slots[k] = 1
                        budget -= 1
                        if phase == "dec":
                            adm += 1
                    else:
                        ghost[k] = 1
                        ghost.move_to_end(k)
                        if len(ghost) > B:
                            ghost.popitem(last=False)
                if phase == "dec":
                    n += 1
    return hit / n, adm / max(1, len(tr["dec"]))

print("\n== policies at equal bytes (served-from-bank share of decode routings)")
hdr = "K  B     prompt  (a)static  (b)hot-LOO  (b)hot-in  " + "  ".join(f"(c)LRU R={r} [adm/tok]" for r in RS)
print(hdr)
summary = collections.defaultdict(list)
for K in KS:
    B = K * NE
    for name, tr in traces.items():
        a = score(tr["dec"], {(l, e) for l in range(K) for e in range(NE)})
        loo = collections.Counter()
        for o, c in allfreq.items():
            if o != name:
                loo.update(c)
        hot = {k for k, _ in loo.most_common(B)}
        b = score(tr["dec"], hot)
        hin = {k for k, _ in allfreq[name].most_common(B)}
        bi = score(tr["dec"], hin)
        warm = [(l, e) for l in range(K) for e in range(NE)]
        cs = [lru(tr, B, r, warm) for r in RS]
        summary[K].append((a, b, bi, [c[0] for c in cs]))
        print(f"{K:2} {B:5} {name:6} {a:10.3f} {b:11.3f} {bi:10.3f}  " +
              "  ".join(f"{c[0]:12.3f} [{c[1]:4.1f}]" for c in cs))
    m = lambda i: sum(s[i] for s in summary[K]) / len(summary[K])
    mc = [sum(s[3][j] for s in summary[K]) / len(summary[K]) for j in range(len(RS))]
    print(f"{K:2} {B:5} {'mean':6} {m(0):10.3f} {m(1):11.3f} {m(2):10.3f}  " + "  ".join(f"{c:12.3f}       " for c in mc))
    if MODEL:
        Ts, te = MODEL
        def gain(h):
            return Ts / (Ts - (h - m(0)) * NL * TOPK * te)
        print(f"   implied decode gain over (a): (b) {gain(m(1)):.3f}x  (b in-sample) {gain(m(2)):.3f}x  " +
              "  ".join(f"(c) R={r} {gain(c):.3f}x" for r, c in zip(RS, mc)) + "   kill below 1.300x")
