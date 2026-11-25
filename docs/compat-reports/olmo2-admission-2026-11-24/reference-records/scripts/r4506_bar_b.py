#!/usr/bin/env python3
"""R4.50.6 bar (b) on one file: Runner's mean KL vs the publisher's float32 NOT WORSE than llama.cpp's by separated
intervals (FAIL only if Runner's 95% interval lies wholly above llama.cpp's). Each side's 95% interval: a percentile
bootstrap of the mean over its 2000 per-position KLs (10,000 resamples, seed 4506), the method of compare_full.py's
kl_ci95. Also reported, as information: the positions are the same on both sides, so the paired mean difference and
its bootstrap interval. Also (a): margin-qualified top-1 of each side.
usage: r4506_bar_b.py RUNNER_GOLD.json LLAMA_GOLD.json"""
import json, sys
import numpy as np


def side(path):
    d = json.load(open(path))
    rows = d["rows"]
    kl = np.array([r["a"]["kld"] for r in rows], dtype=np.float64)
    pos = [r["pos"] for r in rows]
    return d, kl, pos


def ci(x, rng, B=10000):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


dr, kr, pr = side(sys.argv[1])
dl, kl, pl = side(sys.argv[2])
assert pr == pl, "the two reads do not cover the same positions"
rng = np.random.default_rng(4506)
cr, cl = ci(kr, rng), ci(kl, rng)
dlt = kr - kl
cd = ci(dlt, np.random.default_rng(4507))
sr, sl = dr["summary"]["sides"]["a"], dl["summary"]["sides"]["a"]
print("positions %d (the same on both sides)" % len(kr))
print("Runner     mean KL %.4e  95%% [%.4e, %.4e]  top-1 %.1f%%  margin-qualified %.1f%%" % (kr.mean(), cr[0], cr[1], sr["top1_agreement_pct"], sr["margin_qualified_top1_pct"]))
print("llama.cpp  mean KL %.4e  95%% [%.4e, %.4e]  top-1 %.1f%%  margin-qualified %.1f%%" % (kl.mean(), cl[0], cl[1], sl["top1_agreement_pct"], sl["margin_qualified_top1_pct"]))
print("paired (information): Runner - llama.cpp mean %.3e  95%% [%.3e, %.3e]" % (dlt.mean(), cd[0], cd[1]))
b_fail = cr[0] > cl[1]
print("(a) margin-qualified top-1 100%%: %s" % ("PASS" if sr["margin_qualified_top1_pct"] == 100.0 else "FAIL"))
print("(b) Runner's interval wholly above llama.cpp's: %s -> %s" % ("yes" if b_fail else "no", "FAIL" if b_fail else "PASS"))
