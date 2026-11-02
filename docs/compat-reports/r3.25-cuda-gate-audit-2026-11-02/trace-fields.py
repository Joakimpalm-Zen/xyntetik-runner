#!/usr/bin/env python3
"""Measure CPU-vs-CUDA RUNNER_MOE_TRACE field agreement per model x arm x mode (sizing the gate's tolerances).
trace-fields.py RUNNER MODEL [MODEL ...]  (run from the runner tree root)"""
import importlib.util
import json
import os
import sys
import tempfile

spec = importlib.util.spec_from_file_location("mr", "scripts/cuda-moe-replay.py")
mr = importlib.util.module_from_spec(spec)
sys.path.insert(0, "scripts")
spec.loader.exec_module(mr)


def recs(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        out[(r["pos"], r["layer"])] = r
    return out


runner, models = sys.argv[1], sys.argv[2:]
with tempfile.TemporaryDirectory() as tmp:
    for model in models:
        for arm in mr.ARMS:
            for mode in mr.MODES:
                env = dict(mr.ARMS[arm], **mr.MODES[mode])
                tg, tc = os.path.join(tmp, "g.jsonl"), os.path.join(tmp, "c.jsonl")
                for t in (tg, tc):
                    if os.path.exists(t):
                        os.remove(t)
                mr.score(runner, model, True, dict(env, RUNNER_MOE_TRACE=tg))
                mr.score(runner, model, False, dict(mr.MODES[mode], RUNNER_MOE_TRACE=tc))
                g, c = recs(tg), recs(tc)
                keys_same = set(g) == set(c)
                ex = sum(g[k]["experts"] != c[k]["experts"] for k in g if k in c)
                empty = sum(not g[k]["norms"] for k in g)
                dg = max((abs(a - b) for k in g if k in c for a, b in zip(g[k]["gates"], c[k]["gates"])), default=0)
                dn = max((abs(a - b) / max(abs(b), 1e-30) for k in g if k in c
                          for a, b in zip(g[k]["norms"], c[k]["norms"])), default=0)
                print("%-34s %-7s %-7s recs %4d keys %s experts-diff %d empty-norms %d gate-absdiff %.2e norm-reldiff %.2e"
                      % (os.path.basename(model), arm, mode, len(g), "same" if keys_same else "DIFFER", ex, empty, dg, dn),
                      flush=True)
