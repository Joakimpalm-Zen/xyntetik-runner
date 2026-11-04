#!/usr/bin/env python3
"""Check RUNNER_CUDA_ATTN_DUMP output (a development switch, R1.25).

    attn-dump-compare.py DIR_A [DIR_B] [--detail N]

For every dumped prompt-tile attention call, the output is checked against a
float64 attention computed from that call's own dumped inputs (Q, and the K
and V cache rows as stored in fp16). With two directories (one run at
RUNNER_CUDA_ATTN_TC_WIDE=1, one at =0), the calls are paired by order and the
script also says where the inputs first differ, which is the first layer whose
upstream already diverged. --detail N prints per-head and per-token errors and
the Q rows' ranges for call N.
"""
import json
import sys

import numpy as np


def load(d):
    calls = [json.loads(x) for x in open(f"{d}/meta.jsonl")]
    for c in calls:
        base = f"{d}/c{c['call']:04d}-L{c['layer']:02d}"
        tn, qd = c["tn"], c["q_dim"]
        c["q"] = np.fromfile(base + ".q", np.float32).reshape(tn, qd)
        c["o"] = np.fromfile(base + ".o", np.float32).reshape(tn, qd)
        if c["kv_fp16"]:
            rows, kvd = c["pos"] + tn, c["n_kv"] * c["head_dim"]
            c["k"] = np.fromfile(base + ".k", np.float16).reshape(rows, kvd)
            c["v"] = np.fromfile(base + ".v", np.float16).reshape(rows, kvd)
    return calls


def reference(c):
    """float64 attention over the dumped cache, per token and head."""
    tn, hd, nh, nkv = c["tn"], c["head_dim"], c["n_head"], c["n_kv"]
    G = nh // nkv
    q = c["q"].astype(np.float64).reshape(tn, nh, hd)
    k = c["k"].astype(np.float64).reshape(-1, nkv, hd)
    v = c["v"].astype(np.float64).reshape(-1, nkv, hd)
    out = np.zeros((tn, nh, hd))
    for t in range(tn):
        p = c["pos"] + t
        lo = max(0, p - c["window"] + 1) if c["window"] > 0 else 0
        for h in range(nh):
            kh = h // G
            s = (k[lo:p + 1, kh] @ q[t, h]) * c["scale"]
            w = np.exp(s - s.max())
            out[t, h] = (w / w.sum()) @ v[lo:p + 1, kh]
    return out.reshape(tn, nh * hd)


def rel(a, b):
    den = np.abs(b).max()
    return float(np.abs(a - b).max() / den) if den > 0 else float(np.abs(a - b).max())


def detail(c, ref):
    tn, hd, nh = c["tn"], c["head_dim"], c["n_head"]
    e = np.abs(c["o"].astype(np.float64) - ref).reshape(tn, nh, hd).max(axis=2)
    scale = np.abs(ref).max()
    print(f"  call {c['call']} layer {c['layer']} pos {c['pos']} tcw {c['tcw']}: "
          f"error / max|ref| by head (worst token):")
    for h in range(nh):
        t = int(e[:, h].argmax())
        print(f"    head {h:2d}: {e[t, h] / scale:.3e} at token {t}")
    q = c["q"].reshape(tn, nh, hd)
    mx = np.abs(q).max(axis=2)
    rms = np.sqrt((q.astype(np.float64) ** 2).mean(axis=2))
    print(f"  Q rows: max|q| {mx.min():.3g} to {mx.max():.3g}, max/rms {np.nanmin(mx / rms):.3g} "
          f"to {np.nanmax(mx / rms):.3g}; any non-finite: {not np.isfinite(q).all()}")
    if "k" in c:
        print(f"  K max|k| {np.abs(c['k'].astype(np.float32)).max():.3g}, "
              f"V max|v| {np.abs(c['v'].astype(np.float32)).max():.3g}")


def main():
    args = sys.argv[1:]
    want = None
    if "--detail" in args:
        i = args.index("--detail")
        want = int(args[i + 1])
        del args[i:i + 2]
    runs = [load(d) for d in args]
    a = runs[0]
    b = runs[1] if len(runs) > 1 else None
    first_in = None
    print("call layer  pos  tn tcw  A vs ref    " + ("B vs ref    A vs B out  A vs B in" if b else ""))
    for i, c in enumerate(a):
        ref = reference(c) if c["kv_fp16"] else None
        line = f"{c['call']:4d} {c['layer']:5d} {c['pos']:4d} {c['tn']:3d} {c['tcw']:3d}  "
        line += f"{rel(c['o'], ref):.3e}   " if ref is not None else "   n/a      "
        if b and i < len(b):
            d = b[i]
            refb = reference(d) if d["kv_fp16"] else None
            line += f"{rel(d['o'], refb):.3e}   " if refb is not None else "   n/a      "
            line += f"{rel(c['o'], d['o']):.3e}   "
            din = rel(c["q"], d["q"])
            if "k" in c and "k" in d:
                din = max(din, rel(c["k"].astype(np.float32), d["k"].astype(np.float32)),
                          rel(c["v"].astype(np.float32), d["v"].astype(np.float32)))
            line += f"{din:.3e}"
            if first_in is None and din > 0:
                first_in = c["call"]
        print(line)
        if want == c["call"] and ref is not None:
            detail(c, ref)
            if b and i < len(b) and b[i]["kv_fp16"]:
                detail(b[i], reference(b[i]))
    if b:
        print(f"first call whose inputs differ between the runs: {first_in}")


if __name__ == "__main__":
    main()
