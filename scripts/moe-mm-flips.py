#!/usr/bin/env python3
"""Routing-flip account for the grouped-MMA MoE prefill kernels.

The grouped-MMA path (RUNNER_METAL_MOE_MM) perturbs the residual stream at
half/float-staging rounding scale; the router itself runs f32, but discrete
top-k selection can flip on a perturbed input at a near-tie. This harness
measures exactly that: it runs the same prompt prefill twice — matvec path
vs grouped-MMA — with RUNNER_MOE_TRACE capturing every (position, layer)
routing record (selected experts, gate weights, full router logits), then
reports how many selections moved and at what router-logit margins.

Prefill IS teacher-forcing on the prompt (every position's input is the
prompt token, not a sampled one), so the two traces are position-aligned by
construction until generation starts; only the prompt span is compared.

The margin question is the whole point: a flip at a 0.03-nat near-tie is
the model exercising genuine indifference (the same class the 235B census
found for the SCALAR cross-backend gap), while a flip at a wide margin
would indict the kernel. The report separates them.

Usage:
  scripts/moe-mm-flips.py --model models/X.gguf [--text FILE] [--arm 1|half]
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_arm(runner, model, text, ctx, batch, trace_path, mm_env):
    env = dict(os.environ)
    env.pop("RUNNER_METAL_MOE_MM", None)
    if mm_env:
        env["RUNNER_METAL_MOE_MM"] = mm_env
    env["RUNNER_MOE_TRACE"] = str(trace_path)
    proc = subprocess.run(
        [runner, "-m", model, "-f", text, "-c", str(ctx), "-n", "1",
         "--temp", "0", "-b", str(batch), "--no-tray"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"runner failed ({mm_env or 'off'} arm): "
                         f"{proc.stderr.decode(errors='replace')[-500:]}")
    if mm_env and b"grouped-MMA prefill kernels on" not in proc.stderr:
        raise SystemExit("the grouped-MMA arm never engaged — nothing to measure")


def load_trace(path):
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs[(r["pos"], r["layer"])] = r
    return recs


def boundary_margin(logits, used):
    """kth minus (k+1)th largest router logit: the decision boundary width."""
    top = sorted(logits, reverse=True)
    return top[used - 1] - top[used]


def compare(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no aligned (pos, layer) records between the arms")
    set_flips, order_flips, margins_all, margins_flip = [], [], [], []
    dlogits = []
    for k in keys:
        ra, rb = a[k], b[k]
        used = len(ra["experts"])
        m = boundary_margin(ra["logits"], used)
        margins_all.append(m)
        dlogits.extend(abs(x - y) for x, y in zip(ra["logits"], rb["logits"]))
        if set(ra["experts"]) != set(rb["experts"]):
            set_flips.append((k, m))
            margins_flip.append(m)
        elif ra["experts"] != rb["experts"]:
            order_flips.append((k, m))
    n = len(keys)
    margins_all.sort()
    buckets = [1e-3, 1e-2, 0.05, 0.1]
    hist = {}
    lo = 0.0
    for hi in buckets:
        hist[f"[{lo:g},{hi:g})"] = sum(1 for m in margins_flip if lo <= m < hi)
        lo = hi
    hist[f">={lo:g}"] = sum(1 for m in margins_flip if m >= lo)
    out = {
        "records": n,
        "router_logit_mean_abs_delta": sum(dlogits) / len(dlogits),
        "router_logit_max_abs_delta": max(dlogits),
        "set_flips": len(set_flips),
        "set_flip_rate": len(set_flips) / n,
        "order_only_flips": len(order_flips),
        "boundary_margin_median": margins_all[len(margins_all) // 2],
        "boundary_margin_p10": margins_all[len(margins_all) // 10],
        "flip_margin_histogram": hist,
    }
    if margins_flip:
        out["flip_margin_max"] = max(margins_flip)
        out["flip_margin_median"] = sorted(margins_flip)[len(margins_flip) // 2]
    return out, set_flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--text", default=str(ROOT / "docs/determinism-scope.md"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--arm", default="1", choices=("1", "half"),
                    help="grouped-MMA staging to compare against the matvec path")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        t_off = pathlib.Path(td) / "off.jsonl"
        t_on = pathlib.Path(td) / "on.jsonl"
        run_arm(args.runner, args.model, args.text, args.ctx, args.batch,
                t_off, None)
        run_arm(args.runner, args.model, args.text, args.ctx, args.batch,
                t_on, args.arm)
        summary, flips = compare(load_trace(t_off), load_trace(t_on))

    summary["model"] = os.path.basename(args.model)
    summary["arm"] = f"grouped-MMA {'half' if args.arm == 'half' else 'f32'}-staged"
    print(json.dumps(summary, indent=1))
    if flips:
        worst = sorted(flips, key=lambda kf: -kf[1])[:8]
        print("\nwidest-margin flips (pos, layer, off-arm boundary margin):",
              file=sys.stderr)
        for (pos, layer), m in worst:
            print(f"  pos {pos:5d} layer {layer:3d}  margin {m:.5f}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
