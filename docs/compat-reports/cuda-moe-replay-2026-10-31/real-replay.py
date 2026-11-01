"""R3.21.2 real-model check (docs/compat-reports/cuda-moe-replay-2026-10-31): RUNNER_MOE_TRACE / RUNNER_MOE_REPLAY on a real MoE.

Usage: real-replay.py RUNNER MODEL OUTDIR [extra env K=V ...]
Greedy generation, GPU auto (layers past the card's memory stay on the CPU,
whose routers pin the same way). Plain, traced, self-replayed (0 differed and
0 uncovered required, same text), and a rotated-layer control (layer L takes
layer L+1's recorded experts: the text must change, every pair still covered).
"""
import json
import os
import subprocess
import sys

PROMPT = "Explain in three sentences why the sky is blue, then name one consequence for photographers."
N = 64


def run(runner, model, env, tag, out):
    e = {k: v for k, v in os.environ.items() if k not in ("RUNNER_MOE_TRACE", "RUNNER_MOE_REPLAY")}
    e.update(env)
    p = subprocess.run([runner, "-m", model, "-p", PROMPT, "-n", str(N), "--temp", "0",
                        "--no-tray", "-c", "1024"], env=e, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=1800)
    err = p.stderr.decode(errors="replace")
    open(os.path.join(out, tag + ".err"), "w", encoding="utf-8").write(err)
    open(os.path.join(out, tag + ".out"), "wb").write(p.stdout)
    if p.returncode != 0:
        sys.exit("FAIL %s: exit %d\n%s" % (tag, p.returncode, err[-1500:]))
    return p.stdout, err


def counts(err):
    for ln in err.splitlines():
        if ln.startswith("moe-replay: ") and "forced" in ln:
            n = [int(w) for w in ln.replace(",", " ").split() if w.isdigit()]
            return n[0], n[1], n[2]
    sys.exit("FAIL: no moe-replay count line")


def main():
    runner, model, out = sys.argv[1:4]
    extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
    os.makedirs(out, exist_ok=True)
    trace = os.path.join(out, "trace.jsonl")
    if os.path.exists(trace):
        os.remove(trace)
    t0, err = run(runner, model, dict(extra), "plain", out)
    for ln in err.splitlines():
        if ln.startswith(("gpu-split:", "gpu: CUDA backend", "loaded ")):
            print(ln)
    t1, _ = run(runner, model, dict(extra, RUNNER_MOE_TRACE=trace), "traced", out)
    recs = [json.loads(l) for l in open(trace, encoding="utf-8")]
    layers = sorted({r["layer"] for r in recs})
    used = len(recs[0]["experts"])
    print("trace: %d records, %d MoE layers, %d experts each, positions 0..%d" %
          (len(recs), len(layers), used, max(r["pos"] for r in recs)))
    t2, err = run(runner, model, dict(extra, RUNNER_MOE_REPLAY=trace), "self", out)
    f, d, u = counts(err)
    print("self-replay: forced %d (want %d), differed %d, uncovered %d, text %s" %
          (f, len(recs) * used, d, u, "same" if t2 == t0 else "CHANGED"))
    nxt = {l: layers[(i + 1) % len(layers)] for i, l in enumerate(layers)}
    by = {(r["pos"], r["layer"]): r["experts"] for r in recs}
    rot = os.path.join(out, "rotated.jsonl")
    with open(rot, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps({"pos": r["pos"], "layer": r["layer"],
                                 "experts": by[(r["pos"], nxt[r["layer"]])]}) + "\n")
    t3, err = run(runner, model, dict(extra, RUNNER_MOE_REPLAY=rot), "rotated", out)
    rf, rd, ru = counts(err)
    print("rotated-layer control: forced %d, differed %d, uncovered %d, text %s" %
          (rf, rd, ru, "changed" if t3 != t0 else "UNCHANGED"))
    ok = (t1 == t0 and t2 == t0 and f == len(recs) * used and d == 0 and u == 0
          and t3 != t0 and rd > 0 and ru == 0)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
