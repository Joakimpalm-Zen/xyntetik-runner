#!/usr/bin/env python3
"""Where do the CPU and the GPU first part, and by how much?

`cpu_cuda_check.py` answers yes or no: the greedy tokens are identical or
they are not. When they are not, the next question is whether the two
backends drifted apart over many positions or disagreed at one near-tie.
This prints the per-position difference in the log-probability of the SAME
token sequence on both backends, so the answer is a number per position.

The sequence is the CPU's own greedy continuation of one of the gate's
prompts, teacher-forced through `--score` on each backend under the gate's
pins (RUNNER_CUDA_TC=0, eager MoE routing).

    cpu_cuda_delta.py MODEL [--prompt N | --contains TEXT] [--tokens 128]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cpu_cuda_check import PROMPTS  # noqa: E402


def run(runner, args, env):
    p = subprocess.run([runner, *args, "--no-tray"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.exit("runner failed: %s\n%s" % (" ".join(args), p.stderr.decode(errors="replace")[-2000:]))
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--prompt", type=int, help="index into cpu_cuda_check.PROMPTS")
    ap.add_argument("--contains", help="pick the prompt containing this text")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.contains:
        hits = [i for i, p in enumerate(PROMPTS) if a.contains in p]
        if len(hits) != 1:
            sys.exit("--contains matched %d prompts" % len(hits))
        idx = hits[0]
    else:
        idx = a.prompt if a.prompt is not None else 0
    prompt = PROMPTS[idx]

    env = dict(os.environ, RUNNER_CUDA_TC="0", RUNNER_MOE_EAGER="1")
    cont = run(a.runner, ["-m", a.model, "-p", prompt, "-n", str(a.tokens),
                          "--temp", "0", "--gpu", "off"], env).decode("utf-8", "replace")
    text = prompt + cont
    cpu = json.loads(run(a.runner, ["-m", a.model, "--score", "-p", text, "--gpu", "off"], env))
    gpu = json.loads(run(a.runner, ["-m", a.model, "--score", "-p", text, "--gpu", "auto"], env))
    lc, lg = cpu["logprobs"], gpu["logprobs"]
    if len(lc) != len(lg):
        sys.exit("the two backends scored %d and %d positions" % (len(lc), len(lg)))
    d = [abs(x - y) for x, y in zip(lc, lg)]
    first = next((i for i, v in enumerate(d) if v > a.threshold), None)
    worst = max(range(len(d)), key=d.__getitem__)
    over = sum(v > a.threshold for v in d)
    rec = {"schema": "xyntetik.runner.cpu-cuda-delta.v1", "prompt_index": idx,
           "positions": len(d), "threshold": a.threshold,
           "first_over": first, "over": over,
           "max_abs_delta": d[worst], "max_at": worst,
           "mean_abs_delta": sum(d) / len(d),
           "cpu_top1_rate": cpu.get("top1_rate"), "gpu_top1_rate": gpu.get("top1_rate"),
           "abs_delta": d}
    print("positions %d  over %.0e: %d  first at %s  max %.3e at %d  mean %.3e" % (
        len(d), a.threshold, over, first, d[worst], worst, rec["mean_abs_delta"]))
    print("cpu top1 %s  gpu top1 %s" % (cpu.get("top1"), gpu.get("top1")))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
