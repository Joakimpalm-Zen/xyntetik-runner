#!/usr/bin/env python3
"""TC2 against the tensor-core route on this CUDA device, byte for byte.

TC2 runs by default on compute capability 8.0 and newer through the
compute_80 kernels, which add each 16-deep step in the tensor-core route's
order, so their outputs are that route's bytes on every device measured
(sm_86, sm_120). The CPU-against-GPU harnesses pin the reference route, so
without this check a card nobody measured would run the default kernels
unchecked. For each model: chunked --score with TC2 off and in its default
mode must give the same stdout bytes, and the default arm's `score: route`
line must show TC2 launches exactly when the compute_80 module loaded
(`tc2=on-sm80` in its backend detail): auto mode runs no compute_75 kernel.

Usage: cuda-tc2-bytes.py RUNNER MODEL [MODEL ...]
Exit status 0 when every model passes; prints a skip line and exits 0 when
the runner has no CUDA device.
"""
import json
import os
import subprocess
import sys

PROMPT = ("The runner reads a prompt in windows of up to five hundred and "
          "twelve tokens, and every weight matrix of the model is multiplied "
          "against all of the window's columns in one launch of the GEMM. ") * 2


def score(runner, model, tc2):
    env = dict(os.environ, RUNNER_SCORE_CHUNKED="1", RUNNER_GPU_FALLBACK="fail")
    env.pop("RUNNER_CUDA_ROUTE", None)
    if tc2 is None:
        env.pop("RUNNER_CUDA_TC2", None)
    else:
        env["RUNNER_CUDA_TC2"] = tc2
    p = subprocess.run([runner, "-m", model, "--score", "-p", PROMPT, "-c", "1024",
                        "--no-tray"], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=300)
    if p.returncode != 0:
        sys.exit("FAIL %s (TC2 %s): exit %d\n%s" % (model, tc2 or "default", p.returncode,
                                                   p.stderr.decode(errors="replace")))
    route = None
    for ln in p.stderr.decode(errors="replace").splitlines():
        if ln.startswith("score: route "):
            route = json.loads(ln[len("score: route "):])
    if route is None:
        sys.exit("FAIL %s: no score route line" % model)
    return p.stdout, route


def main():
    runner, models = sys.argv[1], sys.argv[2:]
    caps = json.loads(subprocess.run([runner, "--caps"], stdout=subprocess.PIPE,
                                     check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        print("tc2 bytes: skipped (no CUDA device)")
        return 0
    for model in models:
        off, r_off = score(runner, model, "0")
        on, r_on = score(runner, model, None)
        if r_on.get("backend") != "cuda":
            sys.exit("FAIL %s: did not run on the device (%s)" % (model, r_on.get("backend")))
        n = len(json.loads(off).get("logprobs") or [])
        t_off, t_on = r_off["kernels"].get("cuda_tc2"), r_on["kernels"].get("cuda_tc2")
        sm80 = "tc2=on-sm80" in r_on.get("backend_detail", "")
        if n < 100 or off != on:
            sys.exit("FAIL %s: %d positions, TC2 default %s the TC route's bytes" %
                     (model, n, "matches" if off == on else "DIFFERS FROM"))
        if t_off or (sm80 and not t_on) or (not sm80 and t_on):
            sys.exit("FAIL %s: TC2 launches off %s, default %s, compute_80 module %s" %
                     (model, t_off, t_on, "loaded" if sm80 else "not loaded"))
        print("tc2 bytes %s: %d positions identical, TC2 launches %d (%s)" %
              (os.path.basename(model), n, t_on,
               "compute_80 kernels" if sm80 else "below compute capability 8.0: none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
