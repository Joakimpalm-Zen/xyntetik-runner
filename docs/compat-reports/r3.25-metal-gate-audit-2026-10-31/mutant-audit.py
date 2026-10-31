#!/usr/bin/env python3
"""R3.25 Metal half: structural mutants against every tolerance-bounded
identity gate make test runs on Metal. Run from the runner tree root."""
import os, re, shutil, subprocess, sys, time
ROOT = os.getcwd()
OUT = sys.argv[1]
K, M = "src/kernels.metal", "src/metal.m"
MUTANTS = [
 ("rmsnorm-weight-ignored", K, "    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r * w[i];\n}", "    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r;\n}"),
 ("rmsnorm-eps-dropped", K, "    float r = rsqrt(red[0] / n + eps);", "    float r = rsqrt(red[0] / n);"),
 ("qknorm-weight-ignored", K, "    for (int i = tid; i < hd; i += tpg) x[i] = x[i] * r * w[i];\n}\n\nkernel void k_head_rmsnorm", "    for (int i = tid; i < hd; i += tpg) x[i] = x[i] * r;\n}\n\nkernel void k_head_rmsnorm"),
 ("silu-dropped", K, "        gp[gid.x] = (x / (1.0f + exp(-x))) * up[gid.x];", "        gp[gid.x] = x * up[gid.x];"),
 ("rope-skipped", K, "    p[i0] = y.x;\n    p[i1] = y.y;", "    (void)y;"),
 ("rope-position-plus-one", K, "    float ang = (a.pos + col) * fr[j];", "    float ang = (a.pos + col + 1) * fr[j];"),
 ("attn-scale-x1.05", M, "model_attn_scale(m, l)", "model_attn_scale(m, l) * 1.05f"),
 ("matvec-bias-dropped", K, "    if (tiisg == 0) y[row] = a.has_bias ? s + bias[row] : s; \\", "    if (tiisg == 0) y[row] = s; \\"),
 ("rope-store-position-plus-one", K, "        float ang = a.pos * fr[j];", "        float ang = (a.pos + 1) * fr[j];"),
 ("sigmoid-gate-dropped", K, "        xp[gid.x] *= 1.0f / (1.0f + exp(-gp[gid.x]));", "        (void)gp;"),
]
GATES = [
 ("gpu-id test.gguf", ["./test-gpu-identity"], "1e-5"),
 ("gpu-id noisy-norms", ["./test-gpu-identity", "test-noisy-norms.gguf"], "1e-5"),
 ("gpu-id test.gguf 0 1", ["./test-gpu-identity", "test.gguf", "0", "1"], "1e-5"),
 ("attn-tol", ["./test-attn-tol"], None),
 ("gpu-id gptoss-top1", ["./test-gpu-identity", "test-moe-fixture.gptoss-top1.gguf"], "1e-5"),
 ("gpu-id k_off", ["./test-gpu-identity", "test-attn/k_off.gguf"], "1e-5"),
 ("gpu-id k_nope", ["./test-gpu-identity", "test-attn/k_nope.gguf"], "1e-5"),
 ("gpu-id k_half", ["./test-gpu-identity", "test-attn/k_half.gguf"], "1e-5"),
 ("gpu-id k_temp_live", ["./test-gpu-identity", "test-attn/k_temp_live.gguf"], "1e-5"),
 ("gpu-id spark", ["./test-gpu-identity", "test-spark.gguf"], "2e-5"),
 ("gpu-id spark 0 1", ["./test-gpu-identity", "test-spark.gguf", "0", "1"], "2e-5"),
 ("gpu-id spark-swa", ["./test-gpu-identity", "test-spark-swa.gguf"], "2e-5"),
 ("gpu-id ornith (qwen35)", ["./test-gpu-identity", "test-ornith.gguf"], None),
]
BINS = ["runner", "test-gpu-identity", "test-mv-tol", "test-attn-tol", "test-moe-tol"]

def build():
    subprocess.run([sys.executable, "scripts/embed-metal.py"], check=True, capture_output=True)
    r = subprocess.run(["make", "-j6", *BINS], capture_output=True, text=True)
    if r.returncode: raise SystemExit("build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])

def run_gates():
    res = {}
    for name, cmd, lim in GATES:
        env = dict(os.environ)
        if lim: env["GPU_ID_LIMIT"] = lim
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        txt = (r.stdout + r.stderr).strip().splitlines()
        last = txt[-1] if txt else ""
        stat = "PASS" if r.returncode == 0 else "FAIL"
        if "skipped" in (r.stdout + r.stderr) and r.returncode == 0: stat = "SKIP"
        frac = re.findall(r"= ([0-9.e+-]+) of mean range", r.stdout + r.stderr)
        res[name] = (stat, frac[-1] if frac else "", last[-90:])
    return res

orig = {f: open(f).read() for f in (K, M)}
with open(OUT, "w") as log:
    try:
        build()
        base = run_gates()
        log.write("baseline\n" + "".join(f"  {g}: {s} {f}\n" for g, (s, f, _) in base.items())); log.flush()
        for name, f, old, new in MUTANTS:
            src = orig[f]
            assert src.count(old) >= 1, (name, "anchor missing")
            open(f, "w").write(src.replace(old, new))
            time.sleep(1); os.utime(f)
            build()
            res = run_gates()
            open(f, "w").write(orig[f])
            caught = [g for g, (s, _, _) in res.items() if s == "FAIL"]
            log.write(f"{name}: caught by {len(caught)}/{len(GATES)}\n" + "".join(
                f"  {g}: {s} {fr} | {last}\n" for g, (s, fr, last) in res.items())); log.flush()
    finally:
        for f, t in orig.items(): open(f, "w").write(t)
        time.sleep(1)
        for f in orig: os.utime(f)
        build()
        log.write("restored\n")
