#!/usr/bin/env python3
"""R3.25 CUDA half: structural mutants against the CUDA device identity gates.
Run from the runner tree root on a CUDA host: cuda-mutant-audit.py OUT.log [GPU_ID_LIMIT]
With a limit, the gpu-identity gates run at that bound (pass 2); without, at their defaults (pass 1).
Each mutant is applied to src/kernels.cu (then the PTX is regenerated with the local nvcc) or src/cuda.c,
checked to have applied, rebuilt, and run against every gate; the sources are restored after each one."""
import os, re, subprocess, sys, time
OUT = sys.argv[1]
LIMIT = sys.argv[2] if len(sys.argv) > 2 else None
K, C = "src/kernels.cu", "src/cuda.c"
PY = os.environ.get("PYTHON", sys.executable)
MAKE = ["make", "--no-print-directory", "CC=clang", "PYTHON=" + PY]
GELU = "(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))) * u[i]"
MUTANTS = [  # name, file, old, new, expected number of replacements
 ("attn-residual-dropped", C, "ok = ok && enc_add(g, g->x, g->xb, n_embd, tn, n_embd, xdim);", "ok = ok;", 3),
 ("ssm-D-skip-dropped", K, "= sumf + D[h] * x;", "= sumf;", 2),
 ("silu-to-gelu (dense FFN)", K, "g[i] = (x / (1.0f + expf(-x))) * u[i];", "g[i] = " + GELU + ";", 1),
 ("rmsnorm-skipped (scale 1)", K, "    return rsqrtf(red[0] / n + eps);\n}", "    return 1.0f + 0.0f * eps;\n}", 1),
 ("rmsnorm-eps-dropped", K, "    return rsqrtf(red[0] / n + eps);\n}", "    return rsqrtf(red[0] / n) + 0.0f * eps;\n}", 1),
 ("rope-position-plus-one (k_rope)", K, "    int pos = *posp + blockIdx.z;\n    float ang = pos * fr[j];",
  "    int pos = *posp + blockIdx.z + 1;\n    float ang = pos * fr[j];", 1),
 ("attn-scale-x1.05", C, "model_attn_scale(m, l),", "model_attn_scale(m, l) * 1.05f,", 2),
 ("matvec-bias-dropped (all)", K, "a.has_bias ? s + bias[row] : s", "s", None),
 # k_rope rotates Q and K at the same position, so the +1 above shifts both and leaves every
 # relative angle intact (an equivalent mutant); these two break the rotation for real
 ("rope-K-skipped", C, "        ok = ok && enc_rope(g, m, g->kt, n_kv, tn, kv_dim, l);", "        ok = ok;", 1),
 ("rope-skipped (Q and K)", K, "    p[i0] = x0 * c - x1 * s;\n    p[i1] = x0 * s + x1 * c;",
  "    (void)x0; (void)x1; (void)c; (void)s;", 2),
]
ONLY = [n for n in os.environ.get("ONLY", "").split(",") if n]
if ONLY:  # a supplementary pass over named mutants (baseline skipped)
    MUTANTS = [m for m in MUTANTS if m[0] in ONLY]
    assert len(MUTANTS) == len(ONLY), ("unknown mutant in ONLY", ONLY)
G = "./test-gpu-identity"
GATES = [  # name, command, kind
 ("gpu-id test.gguf", [G], "id"),
 ("gpu-id test.gguf 0 1", [G, "test.gguf", "0", "1"], "id"),
 ("gpu-id noisy-norms", [G, "test-noisy-norms.gguf"], "id"),
 ("gpu-id gptoss-top1", [G, "test-moe-fixture.gptoss-top1.gguf"], "id"),
 ("gpu-id k_off", [G, "test-attn/k_off.gguf"], "id"),
 ("gpu-id k_nope", [G, "test-attn/k_nope.gguf"], "id"),
 ("gpu-id k_half", [G, "test-attn/k_half.gguf"], "id"),
 ("gpu-id k_temp_live", [G, "test-attn/k_temp_live.gguf"], "id"),
 ("gpu-id spark", [G, "test-spark.gguf"], "id"),
 ("gpu-id spark 0 1", [G, "test-spark.gguf", "0", "1"], "id"),
 ("gpu-id spark-swa", [G, "test-spark-swa.gguf"], "id"),
 ("gpu-id ornith (qwen35)", [G, "test-ornith.gguf"], "id"),
 ("gpu-id mamba2 s40", [G, "test-m2-s40-g1.gguf"], "id"),
 ("gpu-id nemotron-h", [G, "test-m2-nemotron.gguf"], "id"),
 ("prefix chunk 4", ["./test-prefix"], "prefix"),
 ("prefix chunk 64", ["./test-prefix", "test.gguf", "64"], "prefix"),
 ("cuda-nvfp4", MAKE + ["test-cuda-nvfp4"], "make"),
 ("cuda-tc-kq (TC prefill)", MAKE + ["test-cuda-tc-kq"], "make"),
 ("cuda-moe-replay", MAKE + ["test-cuda-moe-replay"], "make"),
]
# Pass 2's per-fixture bounds: LIMIT for the toy fixtures and 2e-5 for spark (CUDA residue 3.7e-6). With
# TEMP_DEFAULT=1, k_temp_live keeps the cross-backend 2e-3, as it had to before the decode graph's
# temperature fix (its residue was 8.0e-4, see the README)
SPECIAL = {"gpu-id spark": "2e-5", "gpu-id spark 0 1": "2e-5", "gpu-id spark-swa": "2e-5"}
if os.environ.get("TEMP_DEFAULT"):
    SPECIAL["gpu-id k_temp_live"] = None
if LIMIT:  # the limit only moves the gpu-identity gates, so pass 2 runs those alone
    GATES = [g for g in GATES if g[2] == "id"]
BINS = ["runner", "test-gpu-identity", "test-prefix", "test-tc-tol", "test-moe-tol"]


def build(ptx):
    if ptx:
        r = subprocess.run(MAKE + ["ptx", "NVCC=/usr/local/cuda/bin/nvcc"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("ptx failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    r = subprocess.run(MAKE + ["-j16", *BINS], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def dev(kind, txt):
    if kind == "id":
        v = re.findall(r"= ([0-9.e+-]+) of mean range", txt)
    elif kind == "prefix":
        v = re.findall(r"([0-9.e+-]+) of range", txt)
    else:
        v = re.findall(r"(?:max diff|distance) ([0-9.e+-]+)", txt)
    vals = [float(x) for x in v if re.match(r"^[0-9.e+-]+$", x)]
    return "%.2e" % max(vals) if vals else ""


CUR = "baseline"


def run_gates():
    res = {}
    for name, cmd, kind in GATES:
        env = dict(os.environ)
        lim = SPECIAL.get(name, LIMIT) if LIMIT and kind == "id" else None
        if lim:
            env["GPU_ID_LIMIT"] = lim
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            txt, rc = r.stdout + r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            txt, rc = "timeout", -1
        stat = "PASS" if rc == 0 else "FAIL"
        if rc == 0 and re.search(r"\bskipped\b", txt) and kind != "make":
            stat = "SKIP"
        if stat == "FAIL":  # the full output: a make gate's last line is only make's own "Error 1"
            os.makedirs(OUT + ".fail", exist_ok=True)
            open(os.path.join(OUT + ".fail", re.sub(r"\W+", "_", CUR + "__" + name) + ".txt"), "w").write(txt)
        last = (txt.strip().splitlines() or [""])[-1]
        res[name] = (stat, dev(kind, txt), round(time.time() - t0), last[-110:])
    return res


def write(log, title, res):
    log.write(title + "\n" + "".join("  %-26s %s %-9s %4ss | %s\n" % (g, s, d, t, last) for g, (s, d, t, last) in res.items()))
    log.flush()


orig = {f: open(f).read() for f in (K, C)}
with open(OUT, "w") as log:
    log.write("R3.25 CUDA half, %s, GPU_ID_LIMIT=%s%s\n" % (time.strftime("%F %T"), LIMIT or "default (2e-3)",
              " (spark 2e-5%s)" % (", k_temp_live default" if os.environ.get("TEMP_DEFAULT") else "")
              if LIMIT else ""))
    try:
        build(True)
        if not ONLY:
            write(log, "baseline (PTX regenerated with the local nvcc)", run_gates())
        stale = False   # the PTX still holds the last kernels.cu mutant (restored in the source only)
        for name, f, old, new, n in MUTANTS:
            src = orig[f]
            count = src.count(old)
            assert count >= 1 and (n is None or count == n), (name, "anchor count", count, n)
            open(f, "w").write(src.replace(old, new))
            time.sleep(1)
            os.utime(f)
            build(f == K or stale)
            stale = False
            CUR = name
            res = run_gates()
            open(f, "w").write(orig[f])
            stale = f == K
            caught = [g for g, (s, _, _, _) in res.items() if s == "FAIL"]
            write(log, "%s (%d replacements): caught by %d/%d" % (name, count, len(caught), len(GATES)), res)
    finally:
        for f, t in orig.items():
            open(f, "w").write(t)
        time.sleep(1)
        for f in orig:
            os.utime(f)
        build(True)
        log.write("restored %s\n" % time.strftime("%F %T"))
