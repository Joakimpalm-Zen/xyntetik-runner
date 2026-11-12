"""Mamba-2 hybrid admission and the granitehybrid decode step (tracer 2).

`granitehybrid` (Granite-4 h-series) now RUNS: the runner interleaves GQA
attention with a Mamba-2 selective-SSD recurrence, applies the granite muP
scaling and treats the attention layers as NoPE, gated token-identically
against llama.cpp b10353 on the real granite-4.0-h-small (a scripted check;
this file is the unit level). `nemotron_h_moe` (Nemotron-3.5 Lightning) shares
the tensor set but adds a grouped scan this tracer has not certified, so it is
still RECOGNIZED and refused with a specific "forward not yet implemented"
message — never the generic unknown-architecture refusal, which would read as a
typo. Either way, a missing required SSM tensor must FAIL CLOSED naming it (the
hostile-GGUF discipline). No download: the fixtures are generated and mirror a
real granite-4.0-h / Nemotron GGUF (sparse-MoE, NoPE, inner == 2*embd).
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("hybrid")
    out = {}
    for arch in ("granitehybrid", "nemotron_h_moe"):
        pfx = d / arch
        subprocess.run(
            [sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
             "--arch", arch], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        out[arch] = (pfx.with_suffix(".gguf"),
                     d / (arch + ".missing-ssm_d.gguf"))
    # granite-4.0-h-micro shape: expert_count 0, dense gated MLP on every layer
    pfx = d / "granitehybrid-dense"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
         "--dense"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out["granitehybrid-dense"] = (pfx.with_suffix(".gguf"), None)
    return out


def _run(runner_bin, model, n="1"):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", "hi", "-n", n, "--temp", "0",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)


# A prompt long enough that the byte-level fixture tokenizer yields > 100 tokens,
# so XR_SSM_CHUNK=2 splits the Mamba-2 prefill into dozens of chunks and crosses
# the inter-chunk state boundary many times.
CHUNK_PROMPT = ("hello world this is a somewhat longer prompt used to exercise "
                "the chunked mamba-2 prefill scan over several tokens")


def _run_p(runner_bin, model, prompt, n, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", prompt, "-n", str(n), "--temp", "0",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        env=e)


def _act_trace(stderr: bytes) -> bytes:
    """The RUNNER_DEBUG_ACT per-layer activation dump for the prefill pass -- a
    numeric fingerprint (per-layer sum to 6 dp + boundary values) far more
    sensitive than the greedy argmax, which on a tiny random fixture is too
    robust to reveal sub-argmax prefill drift (measured: a dropped inter-chunk
    state carry perturbs the trace but not the 16 greedy tokens)."""
    return b"\n".join(l for l in stderr.splitlines() if l.startswith(b"ACT "))


def _assert_chunked_equals_serial(runner_bin, model):
    """Tracer 3 gate: the chunked-scan prefill must be BIT-IDENTICAL to the serial
    per-token sweep -- chunking changes speed, not output. XR_SSM_SERIAL forces
    the reference path; the default and small chunk sizes take the chunked path
    (tiny sizes cross many inter-chunk state boundaries). Both the prefill
    activation trace AND the greedy token stream must match, at every chunk size.
    A regression in the chunked scan, the conv batching, or the inter-chunk
    state/ring carry moves the activation trace here."""
    dbg = {"RUNNER_DEBUG_ACT": "1"}
    ref = _run_p(runner_bin, model, CHUNK_PROMPT, 16, {"XR_SSM_SERIAL": "1", **dbg})
    assert ref.returncode == 0, ref.stderr.decode(errors="replace")
    ref_act = _act_trace(ref.stderr)
    assert ref_act, "no ACT trace captured; RUNNER_DEBUG_ACT not honored"
    m = re.search(rb"prompt: (\d+) tok", ref.stderr)
    assert m and int(m.group(1)) > 8, \
        "prompt must tokenize to many tokens so XR_SSM_CHUNK=2/3 spans chunks"
    for extra in ({}, {"XR_SSM_CHUNK": "2"}, {"XR_SSM_CHUNK": "3"},
                  {"XR_SSM_CHUNK": "128"}):
        got = _run_p(runner_bin, model, CHUNK_PROMPT, 16, {**dbg, **extra})
        assert got.returncode == 0, got.stderr.decode(errors="replace")
        label = extra or "default-chunk"
        assert _act_trace(got.stderr) == ref_act, (
            f"chunked prefill (env={label}) diverged from the serial sweep in the "
            f"activation trace")
        assert got.stdout == ref.stdout, (
            f"chunked prefill (env={label}) diverged from the serial sweep in the "
            f"token stream: {got.stdout[:200]!r} != {ref.stdout[:200]!r}")


def test_granitehybrid_loads_and_decodes(runner_bin, models):
    """The forward is implemented: granitehybrid must LOAD (not refuse) and
    decode. A build that regressed the loader back to a refusal, or that could
    not run the Mamba-2 step at all, fails here."""
    good, _ = models["granitehybrid"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    # the load announces the hybrid layout, not a refusal
    assert "granitehybrid" in err
    assert "forward not yet implemented" not in err


def test_granitehybrid_dense_loads_and_decodes(runner_bin, models):
    """The DENSE h-variant (granite-4.0-h-micro: expert_count 0, gated MLP on
    every layer). The loader used to bind ffn_gate/ffn_up only inside the
    attention branch, so a dense model's recurrent layers decoded with NULL
    FFN weights — a segfault on the first Mamba-2 layer of the real h-micro."""
    good, _ = models["granitehybrid-dense"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "granitehybrid" in err
    assert "forward not yet implemented" not in err


def test_granitehybrid_dense_decode_is_deterministic(runner_bin, models):
    good, _ = models["granitehybrid-dense"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_granitehybrid_decode_is_deterministic(runner_bin, models):
    """Greedy decode must be reproducible run to run (the property the real
    token-identity gate builds on)."""
    good, _ = models["granitehybrid"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_nemotron_h_moe_loads_and_decodes(runner_bin, models):
    """The forward is implemented: nemotron_h_moe (Nemotron-3.5 Lightning — a
    gate-less squared-ReLU MoE hybrid with an always-on shared expert) must LOAD
    (not refuse) and decode. A regression back to the old refusal fails here."""
    good, _ = models["nemotron_h_moe"]
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "nemotron_h_moe" in err
    assert "forward not yet implemented" not in err


def test_nemotron_h_moe_decode_is_deterministic(runner_bin, models):
    good, _ = models["nemotron_h_moe"]
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


@pytest.mark.parametrize("arch", ["granitehybrid", "nemotron_h_moe"])
def test_missing_ssm_tensor_names_the_tensor(runner_bin, models, arch):
    _, broken = models[arch]
    proc = _run(runner_bin, broken)
    assert proc.returncode != 0, "a missing required SSM tensor must fail closed"
    err = proc.stderr.decode(errors="replace")
    assert "ssm_d" in err, "the failure must name the absent tensor"
    assert arch in err, "still recognized as the hybrid, just malformed"


# (nemotron_h_moe is now an admitted, implemented arch — the old
# "RUNNER_ALLOW_UNKNOWN_ARCH must not smuggle it into llama math" test retired
# with the refusal it guarded; the loads-and-decodes test above covers it.)


# --------------------------------------------------------------------------
# nemotron_h (Nemotron-Nano-9B-v2 family) — NON-MoE Mamba-2 / attention / MLP
# hybrid, graduated from refuse to load+decode (SSM tracer 3). Its grouped scan
# (ssm.group_count=2 in the fixture, 8 in the real model) is the crux Lightning
# was refused on; certified token-identically at the noise floor vs llama.cpp
# b10353 on the real Nemotron-Nano-9B-v2 Q8_0 (a scripted check; this is the
# unit level). Distinct fixture generator (non-MoE, three-way layer typing,
# gate-less squared-ReLU MLP, group_count>1, inner != 2*embd).
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nemo_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("nemotron_h")
    pfx = d / "nemotron_h"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-nemotron.py", str(pfx)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return (pfx.with_suffix(".gguf"), d / "nemotron_h.missing-ssm_d.gguf")


def test_nemotron_h_loads_and_decodes(runner_bin, nemo_model):
    """nemotron_h (NON-MoE) RUNS: it must LOAD (not refuse) and decode the
    grouped Mamba-2 scan. A regression to a refusal, or an inability to run the
    n_group>1 step / the three-way (SSM|attn|MLP) block typing, fails here."""
    good, _ = nemo_model
    proc = _run(runner_bin, good, n="8")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "nemotron_h" in err
    assert "grouped scan" in err
    assert "forward not yet implemented" not in err
    assert "refusing to run it through llama-style math" not in err


def test_verbose_report_on_a_mamba2_first_layout(runner_bin, nemo_model):
    """-v's summary names block 0's weight type. It read block 0's wqkv (if
    recurrent) or wq, and a Mamba-2 block has neither (its matrices are ssm_in
    and ssm_out), so -v crashed on any file whose first block is Mamba-2:
    found on Nemotron 3 Nano 30B-A3B (R4.27.4.1). This fixture's block 0 is
    recurrent."""
    good, _ = nemo_model
    proc = subprocess.run(
        [runner_bin, "-m", str(good), "-p", "hi", "-n", "1", "--temp", "0",
         "--gpu", "off", "-v"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, err
    line = next((l for l in err.splitlines() if l.startswith("weight type")), "")
    assert line.split()[-1] not in ("", "type"), err

def test_nemotron_h_decode_is_deterministic(runner_bin, nemo_model):
    """Greedy decode must be reproducible run to run (the property the real
    token-identity gate builds on)."""
    good, _ = nemo_model
    a = _run(runner_bin, good, n="8")
    b = _run(runner_bin, good, n="8")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_nemotron_h_missing_ssm_tensor_fails_closed(runner_bin, nemo_model):
    """The hostile-GGUF discipline holds for the graduated arch too: a missing
    required SSM tensor must FAIL CLOSED naming it, not decode past the map."""
    _, broken = nemo_model
    proc = _run(runner_bin, broken)
    assert proc.returncode != 0, "a missing required SSM tensor must fail closed"
    err = proc.stderr.decode(errors="replace")
    assert "ssm_d" in err, "the failure must name the absent tensor"
    assert "nemotron_h" in err


# --------------------------------------------------------------------------
# Tracer 3 -- Mamba-2 chunked-scan prefill. The chunked path (conv batched, scan
# tiled into chunks and run in parallel across heads, SSD state + conv ring
# carried across chunk boundaries) is bit-identical to the serial per-token
# sweep by construction; these pin that so a chunking regression cannot ship as
# a silent speed-for-correctness trade. Covers n_group=1 (granite) and n_group>1
# (nemotron) grouped scans, and tiny chunk sizes that cross many boundaries.
# --------------------------------------------------------------------------
def test_granitehybrid_chunked_prefill_matches_serial(runner_bin, models):
    good, _ = models["granitehybrid"]
    _assert_chunked_equals_serial(runner_bin, good)


def test_nemotron_h_chunked_prefill_matches_serial(runner_bin, nemo_model):
    good, _ = nemo_model
    _assert_chunked_equals_serial(runner_bin, good)


# --------------------------------------------------------------------------
# --cpu-moe on CUDA: Granite 4 hybrid MoE runs its attention and Mamba-2
# layers on the device and its experts on the host. Two defects kept it off
# the device: the device tensor lists had no ssm_in, so the first recurrent
# layer failed and the run continued on the CPU, and the --cpu-moe auto fit
# placed banks whose ungated shared expert the device cannot run.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [["--cpu-moe"], ["--cpu-moe", "auto"]])
def test_granitehybrid_cpu_moe_runs_on_the_device(runner_bin, models, cfg):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True,
                                     text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("--cpu-moe placement is CUDA-side")
    good, _ = models["granitehybrid"]
    cpu = _run(runner_bin, good, n="8")
    assert cpu.returncode == 0, cpu.stderr
    proc = subprocess.run(
        [runner_bin, "-m", str(good), "-p", "hi", "-n", "8", "--temp", "0", *cfg],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, err
    assert "CUDA backend" in err, err
    for fallback in ("not resident", "falling back", "continuing on CPU"):
        assert fallback not in err, err
    # the shared expert has no gate: every bank stays on the host
    assert re.search(r"experts 0/\d+ layers on GPU", err), err
    assert proc.stdout == cpu.stdout


# --cpu-moe on CUDA for nemotron_h_moe (R3.43a): its router (a sigmoid with a
# selection bias) has no device kernel, so cuda_admit refused the whole model,
# even with every bank on the host where model_moe_ffn_cpu routes it. Now the
# refusal applies only to a bank on the device: --cpu-moe runs the Mamba-2 and
# attention blocks on the device and routes on the host. Two defects stood in
# the way: the admission refusal, and the forward's jump for an MLP-only block
# straight to the device FFN, past the host branch (a run read the router from
# the device and fell back).
@pytest.mark.parametrize("cfg", [["--cpu-moe"], ["--cpu-moe", "auto"]])
def test_nemotron_h_moe_cpu_moe_runs_the_trunk_on_the_device(runner_bin, models, cfg):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True,
                                     text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("--cpu-moe placement is CUDA-side")
    good, _ = models["nemotron_h_moe"]
    prompt = "the quick brown fox jumps over the lazy dog"

    def score(*extra):
        env = dict(os.environ, RUNNER_GPU_FALLBACK="fail")
        p = subprocess.run([runner_bin, "-m", str(good), "--score", "4", "-p", prompt,
                            "--no-tray", *extra], cwd=ROOT, capture_output=True,
                           timeout=120, env=env)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        return json.loads(p.stdout), p.stderr.decode(errors="replace")

    cpu, _ = score("--gpu", "off")
    dev, err = score(*cfg)
    assert "CUDA backend" in err, err
    for fallback in ("not resident", "falling back", "continuing on CPU"):
        assert fallback not in err, err
    # the router has no device kernel: every bank stays on the host
    assert re.search(r"experts 0/\d+ layers on GPU", err), err
    # the experts run the CPU's own code; only the trunk's device arithmetic
    # differs (measured 9.5e-7 on the fixture)
    assert max(abs(a - b) for a, b in zip(cpu["logprobs"], dev["logprobs"])) <= 1e-4
    assert cpu["argmax"] == dev["argmax"]


def _cuda_or_skip(runner_bin):
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True,
                                     text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the default placement is CUDA-side")


def test_nemotron_h_moe_takes_the_split_by_default_on_cuda(runner_bin, models):
    """R3.43a.2: with no placement named, a GPU load takes the --cpu-moe split
    (its router has no device kernel), and says so."""
    _cuda_or_skip(runner_bin)
    good, _ = models["nemotron_h_moe"]
    prompt = "the quick brown fox jumps over the lazy dog"

    def score(*extra, env_extra=None):
        env = dict(os.environ, RUNNER_GPU_FALLBACK="fail", **(env_extra or {}))
        p = subprocess.run([runner_bin, "-m", str(good), "--score", "4", "-p", prompt,
                            "--no-tray", *extra], cwd=ROOT, capture_output=True,
                           timeout=120, env=env)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        return json.loads(p.stdout), p.stderr.decode(errors="replace")

    cpu, _ = score("--gpu", "off")
    dev, err = score()
    assert "its experts and their routing stay on the host" in err, err
    assert "CUDA backend" in err and re.search(r"experts 0/\d+ layers on GPU", err), err
    assert max(abs(a - b) for a, b in zip(cpu["logprobs"], dev["logprobs"])) <= 1e-4
    assert cpu["argmax"] == dev["argmax"]


def test_nemotron_h_moe_cpu_moe_default_0_keeps_the_cpu_and_names_the_flag(
        runner_bin, models):
    _cuda_or_skip(runner_bin)
    good, _ = models["nemotron_h_moe"]
    proc = subprocess.run([runner_bin, "-m", str(good), "-p", "hi", "-n", "2", "--temp", "0",
                           "--no-tray"], cwd=ROOT, capture_output=True, timeout=120,
                          env=dict(os.environ, RUNNER_CPU_MOE_DEFAULT="0"))
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, err
    assert "has no device kernel" in err and "--cpu-moe keeps the experts" in err, err
    assert "CUDA backend" not in err, err


def test_the_fit_says_where_the_experts_go(runner_bin, models, tmp_path):
    """R3.43a.2 and R3.58: --info fit names a load that keeps every expert bank
    on the host (nemotron_h_moe's default, or --cpu-moe) and sizes it as that
    load (tests/test_cuda_fit_hybrid.py holds the bytes); --cpu-moe auto and N
    keep only some banks on the host, which the fit does not model: it sizes a
    full offload for them and says so."""
    _cuda_or_skip(runner_bin)
    host = "experts       on the host, with their routing"
    note = "the split above sizes a full offload"

    def fit(model, *extra, env_extra=None):
        p = subprocess.run([runner_bin, "--info", "fit", "-m", str(model), *extra], cwd=ROOT,
                           capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, **(env_extra or {})))
        assert p.returncode == 0, p.stderr
        return p.stdout

    nemo, _ = models["nemotron_h_moe"]
    assert host in fit(nemo) and "(the default for this router)" in fit(nemo)
    assert host not in fit(nemo, env_extra={"RUNNER_CPU_MOE_DEFAULT": "0"})
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(tmp_path / "m")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    plain = tmp_path / "m.moe4.gguf"
    out = fit(plain)
    assert host not in out and note not in out
    out = fit(plain, "--cpu-moe")
    assert host in out and "(--cpu-moe)" in out and note not in out
    out = fit(plain, "--cpu-moe", "auto")
    assert note in out and "the budget leaves out" in out and host not in out
    out = fit(plain, "--cpu-moe", "1")
    assert note in out and "the deepest expert banks" in out and host not in out


def test_nemotron_h_moe_routes_with_a_sigmoid_when_the_file_names_no_gating(
        runner_bin, tmp_path):
    """The publisher's router (NemotronHTopkRouter) is a sigmoid, and the
    published GGUFs carry no expert_gating_func key; llama.cpp's graph
    hardcodes the sigmoid. Runner read the missing key as softmax: on the real
    Lightning-30B Q8_0 it scored a perplexity of 18.31 where llama.cpp scored
    11.73 (top-1 agreement 63%, 2026-11-03). The anchor is the key's own
    numbering (1 softmax, 2 sigmoid): a file without the key must score
    exactly as the same file declaring 2, and not as one declaring 1. The
    fixture's router is random, so the two functions route differently."""
    scores = {}
    for tag, gating in (("absent", None), ("sigmoid", "2"), ("softmax", "1")):
        pfx = tmp_path / tag
        env = dict(os.environ)
        env.pop("HYBRID_TEST_GATING", None)
        if gating:
            env["HYBRID_TEST_GATING"] = gating
        subprocess.run([sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
                        "--arch", "nemotron_h_moe"], check=True, cwd=ROOT, env=env,
                       stdout=subprocess.DEVNULL)
        p = subprocess.run([runner_bin, "-m", str(pfx.with_suffix(".gguf")), "--score",
                            "-p", "the quick brown fox jumps over the lazy dog", "--gpu", "off",
                            "--no-tray"], cwd=ROOT, capture_output=True, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        scores[tag] = json.loads(p.stdout)["logprobs"]
    assert scores["absent"] == scores["sigmoid"]
    assert scores["absent"] != scores["softmax"]


# R6.68.62: the router's grouping is validated for every MoE family. The
# nemotron_h binder read expert_group_count and expert_group_used_count and
# checked neither, so a grouping every other family refuses loaded: groups
# that do not divide the experts, or no used group at all (moe_route then
# masks every group and takes expert 0 for each pick). granitehybrid, through
# the shared MoE binder, refused them all along.
@pytest.mark.parametrize("arch", ["nemotron_h_moe", "granitehybrid"])
@pytest.mark.parametrize("groups", ["3,1", "2,0"])
def test_an_expert_grouping_the_router_cannot_run_is_refused(runner_bin, tmp_path, arch, groups):
    pfx = tmp_path / "g"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx), "--arch", arch],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, HYBRID_TEST_EXPERT_GROUPS=groups))
    p = _run(runner_bin, pfx.with_suffix(".gguf"))
    assert p.returncode != 0, p
    assert b"invalid MoE expert grouping" in p.stderr, p.stderr
