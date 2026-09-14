"""Codebook i-quants (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S): dequant parity.

The dequant paths are transcriptions of llama.cpp b10353's per-block
arithmetic over verbatim codebook grids, so the gate is differential, not
unit-shaped: quantize ONE tiny f32 model into every claimed i-quant type
with llama.cpp's own quantizer, then require the runner's CPU path to
agree with llama-server on the same file at every teacher-forced position
of a short prompt, log P(token | prefix) within a tolerance that
llama.cpp's Q8_K activation quantization needs and a decode slip cannot
hide under. A transcription slip (wrong grid index math, wrong sign
table, wrong scale packing) moves every logit by O(1) on a random model
and cannot survive that comparison, while a unit test against
hand-computed blocks would only re-state the transcription.

Needs a llama.cpp build directory in RUNNER_LLAMA_CPP_BIN (llama-quantize,
llama-imatrix, llama-server); skipped when absent. IQ2/IQ1 quantization
requires an importance matrix, generated here from the fixture itself.
The reference is queried through llama-server's /completion (llama-cli's
interactive TUI cannot be scripted reliably), one request per position
with n_probs over the whole vocabulary and cache_prompt off; the runner
side is its --score output.

The base must be the --wide fixture: a codebook block is 256 weights, and
on the default 64-wide rows llama-quantize silently substitutes IQ4_NL or
Q4_0 for every tensor, so the seven "i-quant" files held no i-quant block
at all and the comparison passed against the wrong decoder (found
2026-09-14 while adding the CUDA kernels; the real-file anchor in
docs/muse-glimmer-cert-2026-08-11.md was the gate that actually held).
The files are named by the llama-quantize ftype requested, and the ftype
does not decide every tensor's type (on this fixture the IQ2_S recipe
stores IQ2_XS and IQ3_S tensors, while the IQ3_XXS recipe stores four
IQ2_S ones), so the fixture reads each file's real tensor types and
requires the union to cover all seven decoders before any file is used.

With a CUDA or Metal device present, the same files also feed the
CPU/GPU logit gate (tests/test_gpu_identity.c) through
test_iquant_gpu_matches_cpu, so the device twins of the decoders are
measured on every claimed type, not only on the ones a real file happens
to mix.
"""
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("RUNNER_LLAMA_CPP_BIN")

IQ_TYPES = ["IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M"]

# The three llama.cpp tools this gate drives. A Windows build ships them as
# llama-quantize.exe and friends, and Path.exists() on the bare name does
# not search executable suffixes, so until 2026-09-14 a directory holding
# exactly the three .exe files skipped the whole module (the v0.5.3 review's
# P2). Each tool is resolved once, here, and the resolved path is what the
# subprocess calls use.
TOOLS = ("llama-quantize", "llama-imatrix", "llama-server")


def find_tool(bin_dir, name):
    """The tool's path under bin_dir, bare or with the Windows suffix, or
    None when neither is a file."""
    for cand in (name, name + ".exe"):
        p = pathlib.Path(bin_dir) / cand
        if p.is_file():
            return p
    return None


def resolve_tools(bin_dir):
    if not bin_dir:
        return {n: None for n in TOOLS}
    return {n: find_tool(bin_dir, n) for n in TOOLS}


TOOL = resolve_tools(BIN)

# A missing prerequisite is a skip on a developer machine and a FAILURE on
# the box whose job it is to run the gate: RUNNER_REQUIRE_IQ_GATES names
# the legs that must run ("gpu", "llamacpp", or "all"). A required device
# run that silently skipped would read as green in the release evidence,
# which is the failure mode the device ledger exists to stop.
REQUIRE = {r for r in os.environ.get("RUNNER_REQUIRE_IQ_GATES", "").split(",") if r}

# Pre-quantized fixtures (m-<TYPE>.gguf for every IQ_TYPES entry) let a box
# without llama.cpp run the GPU leg; without it the files are quantized here.
FIXTURE_DIR = os.environ.get("RUNNER_IQ_FIXTURES")


def unavailable(leg, reason):
    if leg in REQUIRE or "all" in REQUIRE:
        pytest.fail(f"{reason} (required by RUNNER_REQUIRE_IQ_GATES)")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def iq_files(tmp_path_factory):
    files = {}
    if FIXTURE_DIR:
        for t in IQ_TYPES:
            out = pathlib.Path(FIXTURE_DIR) / f"m-{t}.gguf"
            if not out.is_file():
                unavailable("gpu", f"RUNNER_IQ_FIXTURES has no {out.name}")
            files[t] = (out, _tensor_types(out))
    else:
        missing = [n for n in ("llama-quantize", "llama-imatrix") if not TOOL[n]]
        if missing:
            unavailable("gpu", f"RUNNER_LLAMA_CPP_BIN lacks {', '.join(missing)} "
                        "and RUNNER_IQ_FIXTURES is unset")
        tmp = tmp_path_factory.mktemp("iq")
        base = tmp / "base.gguf"
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide",
                        str(base)],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        corpus = tmp / "corpus.txt"
        corpus.write_text("the quick brown fox jumps over the lazy dog " * 40)
        imatrix = tmp / "imatrix.gguf"
        subprocess.run([TOOL["llama-imatrix"], "-m", base,
                        "-f", corpus, "-o", imatrix, "--ctx-size", "128"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=300)
        for t in IQ_TYPES:
            out = tmp / f"m-{t}.gguf"
            subprocess.run([TOOL["llama-quantize"], "--imatrix", imatrix,
                            str(base), str(out), t],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=300)
            files[t] = (out, _tensor_types(out))
    covered = set().union(*(types for _, types in files.values()))
    missing = set(IQ_TYPES) - covered
    assert not missing, (f"no fixture carries {sorted(missing)}; per file: "
                         + ", ".join(f"{t}={sorted(ty)}" for t, (_, ty) in files.items()))
    return files


def _tensor_types(model):
    """The set of tensor type names in a GGUF, from the project's own
    header reader (scripts/gguf-inspect.py)."""
    out = subprocess.run(
        [sys.executable, ROOT / "scripts/gguf-inspect.py", str(model)],
        check=True, cwd=ROOT, stdout=subprocess.PIPE, text=True).stdout
    for line in out.splitlines():
        if line.startswith("tensor type histogram"):
            return set(re.findall(r"'([A-Z0-9_]+)'", line))
    raise AssertionError(f"no tensor type histogram for {model}")


def _server(model):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [TOOL["llama-server"], "-m", model,
         "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc, port
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("llama-server did not come up")


def _ref_logprobs(model, tokens, n_vocab):
    """log P(tokens[i+1] | tokens[:i+1]) from llama-server, one teacher-forced
    request per position: the prompt is the token prefix, n_probs the whole
    vocabulary, and top_logprobs are computed from the raw logits before
    sampling, so the actual next token's logprob is in the list whatever
    the sampler picks. The picked token is irrelevant here, but its text
    must be valid UTF-8 or the server's content parser refuses the whole
    response (HTTP 500); on a random fixture the greedy token is a lone
    byte >= 0x80 for the sparsest types, so the fixture's byte-fallback
    tokens for those bytes (id = byte + 3) are banned from SAMPLING with
    logit_bias, which leaves the reported logits alone."""
    ban = [[3 + b, False] for b in range(0x80, 0x100) if 3 + b < n_vocab]
    proc, port = _server(model)
    try:
        out = []
        for i in range(len(tokens) - 1):
            body = json.dumps({
                "prompt": tokens[:i + 1], "n_predict": 1, "n_probs": n_vocab,
                "temperature": 0, "cache_prompt": False, "logit_bias": ban,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            top = d["completion_probabilities"][0]["top_logprobs"]
            lp = [q["logprob"] for q in top if q["id"] == tokens[i + 1]]
            assert lp, f"position {i}: token {tokens[i + 1]} not in top_logprobs"
            out.append(lp[0])
        return out
    finally:
        proc.kill()
        proc.wait()


# llama.cpp's i-quant dot products quantize the activations to Q8_K; the
# runner's generic dequant-then-dot keeps them f32. Measured 2026-09-14 on
# these fixtures (16 positions, 7 files): worst |dlogprob| 0.0163 (IQ2_XS),
# mean 0.003 to 0.005 per file. A decode slip is not a rounding effect: a
# wrong scale, sign or grid index moves every logit by O(1) on a random
# model.
LOGPROB_TOL = 0.05


def worst_deviation(ours, ref, n_expected):
    """max |ours - ref| over exactly n_expected positions. Every count and
    every value is checked first: `max(abs(a - b) for a, b in zip(...))`
    truncates to the shorter list and, once a finite maximum is in hand,
    lets a later NaN through (both reproduced by the v0.5.3 review's P2),
    so a reference that answered fewer positions, or a score that was NaN,
    read as a pass."""
    assert len(ours) == n_expected, f"runner scored {len(ours)} of {n_expected}"
    assert len(ref) == n_expected, f"reference scored {len(ref)} of {n_expected}"
    for i, (a, b) in enumerate(zip(ours, ref)):
        assert isinstance(a, (int, float)) and math.isfinite(a), f"runner logprob {i}: {a!r}"
        assert isinstance(b, (int, float)) and math.isfinite(b), f"reference logprob {i}: {b!r}"
    devs = [abs(a - b) for a, b in zip(ours, ref)]
    assert all(math.isfinite(d) for d in devs), devs
    return max(devs)


def test_worst_deviation_rejects_short_or_nonfinite_scores():
    nan = float("nan")
    assert worst_deviation([0.25, 0.5], [0.25, 0.75], 2) == 0.25
    for ours, ref, n in [
        ([0.25, nan], [0.25, 0.25], 2),        # NaN after a finite maximum
        ([0.25, 0.25], [0.25, nan], 2),
        ([0.25], [0.25, 0.25], 2),             # zip would truncate
        ([0.25, 0.25], [0.25], 2),
        ([0.25, 0.25], [0.25, 0.25], 3),       # fewer positions than scored
        ([0.25, float("inf")], [0.25, 0.25], 2),
        ([0.25, None], [0.25, 0.25], 2),       # a missing score
    ]:
        with pytest.raises(AssertionError):
            worst_deviation(ours, ref, n)


def test_find_tool_accepts_the_windows_suffix(tmp_path):
    for n in TOOLS:
        (tmp_path / (n + ".exe")).write_bytes(b"")
    found = resolve_tools(tmp_path)
    assert all(found[n] == tmp_path / (n + ".exe") for n in TOOLS), found
    (tmp_path / "llama-quantize").write_bytes(b"")
    assert find_tool(tmp_path, "llama-quantize") == tmp_path / "llama-quantize"
    assert find_tool(tmp_path, "llama-cli") is None
    assert resolve_tools(None) == {n: None for n in TOOLS}


@pytest.mark.parametrize("t", IQ_TYPES)
def test_iquant_matches_llamacpp_logprobs(runner_bin, iq_files, t):
    """Teacher-forced per-position log P(token | prefix) on the runner's CPU
    path against llama-server's raw logits, at every position of a short
    prompt. Replaces the greedy-text comparison of the first version: on the
    sparsest fixtures the greedy token is a lone byte >= 0x80, which
    llama-server refuses to return as content, and a text comparison was
    blind to everything below the argmax anyway."""
    if not TOOL["llama-server"]:
        unavailable("llamacpp", "RUNNER_LLAMA_CPP_BIN lacks llama-server")
    model, types = iq_files[t]
    ours = subprocess.run(
        [runner_bin, "-m", model, "-p", "hello world", "--score",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert ours.returncode == 0, ours.stderr.decode(errors="replace")
    score = json.loads(ours.stdout.decode())
    n_expected = len(score["tokens"]) - 1
    assert n_expected >= 8 and score["n_scored"] == n_expected, score
    ref = _ref_logprobs(model, score["tokens"], score["n_vocab"])
    worst = worst_deviation(score["logprobs"], ref, n_expected)
    assert worst <= LOGPROB_TOL, (
        f"{t} ({sorted(types)}): worst |dlogprob| {worst:.4g} over "
        f"{len(ref)} positions; runner={score['logprobs']} llama.cpp={ref}")


@pytest.fixture(scope="module")
def gpu_identity_bin():
    exe = ROOT / ("test-gpu-identity.exe" if sys.platform == "win32"
                  else "test-gpu-identity")
    if not exe.exists():
        unavailable("gpu", "test-gpu-identity not built (make test-gpu-identity)")
    return exe


@pytest.mark.parametrize("t", IQ_TYPES)
def test_iquant_gpu_matches_cpu(runner_bin, gpu_identity_bin, iq_files, t):
    """Device decoders against the host ones at logit precision. The gate
    itself skips, never passes, when there is no device or the model fell
    back to the CPU; that skip is surfaced here rather than counted."""
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True).stdout)
    if not caps.get("gpu"):
        unavailable("gpu", "no GPU backend on this machine")
    model, types = iq_files[t]
    lacking = sorted(types - set(caps.get("gpu_quants", [])))
    if lacking:
        unavailable("gpu", f"{t}: {lacking} have no kernel on the "
                    f"{caps['gpu'].get('backend')} backend")
    p = subprocess.run([gpu_identity_bin, model], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=600)
    log = p.stdout.decode(errors="replace")
    if "gpu-identity: ok (skipped)" in log:
        unavailable("gpu", f"{t}: the gate skipped (no device, or CPU fallback)")
    assert p.returncode == 0, log
    assert "gpu-identity: ok" in log, log


@pytest.fixture(scope="module")
def tc_tol_bin():
    exe = ROOT / ("test-tc-tol.exe" if sys.platform == "win32" else "test-tc-tol")
    if not exe.exists():
        unavailable("tc", "test-tc-tol not built (make test-tc-tol)")
    return exe


# The codebook formats whose prefill is claimed to run on the tensor
# cores. A type listed here must dispatch its batched GEMM in the forced-on
# arm of the tolerance gate on every fixture file that carries it in a
# block, and the gate's numbers must pass; a type not listed is the scalar
# path's and is not asked to. IQ3_S first (the largest tensor group of the
# GSQ-RCO file that motivated the CUDA i-quants); the leg still runs the
# gate wherever a file's other block types have one.
IQ_TC_TYPES = ["IQ3_S"]


@pytest.mark.parametrize("t", IQ_TYPES)
def test_iquant_tc_matches_scalar(runner_bin, tc_tol_bin, iq_files, t):
    """The forced-tensor-core gate (tests/test_tc_tol.c) on each fixture:
    a separate gate from the scalar identity above, which pins TC off. The
    backend's own kernel table decides which of the file's block types
    must dispatch; the gate then requires every one of them to have run in
    the forced-on arm and none in the forced-off arms, and holds the logits
    to its tolerance and the free-running greedy text to identity."""
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True).stdout)
    if not caps.get("gpu"):
        unavailable("tc", "no GPU backend on this machine")
    model, types = iq_files[t]
    lacking = sorted(types - set(caps.get("gpu_quants", [])))
    if lacking:
        unavailable("tc", f"{t}: {lacking} have no kernel on the "
                    f"{caps['gpu'].get('backend')} backend")
    backend_tc = set(subprocess.run(
        [tc_tol_bin, "--types"], cwd=ROOT, stdout=subprocess.PIPE, check=True,
        text=True).stdout.split()[1:])
    claimed = sorted(set(IQ_TC_TYPES) & types)
    not_carried = [x for x in claimed if x not in backend_tc]
    assert not not_carried, (
        f"{t}: {not_carried} are claimed tensor-core formats but the "
        f"{caps['gpu'].get('backend')} backend's kernel table lacks them "
        f"(it lists {sorted(backend_tc)})")
    p = subprocess.run([tc_tol_bin, model], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=900)
    log = p.stdout.decode(errors="replace")
    if "no block tensor of a type this backend has a batched GEMM" in log:
        # the file's only TC-capable tensor sits outside the blocks (the
        # Q6_K embedding on these fixtures), which the gate rightly does
        # not count; a claimed format IS a block type here, so it may not
        # be the reason
        assert not claimed, f"{t}: {claimed} present yet the gate found no eligible block type\n{log}"
        pytest.skip(f"{t}: no block type of this file has a batched GEMM")
    if "GPU or config unavailable" in log:
        unavailable("tc", f"{t}: the gate could not load the model on the device")
    assert p.returncode == 0 and "tc-tol: ok" in log, log
    assert "ok (skipped)" not in log, log
    for x in claimed:
        assert re.search(rf"tc-b64\s+TC dispatches:.*\b{x}=[1-9]", log), (
            f"{t}: {x} never dispatched its tensor-core GEMM in the forced-on arm\n{log}")
