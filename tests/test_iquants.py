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

pytestmark = pytest.mark.skipif(
    not BIN or not (pathlib.Path(BIN) / "llama-quantize").exists(),
    reason="RUNNER_LLAMA_CPP_BIN with llama-quantize/llama-imatrix/llama-cli required")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def iq_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("iq")
    base = tmp / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide",
                    str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    corpus = tmp / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 40)
    imatrix = tmp / "imatrix.gguf"
    subprocess.run([pathlib.Path(BIN) / "llama-imatrix", "-m", base,
                    "-f", corpus, "-o", imatrix, "--ctx-size", "128"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=300)
    files = {}
    for t in IQ_TYPES:
        out = tmp / f"m-{t}.gguf"
        subprocess.run([pathlib.Path(BIN) / "llama-quantize", "--imatrix", imatrix,
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
        [pathlib.Path(BIN) / "llama-server", "-m", model,
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


@pytest.mark.parametrize("t", IQ_TYPES)
def test_iquant_matches_llamacpp_logprobs(runner_bin, iq_files, t):
    """Teacher-forced per-position log P(token | prefix) on the runner's CPU
    path against llama-server's raw logits, at every position of a short
    prompt. Replaces the greedy-text comparison of the first version: on the
    sparsest fixtures the greedy token is a lone byte >= 0x80, which
    llama-server refuses to return as content, and a text comparison was
    blind to everything below the argmax anyway."""
    model, types = iq_files[t]
    ours = subprocess.run(
        [runner_bin, "-m", model, "-p", "hello world", "--score",
         "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert ours.returncode == 0, ours.stderr.decode(errors="replace")
    score = json.loads(ours.stdout.decode())
    assert score["n_scored"] >= 8, score
    ref = _ref_logprobs(model, score["tokens"], score["n_vocab"])
    worst = max(abs(a - b) for a, b in zip(score["logprobs"], ref))
    assert worst <= LOGPROB_TOL, (
        f"{t} ({sorted(types)}): worst |dlogprob| {worst:.4g} over "
        f"{len(ref)} positions; runner={score['logprobs']} llama.cpp={ref}")


@pytest.fixture(scope="module")
def gpu_identity_bin():
    exe = ROOT / ("test-gpu-identity.exe" if sys.platform == "win32"
                  else "test-gpu-identity")
    if not exe.exists():
        pytest.skip("test-gpu-identity not built (make test-gpu-identity)")
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
        pytest.skip("no GPU backend on this machine")
    model, types = iq_files[t]
    lacking = sorted(types - set(caps.get("gpu_quants", [])))
    if lacking:
        pytest.skip(f"{t}: {lacking} have no kernel on the "
                    f"{caps['gpu'].get('backend')} backend")
    p = subprocess.run([gpu_identity_bin, model], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=600)
    log = p.stdout.decode(errors="replace")
    if "gpu-identity: ok (skipped)" in log:
        pytest.skip(f"{t}: the gate skipped (no device, or CPU fallback)")
    assert p.returncode == 0, log
    assert "gpu-identity: ok" in log, log
