"""qwen4exp's PLE n-gram history at a prefix fork (R4.48.22.1).

qwen4exp's PLE block hashes each token's n-grams with the token's predecessors, read from the token history the feed
writes (model.c ple4_prev). The recurrent blob a prefix entry carries holds the block's convolution window but not
that history, and a fork did not rewrite it: the first ngram - 1 tokens after a fork hashed whatever the slot's history
held before the fork's end. That is the slot's previous prompt's tokens after a longer one, and nothing (-1) in a new
process, so a context imported from a snapshot, a pinned context forked after a longer prompt, and the automatic
prefix cache's own fork after one each answered with another PLE row than the prompt prefilled cold (logprobs about
1e-4 apart, the same text). A fork now rebuilds the history from the forked tokens.

The anchors, on the qwen4exp fixture:
- on the CPU, the cold prefill of the same prompt in a fresh process: each fork's tokens and logprobs equal it bit for
  bit (measured: a fork whose history is right is the cold prefill bit for bit on the CPU). The forks: (a) a pinned
  context, then a prompt longer than it, then the request naming it; (b) the context exported to a snapshot file and
  imported into a new process; (c) the automatic tier: the context's prompt, a longer prompt, then the context's
  prompt and the question.
  Each fork's recurrent fold after the prompt (RUNNER_TEST_REC_DUMP: the conv ring, the state and the PLE window)
  equals the cold prefill's word for word too (R4.48.34): six logprobs alone are a weak detector.
- on CUDA, the automatic tier: (c) and the fork made with nothing in between each give the cold prefill's tokens,
  its logprobs within LOGPROB_TOL and its fold within FOLD_REL_TOL (R4.48.34, the advisor's ruling C). R3.60's prefill
  identity is a CPU-only run's (engine_prompt_prefill): on a device, a continued prompt's host steps and kernels may
  round otherwise than a fresh one's (the bounds below say by how much). The PLE history's proof is the CPU arm's,
  bit for bit. Every server's log shows the
  device and RUNNER_GPU_FALLBACK=fail makes a fall to the CPU an error. A pin and an import of a recurrent model on
  the device are held by tests/test_recurrent_device_contexts.py (R4.48.22), qwen4exp's import to its pin's bits.
A fold installed with no tokens named is refused on qwen4exp (model_recurrent_blob_load returns false and changes
nothing), so the next tokens never hash a history nobody named: make test's test-spec-identity on the qwen4exp
fixture asserts the refusal, and its reference names its sequence on every reload, byte for byte against the verify
rows.
"""
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

SYS = ("You are a careful assistant for a warehouse inventory team. Answer "
       "with the item code first, then one short sentence. ")
QUESTION = "Where is item 42?"
# longer than SYS, so the slot's token history past SYS's end is this prompt's
LONG = "unrelated text that shares nothing at all with the context above, " * 4
# R4.48.34, the advisor's ruling C (CUDA only; the CPU stays bit for bit):
# - a logprob within 1e-5 absolute;
# - a fold word within FOLD_REL_TOL, measured per word as |a - b| / max(|a|, |b|).
# Both bounds come from this test's own CUDA arm, "the fork with nothing in between", on an sm_120 MIG
# slice on 2026-11-27: a logprob gap of 4.8e-7, and a worst fold word of 6.77e-5 (484 of 2,496 words differed).
# FOLD_REL_TOL is 8x that word, rounded down.
# For context: the pin method (a pin of SYS, then SYS + QUESTION) measured 1.89e-6 on the same slice.
# No seam reproduces the PLE history defect, so the CPU arm carries that proof. With R4.48.22.1's rebuild undone in a
# scratch build, the defect moved the CPU fold by 0.963 at worst (704 of 2,496 words; the logprobs by 6.15e-5),
# about 1,800x over FOLD_REL_TOL.
LOGPROB_TOL = 1e-5
FOLD_REL_TOL = 5.4e-4
RECDUMP = re.compile(r"recdump: pos (\d+) words (\d+) fnv ([0-9a-f]{16}) hex ([0-9a-f]+)")


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("ple") / "q4e.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-qwen4exp.py", str(m)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return m


def _req(srv, method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(srv.base_url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _complete(srv, prompt, **extra):
    st, b = _req(srv, "POST", "/v1/completions", {
        "prompt": prompt, "max_tokens": 6, "temperature": 0, "logprobs": 2, **extra})
    assert st == 200, b
    c = b["choices"][0]
    return (c["text"], c["logprobs"]["token_logprobs"]), b["runner_telemetry"]


def _serve_in(runner_bin, model, logs, tmp_path, gpu, *extra):
    logs.append(tmp_path / ("server-%d.log" % len(logs)))
    return RunnerServer(runner_bin, model, ctx=512,
                        env=dict(os.environ, RUNNER_GPU_FALLBACK="fail", RUNNER_TEST_REC_DUMP="1"),
                        log_path=str(logs[-1]), extra_args=[*gpu, "-t", "2", *extra])


def _cold(runner_bin, model, logs, tmp_path, gpu):
    with _serve_in(runner_bin, model, logs, tmp_path, gpu) as srv:
        cold, tel = _complete(srv, SYS + QUESTION, cache_prompt=False)
    assert tel["prompt_cached_tokens"] == 0, tel
    return cold


def _auto(runner_bin, model, logs, tmp_path, gpu, between):
    # the automatic tier: SYS is published whole after its prefill; P + the question is an exact full-prefix hit
    with _serve_in(runner_bin, model, logs, tmp_path, gpu) as srv:
        _complete(srv, SYS)
        if between:
            _complete(srv, between)
        got, tel = _complete(srv, SYS + QUESTION)
    assert tel["prompt_reuse"] == "prefix_cache" and tel["prompt_cached_tokens"] > 16, tel
    return got


def fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return h


def _folds(log):
    """[(pos, raw bytes)] of every RUNNER_TEST_REC_DUMP line in a server's log, in order. Each line's hex hashes to the
    hash printed beside it (the seam reads what it prints)."""
    out = []
    for pos, words, fnv, hx in RECDUMP.findall(pathlib.Path(log).read_text(errors="replace")):
        raw = bytes.fromhex(hx)
        assert len(raw) == 4 * int(words) and fnv1a64(raw) == int(fnv, 16), (pos, words, fnv)
        out.append((int(pos), raw))
    assert out, "no recdump line in " + str(log)
    return out


def _fold_rel(a, b):
    """The largest relative difference of two folds' f32 words, against the larger of each pair."""
    worst = 0.0
    for x, y in zip(struct.unpack("<%df" % (len(a) // 4), a), struct.unpack("<%df" % (len(b) // 4), b)):
        if x != y:
            worst = max(worst, abs(x - y) / max(abs(x), abs(y)))
    return worst


def test_a_fork_hashes_the_tokens_it_forked_on_the_cpu(runner_bin, model, tmp_path):
    gpu, logs, prompt = ["--gpu", "off"], [], SYS + QUESTION
    cold = _cold(runner_bin, model, logs, tmp_path, gpu)
    snaps = tmp_path / "snaps"
    with _serve_in(runner_bin, model, logs, tmp_path, gpu, "--kv-snapshots", str(snaps)) as srv:
        st, pin = _req(srv, "POST", "/v1/runner/contexts", {"id": "sys", "prompt": SYS})
        assert st == 200, pin
        n = pin["tokens"]
        _complete(srv, LONG)
        pinned, tel = _complete(srv, prompt, context_id="sys")
        assert tel["prompt_cached_tokens"] == n and tel["prompt_reuse"] == "prefix_cache", tel
        st, snap = _req(srv, "POST", "/v1/runner/contexts/sys/snapshot", {})
        assert st == 200, snap
    with _serve_in(runner_bin, model, logs, tmp_path, gpu, "--kv-snapshots", str(snaps)) as srv:
        st, loaded = _req(srv, "POST", "/v1/runner/contexts", {"id": "sys2", "snapshot": "sys"})
        assert st == 200 and loaded["tokens"] == n, loaded
        imported, tel = _complete(srv, prompt, context_id="sys2")
        assert tel["prompt_cached_tokens"] == n and tel["prompt_reuse"] == "prefix_cache", tel
    auto = _auto(runner_bin, model, logs, tmp_path, gpu, LONG)
    for name, got in (("(a) pinned, forked after a longer prompt", pinned),
                      ("(b) imported in a new process", imported),
                      ("(c) the automatic tier after a longer prompt", auto)):
        assert got == cold, (name, got, cold)
    # the state: each fork's fold after the prompt (its server's last prompt feed) is the cold prefill's, word for word
    cold_fold = _folds(logs[0])[-1]
    for name, log in (("(a) pinned", logs[1]), ("(b) imported", logs[2]), ("(c) the automatic tier", logs[3])):
        pos, raw = _folds(log)[-1]
        assert pos == cold_fold[0], (name, pos, cold_fold[0])
        assert raw == cold_fold[1], (name, "fold words differ", _fold_rel(raw, cold_fold[1]))
    # the instrument's premise: the fold after SYS alone (the automatic tier's first prompt) is not the fold after
    # the whole prompt, so a seam printing one constant cannot pass the comparison above
    sys_fold = _folds(logs[3])[0]
    assert sys_fold[0] < cold_fold[0] and sys_fold[1] != cold_fold[1], (sys_fold[0], cold_fold[0])


def test_the_automatic_tier_on_cuda_hashes_the_tokens_it_forked(runner_bin, model, tmp_path):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    gpu, logs = ["--gpu", "99"], []
    cold = _cold(runner_bin, model, logs, tmp_path, gpu)
    direct = _auto(runner_bin, model, logs, tmp_path, gpu, None)
    after = _auto(runner_bin, model, logs, tmp_path, gpu, LONG)
    cold_fold = _folds(logs[0])[-1]
    for name, got, log in (("the fork with nothing in between", direct, logs[1]),
                           ("(c) the automatic tier after a longer prompt", after, logs[2])):
        # ruling C: the same tokens, the logprobs and the fold within their bounds (not bit for bit on a device)
        assert got[0] == cold[0], (name, got, cold)
        assert len(got[1]) == len(cold[1]), (name, got, cold)
        gap = max(abs(a - b) for a, b in zip(got[1], cold[1]))
        assert gap <= LOGPROB_TOL, (name, "logprob gap", gap, got, cold)
        pos, raw = _folds(log)[-1]
        assert pos == cold_fold[0] and len(raw) == len(cold_fold[1]), (name, pos, cold_fold[0])
        rel = _fold_rel(raw, cold_fold[1])
        print(f"{name}: logprob gap {gap:.3g}, fold worst relative word {rel:.3g}")
        assert rel <= FOLD_REL_TOL, (name, "fold relative gap", rel)
    for log in logs:
        text = pathlib.Path(log).read_text(errors="replace")
        assert '"backend":"cuda"' in text or "CUDA backend" in text, text[-1500:]
        assert "continuing on the CPU" not in text, text[-1500:]