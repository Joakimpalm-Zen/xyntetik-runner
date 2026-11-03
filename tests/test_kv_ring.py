"""`RUNNER_KV_RING=1`: give sliding layers only the rows they can read.

A sliding layer clamps its attention start to `p - swa_window + 1`, so rows
older than the window are written once and never read. The ring allocates
`swa_window + n_batch` rows for those layers and indexes them modulo that
count, which on gemma-3-4b at ctx 32768 takes the cache from 4563 MB to 800 MB.

THE ANCHOR IS THE FLAT PATH. The ring must be output-identical to the default
allocation, because it holds exactly the rows the flat allocation would have
been read from -- no more, no less. That makes the existing, shipped, unringed
engine the reference implementation, which is an anchor outside the code under
test rather than the ring agreeing with itself.

The identity only means something if the ring actually WRAPS, so these tests
assert the ring engaged and use a prompt many times longer than the window; a
prompt shorter than the ring would pass without exercising a single modulo.

Partial rewind and the prefix cache resume from what the ring still holds
(R5.5, docs/kv-ring.md), gated at the end of this file through the server.
CUDA's former two flat-copy sites mirror a ring as a whole.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

SWA_WINDOW = 32
N_CTX = 1024
# 14 repeats tokenizes to ~869 tokens on the fixture vocabulary, which is many
# times the 96-row ring: the modulo wraps ~9 times during prefill alone.
PROMPT = " ".join(["the quick brown fox jumps over the lazy dog"] * 14)


def _make(tmp_path_factory, name, extra):
    p = tmp_path_factory.mktemp(name) / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(p), *extra], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return p


@pytest.fixture(scope="module")
def swa_model(tmp_path_factory):
    return _make(tmp_path_factory, "ring", ["--swa", f"{SWA_WINDOW},2"])


@pytest.fixture(scope="module")
def dense_model(tmp_path_factory):
    return _make(tmp_path_factory, "ringdense", [])


def _run(runner_bin, model, ring, args, gpu="off"):
    env = dict(os.environ)
    if ring:
        env["RUNNER_KV_RING"] = "1"
    else:
        env.pop("RUNNER_KV_RING", None)
    return subprocess.run(
        [runner_bin, "-m", str(model), "-c", str(N_CTX), "-t", "2",
         "--gpu", gpu, *args],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300)


def _generate(runner_bin, model, ring):
    return _run(runner_bin, model, ring,
                ["-p", PROMPT, "-n", "12", "--temp", "0"])


def _info(runner_bin, model, ring):
    p = _run(runner_bin, model, ring, ["-v", "-p", "hi", "-n", "1"])
    return p.stderr.decode(errors="replace")


def _mb(text, label):
    m = re.search(rf"^{label}\s+([0-9.]+) MB", text, re.M)
    return float(m.group(1)) if m else None


def test_ring_engages_and_is_off_by_default(runner_bin, swa_model):
    assert "kv ring" not in _info(runner_bin, swa_model, ring=False)
    line = next((l for l in _info(runner_bin, swa_model, ring=True).splitlines()
                 if l.startswith("kv ring")), "")
    assert line, "ring did not engage"
    rows = int(re.search(r"kv ring\s+(\d+) rows", line).group(1))
    # window plus one in-flight batch; must exceed the window or a single
    # batch could overwrite rows its own first token still has to read
    assert rows > SWA_WINDOW, line
    assert rows < N_CTX, line


def test_ring_logprobs_are_identical_to_the_flat_allocation(
        runner_bin, swa_model, tmp_path):
    """The whole correctness claim, against the shipped engine as reference.

    Scored rather than generated, because GENERATION IS NOT SENSITIVE ENOUGH.
    Measured while building this: with the ring deliberately shrunk to half the
    window -- a ring that provably loses rows attention still reads -- greedy
    output stayed byte-identical, because the fixture echoes its prompt whatever
    the KV says. The same mutation moves a scored logprob by 7.6e-2 while a
    correct ring moves it by exactly 0.0 across 869 positions. A gate that
    cannot see a broken ring is not a gate.
    """
    assert "kv ring" in _info(runner_bin, swa_model, ring=True), "ring off"
    pf = tmp_path / "prompt.txt"
    pf.write_text(PROMPT)
    args = ["--score", "-f", str(pf)]
    flat = _run(runner_bin, swa_model, False, args)
    ring = _run(runner_bin, swa_model, True, args)
    assert flat.returncode == 0, flat.stderr.decode(errors="replace")
    assert ring.returncode == 0, ring.stderr.decode(errors="replace")
    a = json.loads(flat.stdout)
    b = json.loads(ring.stdout)
    # the ring must wrap several times or the modulo is never exercised
    assert a["n_scored"] > 8 * SWA_WINDOW, a["n_scored"]
    assert b["n_scored"] == a["n_scored"]
    worst = max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))
    assert worst == 0.0, f"ring changed logprobs by {worst}"
    assert b["top1"] == a["top1"]


def test_ring_generation_is_identical_too(runner_bin, swa_model):
    """Weaker than the scored gate above, kept because it is the user-visible
    property: the same prompt must still produce the same tokens."""
    flat = _generate(runner_bin, swa_model, ring=False)
    ring = _generate(runner_bin, swa_model, ring=True)
    assert flat.returncode == 0, flat.stderr.decode(errors="replace")
    assert len(flat.stdout) > 200, f"nothing generated: {flat.stdout!r}"
    assert ring.stdout == flat.stdout


def test_ring_actually_shrinks_the_allocation(runner_bin, swa_model):
    flat = _mb(_info(runner_bin, swa_model, ring=False), "kv cache")
    ring = _mb(_info(runner_bin, swa_model, ring=True), "kv cache")
    assert flat and ring
    assert ring < flat, f"ring {ring} MB not smaller than flat {flat} MB"


def test_ring_gpu_contract_matches_the_backend(runner_bin, swa_model,
                                                tmp_path):
    """Every device backend must RUN the ring, bit-identically.

    CUDA resolves device cache addresses through ``kv_slot`` (2026-08-30);
    Metal joined on 2026-09-01 with the same modulo through ``kv_row_off``
    in its store and attention kernels — the former Metal refusal is gone,
    and a reappearing refusal is a regression this asserts against. A
    machine without a device exercises the already-gated CPU fallback only.
    """
    p = _run(runner_bin, swa_model, True,
             ["-v", "-p", "hi", "-n", "1"], gpu="auto")
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    engaged = any(l.startswith("kv ring ") for l in err.splitlines())
    refused = "kv ring: refused" in err
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE).stdout)
    backend = caps["gpu"]["backend"] if caps.get("gpu") else None

    assert engaged and not refused, err
    if backend not in ("cuda", "metal"):
        return

    # Absolute anchor: the shipped flat allocation is the reference, and the
    # long prompt wraps this fixture's ring many times. The device path must
    # preserve each teacher-forced logprob exactly, not merely generate the
    # same text.
    pf = tmp_path / "gpu-prompt.txt"
    pf.write_text(PROMPT)
    args = ["--score", "-f", str(pf)]
    flat = _run(runner_bin, swa_model, False, args, gpu="auto")
    ring = _run(runner_bin, swa_model, True, args, gpu="auto")
    assert (b"CUDA backend" in flat.stderr + ring.stderr or
            b"Metal backend" in flat.stderr + ring.stderr)
    assert flat.returncode == ring.returncode == 0
    a, b = json.loads(flat.stdout), json.loads(ring.stdout)
    assert a["n_scored"] > 8 * SWA_WINDOW
    assert b["logprobs"] == a["logprobs"]
    assert b["top1"] == a["top1"]


def test_ring_leaves_a_model_without_sliding_layers_alone(
        runner_bin, dense_model):
    """Nothing to recycle: the flag must be inert, not merely harmless."""
    info = _info(runner_bin, dense_model, ring=True)
    assert "kv ring" not in info, info
    flat = _generate(runner_bin, dense_model, ring=False)
    ring = _generate(runner_bin, dense_model, ring=True)
    assert ring.stdout == flat.stdout


# R5.5: the paths that used to refuse under a ring -- partial rewind and the
# shared prefix cache -- now resume from what the ring still holds, by one
# rule (docs/kv-ring.md): a state at `pos` resumes at `keep` when
# keep == 0, pos <= R, or keep >= pos - n_batch - 1. The anchor is the same
# as above, the flat allocation, through the server: every request's
# continuation logprobs (and top-5 alternatives) must be bit-identical, ring
# against flat, while the ring server reports which path it took. The flat
# reference takes the same path: where the ring must miss, the flat request
# is sent with cache_prompt false, a cold recompute, because a flat hit
# prefills the tail in another batch geometry and CPU prefill is not
# batch-invariant in the last bits (measured here before the change: a ring
# reset against a flat rewind differed by 1e-15 in a top-5 logprob).
# Mutation record (2026-11-03, CPU and Metal): installing a snapshot's ring
# rows one slot off fails the fork request, a resume rule that always says yes
# fails the miss request, and a watermark that forgets earlier writes fails
# only the watermark test below.

sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

BATCH = 16                     # R = SWA_WINDOW + BATCH = 48 rows


def _ids(seed, n):
    import random
    r = random.Random(seed)
    return [r.randrange(3, 250) for _ in range(n)]


T, V, W, Y = _ids(1, 600), _ids(2, 12), _ids(3, 24), _ids(4, 12)
# r1 cold; r2 keeps the slot's 600 (a rewind past the 8 generated tokens);
# r3 shares only 300 tokens, which the ring no longer holds: a miss under the
# ring, a hit flat; r4 shares 605 with r2's cached entry (n = 612), within
# n_batch + 1 of its end: a shared-cache fork under both
REQUESTS = [(T, 8), (T + V, 8), (T[:300] + W, 8), (T + V[:5] + Y, 8)]
FLAT_COLD = {2}                # the flat reference recomputes where the ring misses
# The watermark: after 16 generated tokens the ring has recycled the slots of
# positions up to ~568; a one-token extension then leaves the slot at ~602,
# where a rule that assumed the ring holds [pos - R, pos) would let a rewind to
# 590 read positions 559-567 from slots later writes took
HOLE = [(T, 16), (T + V[:1], 1), (T[:590] + W, 8)]


def _serve(runner_bin, model, ring, gpu, requests=REQUESTS, flat_cold=FLAT_COLD):
    env = dict(os.environ)
    if ring:
        env["RUNNER_KV_RING"] = "1"
    else:
        env.pop("RUNNER_KV_RING", None)
    out = []
    with RunnerServer(runner_bin, model, ctx=N_CTX, parallel=1, env=env,
                      extra_args=["-b", str(BATCH), "-t", "2", "--gpu", gpu]) as srv:
        for i, (toks, n_out) in enumerate(requests):
            body = {"prompt": toks, "max_tokens": n_out, "temperature": 0, "logprobs": 5}
            if not ring and i in flat_cold:
                body["cache_prompt"] = False
            req = urllib.request.Request(
                srv.base_url + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            lp = d["choices"][0]["logprobs"]
            t = d["runner_telemetry"]
            out.append({"tokens": lp["tokens"], "lp": lp["token_logprobs"],
                        "top": lp["top_logprobs"], "how": t["prompt_reuse"],
                        "cached": t["prompt_cached_tokens"]})
    return out


@pytest.mark.parametrize("gpu", ["off", "auto"])
def test_ring_rewind_and_prefix_cache_match_the_flat_cache(runner_bin, swa_model, gpu):
    if gpu == "auto":
        caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT, check=True,
                                         stdout=subprocess.PIPE).stdout)
        if (caps.get("gpu") or {}).get("backend") not in ("cuda", "metal"):
            pytest.skip("no CUDA or Metal device: the CPU cell above is the run")
    flat = _serve(runner_bin, swa_model, False, gpu)
    ring = _serve(runner_bin, swa_model, True, gpu)
    for i, (a, b) in enumerate(zip(flat, ring)):
        assert b["tokens"] == a["tokens"], (i, a["tokens"], b["tokens"])
        assert b["lp"] == a["lp"], (i, a["lp"], b["lp"])
        assert b["top"] == a["top"], i
    hows = [(r["how"], r["cached"]) for r in ring]
    # the rewind keeps the slot's prompt, the uncovered prefix is recomputed,
    # and the covered one forks from the shared cache
    assert hows[1] == ("kv", 600), hows
    assert hows[2][1] == 0, hows
    assert hows[3] == ("prefix_cache", 605), hows
    # and the flat reference took the same paths where it was not sent cold
    assert [(flat[i]["how"], flat[i]["cached"]) for i in (1, 3)] == [("kv", 600), ("prefix_cache", 605)]


@pytest.mark.parametrize("gpu", ["off", "auto"])
def test_ring_resume_rule_follows_what_later_writes_recycled(runner_bin, swa_model, gpu):
    if gpu == "auto":
        caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT, check=True,
                                         stdout=subprocess.PIPE).stdout)
        if (caps.get("gpu") or {}).get("backend") not in ("cuda", "metal"):
            pytest.skip("no CUDA or Metal device: the CPU cell above is the run")
    flat = _serve(runner_bin, swa_model, False, gpu, HOLE, {2})
    ring = _serve(runner_bin, swa_model, True, gpu, HOLE, {2})
    for i, (a, b) in enumerate(zip(flat, ring)):
        assert b["tokens"] == a["tokens"] and b["lp"] == a["lp"] and b["top"] == a["top"], i
    hows = [(r["how"], r["cached"]) for r in ring]
    assert hows[1] == ("kv", 600) and hows[2] == ("ring_reset", 0), hows
