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

Ring-off is the default and stays so: three call sites address KV as flat
absolute rows (model.h, `model_kv_byte_off`) and refuse under a ring, costing a
server its shared prefix cache and partial rewind. That is a real trade.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

SWA_WINDOW = 32
N_CTX = 1024
# 14 repeats tokenizes to ~869 tokens on the fixture vocabulary, which is many
# times the 96-row ring: the modulo wraps ~9 times during prefill alone.
PROMPT = " ".join(["the quick brown fox jumps over the lazy dog"] * 14)


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


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


def _run(runner_bin, model, ring, args):
    env = dict(os.environ)
    if ring:
        env["RUNNER_KV_RING"] = "1"
    else:
        env.pop("RUNNER_KV_RING", None)
    return subprocess.run(
        [runner_bin, "-m", str(model), "-c", str(N_CTX), "-t", "2",
         "--gpu", "off", *args],
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


def test_ring_is_refused_when_the_gpu_is_in_play(runner_bin, swa_model):
    """CPU-only, and refused rather than silently wrong.

    The device attention kernels address the cache by absolute position
    (`kc + base + t * row_b`, kernels.cu) and so does their row store, so a
    ring layer on the GPU reads rows holding a different token. Measured on an
    RTX 3070 partial split (20/34 layers, 2026-08-30): every scored position
    came back nan, while the same build on the CPU path was bit-identical to
    the flat allocation. A silent wrong answer is the one outcome this whole
    change must not produce, so the ring says no and says why.
    """
    env = dict(os.environ, RUNNER_KV_RING="1")
    p = subprocess.run(
        [runner_bin, "-m", str(swa_model), "-c", str(N_CTX), "-t", "2",
         "-v", "-p", "hi", "-n", "1"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    # no --gpu off here: whatever backend this host has, the ring must either
    # stay off or explain itself. It must never quietly engage beside a device.
    engaged = any(l.startswith("kv ring ") for l in err.splitlines())
    refused = "kv ring: refused" in err
    assert refused or not engaged, err


def test_ring_leaves_a_model_without_sliding_layers_alone(
        runner_bin, dense_model):
    """Nothing to recycle: the flag must be inert, not merely harmless."""
    info = _info(runner_bin, dense_model, ring=True)
    assert "kv ring" not in info, info
    flat = _generate(runner_bin, dense_model, ring=False)
    ring = _generate(runner_bin, dense_model, ring=True)
    assert ring.stdout == flat.stdout
