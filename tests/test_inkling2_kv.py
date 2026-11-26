"""R4.48.8.5: Inkling 2 Mini's --kv q8 cache and its fit, on the tiny inkling2_loop file (tests/inkling2_tiny.py).

Gate (b): q8 engages, and it keeps every bit invariance the float32 cache holds (thread counts, prompt chunk sizes),
asserted pair by pair against what the float32 cache shows. The house's other formats (f16, k8v4, fp4) are not this
engine's: an asked one keeps f32 with a note. Gate (a), bit for bit against the float32 cache over the round-tripped
rows, is tests/test_inkling2_kvq8.c.

The fit: --info fit counts the cache the load allocates, byte for byte, for the kind --kv asks, every slot --parallel
asks, and the sliding applications by the file's own pattern.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_tiny import FULL, H, HD, K, LOOP_END, LOOP_START, N_BLOCK, NKV, PASSES, WIN, _kvs, _tensors, _write  # noqa: E402

# 48 byte tokens: six windows of 8, so the sliding buffers slide
PROMPT = "the q8 cache slides past the window, chunk by chunk"
CTX = 64


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("ink2kv")
    return _write(d / "tiny.gguf", _kvs(), _tensors())


def _run(runner_bin, model, *args):
    return subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "off", "-c", str(CTX), *map(str, args)],
                          capture_output=True, encoding="utf-8", errors="replace", timeout=600)


def _score(runner_bin, model, *extra, threads=2, batch=8):
    r = _run(runner_bin, model, "-t", threads, "-b", batch, "--score", "1", "-p", PROMPT, *extra)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)["logprobs"], r.stderr


def test_q8_engages(runner_bin, tiny):
    f32, _ = _score(runner_bin, tiny)
    q8, err = _score(runner_bin, tiny, "--kv", "q8")
    assert "K/V q8" in err and "keeps f32" not in err, err[-2000:]
    assert q8 != f32, "--kv q8 changed no score: it did not engage"


CONFIGS = {"t1 b8": (1, 8), "t4 b8": (4, 8), "t2 b1": (2, 1), "t2 b8": (2, 8)}


def test_q8_keeps_every_bit_invariance_the_f32_cache_holds(runner_bin, tiny):
    f32 = {n: _score(runner_bin, tiny, threads=t, batch=b)[0] for n, (t, b) in CONFIGS.items()}
    q8 = {n: _score(runner_bin, tiny, "--kv", "q8", threads=t, batch=b)[0] for n, (t, b) in CONFIGS.items()}
    names = list(CONFIGS)
    held = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if f32[a] == f32[b]:
                held += 1
                assert q8[a] == q8[b], f"the float32 cache gives {a} and {b} the same bits; q8 does not"
    assert held > 0, "the float32 cache held no invariance here: the test would assert nothing"


@pytest.mark.parametrize("kv", ["fp4", "k8v4", "f16"])
def test_another_format_keeps_f32_with_a_note(runner_bin, tiny, kv):
    f32, _ = _score(runner_bin, tiny)
    got, err = _score(runner_bin, tiny, "--kv", kv)
    assert f"--kv {kv} is not one of its formats, so it keeps f32" in err, err[-2000:]
    assert got == f32


def _bytes(text, where):
    m = re.search(r"cache\s+[0-9.]+ Gi?B \((\d+) bytes", text)
    assert m, f"no exact cache bytes in the {where}:\n{text[-2000:]}"
    return int(m.group(1))


def _fit(runner_bin, model, *extra):
    r = _run(runner_bin, model, "--info", "fit", *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _apps():
    """(global, sliding) applications of the tiny file: its blocks with the loop's passes."""
    blocks = list(range(LOOP_START)) + list(range(LOOP_START, LOOP_END)) * PASSES + list(range(LOOP_END, N_BLOCK))
    g = sum(1 for b in blocks if FULL[b])
    return g, len(blocks) - g


def _expected(kv, batch, slots=1):
    C = NKV * HD
    row = C // 32 * 34 if kv == "q8" else C * 4
    g, sl = _apps()
    local = min(CTX, WIN - 1 + batch)
    conv = (g + sl) * 4 * (K - 1) * max(C, H) * 4
    return slots * (2 * g * CTX * row + 2 * sl * local * row + conv)


@pytest.mark.parametrize("kv", ["f32", "q8"])
@pytest.mark.parametrize("batch", [1, 8])
def test_the_fit_counts_the_cache_the_load_allocates(runner_bin, tiny, kv, batch):
    fit = _bytes(_fit(runner_bin, tiny, "--kv", kv, "-b", batch), "fit")
    r = _run(runner_bin, tiny, "--kv", kv, "-b", batch, "-t", 2, "-p", "hi", "-n", 1, "--temp", 0)
    assert r.returncode == 0, r.stderr[-2000:]
    assert fit == _bytes(r.stderr, "load") == _expected(kv, batch)


def test_the_fit_names_the_sliding_applications_by_the_files_pattern(runner_bin, tiny):
    _, sl = _apps()
    out = _fit(runner_bin, tiny)
    assert f"the {sl} sliding ones" in out, out


def test_the_fit_counts_every_slot(runner_bin, tiny):
    one = _bytes(_fit(runner_bin, tiny, "--kv", "q8", "-b", 8), "fit")
    two = _bytes(_fit(runner_bin, tiny, "--kv", "q8", "-b", 8, "--parallel", 2), "fit")
    assert two == 2 * one == _expected("q8", 8, slots=2)


def test_the_fit_says_what_another_format_and_the_ring_switch_do(runner_bin, tiny):
    out = _fit(runner_bin, tiny, "--kv", "fp4", "--kv-ring", "off", "-b", 8)
    assert "keeps f32" in out and "always keep a window ring" in out, out
    assert _bytes(out, "fit") == _expected("f32", 8)


def test_the_cuda_fit_places_the_file_as_its_load_does(runner_bin, tiny):
    """On a CUDA host: --info fit's device section is the load's own placement run dry, so its placement is the load's
    (every weight on the device, or the expert stacks on the host, or the CPU), at --kv q8, and its non-weight line
    carries exact bytes."""
    import os
    env = dict(os.environ, RUNNER_INKLING2_CUDA="1")
    env.pop("RUNNER_INKLING2_ROUTE", None)
    common = [str(runner_bin), "-m", str(tiny), "--gpu", "auto", "-c", str(CTX), "-b", "8", "--kv", "q8"]
    fit = subprocess.run(common + ["--info", "fit"], capture_output=True, encoding="utf-8", errors="replace", env=env,
                         timeout=600)
    m = re.search(r"device\s+CUDA route: (.*)", fit.stdout)
    if not m:
        pytest.skip("no CUDA device here: the device fit is a CUDA host's (" + fit.stdout.strip().splitlines()[-1] + ")")
    assert re.search(r"non-weight\s+[0-9.]+ GiB \(\d+ bytes", fit.stdout), fit.stdout
    # the whole placement's own non-weight bytes, whichever placement the device took
    assert re.search(r"whole\s+[0-9.]+ GiB of weights \(\d+ bytes\) and [0-9.]+ GiB besides them \(non-weight \d+ bytes\)",
                     fit.stdout), fit.stdout
    load = subprocess.run(common + ["-t", "2", "-p", "hi", "-n", "1", "--temp", "0"], capture_output=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)
    assert load.returncode == 0, load.stderr[-2000:]
    on_device = "inkling2_loop: the weights on CUDA" in load.stderr
    assert m.group(1).startswith("every weight on the device") == on_device, (fit.stdout, load.stderr[-2000:])


def test_q8_engages_on_the_device(runner_bin, tiny):
    """On a CUDA host: --kv q8 under RUNNER_INKLING2_CUDA=1 stores q8 rings on the device, with no test switch: the load
    line says K/V q8, no note keeps f32, and the weights are on CUDA (the bind banner). Gate (c) in
    tests/test_inkling2_cuda.c holds those rings to the CPU's q8 cache."""
    import os
    env = dict(os.environ, RUNNER_INKLING2_CUDA="1")
    env.pop("RUNNER_INKLING2_ROUTE", None)
    common = [str(runner_bin), "-m", str(tiny), "--gpu", "auto", "-c", str(CTX), "-b", "8", "--kv", "q8"]
    fit = subprocess.run(common + ["--info", "fit"], capture_output=True, encoding="utf-8", errors="replace", env=env,
                         timeout=600)
    if "CUDA route:" not in fit.stdout:
        pytest.skip("no CUDA device here: the device rings are a CUDA host's")
    load = subprocess.run(common + ["-t", "2", "-p", "hi", "-n", "1", "--temp", "0"], capture_output=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)
    assert load.returncode == 0, load.stderr[-2000:]
    assert "K/V q8" in load.stderr and "keeps f32" not in load.stderr, load.stderr[-2000:]
    assert "inkling2_loop: the weights on CUDA" in load.stderr, load.stderr[-2000:]
    assert "slots' K/V rings in q8" in fit.stdout, fit.stdout
