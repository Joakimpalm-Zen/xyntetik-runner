"""Sliding-window layers are allocated KV they can never read.

Every layer gets `n_ctx` rows, but a sliding layer clamps its attention start
to `p - swa_window + 1`, so rows older than the window are written once and
never read again. On a model whose sliding layers are more numerous AND wider
in KV than its full ones this is most of the cache: measured externally on
gemma-4-31B, 91% of the KV budget serving a 1024-token window, and 29.53 GB
wanted at ctx 32k against 3.52 GB reachable, which is the difference between a
4k toy and a 32k agentic context on a 24 GB device.

This gate pins the REPORT, not a fix. The fix is a ring layout, and three call
sites assume flat absolute rows (model.h, `model_kv_byte_off`), so measuring
before changing the layout is deliberate rather than partial.

THE ANCHOR. The expected byte counts are computed here from the geometry the
model itself publishes -- kv heads, head dim, context, window -- and the
definition of a sliding window, not from anything the KV code does. A cache
row is `n_head_kv * head_dim` f16 elements, K and V both, so:

    allocated = sum over layers of n_ctx rows
    reachable = sum over layers of (sliding ? min(n_ctx, window) : n_ctx) rows

An engine that computed reachability its own way would agree with itself and
fail here, which is the point: a gate whose expected value comes from the
system under test proves only that the system is self-consistent.
"""
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

N_CTX = 16384
SWA_WINDOW = 32
SWA_PERIOD = 2          # every layer whose 1-based index is not a multiple
                        # of 2 slides, so layer 0 slides and layer 1 does not


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
    return _make(tmp_path_factory, "kvreach",
                 ["--swa", f"{SWA_WINDOW},{SWA_PERIOD}"])


@pytest.fixture(scope="module")
def dense_model(tmp_path_factory):
    return _make(tmp_path_factory, "kvdense", [])


def _verbose(runner_bin, model):
    p = subprocess.run(
        [runner_bin, "-m", str(model), "-v", "-p", "hi", "-n", "1",
         "-c", str(N_CTX), "-t", "2", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=120)
    return p.stderr.decode(errors="replace")


def _mb(text, label):
    m = re.search(rf"^{label}\s+([0-9.]+) MB", text, re.M)
    return float(m.group(1)) if m else None


def _geometry(text):
    """kv heads, head dim, layers and context, as the model reports them."""
    heads = re.search(r"^heads\s+(\d+) \((\d+) kv\)", text, re.M)
    hd = re.search(r"^head dim\s+(\d+)", text, re.M)
    layers = re.search(r"^layers\s+(\d+)", text, re.M)
    ctx = re.search(r"^context\s+(\d+)", text, re.M)
    assert heads and hd and layers and ctx, text
    return (int(heads.group(2)), int(hd.group(1)),
            int(layers.group(1)), int(ctx.group(1)))


def test_reachable_kv_matches_the_sliding_window_definition(
        runner_bin, swa_model):
    out = _verbose(runner_bin, swa_model)
    n_kv, head_dim, n_layer, n_ctx = _geometry(out)

    # K and V, f16, one row per position per layer
    row_bytes = n_kv * head_dim * 2
    both = 2                                   # kcache and vcache
    sliding = [(i + 1) % SWA_PERIOD != 0 for i in range(n_layer)]
    assert any(sliding) and not all(sliding), sliding

    alloc_rows = n_layer * n_ctx
    reach_rows = sum(min(n_ctx, SWA_WINDOW) if s else n_ctx for s in sliding)
    expect_alloc = both * alloc_rows * row_bytes / 1e6
    expect_reach = both * reach_rows * row_bytes / 1e6

    allocated = _mb(out, "kv cache")
    reachable = _mb(out, "kv reachable")
    assert allocated is not None, f"no 'kv cache' line in:\n{out}"
    assert reachable is not None, f"no 'kv reachable' line in:\n{out}"
    # half of the reported precision (one decimal place)
    assert allocated == pytest.approx(expect_alloc, abs=0.05)
    assert reachable == pytest.approx(expect_reach, abs=0.05)
    assert reachable < allocated


def test_report_names_the_sliding_layers_and_the_window(runner_bin, swa_model):
    out = _verbose(runner_bin, swa_model)
    _, _, n_layer, _ = _geometry(out)
    n_swa = sum((i + 1) % SWA_PERIOD != 0 for i in range(n_layer))
    line = next((l for l in out.splitlines() if l.startswith("kv reachable")),
                "")
    assert f"{n_swa} of {n_layer} layers slide" in line, line
    assert f"{SWA_WINDOW}-token window" in line, line


def test_a_model_without_sliding_layers_reports_no_waste(
        runner_bin, dense_model):
    """The line must not appear when there is nothing to report."""
    out = _verbose(runner_bin, dense_model)
    assert _mb(out, "kv cache") is not None
    assert _mb(out, "kv reachable") is None, out
