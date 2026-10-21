"""A partial GPU split must not depend on the file's tensor order.

The split planner sums the offloaded layers' own bytes, but the upload used
to be one contiguous file prefix through the farthest tensor any of those
layers owns. A file stored in name order (blk.0, blk.1, blk.10, ...) makes
that prefix reach nearly the whole file, and the one allocation then asks for
far more than the plan: a 14B bf16 planned at 24.8 GB on a 24 GB slice
failed its weight allocation and served from the CPU (the lab, 2026-09-23).
Now the runner uploads the stretches the device's tensors occupy, one
allocation a stretch, when they do not lie together.

Needs a CUDA device; skips otherwise. Same greedy text from the block-order
and the name-order fixture, on the CPU and on a forced partial split.

Twelve blocks and ten on the device: in name order blk.10 and blk.11 sort
between blk.1 and blk.2, so the device layers' span holds host layers. With
two blocks the spread came from the token embedding (sorted last), which the
device has not read since 2026-10-21 (the image is the stretch of the tensors
it reads), so the split's one layer became a contiguous span. The layers are
the wide fixture's (2.4 MB each): the image takes a gap of up to 2 MiB along
rather than start a new stretch, so the two host layers between blk.1 and
blk.2 (4.7 MB) must be wider than that for the name-order file to upload as
runs. With the narrow fixture's layers the gap was a few kilobytes, the image
one stretch, and the test asked for runs that should not exist.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def _caps():
    p = subprocess.run([str(EXE), "--caps"], capture_output=True, text=True, timeout=60)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {}


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner not built")
    caps = _caps()
    backends = json.dumps(caps).lower()
    if "cuda" not in backends:
        pytest.skip("no CUDA device")
    d = tmp_path_factory.mktemp("order")
    block = d / "block.gguf"
    alpha = d / "alpha.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--layers", "12",
                    "--wide", str(block)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--layers", "12",
                    "--wide", "--alpha-order", str(alpha)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return block, alpha


def _run(model, *extra):
    # The toy model's greedy text is byte tokens that need not form UTF-8
    # (twelve layers emit lone 0xC4 bytes), and the CLI writes the model's
    # bytes as they are: compare them as bytes, not as decoded text.
    p = subprocess.run([str(EXE), "-m", str(model), "-p", "The quick brown", "-n", "8",
                        "--temp", "0", "-t", "2", "--no-tray", *extra],
                       capture_output=True, timeout=300, cwd=ROOT)
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 0, err
    return p.stdout.strip(), err


def test_name_order_partial_split_uploads_runs_and_matches(fixtures):
    block, alpha = fixtures
    cpu_text, _ = _run(block, "--gpu", "off")
    cpu_alpha, _ = _run(alpha, "--gpu", "off")
    assert cpu_alpha == cpu_text, "the name-order file is the same model"
    # a forced partial split: the last two layers stay on the CPU
    gpu_block, err_block = _run(block, "--gpu", "auto", "--gpu-layers", "10")
    gpu_alpha, err_alpha = _run(alpha, "--gpu", "auto", "--gpu-layers", "10")
    assert "CUDA backend" in err_block and "CUDA backend" in err_alpha, (err_block, err_alpha)
    assert "uploading them as runs" not in err_block
    assert "uploading them as runs" in err_alpha, err_alpha
    assert gpu_alpha == gpu_block, (gpu_alpha, gpu_block)


def test_gpu_layers_past_the_fitted_split_says_so(fixtures):
    # An explicit --gpu-layers is honoured whatever the offload budget holds;
    # past it a discrete card fails the allocation or, under a Windows
    # driver, pages the excess to system memory (an 8 GB RTX 3070 decoded
    # 2.9 tok/s against 5.6 at the fitted split). A small --reserve-vram
    # stands in for a model that does not fit.
    block, _ = fixtures
    _, fitted = _run(block, "--gpu", "auto", "--reserve-vram", "1")
    split = next((l for l in fitted.splitlines() if l.startswith("gpu-split:") and " G=" in l), "")
    if split:
        g, n = split.split(" G=")[1].split()[0].split("/")
        if g == n:
            pytest.skip("1% of this device's memory holds the whole fixture")
    else:
        assert "not even one layer fits" in fitted, fitted
        g = "0"
    assert "past the" not in fitted, fitted
    _, forced = _run(block, "--gpu", "auto", "--reserve-vram", "1", "--gpu-layers", "99")
    assert "--gpu-layers 99 places" in forced and f"past the {g} the offload budget holds" in forced, forced
    _, unforced = _run(block, "--gpu", "auto")
    assert "past the" not in unforced, unforced
