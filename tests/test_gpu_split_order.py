"""A partial GPU split must not depend on the file's tensor order.

The split planner sums the offloaded layers' own bytes, but the upload used
to be one contiguous file prefix through the farthest tensor any of those
layers owns. A file stored in name order (blk.0, blk.1, blk.10, ...) makes
that prefix reach nearly the whole file, and the one allocation then asks for
far more than the plan: a 14B bf16 planned at 24.8 GB on a 24 GB slice
failed its weight allocation and served from the CPU (the lab, 2026-09-23).
Now the runner measures the spread and uploads per tensor when it is real.

Needs a CUDA device; skips otherwise. Same greedy text from the block-order
and the name-order fixture, on the CPU and on a forced partial split.
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
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(block)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--alpha-order", str(alpha)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return block, alpha


def _run(model, *extra):
    p = subprocess.run([str(EXE), "-m", str(model), "-p", "The quick brown", "-n", "8",
                        "--temp", "0", "-t", "2", "--no-tray", *extra],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip(), p.stderr


def test_name_order_partial_split_uploads_per_tensor_and_matches(fixtures):
    block, alpha = fixtures
    cpu_text, _ = _run(block, "--gpu", "off")
    cpu_alpha, _ = _run(alpha, "--gpu", "off")
    assert cpu_alpha == cpu_text, "the name-order file is the same model"
    # a forced partial split: the last layer stays on the CPU
    gpu_block, err_block = _run(block, "--gpu", "auto", "--gpu-layers", "1")
    gpu_alpha, err_alpha = _run(alpha, "--gpu", "auto", "--gpu-layers", "1")
    assert "CUDA backend" in err_block and "CUDA backend" in err_alpha, (err_block, err_alpha)
    assert "uploading per tensor" not in err_block
    assert "uploading per tensor" in err_alpha, err_alpha
    assert gpu_alpha == gpu_block, (gpu_alpha, gpu_block)
