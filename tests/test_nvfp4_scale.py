"""NVFP4 is two-level: block scales inside the blocks AND a per-tensor F32
companion `<base>.scale` beside the weight. The block decode was verified
byte-identical against llama.cpp; what shipped wrong was never applying the
companion, so every NVFP4 weight came out too large by 1/scale (measured at
roughly 7,200x on a real file). This gate holds an absolute anchor: the same
values written as plain F32 with the companion folded in."""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "make-test-model.py"
PROBE = ROOT / "scripts" / "nvfp4-probe.py"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    d = tmp_path_factory.mktemp("nvfp4")
    out = {}
    for mode in ("nvfp4", "nvfp4-dequant"):
        path = d / f"{mode}.gguf"
        subprocess.run([sys.executable, GEN, path, "--quant", mode],
                       check=True, cwd=ROOT)
        out[mode] = path
    return out


def score(runner_bin, model, gpu="off"):
    r = subprocess.run([runner_bin, "-m", model, "--score", "-p",
                        "the quick brown fox jumps over the lazy dog",
                        "--gpu", gpu, "--no-tray"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), r.stderr


def test_companion_scale_is_applied(runner_bin, fixtures):
    nv, _ = score(runner_bin, fixtures["nvfp4"])
    f32, _ = score(runner_bin, fixtures["nvfp4-dequant"])
    assert nv["n_scored"] == f32["n_scored"] > 3
    diffs = [abs(a - b) for a, b in zip(nv["logprobs"], f32["logprobs"])]
    # block decode x companion in F32 versus dot(raw) * companion: the two
    # differ only by float reassociation, never by the 256x the companion is
    assert max(diffs) < 1e-4, max(diffs)


def _gpu_backend(runner_bin):
    caps = subprocess.run([runner_bin, "--caps"], capture_output=True,
                          text=True, timeout=30, cwd=ROOT)
    gpu = json.loads(caps.stdout).get("gpu") or {}
    return gpu.get("backend")


def test_gpu_auto_matches_the_cpu_and_says_which_path_ran(runner_bin, fixtures, tmp_path):
    """CUDA applies the companion in its matvec tails, so under --gpu auto a
    CUDA box must actually run the NVFP4 fixture on the device (gpu_layers in
    the receipt profile) and agree with the CPU run to reduction-order
    residue. Metal has no NVFP4 kernel and keeps the whole model on the CPU,
    so there --gpu auto IS the CPU run, bit for bit; the same on a box with
    no GPU at all. A silently unscaled device run would be off by 256x."""
    off, _ = score(runner_bin, fixtures["nvfp4"], gpu="off")
    auto, _ = score(runner_bin, fixtures["nvfp4"], gpu="auto")
    backend = _gpu_backend(runner_bin)
    if backend == "cuda":
        rec = tmp_path / "auto.json"
        r = subprocess.run([runner_bin, "-m", fixtures["nvfp4"], "-p", "the quick",
                            "-n", "2", "--temp", "0", "--gpu", "auto", "--no-tray",
                            "--transcript", str(rec)],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        prof = json.loads(rec.read_bytes())["profile"]
        assert prof["gpu_layers"] > 0, "CUDA declined the NVFP4 fixture: " + r.stderr
        diffs = [abs(a - b) for a, b in zip(off["logprobs"], auto["logprobs"])]
        assert max(diffs) < 1e-3, max(diffs)
    else:
        assert off["logprobs"] == auto["logprobs"]


def test_probe_reads_the_companion(fixtures):
    r = subprocess.run([sys.executable, PROBE, fixtures["nvfp4"]],
                       capture_output=True, text=True, check=True)
    assert ".scale = 0.00390625" in r.stdout and "plausible" in r.stdout \
        and "NOT plausible" not in r.stdout, r.stdout
