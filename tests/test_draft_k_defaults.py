"""The --draft-k default per backend and draft source (R3.3.7).

With no --draft-k the width is 4, except on CUDA: 7 for the lookup and 2 for
the NextN head (measured: docs/compat-reports/r3.3.7-draft-width-defaults-2026-11-03).
The startup line of each source prints the width its rounds take; this pins
those numbers so a refactor cannot quietly return every source to 4. An
explicit --draft-k always wins. The CUDA half runs where a CUDA device is.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPEAT = "the cat sat on the mat and the dog sat on the mat and the cat sat on the"


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    plain = ROOT / "test.gguf"
    if not plain.exists():
        pytest.skip("test.gguf fixture not built")
    mtp = tmp_path_factory.mktemp("dk") / "mtp.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--mtp-layers", "1", str(mtp)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return {"plain": plain, "mtp": mtp}


def backend(runner_bin):
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    gpu = json.loads(caps.stdout.decode()).get("gpu") or {}
    return gpu.get("backend")


def width(runner_bin, models, source, gpu, *extra):
    model = models["mtp"] if source == "mtp" else models["plain"]
    spec = {"lookup": ["--draft", "lookup"], "mtp": ["--draft", "mtp"],
            "draft": ["--draft", str(models["plain"])]}[source]
    p = subprocess.run([runner_bin, "-m", str(model), "-p", REPEAT, "-n", "8", "-c", "512",
                        "--temp", "0", "--gpu", gpu, *spec, *extra],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    pat = {"lookup": r"lookup: drafting .* up to (\d+) per round",
           "mtp": r"mtp: drafting from the model's NextN head \((\d+) tokens per round\)",
           "draft": r"draft: .* \((\d+) tokens per round\)"}[source]
    m = re.search(pat, err)
    assert m, err
    return int(m[1])


@pytest.mark.parametrize("source", ["lookup", "mtp", "draft"])
def test_cpu_default_is_four(runner_bin, models, source):
    assert width(runner_bin, models, source, "off") == 4


@pytest.mark.parametrize("source", ["lookup", "mtp", "draft"])
def test_explicit_draft_k_wins(runner_bin, models, source):
    assert width(runner_bin, models, source, "off", "--draft-k", "3") == 3


@pytest.mark.parametrize("source,want", [("lookup", 7), ("mtp", 2), ("draft", 4)])
def test_cuda_defaults(runner_bin, models, source, want):
    if backend(runner_bin) != "cuda":
        pytest.skip("needs a CUDA device")
    assert width(runner_bin, models, source, "99") == want
    assert width(runner_bin, models, source, "99", "--draft-k", "5") == 5


@pytest.mark.parametrize("source", ["lookup", "draft"])
def test_cuda_partial_offload_keeps_four(runner_bin, models, source):
    # the CUDA widths assume every layer on the device: CPU layers still pay
    # 0.45 to 0.78 of a step a verify row (the NextN head refuses a partial
    # offload outright)
    if backend(runner_bin) != "cuda":
        pytest.skip("needs a CUDA device")
    assert width(runner_bin, models, source, "1") == 4


def test_draft_k_zero_is_still_refused(runner_bin, models):
    p = subprocess.run([runner_bin, "-m", str(models["plain"]), "-p", "x", "-n", "1",
                        "--gpu", "off", "--draft", "lookup", "--draft-k", "0"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert "--draft-k" in p.stderr.decode(errors="replace")
