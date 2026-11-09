"""The route line's tc= label names the route the default takes (R3.39.2).

With RUNNER_CUDA_TC unset the line read tc=promoted for every non-MoE architecture, the listed ones or not, so an
unpromoted model's line claimed a tensor-core prefill it never ran. It now reads taken only when the default takes
the tensor cores for one of the model's weight matrices (tc_promoted), and off otherwise. The label is checked against
the route itself: a chunked score's receipt counts the tensor-core GEMMs by type (kernels.gemm_by_type).

- A llama fixture (a promoted architecture): taken, and its matrices take the tensor cores.
- A wide spark2_5 fixture in Q4_K: promoted for Q4_K and Q6_K by R3.39.2, so taken: Q4_K takes the tensor cores and its
  Q5_0 fallback rows do not.
- A narrow spark2_5 fixture (Q4_K falls back to Q5_0 on its short rows): no promoted type, so off and no tensor-core
  GEMM.
- A deepseek2 fixture (not promoted): off and no tensor-core GEMM, where it read "promoted" before.

The checks need a CUDA device at full offload; elsewhere the module skips.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT = (ROOT / "tests/fixtures/gold-corpus-2k.txt").read_text(encoding="utf-8")[:1500]


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def models(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    tmp = tmp_path_factory.mktemp("tclabel")
    out = {}
    for name, extra in (("llama", []), ("spark-wide", ["--spark2-5", "--wide"]), ("spark", ["--spark2-5"]),
                        ("deepseek2", ["--arch", "deepseek2", "--mla-wide"])):
        f = tmp / f"{name}.gguf"
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", *extra, str(f)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        q = tmp / f"{name}-q4k.gguf"
        subprocess.run([str(runner_bin), "-m", str(f), "--quantize", str(q), "--quant", "q4_k"], check=True,
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out[name] = q
    (tmp / "text.txt").write_text(TEXT, encoding="utf-8")
    out["text"] = tmp / "text.txt"
    return out


def _route(runner_bin, model, text):
    p = subprocess.run([str(runner_bin), "-m", str(model), "--score", "-f", str(text), "--score-tokens", "120",
                        "--gpu", "99", "--no-tray", "-t", "2"], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "RUNNER_SCORE_CHUNKED": "1", "RUNNER_DEBUG_ROUTE": "1"})
    assert p.returncode == 0, p.stderr[-1500:]
    label = re.search(r" tc=([a-z-]+)", p.stderr)
    gemm = re.search(r'"gemm_by_type":(\{[^}]*\})', p.stderr)
    assert label and gemm, p.stderr[-1500:]
    return label.group(1), json.loads(gemm.group(1))


@pytest.mark.parametrize("name, label, took", [
    ("llama", "taken", None),
    ("spark-wide", "taken", {"Q4_K"}),
    ("spark", "off", set()),
    ("deepseek2", "off", set()),
])
def test_the_label_names_the_route_the_default_takes(runner_bin, models, name, label, took):
    got, gemm = _route(runner_bin, models[name], models["text"])
    assert got == label, (name, got, gemm)
    assert bool(gemm) == (label == "taken"), (name, got, gemm)
    if took is not None:
        assert set(gemm) == took, (name, gemm)
