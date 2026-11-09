"""spark2_5's tensor-core prefill is the default for Q4_K and Q6_K, and only for those (R3.39.2).

The promotion rests on a float32-reference row and a speed row for XHToken's Spark-X2.5-4B Q4_K_M, whose matrices
are Q4_K and Q6_K, so it covers exactly those two types. A chunked score's receipt counts the tensor-core GEMMs by
type (kernels.gemm_by_type), which is what actually ran:
- a wide spark2_5 fixture in Q4_K takes the tensor cores for its Q4_K matrices and not for its Q5_0 fallback rows;
- with RUNNER_CUDA_TC=0 it takes none;
- a narrow spark2_5 fixture (its short rows fall back to Q5_0) takes none.

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
    tmp = tmp_path_factory.mktemp("tcspark")
    out = {}
    for name, extra in (("wide", ["--spark2-5", "--wide"]), ("narrow", ["--spark2-5"])):
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


def _gemm(runner_bin, model, text, **env):
    p = subprocess.run([str(runner_bin), "-m", str(model), "--score", "-f", str(text), "--score-tokens", "120",
                        "--gpu", "99", "--no-tray", "-t", "2"], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "RUNNER_SCORE_CHUNKED": "1", **env})
    assert p.returncode == 0, p.stderr[-1500:]
    gemm = re.search(r'"gemm_by_type":(\{[^}]*\})', p.stderr)
    assert gemm, p.stderr[-1500:]
    return json.loads(gemm.group(1))


def test_q4_k_takes_the_tensor_cores_by_default_and_q5_0_does_not(runner_bin, models):
    gemm = _gemm(runner_bin, models["wide"], models["text"])
    assert set(gemm) == {"Q4_K"} and gemm["Q4_K"] > 0, gemm


def test_the_opt_out_holds_the_default_route(runner_bin, models):
    assert _gemm(runner_bin, models["wide"], models["text"], RUNNER_CUDA_TC="0") == {}


def test_a_type_the_row_did_not_exercise_stays_off(runner_bin, models):
    assert _gemm(runner_bin, models["narrow"], models["text"]) == {}
