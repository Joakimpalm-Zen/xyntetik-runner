"""clef's tensor-core prefill is the default for Q4_K and Q8_0, and only for those (R3.39.2).

The promotion rests on a float32-reference row and a speed row for ggml-org's Clef-Flash Q4_K_M, whose matrices are
Q8_0 and Q4_K, so it covers exactly those two types, under the obligation in docs/kernel-routes.md (mean KL within
1.01x, paired Wilcoxon and McNemar not worse at p < 0.05). A chunked score's receipt counts the tensor-core GEMMs by
type (kernels.gemm_by_type), which is what actually ran, on scripts/make-test-clef.py's fixture:
- in Q8_0 it takes the tensor cores by default, and with RUNNER_CUDA_TC=0 it takes none;
- quantized to Q4_K, its short rows fall back to Q5_0, a type the row did not exercise: none.
(The fixture's rows are too short for Q4_K blocks; tests/test_tc_spark_promotion.py pins a Q4_K promotion's dispatch.)

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
    tmp = tmp_path_factory.mktemp("tcclef")
    f = tmp / "clef.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-clef.py", str(f)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    out = {}
    for q in ("q8_0", "q4_k"):
        m = tmp / f"clef-{q}.gguf"
        subprocess.run([str(runner_bin), "-m", str(f), "--quantize", str(m), "--quant", q], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out[q] = m
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


def test_q8_0_takes_the_tensor_cores_by_default(runner_bin, models):
    gemm = _gemm(runner_bin, models["q8_0"], models["text"])
    assert set(gemm) == {"Q8_0"} and gemm["Q8_0"] > 0, gemm


def test_the_opt_out_holds_the_default_route(runner_bin, models):
    assert _gemm(runner_bin, models["q8_0"], models["text"], RUNNER_CUDA_TC="0") == {}


def test_a_type_the_row_did_not_exercise_stays_off(runner_bin, models):
    assert _gemm(runner_bin, models["q4_k"], models["text"]) == {}
