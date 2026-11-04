"""CUDA's wide-head tensor-core prefill attention is the same bytes run to run (R3.6.3.1).

`k_attn_tcw` sums its four warps' partial scores in warp order and walks the keys in a fixed tile order
with a running maximum (src/kernels.cu), so scoring a text twice through the prefill route must give
the same bytes. `RUNNER_SCORE_CHUNKED=1` reads the text in 64-row tiles, so every position comes
through the prefill attention. Two head widths, 256 and 512 (make-test-model.py --head-dim, four query
heads on two KV heads), forced on with RUNNER_CUDA_ATTN_TC_WIDE=1 since the fixture is not gemma-4
E-series; the route record must name the kernel, and =0 must not.

Skipped where there is no CUDA device.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_record import body_bytes  # noqa: E402

TEXT = " ".join(["the quick brown fox jumps over the lazy dog"] * 12)


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = json.loads(subprocess.run([str(exe), "--caps"], cwd=ROOT, check=True,
                                     stdout=subprocess.PIPE).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the wide-head tensor-core attention needs a CUDA device")
    return exe


def _score(exe, model, wide):
    env = dict(os.environ, RUNNER_SCORE_CHUNKED="1", RUNNER_SPEC_BATCH="64",
               RUNNER_CUDA_ATTN_TC_WIDE=wide)
    p = subprocess.run([str(exe), "-m", str(model), "--gpu", "auto", "-c", "1024",
                        "--score", "-p", TEXT, "--no-tray"],
                       cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


@pytest.mark.parametrize("hd", [256, 512])
def test_wide_tile_is_the_same_bytes_twice(runner_bin, tmp_path, hd):
    model = tmp_path / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model), "--head-dim", str(hd)],
                   check=True, cwd=ROOT, stdout=subprocess.PIPE)
    a = _score(runner_bin, model, "1")
    b = _score(runner_bin, model, "1")
    rec = json.loads(a)
    assert " attn_wide=on " in rec["score_route"]["backend_detail"], rec["score_route"]
    assert rec["n_scored"] > 300, rec["n_scored"]
    assert body_bytes(a) == body_bytes(b)
    off = json.loads(_score(runner_bin, model, "0"))
    assert "attn_wide" not in off["score_route"]["backend_detail"], off["score_route"]


def _gen_qwen35(path, state):
    # scripts/make-test-ornith.py: the qwen35 hybrid (Gated DeltaNet layers
    # and full-attention ones), its attention heads ORNITH_TEST_STATE wide
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(path)], check=True, cwd=ROOT,
                   stdout=subprocess.PIPE, env=dict(os.environ, ORNITH_TEST_STATE=str(state)))


def test_qwen35_at_head_width_256_takes_the_wide_tile_by_default(runner_bin, tmp_path):
    """The default list (R3.6.3.2): qwen35 at head width 256 runs k_attn_tcw with the switch unset,
    =0 keeps the tile, and a 256-wide model of another architecture stays off by default."""
    model = tmp_path / "q35.gguf"
    _gen_qwen35(model, 256)
    on = json.loads(_score(runner_bin, model, ""))
    assert " attn_wide=on " in on["score_route"]["backend_detail"], on["score_route"]
    off = json.loads(_score(runner_bin, model, "0"))
    assert "attn_wide" not in off["score_route"]["backend_detail"], off["score_route"]
    other = tmp_path / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(other), "--head-dim", "256"],
                   check=True, cwd=ROOT, stdout=subprocess.PIPE)
    dense = json.loads(_score(runner_bin, other, ""))
    assert "attn_wide" not in dense["score_route"]["backend_detail"], dense["score_route"]


def test_qwen35_at_another_head_width_stays_off_by_default(runner_bin, tmp_path):
    # the kernel admits 512, but the default keys on the measured width: a
    # 512-wide qwen35 stays on the tile unless RUNNER_CUDA_ATTN_TC_WIDE=1
    model = tmp_path / "q35n.gguf"
    _gen_qwen35(model, 512)
    rec = json.loads(_score(runner_bin, model, ""))
    assert "attn_wide" not in rec["score_route"]["backend_detail"], rec["score_route"]
