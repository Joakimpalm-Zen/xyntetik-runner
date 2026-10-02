"""The tiled Metal prefill attention is k_attn's arithmetic, not a new route.

`k_attn_tile` lets several prompt columns share each read of the KV history
(src/kernels.metal). It was written to perform, per column, exactly the
operations `k_attn` performs in exactly their order, which is a claim a test
can hold: with the tile on and off, the same prompt must produce the same
tokens AND the same recorded log-probabilities to the last bit, on every
cache format and attention shape the kernel has a branch for.

Skipped where there is no Metal device.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer, find_runner  # noqa: E402

PROMPT = ("the quick brown fox jumps over the lazy dog and keeps running past "
          "the mill, the river and the two bridges until the road ends")


def _metal(exe):
    try:
        caps = json.loads(subprocess.run([exe, "--caps"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, timeout=60).stdout)
    except Exception:
        return False
    return (caps.get("gpu") or {}).get("backend") == "metal"


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    if not _metal(e):
        pytest.skip("no Metal device")
    return e


def _model(tmp_path, *flags):
    m = tmp_path / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", *flags, str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _run(exe, model, tile, extra):
    """Top-5 log-probabilities of 8 generated tokens after a 3-batch prefill,
    and how many tiled attention dispatches the server reported."""
    env = {"RUNNER_METAL_ATTN_TILE": str(tile), "RUNNER_METAL_STATS": "1"}
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["-b", "24", *extra], env=env) as srv:
        req = urllib.request.Request(
            srv.base_url + "/v1/completions",
            data=json.dumps({"prompt": PROMPT, "max_tokens": 8, "temperature": 0,
                             "logprobs": 5}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
        err = srv.stderr_text() if hasattr(srv, "stderr_text") else ""
    ch = body["choices"][0]
    return ch["text"], ch["logprobs"], err


@pytest.mark.parametrize("flags,extra", [
    ((), ()),                                   # f16 cache, full attention
    (("--wide",), ("--kv", "q8")),              # quantized K and V
    (("--wide",), ("--kv", "k8v4")),            # q8 K, fp4 V
    (("--wide",), ("--kv", "fp4")),             # fp4 both
    (("--arch", "qwen3", "--swa", "8,2"), ()),  # sliding window: per-column t0
], ids=["f16", "q8", "k8v4", "fp4", "swa"])
def test_tile_matches_the_one_column_kernel_bit_for_bit(exe, tmp_path, flags, extra):
    model = _model(tmp_path, *flags)
    text1, lp1, _ = _run(exe, model, 1, extra)
    text0, lp0, _ = _run(exe, model, 0, extra)
    assert text1 == text0
    assert lp1["token_logprobs"] and len(lp1["token_logprobs"]) == 8
    # exact equality of every reported number, not a tolerance
    assert lp1 == lp0


def test_the_tiled_kernel_is_the_one_that_runs(exe, tmp_path):
    """The comparison above means nothing if both arms ran k_attn."""
    model = _model(tmp_path)
    counts = {}
    for tile in (1, 0):
        env = dict(os.environ, RUNNER_METAL_ATTN_TILE=str(tile), RUNNER_METAL_STATS="1")
        p = subprocess.run([exe, "-m", str(model), "-p", PROMPT, "-n", "2", "--temp", "0",
                            "-b", "24", "--no-tray"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        assert p.returncode == 0
        tiles = [int(line.split("(tile ")[1].split(")")[0])
                 for line in p.stderr.decode(errors="replace").splitlines()
                 if "metal-census" in line]
        counts[tile] = max(tiles) if tiles else -1
    assert counts[1] > 0, counts
    assert counts[0] == 0, counts
