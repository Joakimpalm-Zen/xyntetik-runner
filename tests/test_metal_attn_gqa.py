"""Grouped-query decode attention on Metal is k_attn_chunk_coop's arithmetic.

`k_attn_chunk_gqa` serves every query head of a KV head from one threadgroup,
so the group's K and V rows are fetched once instead of once per query head
(src/kernels.metal). Per query head it performs the operations
`k_attn_chunk_coop` performs, in their order, so with the kernel on and off
the same prompt must produce the same tokens AND the same recorded
log-probabilities to the last bit.

The fixture has four query heads on two KV heads. RUNNER_METAL_ATTN_CHUNK
makes its short context take the chunked decode path both kernels live on.

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


def _run(exe, model, gqa, chunk):
    env = {"RUNNER_METAL_ATTN_GQA": str(gqa), "RUNNER_METAL_ATTN_CHUNK": str(chunk)}
    with RunnerServer(exe, model, ctx=1024, parallel=1, env=env) as srv:
        req = urllib.request.Request(
            srv.base_url + "/v1/completions",
            data=json.dumps({"prompt": PROMPT, "max_tokens": 24, "temperature": 0,
                             "logprobs": 5}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
    ch = body["choices"][0]
    return ch["text"], ch["logprobs"]


@pytest.mark.parametrize("flags,chunk", [
    ((), 16),                                    # several chunks per head
    ((), 50),                                    # a ragged last chunk
    (("--arch", "qwen3", "--swa", "40,2"), 16),  # sliding window: chunks from t0
], ids=["chunk16", "chunk50", "swa"])
def test_grouped_kernel_matches_the_per_head_kernel_bit_for_bit(exe, tmp_path, flags, chunk):
    model = _model(tmp_path, *flags)
    text1, lp1 = _run(exe, model, 1, chunk)
    text0, lp0 = _run(exe, model, 0, chunk)
    assert text1 == text0
    assert lp1["token_logprobs"] and len(lp1["token_logprobs"]) == 24
    assert lp1 == lp0   # every reported number, exactly


def test_the_grouped_kernel_is_the_one_that_runs(exe, tmp_path):
    """The comparison above means nothing if both arms ran the same kernel."""
    model = _model(tmp_path)
    counts = {}
    for gqa in (1, 0):
        env = dict(os.environ, RUNNER_METAL_ATTN_GQA=str(gqa),
                   RUNNER_METAL_ATTN_CHUNK="16", RUNNER_METAL_STATS="1")
        p = subprocess.run([exe, "-m", str(model), "-p", PROMPT, "-n", "8", "--temp", "0",
                            "--no-tray"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        assert p.returncode == 0
        seen = [int(line.split("gqa ")[1].split(")")[0])
                for line in p.stderr.decode(errors="replace").splitlines()
                if "metal-census" in line]
        counts[gqa] = max(seen) if seen else -1
    assert counts[1] > 0, counts
    assert counts[0] == 0, counts
