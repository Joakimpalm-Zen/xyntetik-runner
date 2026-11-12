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
import re
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


# gemma-4's E-series shape at 512-wide heads, sliding and full layers alike:
# the family the wide tile is on by default for
WIDE512 = ("--eseries", "3,16", "--head-dim", "512")


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
    (WIDE512, ()),                              # 512-wide heads (R3.6.3.2)
], ids=["f16", "q8", "k8v4", "fp4", "swa", "wide512"])
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


def _tiles(exe, model, **env):
    """The tiled attention dispatches a short prefill made, and the route
    detail a --score receipt names."""
    e = dict(os.environ, RUNNER_METAL_STATS="1", **env)
    p = subprocess.run([exe, "-m", str(model), "-p", PROMPT, "-n", "2", "--temp", "0",
                        "-b", "24", "--no-tray"], env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
    tiles = [int(line.split("(tile ")[1].split(")")[0])
             for line in p.stderr.decode(errors="replace").splitlines()
             if "metal-census" in line]
    s = subprocess.run([exe, "-m", str(model), "--score", "-p", PROMPT, "--no-tray"], env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert s.returncode == 0, s.stderr.decode(errors="replace")[-400:]
    detail = json.loads(s.stdout)["score_route"]["backend_detail"]
    return (max(tiles) if tiles else -1), detail


def test_wide_heads_take_the_tile_on_the_e_series_and_the_switch_keeps_k_attn(exe, tmp_path):
    """R3.6.3.2: heads 257 to 512 wide ran k_attn, a column a threadgroup over
    the whole history. On gemma-4's E-series the tile takes them by default,
    the receipt says attn_wide=on, and RUNNER_METAL_ATTN_TILE_WIDE=0 keeps
    k_attn for them (every layer of the fixture is 512 wide, so no tile)."""
    model = _model(tmp_path, *WIDE512)
    on, detail_on = _tiles(exe, model)
    off, detail_off = _tiles(exe, model, RUNNER_METAL_ATTN_TILE_WIDE="0")
    assert on > 0 and "attn_wide=on" in detail_on, (on, detail_on)
    assert off == 0 and "attn_wide" not in detail_off, (off, detail_off)


def test_a_wide_head_outside_the_family_waits_for_the_switch(exe, tmp_path):
    """The default is the family it was measured on; =1 takes any model."""
    model = _model(tmp_path, "--head-dim", "512")
    dflt, detail = _tiles(exe, model)
    forced, detail_forced = _tiles(exe, model, RUNNER_METAL_ATTN_TILE_WIDE="1")
    assert dflt == 0 and "attn_wide" not in detail, (dflt, detail)
    assert forced > 0 and "attn_wide=on" in detail_forced, (forced, detail_forced)


def test_the_score_scratch_is_bounded_and_a_wide_batch_spans_dispatches(exe, tmp_path):
    """R6.68.88: the attention score scratch holds a bounded number of
    columns, and a batch wider than that runs its attention several
    dispatches a layer, each column at its own position, for the same bytes.
    The scratch was batch x heads x context floats (2 GiB for a 512-column
    batch of a 32-head model at a 32K context), which the load's budget never
    counted. RUNNER_METAL_ATT_BYTES brings the bound down to fixture size."""
    model = _model(tmp_path)
    prompt = PROMPT + " " + PROMPT[:40]
    ident = {"RUNNER_METAL_MM": "0", "RUNNER_METAL_ATTN_COOP": "0", "RUNNER_METAL_MV": "0"}

    def run(gpu, env):
        return subprocess.run(
            [exe, "-m", str(model), "-c", "1024", "-b", "150", "-p", prompt,
             "-n", "8", "--temp", "0", "--gpu", gpu],
            env={**os.environ, **env}, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=300, check=True)

    def read(r):
        err = r.stderr.decode(errors="replace")
        sizes = {int(n): int(b) for n, b in
                 re.findall(r"metal-scratch n=(\d+) att=(\d+) B", err)}
        census = re.findall(r"metal-census n=(\d+) .* attn=(\d+)\(tile (\d+)\)", err)
        return sizes, [(int(n), int(a), int(t)) for n, a, t in census]

    cpu = run("off", {})
    assert cpu.stdout
    whole = run("auto", {**ident, "RUNNER_METAL_STATS": "1"})
    sizes, census = read(whole)
    one = sizes[1]                              # one column: heads x context floats
    width = max(sizes)
    assert width > 128, sizes                   # wide enough for three sub-tiles
    # the default bound: a fixture's whole batch, one dispatch a layer
    assert sizes[width] == width * one
    assert whole.stdout == cpu.stdout
    layers = [a for n, a, t in census if n == width][0]

    bounded = run("auto", {**ident, "RUNNER_METAL_STATS": "1",
                           "RUNNER_METAL_ATT_BYTES": str(64 * one)})
    sizes, census = read(bounded)
    assert sizes[width] == 64 * one, sizes      # 64 columns, whatever the batch
    assert bounded.stdout == cpu.stdout
    # the counters are cumulative: this batch's dispatches, three a layer
    assert [a for n, a, t in census if n == width][0] == 3 * layers
