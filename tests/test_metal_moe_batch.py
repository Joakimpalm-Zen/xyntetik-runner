"""Metal MoE prompt batching: identity and dispatch-count tracer."""

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "abcdefg"  # byte fixture: BOS plus seven bytes is one full batch of 8


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = json.loads(subprocess.run([exe, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE,
                                     check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal backend")
    return exe


@pytest.fixture(scope="module")
def moe_fixture_prefix(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("metalmoebatch") / "f"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(prefix)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return prefix


def _run(runner, model, gpu, *, stats=False, route_trace=False,
         route_batch=True, nan_trace=None, full_trace=None):
    env = dict(os.environ, RUNNER_METAL_MM="0", RUNNER_METAL_ATTN_COOP="0", RUNNER_METAL_MV="0")
    if stats:
        env["RUNNER_METAL_STATS"] = "1"
    if route_trace:
        env["RUNNER_METAL_MOE_ROUTE_TRACE"] = "1"
        env["RUNNER_METAL_MOE_BATCH"] = "1" if route_batch else "0"
    if nan_trace is not None:
        env["RUNNER_METAL_NAN_TRACE"] = str(nan_trace)
    if full_trace is not None:
        env["RUNNER_MOE_TRACE"] = str(full_trace)
    return subprocess.run(
        [runner, "-m", model, "-p", PROMPT, "-n", "1", "-b", "8",
         "--temp", "0", "--gpu", gpu],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300, check=True)


def test_prompt_tile_batches_moe_without_changing_tokens(runner_bin,
                                                          moe_fixture_prefix):
    gemma_moe_fixture = pathlib.Path(f"{moe_fixture_prefix}.gemma4-moe.gguf")
    cpu = _run(runner_bin, gemma_moe_fixture, "off")
    gpu = _run(runner_bin, gemma_moe_fixture, "auto", stats=True)
    assert cpu.stdout == gpu.stdout
    assert b"Metal backend" in gpu.stderr

    match = re.search(rb"metal-census n=8 .* moe=(\d+)", gpu.stderr)
    assert match, gpu.stderr.decode(errors="replace")
    # Two layers, each encoded as one route, two expert projections, one
    # activation, one down projection, and one ordered weighted sum.
    assert int(match.group(1)) == 10


@pytest.mark.parametrize("name, per_layer", [
    # attention's four projections and the router
    ("moe4", 5),
    # and the gemma-4 shared branch's gate, up and down
    ("gemma4-moe", 8),
])
def test_a_batch_encodes_each_decode_bit_matvec_once_a_layer(
        runner_bin, moe_fixture_prefix, name, per_layer):
    """R6.68.87: the router and the gemma-4 shared branch keep a decode
    step's bits in a prompt batch, and go out as one dispatch for the
    batch's columns. They were encoded once a column: 8 columns, 8 times
    the dispatches, for the same bytes."""
    model = pathlib.Path(f"{moe_fixture_prefix}.{name}.gguf")
    cpu = _run(runner_bin, model, "off")
    gpu = _run(runner_bin, model, "auto", stats=True)
    assert cpu.stdout == gpu.stdout
    match = re.search(rb"metal-census n=8 .* mv=(\d+) ", gpu.stderr)
    assert match, gpu.stderr.decode(errors="replace")
    # two layers, and the output head
    assert int(match.group(1)) == 2 * per_layer + 1


def test_a_quantized_shared_branch_keeps_its_bytes_in_a_batch(
        runner_bin, moe_fixture_prefix, tmp_path):
    """The gemma-4 MoE shared branch at Q8_0: with several columns in one
    dispatch it runs the column-inner twin (k_mvc_q8_0), where a one-column
    step runs k_mv_q8_0. The F32 fixtures above only reach the plain column
    walk; this one reaches a twin on the batch path (R6.68.87's second
    read), and Metal admits a twin only after comparing it with its
    one-column kernel on the running device (metal_mvc_self_test)."""
    src = pathlib.Path(f"{moe_fixture_prefix}.gemma4-moe.gguf")
    model = tmp_path / "gemma4-moe-q8.gguf"
    subprocess.run([runner_bin, "-m", src, "--quantize", model, "--quant", "q8_0"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    cpu = _run(runner_bin, model, "off")
    gpu = _run(runner_bin, model, "auto", stats=True)
    assert cpu.stdout and cpu.stdout == gpu.stdout
    assert b"Metal backend" in gpu.stderr
    assert b"not bit-identical on this device" not in gpu.stderr, \
        "the twin was refused here, so this run did not reach it"
    match = re.search(rb"metal-census n=8 .* mv=(\d+) ", gpu.stderr)
    assert match, gpu.stderr.decode(errors="replace")
    assert int(match.group(1)) == 2 * 8 + 1


def test_batched_routes_are_byte_identical_to_serial_routes(
        runner_bin, moe_fixture_prefix):
    model = pathlib.Path(f"{moe_fixture_prefix}.moe4.gguf")
    serial = _run(runner_bin, model, "auto", route_trace=True,
                  route_batch=False)
    batched = _run(runner_bin, model, "auto", route_trace=True,
                   route_batch=True)
    assert serial.stdout == batched.stdout
    pattern = re.compile(rb"^metal-moe-route .*$", re.MULTILINE)
    serial_routes = pattern.findall(serial.stderr)
    batched_routes = pattern.findall(batched.stderr)
    assert serial_routes, serial.stderr.decode(errors="replace")
    assert serial_routes == batched_routes


def test_full_router_logits_survive_metal_layer_reuse(
        runner_bin, moe_fixture_prefix, tmp_path):
    model = pathlib.Path(f"{moe_fixture_prefix}.moe4.gguf")
    trace = tmp_path / "routes.jsonl"
    _run(runner_bin, model, "auto", full_trace=trace)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows
    positions = {row["pos"] for row in rows}
    assert len(positions) >= 8
    assert {(row["pos"], row["layer"]) for row in rows} == {
        (pos, layer) for pos in positions for layer in range(2)
    }
    for row in rows:
        assert len(row["logits"]) == 4
        assert len(row["experts"]) == 2
        assert len(row["gates"]) == 2
        assert all(isinstance(value, float) for value in row["logits"])
    # The two layers carry different routers. If the backend merely reads the
    # reused live buffer after completion, every layer would contain the last
    # layer's vector and this independent-lifetime assertion fails.
    by_pos = {(row["pos"], row["layer"]): row for row in rows}
    assert any(by_pos[pos, 0]["logits"] != by_pos[pos, 1]["logits"]
               for pos in positions)


@pytest.mark.parametrize("level", [3, 4])
def test_gemma_moe_nan_stage_probes_preserve_output(
        runner_bin, moe_fixture_prefix, level):
    """Deep probes replace encoders inside the Gemma MoE helper.

    The replacement must be returned to the outer graph before it appends the
    residual tail.  Output identity makes this an end-to-end ownership gate,
    rather than merely checking the source assignment.
    """
    model = pathlib.Path(f"{moe_fixture_prefix}.gemma4-moe.gguf")
    baseline = _run(runner_bin, model, "auto")
    probed = _run(runner_bin, model, "auto", nan_trace=level)
    assert baseline.stdout == probed.stdout


def _continuation_logprobs(runner, model, *args):
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import RunnerServer
    import urllib.request
    prompt = ("the quick brown fox jumps over the lazy dog and keeps running "
              "past the mill and the river")
    with RunnerServer(runner, model, ctx=256, parallel=1, extra_args=list(args)) as srv:
        req = urllib.request.Request(
            srv.base_url + "/v1/completions",
            data=json.dumps({"prompt": prompt, "max_tokens": 12, "temperature": 0,
                             "logprobs": 1}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            lp = json.load(r)["choices"][0]["logprobs"]
    return lp["tokens"], lp["token_logprobs"]


def test_an_expert_layer_narrower_than_a_tile_step_keeps_the_matvec(runner_bin, tmp_path):
    """The grouped expert tile steps its input 32 wide with no tail step. The
    dense tile is admitted only for an input that is a whole number of steps;
    the expert tile had no such rule, so a prompt batch through an expert
    layer 16 wide read 16 past every weight row and activation row, and the
    continuation's log-probabilities were off the CPU's by up to 0.7 on this
    fixture (512 wide, experts 16 wide). One token at a time never took the
    tile, which is why scoring agreed."""
    model = tmp_path / "ornith-moe-s128.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(model)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, ORNITH_TEST_MOE="1", ORNITH_TEST_STATE="128"))
    cpu_tokens, cpu = _continuation_logprobs(runner_bin, model, "--gpu", "off", "-b", "1")
    for b in ("8", "64"):
        tokens, dev = _continuation_logprobs(runner_bin, model, "-b", b)
        assert tokens == cpu_tokens
        # the tiled gate and up projections reassociate their sums: 2.4e-4 here
        assert max(abs(x - y) for x, y in zip(cpu, dev)) < 5e-3, (b, cpu, dev)
