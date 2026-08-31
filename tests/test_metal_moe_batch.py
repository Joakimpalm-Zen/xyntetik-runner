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
    exe = ROOT / "runner"
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
    env = dict(os.environ, RUNNER_METAL_MM="0", RUNNER_METAL_ATTN_COOP="0")
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
