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
def gemma_moe_fixture(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("metalmoebatch") / "f"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(prefix)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return pathlib.Path(f"{prefix}.gemma4-moe.gguf")


def _run(runner, model, gpu, *, stats=False):
    env = dict(os.environ, RUNNER_METAL_MM="0", RUNNER_METAL_ATTN_COOP="0")
    if stats:
        env["RUNNER_METAL_STATS"] = "1"
    return subprocess.run(
        [runner, "-m", model, "-p", PROMPT, "-n", "1", "-b", "8",
         "--temp", "0", "--gpu", gpu],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300, check=True)


def test_prompt_tile_batches_moe_without_changing_tokens(runner_bin,
                                                          gemma_moe_fixture):
    cpu = _run(runner_bin, gemma_moe_fixture, "off")
    gpu = _run(runner_bin, gemma_moe_fixture, "auto", stats=True)
    assert cpu.stdout == gpu.stdout
    assert b"Metal backend" in gpu.stderr

    match = re.search(rb"metal-census n=8 .* moe=(\d+)", gpu.stderr)
    assert match, gpu.stderr.decode(errors="replace")
    # First tracer slice: one batched route, while the four expert operations
    # remain serial per token: 2 layers * (1 + 4 * 8).
    assert int(match.group(1)) == 66
