"""The CUDA placement plan costs the weights as the upload allocates them (R4.26.18).

Under --cpu-moe the upload joins the device tensors into runs and binds one allocation a run (shared_plan_upload).
The driver rounds each allocation to its 2 MiB granularity. The plan used to cost every tensor as its own rounded
allocation and each expert bank with a nine-granule allowance. On Qwen3-30B-A3B Q6_K that over-planned the 24 GB
slice's weights by about 1.07 GB, two banks and most of the prompt window. The plan now re-costs the weights by runs
after the greedy placement and keeps placing banks while that exact figure fits.

The invariant, read from RUNNER_DEBUG_VRAM's per-category lines on MoE fixtures under --cpu-moe auto:
- The plan's weights by runs equal what the upload's bindings round to, up to the head's own granule (the head joins a
  run or binds alone).
- The greedy estimate is never below the runs' figure: it is the conservative bound the refinement starts from.

The checks need a CUDA device; elsewhere the module skips.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAN = 2 << 20


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def fixtures(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    tmp = tmp_path_factory.mktemp("planruns")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(tmp / "fx")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return tmp


@pytest.mark.parametrize("name", ["moe4", "gptoss-mxfp4", "shexp"])
def test_the_plan_costs_the_weights_as_the_upload_allocates_them(runner_bin, fixtures, name):
    model = fixtures / f"fx.{name}.gguf"
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", "Hello", "-n", "2", "--temp", "0", "--gpu", "99",
                        "--cpu-moe", "auto", "--no-tray", "-t", "2"], cwd=ROOT, capture_output=True, text=True,
                       errors="replace", timeout=300, env={**os.environ, "RUNNER_DEBUG_VRAM": "1"})
    assert p.returncode == 0, p.stderr[-1500:]
    plan = re.search(r"vram-plan: .*weights ([0-9.]+) GB greedy, ([0-9.]+) GB by (\d+) runs, banks (\d+).*"
                     r"\(bytes: weights (\d+) by runs, head (\d+)\)", p.stderr)
    up = re.search(r"vram: (\d+) bindings, .*\(bytes: requested (\d+), rounded (\d+)\)", p.stderr)
    assert plan and up, p.stderr[-2000:]
    greedy_gb, runs_gb = float(plan.group(1)), float(plan.group(2))
    by_runs, head = int(plan.group(5)), int(plan.group(6))
    rounded = int(up.group(3))
    head_gran = (head + GRAN - 1) // GRAN * GRAN
    assert rounded - by_runs in (0, head_gran), (name, by_runs, head, rounded)
    assert greedy_gb >= runs_gb, (name, greedy_gb, runs_gb)
