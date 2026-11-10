"""The shared prefix cache's default budget is sized from the served model (R4.26.13).

The budget was a fixed 512 MB (RUNNER_PREFIX_CACHE_MB), sized from the traffic. An entry over half of it is not stored
whole: a recurrent model stores nothing, a dense one a truncated prefix. On Qwen3.5-4B no prompt past 6,048 tokens was
cached. With RUNNER_PREFIX_CACHE_MB unset, each model a server loads now raises the budget to hold one of its own
full-context entries under that half-budget rule. The budget is never below the 512 MB floor and never above an eighth
of the RAM the process may use.

Anchors outside the runner: the fixture's declared geometry for the entry's bytes (2 layers, K and V, 4 KV heads 512
wide, f16: 16 KiB a token, no recurrent state), and the RAM figure `--caps` reports (the effective limit
plat_ram_bytes reads).
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
from harness import RunnerServer  # noqa: E402

MiB = 1 << 20
FLOOR = 512 * MiB
PER_TOK = 2 * 2 * 4 * 512 * 2   # layers x (K, V) x KV heads x head width x f16
CTX = 24576                     # a 384 MiB entry: twice it is over the floor


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("pfxbudget")
    wide, small = d / "wide.gguf", d / "small.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(wide), "--layers", "2",
                    "--head-dim", "512", "--heads", "4:4"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(small)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return {"wide": wide, "small": small}


def _ram(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=60)
    return json.loads(caps.stdout.decode())["ram_bytes"]


def _budget(runner_bin, model, ctx, log, env_mb=None):
    env = {k: v for k, v in os.environ.items() if k != "RUNNER_PREFIX_CACHE_MB"}
    if env_mb is not None:
        env["RUNNER_PREFIX_CACHE_MB"] = str(env_mb)
    with RunnerServer(runner_bin, model, ctx=ctx, env=env, log_path=log,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/runner/prefix-cache", timeout=60) as r:
            stats = json.load(r)
    return stats["budget_bytes"], pathlib.Path(log).read_text(errors="replace")


def test_a_model_whose_full_context_entry_outgrows_half_the_floor_raises_the_budget(runner_bin, fx, tmp_path):
    want = max(FLOOR, min(2 * PER_TOK * CTX, _ram(runner_bin) // 8))
    budget, log = _budget(runner_bin, fx["wide"], CTX, tmp_path / "srv.log")
    assert budget == want, (budget, want)
    assert f"prefix cache: budget {want >> 20} MB" in log, log[-1500:]


def test_a_small_model_keeps_the_floor(runner_bin, fx, tmp_path):
    budget, log = _budget(runner_bin, fx["small"], 512, tmp_path / "srv.log")
    assert budget == FLOOR
    assert "prefix cache: budget" not in log, "the floor needs no line"


def test_a_set_budget_is_used_as_given(runner_bin, fx, tmp_path):
    budget, log = _budget(runner_bin, fx["wide"], CTX, tmp_path / "srv.log", env_mb=64)
    assert budget == 64 * MiB
    assert "prefix cache: budget" not in log
