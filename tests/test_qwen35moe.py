"""qwen35moe (Qwen3.5/3.6-35B-A3B and their derivatives): the Qwen 3.5 token
mixer with routed experts and a sigmoid-gated shared expert on every block.
The arithmetic is checked against transformers' float32 Qwen3_5MoeForCausalLM
on a converted random model (docs/compat-reports/qwen35moe-toy-2026-10-30);
these pin the binding on the fixture."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("q35moe") / "q35moe.gguf"
    env = dict(os.environ, ORNITH_TEST_MOE="1")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(out)],
                   check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
    return out


def _run(runner_bin, model, *extra):
    return subprocess.run([runner_bin, "-m", str(model), *extra, "--gpu", "off", "-t", "2"],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def test_every_block_routes_its_experts(runner_bin, model, tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("hello world, the quick brown fox jumps over the lazy dog")
    got = _run(runner_bin, model, "--score", "-f", str(corpus))
    assert got.returncode == 0, got.stderr.decode(errors="replace")
    route = next(l for l in got.stderr.decode().splitlines() if l.startswith("score: route "))
    r = json.loads(route[len("score: route "):])
    assert r["layers"] == 4 and r["moe_layers"] == 4, r


def test_decodes_deterministically(runner_bin, model):
    a = _run(runner_bin, model, "-p", "hello there", "-n", "24", "--temp", "0")
    b = _run(runner_bin, model, "-p", "hello there", "-n", "24", "--temp", "0")
    assert a.returncode == 0, a.stderr.decode(errors="replace")
    assert a.stdout == b.stdout and len(a.stdout) > len(b"hello there")
    assert b"qwen35moe" in a.stderr
