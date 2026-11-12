"""Every column of a prompt batch on the device is held to the CPU's.

`--score` feeds one token at a time, and a generated continuation reads only
a batch's last column, so a device kernel that runs only for a batch of
several columns had no gate on what it leaves in the other columns. Mean
pooled embeddings read every column's final row, and on a fully offloaded
model they come from the device's batch (R6.68.107), so a server on the
device against a server with `--gpu off` is that gate, on any backend.

It caught the first defect it was pointed at: the Apple GPU's grouped expert
tile on an expert layer 16 wide (R6.68.109), 7e-2 off on the 512-wide
recurrent MoE fixture. The bound here is for gross errors; the tolerance
gates hold the tiles' reassociated sums to their own, tighter bars.

Skipped where there is no GPU backend.
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

TEXTS = ["the quick brown fox jumps over the lazy dog and keeps running past the mill",
         "hello",
         "two bridges cross the river below the mill and the road ends at the second one"]

# (generator, arguments after the output name, environment, files it writes)
FAMILIES = {
    "dense":            ("make-test-model.py", [], {}, None),
    "eseries":          ("make-test-model.py", ["--eseries", "3,16"], {}, None),
    "recurrent":        ("make-test-ornith.py", [], {}, None),
    "recurrent-wide":   ("make-test-ornith.py", [], {"ORNITH_TEST_STATE": "128"}, None),
    "recurrent-moe":    ("make-test-ornith.py", [], {"ORNITH_TEST_MOE": "1"}, None),
    "recurrent-moe-wide": ("make-test-ornith.py", [],
                           {"ORNITH_TEST_MOE": "1", "ORNITH_TEST_STATE": "128"}, None),
    "mamba2":           ("make-test-hybrid.py", [], {}, ".gguf"),
    "mamba2-dense":     ("make-test-hybrid.py", ["--dense", "--rope"], {}, ".gguf"),
    "nemotron":         ("make-test-nemotron.py", [], {}, ".gguf"),
    "moe":              ("make-test-moe.py", [], {},
                         (".moe1.gguf", ".moe3.gguf", ".shexpg.gguf", ".gemma4-moe.gguf",
                          ".gemma4-moe-hetero.gguf", ".gptoss-mxfp4.gguf")),
}


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    caps = json.loads(subprocess.run([e, "--caps"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, timeout=60).stdout)
    if not caps.get("gpu"):
        pytest.skip("no GPU backend: this test holds the device's batch to the CPU")
    return e


def _models(tmp_path, family):
    script, args, env, writes = FAMILIES[family]
    out = tmp_path / ("f" if writes else "f.gguf")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), str(out), *args],
                   check=True, stdout=subprocess.DEVNULL, cwd=tmp_path,
                   env=dict(os.environ, **env))
    if writes is None:
        return [out]
    names = (writes,) if isinstance(writes, str) else writes
    return [pathlib.Path(str(out) + n) for n in names]


def _embed(exe, model, extra):
    with RunnerServer(exe, model, ctx=256, parallel=1, extra_args=extra) as srv:
        req = urllib.request.Request(
            srv.base_url + "/v1/embeddings", data=json.dumps({"input": TEXTS}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return [d["embedding"] for d in json.load(r)["data"]]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_prompt_batch_on_the_device_leaves_the_cpus_rows(exe, tmp_path, family):
    for model in _models(tmp_path, family):
        assert model.exists(), model
        cpu = _embed(exe, model, ["--gpu", "off"])
        dev = _embed(exe, model, [])
        worst = max(abs(a - b) for u, v in zip(cpu, dev) for a, b in zip(u, v))
        # measured on one Apple GPU: at most 1.2e-3 (a half-staged tile's sums)
        assert worst < 1e-2, (model.name, worst)
