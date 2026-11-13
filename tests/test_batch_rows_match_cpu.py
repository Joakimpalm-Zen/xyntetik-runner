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
import math
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


def _embed(exe, model, extra, env=None, log_path=None):
    with RunnerServer(exe, model, ctx=256, parallel=1, extra_args=extra,
                      env=env, log_path=log_path) as srv:
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


def _worst(cpu, dev):
    # a NaN or an infinity is as far off as it gets
    return max(abs(a - b) if math.isfinite(a - b) else math.inf
               for u, v in zip(cpu, dev) for a, b in zip(u, v))


def test_the_half_staged_expert_tile_without_its_scale_buffer_leaves_the_cpus_rows(
        exe, tmp_path):
    """RUNNER_METAL_MOE_MM=half: the grouped expert tile stages each activation
    row scaled by its max|x|, which a pass writes into the tiled GEMMs' scale
    buffer first. When that buffer could not be made the half-staged tile ran
    anyway and read its scales from the dummy buffer (R6.68.113). The choice
    of the half-staged tile and of its scale buffer is now one decision: a
    tensor whose scale buffer cannot be made takes the f32-staged tile, and
    the log says so once. RUNNER_METAL_INIT_INJECT_FAILURE=const:xsc fails the
    buffer; the dense tile already fell back to the matvec on that failure."""
    caps = json.loads(subprocess.run([exe, "--caps"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, timeout=60).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Apple GPU's half-staged expert tile")
    prefix = tmp_path / "f"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make-test-moe.py"), str(prefix)],
                   check=True, stdout=subprocess.DEVNULL, cwd=tmp_path)
    model = pathlib.Path(str(prefix) + ".gptoss-mxfp4.gguf")   # MXFP4 experts: a half twin
    cpu = _embed(exe, model, ["--gpu", "off"])
    half = dict(os.environ, RUNNER_METAL_MOE_MM="half")
    for name, env in (("half", half),
                      ("no-scales", dict(half, RUNNER_METAL_INIT_INJECT_FAILURE="const:xsc"))):
        log = tmp_path / f"{name}.log"
        dev = _embed(exe, model, [], env=env, log_path=log)
        text = log.read_text(errors="replace")
        assert "Metal backend" in text, (name, text[-2000:])
        worst = _worst(cpu, dev)
        assert worst < 1e-2, (name, worst)
        if name == "half":
            # the control reaches the tile this test is about
            assert "grouped-MMA prefill kernels on (half-staged" in text, text[-2000:]
        else:
            # said once, and the experts still ran a tile: the f32-staged one
            assert text.count("scale buffer could not be made") == 1, text[-2000:]
            assert "grouped-MMA prefill kernels on (f32-staged" in text, text[-2000:]
