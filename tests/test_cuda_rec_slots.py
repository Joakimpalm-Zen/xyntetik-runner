"""A hybrid's recurrent state on the device has a slot per stateful layer only (R3.47).

Every hybrid family kept its recurrent state in per-layer buffers sized for every layer, attention and MLP-only layers
included, on the host, on CUDA and on Metal: qwen35's Gated DeltaNet fold, Mamba-2's conv window and SSM state, and
lfm2's conv ring. The state now has one slot per layer that holds it (model_rec_slot). RUNNER_KV_RECURRENT_ROWS=1
keeps 1.15.0's slot for every layer.

A size or an offset still taken per layer overruns the compact buffers. The first one missed (the Mamba-2 conv window
in cuda_fwd_prepare) passed generation, inside the allocator's rounding, and failed teacher-forced scoring at its
first forward, which then fell back to the CPU. So the checks score on the device:
- the route must stay on CUDA under the compact layout;
- the per-token scores must equal the 1.15.0 layout's, since no layer ever reads another's slot.

The checks need a CUDA device; elsewhere the module skips.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
KNOB = "RUNNER_KV_RECURRENT_ROWS"
TEXT = "the quick brown fox jumps over the lazy dog, and the dog sleeps on in the sun all day long"

# (id, generator, its arguments, its environment)
FAMILIES = [
    ("qwen35", "make-test-ornith.py", [], {}),
    ("qwen35-8", "make-test-ornith.py", [], {"ORNITH_TEST_LAYERS": "8"}),
    ("qwen35moe", "make-test-ornith.py", [], {"ORNITH_TEST_MOE": "1"}),
    ("clef", "make-test-clef.py", [], {}),
    ("qwen4exp", "make-test-qwen4exp.py", [], {}),
    ("granitehybrid", "make-test-hybrid.py", ["--dense"], {}),
    ("nemotron_h", "make-test-nemotron.py", [], {}),
]


def _cuda(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=30)
    return (json.loads(caps.stdout.decode()).get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def models(runner_bin, tmp_path_factory):
    if not _cuda(runner_bin):
        pytest.skip("needs a CUDA device")
    d = tmp_path_factory.mktemp("recslots")
    out = {}
    for name, gen, args, env in FAMILIES:
        stem = d / name
        target = stem if gen == "make-test-hybrid.py" else stem.with_suffix(".gguf")
        subprocess.run([sys.executable, ROOT / "scripts" / gen, str(target), *args], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL, env={**os.environ, **env})
        path = stem.with_suffix(".gguf")
        if not path.exists():   # make-test-nemotron.py writes <arg>.gguf
            path = pathlib.Path(str(target) + ".gguf")
        out[name] = path
    return out


def _score(runner_bin, model, prior):
    env = {k: v for k, v in os.environ.items() if k != KNOB}
    if prior:
        env[KNOB] = "1"
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", TEXT, "--score", "2", "--gpu", "99", "-t", "2",
                        "--no-tray"], cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=300,
                       env=env)
    assert p.returncode == 0, p.stderr[-1500:]
    return json.loads(p.stdout), p.stderr


@pytest.mark.parametrize("name", [f[0] for f in FAMILIES])
def test_scoring_stays_on_the_device_and_matches_the_full_layout(runner_bin, models, name):
    now, err = _score(runner_bin, models[name], False)
    assert "forward failed at runtime" not in err, (name, err[-1500:])
    assert '"backend":"cuda"' in err, (name, err[-1500:])
    full, _ = _score(runner_bin, models[name], True)
    assert now["logprobs"] == full["logprobs"], name
