"""One pool dispatch for Q/K/V and for gate/up is byte-identical (R3.1.6).

forward_layer runs the projections that read the same normed input (Q, the
afmoe Q gate, K and V; gate and up) as ONE thread-pool dispatch over the union
of their rows, where each used to be its own dispatch and barrier. Every row
still runs exactly the job it ran alone, so the logits must not move by a bit.
RUNNER_MV_FUSE=0 restores the separate dispatches; this holds the two against
each other on the batched path (--score, a teacher-forced prefill) and on the
single-token decode path, with the int8 activation route on and off (it runs
on AVX2 hosts, where the fused dispatch quantizes the activations once for all
parts), over fixtures that take each branch: absent V, shared-KV layers,
gemma3's layout and a Q8_0 file.
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

TEXT = ("The quick brown fox jumps over the lazy dog, and then the dog "
        "wakes up and chases the fox across the field until sunset.")

FIXTURES = {
    "plain": [],
    "q8_0": ["--quant", "q8_0"],
    "drop_v": ["--arch", "gemma4", "--drop-v", "1,3"],
    "eseries": ["--eseries", "3,16"],
    "gemma3": ["--gemma3"],
}


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def model(request, tmp_path_factory):
    m = tmp_path_factory.mktemp("fuse") / f"{request.param}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    *FIXTURES[request.param], str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _exe():
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    return exe


def _env(fuse, i8):
    env = dict(os.environ, RUNNER_CPU_I8="1" if i8 else "0")
    if fuse:
        env.pop("RUNNER_MV_FUSE", None)
    else:
        env["RUNNER_MV_FUSE"] = "0"
    return env


@pytest.mark.parametrize("i8", [False, True])
def test_the_batched_path_is_byte_identical(model, i8):
    outs = []
    for fuse in (False, True):
        p = subprocess.run([_exe(), "-m", str(model), "--score", "-p", TEXT,
                            "-t", "4", "--gpu", "off"],
                           capture_output=True, text=True, timeout=300,
                           env=_env(fuse, i8))
        assert p.returncode == 0, p.stderr
        outs.append(p.stdout)
    assert outs[0] == outs[1]


def _decode(model, fuse, i8):
    env = _env(fuse, i8)
    with RunnerServer(_exe(), model, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "4"], env=env) as srv:
        req = urllib.request.Request(
            srv.base_url + "/v1/completions",
            data=json.dumps({"prompt": TEXT, "max_tokens": 16, "temperature": 0,
                             "logprobs": 5}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            c = json.load(r)["choices"][0]
    return json.dumps([c["text"], c["logprobs"]], sort_keys=True)


@pytest.mark.parametrize("i8", [False, True])
def test_the_decode_path_is_byte_identical(model, i8):
    assert _decode(model, False, i8) == _decode(model, True, i8)
