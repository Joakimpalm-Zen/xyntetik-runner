"""RUNNER_CPU_I8_PREFILL: CPU batches of 8 or more rows in int8 (R3.1.5 B).

The int8 prefill tiles quantize the activations, so the route is
tolerance-class and opt-in. What this holds, on a q8_0 fixture with the GPU
off:

- the route line names the route (cpu_act f32 off, f32+int8-prefill on; a
  build without the arm64 kernel ignores the flag and the route tests skip);
- under the flag a chunked score (16-row batches) differs from the f32 route,
  so the tiles ran, and stays close to it with no top-1 change;
- a solo score (one row a forward, the decode arithmetic) is byte-identical
  with and without the flag: the flag touches batches only.

The kernel's arithmetic is held in test_quants_simd (the NEON kernel equals
its scalar twin bit for bit, and both the same-quantization dot).
"""
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEXT = ("The quick brown fox jumps over the lazy dog while the five boxing wizards "
        "jump quickly, and a sphinx of black quartz judges my vow.")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    f = tmp_path_factory.mktemp("i8p") / "wide-q8.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide",
                    "--quant", "q8_0", str(f)], check=True, cwd=ROOT, capture_output=True)
    return f


def _score(runner_bin, model, i8p, chunked):
    env = {k: v for k, v in os.environ.items()
           if k not in ("RUNNER_CPU_I8", "RUNNER_CPU_I8_PREFILL", "RUNNER_SCORE_CHUNKED")}
    if i8p:
        env["RUNNER_CPU_I8_PREFILL"] = "1"
    if chunked:
        env["RUNNER_SCORE_CHUNKED"] = "1"
    r = subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "off", "-t", "2",
                        "--score", "--score-top", "8", "-p", TEXT, "-c", "256"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    m = re.search(r"score: route (\{.*\})", r.stderr)
    assert m, r.stderr[-1500:]
    return json.loads(r.stdout), json.loads(m.group(1))


@pytest.fixture(scope="module")
def has_kernel(runner_bin, model):
    # the flag is ignored, with a notice, on a build without the NEON kernel
    _, on = _score(runner_bin, model, True, True)
    has = on["cpu_act"] == "f32+int8-prefill"
    # every Apple Silicon core has the dot-product extension: a skip there
    # would hide a route that stopped engaging
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert has, on["cpu_act"]
    return has


def test_the_route_line_names_it(runner_bin, model, has_kernel):
    _, off = _score(runner_bin, model, False, True)
    assert off["cpu_act"] == "f32"
    if not has_kernel:
        pytest.skip("no int8 prefill kernel on this build")


def test_chunked_rows_take_the_tiles_and_stay_close(runner_bin, model, has_kernel):
    if not has_kernel:
        pytest.skip("no int8 prefill kernel on this build")
    f32, _ = _score(runner_bin, model, False, True)
    i8, route = _score(runner_bin, model, True, True)
    assert route["paths"]["keep"]["rows_max"] >= 8
    assert i8["logprobs"] != f32["logprobs"], "the int8 tiles never ran"
    assert i8["argmax"] == f32["argmax"]
    worst = max(abs(a - b) for a, b in zip(i8["logprobs"], f32["logprobs"]))
    assert worst < 0.05, worst


def test_solo_decode_is_untouched(runner_bin, model):
    off, _ = _score(runner_bin, model, False, False)
    on, _ = _score(runner_bin, model, True, False)
    assert on["logprobs"] == off["logprobs"]
    assert on["top"] == off["top"]
