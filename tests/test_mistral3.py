"""mistral3 (R4.4): Ministral 3, Devstral Small 2 and Magistral Small 1.2.

The architecture is llama with two additions, both read from the converter's
keys (llama.cpp conversion/mistral3.py):

- the llama 4 attention temperature (transformers' llama_4_scaling_beta,
  attention.temperature_scale) on EVERY layer's Q after its rope, at
  1 + beta * ln(1 + floor(pos / original context)); llama 4 applies the same
  factor only on the layers that skip rope;
- YaRN whose magnitude factor comes from transformers' mscale_all_dim
  (rope.scaling.yarn_log_multiplier): get_mscale(f, 1) / get_mscale(f, m),
  which is 1 for Ministral 3, not the default 1 + 0.1 ln f.

Neither is visible in a short toy's text. The gates read the post-rope Q of
layer 0 (RUNNER_DEBUG_ACT), whose input is the same embedding in every
fixture, so the ratio of two fixtures' Q sums is exactly the factor that
differs between them.
"""
import json
import math
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make(tmp, name, *extra):
    out = tmp / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return out


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mistral3")
    return {
        # BETA,ORIG,FACTOR,LOGMUL; a 2-position original context makes the
        # temperature live from position 2
        "temp": make(tmp, "temp", "--arch", "mistral3", "--mistral3-rope", "0.5,2,16,1"),
        "notemp": make(tmp, "notemp", "--arch", "mistral3", "--mistral3-rope", "0,2,16,1"),
        "nolog": make(tmp, "nolog", "--arch", "mistral3", "--mistral3-rope", "0,2,16,0"),
    }


def q_layer0(runner_bin, model, extra_env=None):
    """(last position, layer 0's post-rope Q sum) for the prompt's last row."""
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", "hello world", "-n", "1",
                        "--temp", "0", "--gpu", "off"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
                       env={**os.environ, "RUNNER_DEBUG_ACT": "1", **(extra_env or {})})
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err[-600:]
    head = re.search(r"^ACT ==== forward n=(\d+) pos=(\d+) arch=mistral3", err, re.M)
    q0 = re.search(r"^ACT L0\s+q-post-rope\s+.*?sum=([+-][0-9.]+)", err, re.M)
    assert head and q0, err[-600:]
    return int(head.group(1)) + int(head.group(2)) - 1, float(q0.group(1))


def test_the_temperature_scales_every_layers_q_after_rope(runner_bin, fx):
    pos, with_t = q_layer0(runner_bin, fx["temp"])
    pos2, without = q_layer0(runner_bin, fx["notemp"])
    assert pos == pos2 and pos >= 2
    want = 1.0 + 0.5 * math.log(1.0 + math.floor(pos / 2))
    # layer 0 ropes (mistral3 has no NoPE layers): under llama 4's rule the
    # factor would not apply here at all and the ratio would be 1
    assert with_t / without == pytest.approx(want, rel=2e-5)


def test_yarn_log_multiplier_sets_the_magnitude_factor(runner_bin, fx):
    _, log1 = q_layer0(runner_bin, fx["notemp"])
    _, nolog = q_layer0(runner_bin, fx["nolog"])
    f = 16.0
    # with mscale_all_dim 1 the factor is 1; without the key, 1 + 0.1 ln f
    assert log1 / nolog == pytest.approx(1.0 / (1.0 + 0.1 * math.log(f)), rel=2e-5)


def test_a_temperature_without_its_original_context_is_refused(runner_bin, tmp_path):
    bad = make(tmp_path, "bad", "--arch", "mistral3", "--mistral3-rope", "0.5,0,16,1")
    p = subprocess.run([str(runner_bin), "-m", str(bad), "-p", "hi", "-n", "1", "--gpu", "off"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert "original_context_length" in p.stderr.decode(errors="replace")


def test_metal_matches_cpu_with_the_temperature_live(runner_bin, fx):
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, timeout=120).stdout.decode())
    if ((caps.get("gpu") or {}).get("backend")) != "metal":
        pytest.skip("no Metal backend on this machine")
    outs = {}
    for gpu in ("off", "auto"):
        p = subprocess.run([str(runner_bin), "-m", str(fx["temp"]), "-p", "hello world", "-n", "12",
                            "--temp", "0", "--gpu", gpu], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        assert p.returncode == 0
        outs[gpu] = (p.stdout, p.stderr.decode(errors="replace"))
    assert "Metal backend" in outs["auto"][1], outs["auto"][1][-400:]
    assert outs["off"][0] == outs["auto"][0], "Metal output differs from CPU"
