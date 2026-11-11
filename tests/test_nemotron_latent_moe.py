"""LatentMoE on nemotron_h_moe (Nemotron 3.5 Super and Bolt, R4.27.4.4).

The routed experts of a LatentMoE block run at moe_latent_size, between
ffn_latent_down (n_embd -> latent) and ffn_latent_up (latent -> n_embd). The
router and the shared expert read the full-width normed input, as llama.cpp
36a0e2f's nemotron-h-moe graph does.

The anchor is linearity, not Runner: scripts/make-test-hybrid.py --latent
writes the latent model and a composed twin with no latent pair, whose experts
are the latent experts multiplied through it (up . down_proj, up_proj . down),
every other tensor the same bytes. The two compute the same function, so their
per-position logprobs agree to f32 rounding on the one-token pass (--score's
default for this model) and on a batched pass (RUNNER_SCORE_CHUNKED=1, 16-row
batches through the experts grouped by expert). A latent pair applied in the wrong
place, in the wrong order or transposed shows as a disagreement. Then the
loader's refusals: a half pair, a pair the metadata does not declare, a latent
width past n_embd, and a projection shorter than the width.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "The capital of France is Paris, and of Italy"
LATENT = 16                      # the fixture's n_embd is 32
# f32 rounding of a 16-term regrouped sum, in logprob units. Measured on gcc
# 13.3 builds: 4.8e-7 to 9.6e-7 against the twin. Other compilers vectorize
# the sums differently, so the bound keeps a wide margin; the zeroed-experts
# control reads 0.33.
TOL = 1e-4
# Batched against the one-token pass: a rounding floor that is the host's
# (9.6e-7 on gcc 13.3 x86-64 Linux; tests/test_nemotron_hetero_moe.py's same
# comparison read about 1e-4 on the hosted Windows runner, R6.66). The bound is
# that floor measured in the same run on the composed twin, times
# FLOOR_FACTOR, plus FLOOR_ABS; on Linux the two are equal to the last bit.
FLOOR_FACTOR = 2
FLOOR_ABS = 1e-6


def make(d, name, brk=None):
    env = dict(os.environ, HYBRID_TEST_AMP="0.6")
    if brk:
        env["HYBRID_TEST_LATENT_BREAK"] = brk
    pfx = d / name
    subprocess.run([sys.executable, ROOT / "scripts/make-test-hybrid.py", str(pfx),
                    "--arch", "nemotron_h_moe", "--latent", str(LATENT)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, env=env)
    return pfx.with_suffix(".gguf"), d / (name + ".composed.gguf")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return make(tmp_path_factory.mktemp("latent"), "lat")


def score(runner_bin, model, chunked=False):
    env = dict(os.environ, RUNNER_SCORE_CHUNKED="1" if chunked else "0")
    p = subprocess.run([str(runner_bin), "-m", str(model), "-p", PROMPT, "--score", "8",
                        "--gpu", "off", "--no-tray"], env=env,
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout), p.stderr.decode(errors="replace")


def max_diff(a, b):
    assert len(a["logprobs"]) == len(b["logprobs"]) > 30
    return max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))


@pytest.mark.parametrize("chunked", [False, True], ids=["token-pass", "batched-pass"])
def test_latent_model_equals_its_composed_twin(runner_bin, pair, chunked):
    lat, comp = pair
    a, err = score(runner_bin, lat, chunked)
    b, _ = score(runner_bin, comp, chunked)
    assert "LatentMoE, the routed experts run at width 16 (n_embd 32)" in err
    d = max_diff(a, b)
    assert d <= TOL, f"latent against its composed twin: max |dlogprob| {d:.3e} > {TOL:.0e}"
    assert a["argmax"] == b["argmax"]


def test_the_batched_pass_equals_the_token_pass(runner_bin, pair):
    lat, comp = pair
    got = max_diff(score(runner_bin, lat)[0], score(runner_bin, lat, True)[0])
    floor = max_diff(score(runner_bin, comp)[0], score(runner_bin, comp, True)[0])
    bound = FLOOR_FACTOR * floor + FLOOR_ABS
    assert got <= bound, (f"latent token against batched: max |dlogprob| {got:.3e}; the composed twin's "
                          f"floor {floor:.3e}; bound {bound:.3e}")


def test_the_twin_is_a_test_the_experts_move_the_logits(runner_bin, pair, tmp_path):
    """The comparison above means something only if the routed experts reach
    the logits: the same latent model with its experts zeroed must read far
    outside the tolerance."""
    lat, _ = pair
    zero, _ = make(tmp_path, "zero", "zero-experts")
    a, _ = score(runner_bin, lat)
    z, _ = score(runner_bin, zero)
    d = max_diff(a, z)
    assert d > 1000 * TOL, f"zeroed experts: max |dlogprob| {d:.3e}, not past {1000 * TOL:.0e}"


@pytest.mark.parametrize("brk,needle", [
    ("no-up", "has an ffn_latent_down and no ffn_latent_up, but moe_latent_size is 16"),
    ("no-key", "has an ffn_latent_down and an ffn_latent_up, but moe_latent_size is 0"),
    ("wide", "moe_latent_size 33 is wider than n_embd 32"),
    ("short-down", "tensor ffn_latent_down in blk.2 has shape [32,15]"),
])
def test_a_broken_latent_layout_is_refused_by_name(runner_bin, tmp_path, brk, needle):
    lat, _ = make(tmp_path, "brk", brk)
    p = subprocess.run([str(runner_bin), "-m", str(lat), "-p", "hi", "-n", "1",
                        "--gpu", "off", "--no-tray"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    err = p.stderr.decode(errors="replace")
    assert p.returncode != 0, err
    assert needle in err, err


def test_a_latent_file_on_cuda_takes_no_split_and_names_the_refusal(runner_bin, pair):
    """R3.43a.2: nemotron_h_moe takes the --cpu-moe split by default on CUDA,
    but CUDA refuses LatentMoE whatever the placement, so a latent file takes
    no split and says only the refusal."""
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the default placement is CUDA-side")
    lat, _ = pair
    p = subprocess.run([str(runner_bin), "-m", str(lat), "-p", "hi", "-n", "1", "--no-tray"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    assert "LatentMoE (experts at width 16" in err, err
    assert "its experts and their routing stay on the host" not in err, err
