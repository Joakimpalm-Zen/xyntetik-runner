"""--score: teacher-forced logprob/NLL scoring (adaptation D1).

The eval/reward primitive: one pass over a fixed token sequence, per-position
log P(token | prefix) under the model, plus NLL/perplexity summary — no
sampling, no gradients. The gates below are the mode's whole contract:

  * the DEFAULT is the solo path — the exact numerics the sampler sees at
    decode time. Measured 2026-08-21: the CPU batched forward is NOT
    bit-identical to solo (max |dlp| ~1e-6, diverging from batch row 2), so
    chunked scoring would quietly reintroduce train/infer mismatch. Chunked
    (RUNNER_SCORE_CHUNKED=1) is the opt-in speed lever; the gate below PINS
    its measured envelope instead of pretending identity.
  * deterministic: same input -> byte-identical output, run to run.
  * the recurrent (DeltaNet hybrid) fixture holds every property too — the
    fold must advance equivalently under chunked and solo scoring.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("score")
    dense = d / "dense.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(dense)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    ornith = d / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py",
                    str(ornith)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return {"dense": dense, "recurrent": ornith}


PROMPT = "the quick brown fox jumps over the lazy dog and keeps on running"


def _score(runner_bin, model, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        [runner_bin, "-m", str(model), "--score", "-p", PROMPT,
         "-t", "2", "--gpu", "off"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_score_shape_and_invariants(runner_bin, models, kind):
    out = json.loads(_score(runner_bin, models[kind]))
    assert out["schema_version"] == "xyntetik.runner.score.v1"
    n = out["n_tokens"]
    assert n >= 2
    assert out["n_scored"] == n - 1
    assert len(out["tokens"]) == n
    assert len(out["logprobs"]) == n - 1
    assert all(lp <= 0.0 for lp in out["logprobs"])
    assert out["nll_total"] == pytest.approx(-sum(out["logprobs"]), rel=1e-6)
    assert out["nll_mean"] == pytest.approx(out["nll_total"] / out["n_scored"],
                                            rel=1e-6)
    assert out["ppl"] == pytest.approx(math.exp(out["nll_mean"]), rel=1e-6)


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_score_top1_is_bounded_by_probability_theory(runner_bin, models, kind):
    """The self-check anchor: --score reports absolute next-token top-1.

    A scorer can return a plausible-looking NLL with a correct `n_scored` and
    exit 0 while being catastrophically wrong (measured 2026-08-30: 161x off a
    reference engine on one model family). Every validity assertion passed,
    because they test that the arm RAN, not that the number is right.

    `top1` fixes that because it is bounded by facts that hold OUTSIDE this
    implementation, for any probability distribution whatsoever:

      * at most one token can carry more than half the mass, so a position
        whose scored token has p > 0.5 IS the argmax. Those positions are a
        lower bound on top1.
      * the largest probability in a distribution over V outcomes is at least
        1/V, so an argmax position must have p >= 1/n_vocab. Positions below
        that cannot be the argmax, giving an upper bound.

    Neither bound comes from the scorer, so a scorer disagreeing with its own
    logprobs fails here rather than reporting a confident wrong number.
    """
    out = json.loads(_score(runner_bin, models[kind]))
    n_vocab = out["n_vocab"]
    assert n_vocab > 1
    top1 = out["top1"]
    assert isinstance(top1, int)
    assert 0 <= top1 <= out["n_scored"]
    assert out["top1_rate"] == pytest.approx(top1 / out["n_scored"], rel=1e-6)

    probs = [math.exp(lp) for lp in out["logprobs"]]
    must_be_argmax = sum(1 for p in probs if p > 0.5)
    could_be_argmax = sum(1 for p in probs if p >= 1.0 / n_vocab)
    assert must_be_argmax <= top1 <= could_be_argmax, (
        f"top1={top1} outside [{must_be_argmax}, {could_be_argmax}]")


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_score_chunked_agrees_with_solo_on_top1(runner_bin, models, kind):
    """Both paths must agree on the discrete answer even where floats differ.

    The logprob envelope above tolerates ~1e-6 drift between the batched and
    solo forwards. An argmax is discrete: drift that small can only flip it at
    a near-exact tie, so a disagreement here means a real batching fault (wrong
    row, wrong position, stale fold) rather than reduction order.
    """
    solo = json.loads(_score(runner_bin, models[kind]))
    chunked = json.loads(_score(runner_bin, models[kind],
                                {"RUNNER_SCORE_CHUNKED": "1"}))
    assert chunked["n_vocab"] == solo["n_vocab"]
    assert chunked["top1"] == solo["top1"]


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_score_chunked_stays_inside_measured_envelope(runner_bin, models, kind):
    solo = json.loads(_score(runner_bin, models[kind]))
    chunked = json.loads(_score(runner_bin, models[kind],
                                {"RUNNER_SCORE_CHUNKED": "1"}))
    assert chunked["tokens"] == solo["tokens"]
    deltas = [abs(a - b) for a, b in
              zip(chunked["logprobs"], solo["logprobs"])]
    # the measured CPU batch-vs-solo reduction-order envelope, with headroom;
    # a real batching bug (wrong row, wrong position, stale fold) lands orders
    # of magnitude beyond this
    assert max(deltas) <= 2e-5, max(deltas)


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_score_chunked_is_deterministic(runner_bin, models, kind):
    env = {"RUNNER_SCORE_CHUNKED": "1"}
    assert (_score(runner_bin, models[kind], env)
            == _score(runner_bin, models[kind], env))


@pytest.mark.parametrize("kind", ["dense", "recurrent"])
def test_score_is_deterministic(runner_bin, models, kind):
    assert _score(runner_bin, models[kind]) == _score(runner_bin, models[kind])


def test_score_rejects_too_short(runner_bin, models):
    p = subprocess.run(
        [runner_bin, "-m", str(models["dense"]), "--score", "-p", "",
         "-t", "2", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert b"--score" in p.stderr
