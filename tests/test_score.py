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


@pytest.mark.parametrize("chunked", [False, True])
def test_score_top_reports_the_distribution_head(runner_bin, models, chunked):
    """--score-top N: each position's N most likely tokens, best first.

    Anchored in probability, not in this implementation: the first entry is
    the position's reported argmax, the entries are sorted, their
    probabilities sum to at most 1, and the scored token is listed with its
    reported logprob whenever that logprob beats the Nth entry."""
    env = dict(os.environ)
    if chunked:
        env["RUNNER_SCORE_CHUNKED"] = "1"
    p = subprocess.run(
        [runner_bin, "-m", str(models["dense"]), "--score", "--score-top", "5",
         "-p", PROMPT, "-t", "2", "--gpu", "off"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    out = json.loads(p.stdout)
    assert len(out["top"]) == out["n_scored"]
    for i, row in enumerate(out["top"]):
        assert len(row) == 5
        ids = [e[0] for e in row]
        lps = [e[1] for e in row]
        assert len(set(ids)) == 5
        assert all(0 <= t < out["n_vocab"] for t in ids)
        assert ids[0] == out["argmax"][i]
        assert all(lps[j] >= lps[j + 1] for j in range(4))
        assert sum(math.exp(x) for x in lps) <= 1.0 + 1e-5
        tok, lp = out["tokens"][i + 1], out["logprobs"][i]
        if lp > lps[-1]:
            assert tok in ids
        if tok in ids:
            assert lps[ids.index(tok)] == pytest.approx(lp, abs=1e-6)


@pytest.mark.parametrize("chunked", [False, True])
def test_score_reports_its_route(runner_bin, models, chunked):
    """The paths and kernels a score ran, on stderr (the record stays
    byte-comparable across routes): a gate that compares two scores proves a
    route switch only when the switch's path and kernels are shown to run."""
    env = dict(os.environ)
    if chunked:
        env["RUNNER_SCORE_CHUNKED"] = "1"
    p = subprocess.run(
        [runner_bin, "-m", str(models["dense"]), "--score", "-p", PROMPT,
         "-t", "2", "--gpu", "off"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    n_scored = json.loads(p.stdout)["n_scored"]
    lines = [ln for ln in p.stderr.decode().splitlines()
             if ln.startswith("score: route ")]
    assert len(lines) == 1
    route = json.loads(lines[0][len("score: route "):])
    assert route["backend"] == "cpu"
    assert "kernels" in route
    paths = route["paths"]
    if chunked:
        assert paths["keep"]["forwards"] >= 1
        assert paths["keep"]["rows_max"] > 1
    else:
        assert set(paths) == {"solo"}
        assert paths["solo"]["forwards"] == n_scored


def test_spec_batch_widens_the_score_chunks(runner_bin, models):
    """RUNNER_SPEC_BATCH sets the rows a chunked score forward carries (16 by
    default, at most 64), so a fidelity run can score at a device's prefill
    widths; out-of-range values leave the default."""
    long_prompt = PROMPT   # byte-level fixture tokens: ~90 positions

    def route(value):
        env = dict(os.environ, RUNNER_SCORE_CHUNKED="1")
        env.pop("RUNNER_SPEC_BATCH", None)
        if value is not None:
            env["RUNNER_SPEC_BATCH"] = value
        p = subprocess.run(
            [runner_bin, "-m", str(models["dense"]), "--score", "-p", long_prompt,
             "-t", "2", "--gpu", "off"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        line = [ln for ln in p.stderr.decode().splitlines()
                if ln.startswith("score: route ")][0]
        return (json.loads(p.stdout)["n_scored"],
                json.loads(line[len("score: route "):])["paths"]["keep"]["rows_max"])

    n, rows = route(None)
    assert n > 40 and rows == 16
    n48, rows48 = route("48")
    assert n48 == n and 16 < rows48 <= 48
    for bad in ("0", "65", "junk"):
        assert route(bad) == (n, 16), bad


def test_score_top_needs_score(runner_bin, models):
    p = subprocess.run(
        [runner_bin, "-m", str(models["dense"]), "--score-top", "5",
         "-p", PROMPT, "-n", "2", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert b"--score-top" in p.stderr and b"needs --score" in p.stderr
