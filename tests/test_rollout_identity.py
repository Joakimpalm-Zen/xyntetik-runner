"""The policy that samples is the policy that trains, bit for bit.

scripts/train-grpo-lite.py samples rollouts from a served model and trains on
them with --train. Its claim is that nothing differs between the two: the
trainer's taped forward, run on the token ids the server sampled, under the
adapter the server sampled with, gives the very logprobs the server reported.
This gate holds that claim on every family the trainer admits, plus a K-quant
base, with the loop's own settings: a CPU server reading the prompt one token
per forward (-b 1), seeded sampling at a temperature, the id form of the
training data, and --train-logprobs at lr 0 so that every example is scored
under the weights that sampled it.

It also keeps itself honest. The adapter must matter (the same ids scored
without it differ), the wire must carry the full float (a logprob rounded to
six decimals would hide a last-bit difference), and the default batched
prompt pass must still be seen to differ on a quantized base. That last check
is the reason the loop passes -b 1: from 8 rows up the CPU reads a quantized
weight through a dequantized f32 tile, exact per output but not the solo dot
(docs/kernel-routes.md), and every logprob after the prompt then moves in the
last bits. If that route is ever made exact, the check fails with a message
saying the -b 1 is no longer needed. It reads the route directly (the same
text scored through it and one position per forward), not through sampled
tokens: those carry only the few differences the f16 KV cache does not round
away, from 32 of 39 logprobs on an Apple M1 to 1 of 39 on an x86 host and
none at all on one CI runner, which failed the check while the route still
differed at about half the positions.

The claim is an identity between two paths of one binary, so its main
assertion has no anchor outside the system, by its nature. The precision check
does: nine significant digits round-trip every IEEE 754 binary32 value, so a
logprob that is not the %.9g of some f32 was not printed from the engine's
float.
"""
import json
import os
import pathlib
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402
from test_train import CORPUS, TRAIN_FAMILIES, _family_fixture  # noqa: E402

# the fixtures' training context: above it YaRN rescales rope, and the server
# and the trainer must run at the same one
CTX = "256"
PROMPTS = ["the runner trains the adapter", "the corpus is learned by"]
SEEDS = [11, 12]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _fixture(runner_bin, tmp_path, family):
    if family == "f32":
        d = tmp_path / "fx"
        d.mkdir()
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        str(d / "base.gguf")], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        (d / "corpus.txt").write_text(CORPUS)
        return d
    if family == "q4_k":
        # 256-wide rows hold whole K-quant blocks
        d = tmp_path / "fx"
        d.mkdir()
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        "--wide", str(d / "wide.gguf")], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        _run(runner_bin, ["-m", str(d / "wide.gguf"), "--quantize",
                          str(d / "base.gguf"), "--quant", "q4_k"])
        (d / "corpus.txt").write_text(CORPUS)
        return d
    return _family_fixture(tmp_path, family)


def _run(runner_bin, args, ok=True):
    p = subprocess.run([str(runner_bin), *args], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=600)
    if ok:
        assert p.returncode == 0, p.stderr.decode(errors="replace")[-2000:]
    return p


def _adapter(runner_bin, d):
    out = d / "ad.gguf"
    _run(runner_bin, ["-m", str(d / "base.gguf"), "--train",
                      str(d / "corpus.txt"), "--train-steps", "4", "--lr",
                      "3e-3", "--train-out", str(out), "-t", "2", "-c", CTX])
    return out


def _sample(srv, prompt, seed, echo):
    """One rollout as train-grpo-lite.py takes it: the sampled ids and their
    logprobs, and with echo the prompt's ids in front."""
    body = {"prompt": prompt, "max_tokens": 12, "temperature": 0.8,
            "seed": seed, "logprobs": 1}
    if echo:
        body["echo"] = True
    st, d = _post(srv, body)
    assert st == 200, d
    lp = d["choices"][0]["logprobs"]
    ids, lps = lp["token_ids"], lp["token_logprobs"]
    n = d["usage"]["prompt_tokens"] if echo else 0
    return ids[:n], ids[n:], lps[n:]


def _post(srv, payload):
    req = urllib.request.Request(srv.base_url + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _prompt_ids(runner_bin, base):
    """Each prompt's token ids, from a separate echoing server: echo scores a
    prompt one token per forward and leaves that KV behind, so a measured
    server that echoed would never run its own prompt pass."""
    with RunnerServer(runner_bin, base, ctx=int(CTX),
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        return [_sample(srv, prompt, 1, True)[0] for prompt in PROMPTS]


def _rollouts(runner_bin, base, adapter, extra, prompt_ids=None):
    """Sampled (prompt ids, ids, logprobs) for every prompt and seed. With
    `prompt_ids` no request echoes, so the server's own prompt pass is the
    one measured; without, every request echoes."""
    args = ["--gpu", "off", "-t", "2", *extra]
    if adapter:
        args += ["--lora", str(adapter)]
    out = []
    with RunnerServer(runner_bin, base, ctx=int(CTX), extra_args=args) as srv:
        for i, prompt in enumerate(PROMPTS):
            for seed in SEEDS:
                pids, ids, lps = _sample(srv, prompt, seed, prompt_ids is None)
                if prompt_ids is not None:
                    pids = prompt_ids[i]
                if ids:
                    out.append((pids, ids, lps))
    assert out, "no rollout sampled a token"
    return out


def _trained(runner_bin, d, rollouts, adapter):
    """The trainer's logprobs for every rollout, under `adapter` (a fresh one
    when None) and lr 0, so that no step moves the weights."""
    data = d / "ids.jsonl"
    data.write_text("".join(
        json.dumps({"prompt_tokens": p, "completion_tokens": c,
                    "weight": 1.0}) + "\n" for p, c, _ in rollouts))
    lp = d / "trained.jsonl"
    args = ["-m", str(d / "base.gguf"), "--train", str(data), "--train-steps",
            str(len(rollouts)), "--lr", "0", "--train-out", str(d / "lr0.gguf"),
            "--train-logprobs", str(lp), "-t", "2", "-c", CTX]
    args += ["--lora", str(adapter)] if adapter else ["--lora-rank", "8"]
    _run(runner_bin, args)
    lines = [json.loads(x) for x in lp.read_text().splitlines()]
    assert [x["step"] for x in lines] == list(range(1, len(rollouts) + 1))
    for x, (p, c, _) in zip(lines, rollouts):
        assert x["first"] == len(p) and len(x["logprobs"]) == len(c)
    return [x["logprobs"] for x in lines]


def _f32_text(v):
    # the wire prints an f32 at %.9g, which round-trips it exactly
    return float("%.9g" % struct.unpack("f", struct.pack("f", v))[0])


FAMILIES = ["f32", "q4_k", *sorted(TRAIN_FAMILIES)]


@pytest.mark.parametrize("family", FAMILIES)
def test_trainer_scores_rollouts_exactly_as_the_sampler(runner_bin, tmp_path, family):
    d = _fixture(runner_bin, tmp_path, family)
    adapter = _adapter(runner_bin, d)
    pids = _prompt_ids(runner_bin, d / "base.gguf")
    rollouts = _rollouts(runner_bin, d / "base.gguf", adapter, ["-b", "1"], pids)
    trained = _trained(runner_bin, d, rollouts, adapter)
    for (_, ids, sampled), got in zip(rollouts, trained):
        for v in sampled:
            assert _f32_text(v) == v, ("logprob not at full precision", v)
        assert got == sampled, (family, ids, sampled, got)
    # anti-vacuity: the same ids under a fresh (no-op) adapter score as the
    # bare base, which the trained adapter must move
    base_scored = _trained(runner_bin, d, rollouts, None)
    assert base_scored != trained, (family, "the adapter changed nothing")


@pytest.mark.parametrize("family", ["q8_0", "q4_k"])
def test_echo_alone_makes_the_prompt_pass_exact(runner_bin, tmp_path, family):
    """The loop's first request per prompt echoes, to learn the prompt's ids:
    echo scores the prompt one token per forward whatever -b says, so those
    rollouts are exact on the default server too."""
    d = _fixture(runner_bin, tmp_path, family)
    adapter = _adapter(runner_bin, d)
    rollouts = _rollouts(runner_bin, d / "base.gguf", adapter, [])
    assert _trained(runner_bin, d, rollouts, adapter) == \
        [lps for _, _, lps in rollouts]


@pytest.mark.parametrize("family", ["f32", "q8_0"])
def test_a_fresh_adapter_scores_as_the_bare_base_sampled(runner_bin, tmp_path, family):
    """Round 0 of the loop samples with no adapter and trains a fresh one,
    whose B is zero: an exact no-op, so step 1 still scores what was sampled."""
    d = _fixture(runner_bin, tmp_path, family)
    pids = _prompt_ids(runner_bin, d / "base.gguf")
    rollouts = _rollouts(runner_bin, d / "base.gguf", None, ["-b", "1"], pids)
    trained = _trained(runner_bin, d, rollouts, None)
    assert trained == [lps for _, _, lps in rollouts]


def _score(runner_bin, d, adapter, text, chunked):
    env = dict(os.environ)
    env.pop("RUNNER_SCORE_CHUNKED", None)
    if chunked:
        env["RUNNER_SCORE_CHUNKED"] = "1"
    p = subprocess.run([str(runner_bin), "-m", str(d / "base.gguf"), "--lora",
                        str(adapter), "--gpu", "off", "-t", "2", "-c", CTX,
                        "--score", "-p", text], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-2000:]
    return json.loads(p.stdout.decode().strip().splitlines()[-1])["logprobs"]


def test_default_batched_prompt_pass_still_differs_on_a_quantized_base(runner_bin, tmp_path):
    """The batch route a default prompt pass takes (RUNNER_SCORE_CHUNKED=1
    scores through it) against one position per forward, the trainer's
    numerics, on the same text, base and adapter."""
    d = _fixture(runner_bin, tmp_path, "q8_0")
    adapter = _adapter(runner_bin, d)
    text = CORPUS[:120]
    solo = _score(runner_bin, d, adapter, text, False)
    batched = _score(runner_bin, d, adapter, text, True)
    assert len(solo) == len(batched) > 8, (len(solo), len(batched))
    deltas = [abs(a - b) for a, b in zip(solo, batched)]
    assert any(deltas), (
        "the default batched prompt pass now matches the trainer bit for bit "
        "on a quantized base: the -b 1 in train-grpo-lite.py (and this "
        "check) are no longer needed")
    assert max(deltas) < 1e-4, max(deltas)


@pytest.mark.parametrize("line, why", [
    ({"prompt_tokens": [1, 5], "completion_tokens": [10**6]},
     "outside the vocabulary"),
    ({"prompt_tokens": [1, -2], "completion_tokens": [5]}, "non-negative integer"),
    ({"prompt_tokens": [1, 2.5], "completion_tokens": [5]}, "non-negative integer"),
    ({"prompt_tokens": [1, 5], "completion_tokens": []}, "at least one"),
    ({"prompt_tokens": [1, 5]}, "one of the two"),
    ({"prompt_tokens": [1, 5], "completion_tokens": [5], "prompt": "x"},
     "one of the two"),
    ({"prompt_tokens": "1 5", "completion_tokens": [5]}, "non-negative integer"),
])
def test_id_form_is_refused_when_malformed(runner_bin, tmp_path, line, why):
    d = _fixture(runner_bin, tmp_path, "f32")
    data = d / "bad.jsonl"
    data.write_text(json.dumps(line) + "\n")
    p = _run(runner_bin, ["-m", str(d / "base.gguf"), "--train", str(data),
                          "--train-steps", "1", "--train-out",
                          str(d / "x.gguf"), "-t", "2"], ok=False)
    assert p.returncode != 0
    assert why in p.stderr.decode(errors="replace")
    assert not (d / "x.gguf").exists()


def test_train_logprobs_without_train_is_refused(runner_bin, tmp_path):
    d = _fixture(runner_bin, tmp_path, "f32")
    p = _run(runner_bin, ["-m", str(d / "base.gguf"), "-p", "hi", "-n", "1",
                          "--train-logprobs", str(d / "lp.jsonl")], ok=False)
    assert p.returncode != 0
    assert "--train-logprobs" in p.stderr.decode(errors="replace")
    assert not (d / "lp.jsonl").exists()
