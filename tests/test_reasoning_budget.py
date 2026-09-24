"""A reasoning budget closes an over-long reasoning turn (R4.12.16).

Budget forcing, s1-style: when a turn has spent `reasoning_max_tokens` inside
its reasoning channel the sampler is restricted to the model's reasoning-close
token, so the turn closes and the model goes on to address its recipient. It
is not a stop: the answer keeps the whole `max_tokens` budget.

Driven over HTTP against the muse fixture, whose reasoning turn is a real
` to=self` channel ending on `<|eom|>` (a single control token, which is what
the feature needs). `enable_thinking: true` primes that turn in the prompt, so
generation starts inside reasoning and every sampled token counts against the
budget -- the fixture's weights are random, so what it reasons ABOUT is
meaningless here; what is measured is where the turn ends and what the server
reports about it.
"""
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

BUDGET = 6


@pytest.fixture(scope="module")
def muse_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("rbudget") / "muse.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--muse-glimmer",
                    "--control", "<|start|>,<|message|>,<|eom|>,<|eot|>", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def plain_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("rbudget") / "plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _serve(model, extra=()):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    srv = RunnerServer(exe, model, ctx=2048, parallel=1,
                       extra_args=["--gpu", "off", "-t", "2", *extra])
    return srv


@pytest.fixture(scope="module")
def muse(muse_model):
    with _serve(muse_model) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def muse_default_budget(muse_model):
    """Same model, server started with --reasoning-budget."""
    with _serve(muse_model, ["--reasoning-budget", str(BUDGET)]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def plain(plain_model):
    with _serve(plain_model) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _chat(server, **extra):
    payload = {"model": server.model_id, "max_tokens": 120, "temperature": 0,
               "enable_thinking": True,
               "messages": [{"role": "user", "content": "Which is warmer, Sydney or Oslo?"}]}
    payload.update(extra)
    req = urllib.request.Request(server.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _budget(body):
    # runner_telemetry rides on the body, not on the choice
    return body.get("runner_telemetry", {}).get("reasoning_budget")


def test_without_a_budget_nothing_is_reported_and_nothing_is_forced(muse):
    code, body = _chat(muse)
    assert code == 200, body
    assert _budget(body) is None


def _reasoning(body):
    """The reasoning channel, with the fixture's word-boundary marker read as
    the space it stands for: this vocabulary is byte-fallback and has no
    U+2581 piece, so it spells a space as the three bytes that decode to
    U+2581 (the same quirk the scripted-reply hook documents). A real
    vocabulary writes the sentence with ordinary spaces."""
    text = body["choices"][0]["message"].get("reasoning_content") or ""
    return text.replace("\u2581", " ")


def test_the_forced_close_writes_a_transition_sentence(muse):
    """A bare terminator drops the model mid sentence and measurably costs
    the answer that follows (llama.cpp's budget PR: HumanEval 93% uncapped,
    ~89% capped with a message, 79% with a bare end tag), so the default
    close writes a short neutral line first. It lands INSIDE the reasoning
    channel, where a reader of the trace can see why it ends there."""
    code, body = _chat(muse, reasoning_max_tokens=BUDGET)
    assert code == 200, body
    assert "stopping deliberation here" in _reasoning(body)


def test_a_caller_chooses_the_sentence_or_asks_for_a_bare_close(muse):
    code, body = _chat(muse, reasoning_max_tokens=BUDGET,
                       reasoning_budget_message="\nEnough. Answering now.\n")
    assert code == 200, body
    assert "Enough. Answering now." in _reasoning(body)
    assert "stopping deliberation here" not in _reasoning(body)

    code, body = _chat(muse, reasoning_max_tokens=BUDGET,
                       reasoning_budget_message="")
    assert code == 200, body
    assert "stopping deliberation" not in _reasoning(body)
    # the close still fired: counting stopped at the cap
    assert _budget(body)["forced_close"] is True


def test_a_sentence_that_cannot_fit_is_refused_not_truncated(muse):
    code, body = _chat(muse, reasoning_max_tokens=BUDGET,
                       reasoning_budget_message="word " * 2000)
    assert code == 400, body
    assert "shorter sentence" in json.dumps(body)
    code, body = _chat(muse, reasoning_max_tokens=BUDGET,
                       reasoning_budget_message=7)
    assert code == 400, body
    assert "reasoning_budget_message" in json.dumps(body)


def test_the_budget_closes_the_reasoning_turn(muse):
    code, body = _chat(muse, reasoning_max_tokens=BUDGET)
    assert code == 200, body
    rb = _budget(body)
    assert rb is not None and rb["max_tokens"] == BUDGET
    # the cap fired, and it fired at the budget rather than somewhere else
    assert rb["forced_close"] is True
    # Counting stops when the turn closes, so a count just past the budget
    # (the cap itself, plus the forced sentence and its close token) is the
    # proof that it DID close: an unforced turn would have counted every one
    # of the generated tokens instead.
    assert BUDGET <= rb["tokens"] <= BUDGET + 256   # the cap, the sentence, the close
    assert body["usage"]["completion_tokens"] > BUDGET
    # not a stop: the turn ran on past the close with its own budget intact
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    assert body["choices"][0]["finish_reason"] != "reasoning_limit"


def test_a_budget_larger_than_the_turn_never_fires(muse):
    code, body = _chat(muse, reasoning_max_tokens=100000, max_tokens=24)
    assert code == 200, body
    rb = _budget(body)
    assert rb["forced_close"] is False
    assert rb["tokens"] <= 24


def test_the_server_default_applies_and_a_request_can_turn_it_off(muse_default_budget):
    code, body = _chat(muse_default_budget)
    assert code == 200, body
    rb = _budget(body)
    assert rb is not None and rb["max_tokens"] == BUDGET and rb["forced_close"] is True

    code, body = _chat(muse_default_budget, reasoning_max_tokens=0)
    assert code == 200, body
    assert _budget(body) is None


def test_a_bad_budget_is_refused(muse):
    for bad in (-1, 1.5, "lots", True):
        code, body = _chat(muse, reasoning_max_tokens=bad)
        assert code == 400, (bad, body)
        assert "reasoning_max_tokens" in json.dumps(body)


def test_a_model_without_a_reasoning_close_token_refuses_the_budget(plain):
    code, body = _chat(plain, reasoning_max_tokens=8)
    assert code == 400, body
    assert "single token" in json.dumps(body)
    # and without the field that model serves normally
    code, body = _chat(plain)
    assert code == 200, body
    assert _budget(body) is None
