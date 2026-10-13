"""What one request leaves behind for the next one on the same slot.

A server slot keeps its engine, its sampler and its KV between requests. Only
two things are meant to carry over: the sampler's RNG state (so an unseeded
stream stays diverse) and the cached prefix (a cost, never a result). Every
other knob a request sets (sampling overrides, stops, constraints, logprobs,
budgets) must be applied for that request and gone for the next. These tests
force one slot (--parallel 1) and a fixed server seed (-s), then compare a
request served after some other request with the same request served on a
fresh server: equal means nothing leaked.

Absolute anchor: the fixed-seed control. Two fresh servers started with the
same -s produce the same unseeded stream, and an explicit seed reproduces its
own output on any slot history. The rest compares this server with itself
under different histories, which is the property under test.
"""
import json
import pathlib
import socket
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

SEED = "42"
SAMPLED = {"prompt": "hello world", "max_tokens": 12, "temperature": 0.9}
CHAT = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10,
        "temperature": 0.9}


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import subprocess
    m = tmp_path_factory.mktemp("seq") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def server(exe, model):
    return RunnerServer(exe, model, ctx=512, parallel=1,
                        extra_args=["--gpu", "off", "-t", "2", "-s", SEED])


def post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def text(resp):
    status, body = resp
    assert status == 200, body
    choice = body["choices"][0]
    return choice["message"]["content"] if "message" in choice else choice["text"]


def served(exe, model, *bodies, path="/v1/completions"):
    """Serve the bodies in order on one fresh single-slot server."""
    with server(exe, model) as srv:
        return [post(srv, path, b) for b in bodies]


# ---------------------------------------------------------------- the RNG

def test_a_fresh_server_with_the_same_seed_repeats_the_unseeded_stream(exe, model):
    first = served(exe, model, SAMPLED, SAMPLED)
    again = served(exe, model, SAMPLED, SAMPLED)
    assert [text(r) for r in first] == [text(r) for r in again]
    # the state carries: the second draw is not a replay of the first
    assert text(first[0]) != text(first[1])


def test_an_explicit_seed_reproduces_on_any_slot_history(exe, model):
    seeded = dict(SAMPLED, seed=5)
    alone = served(exe, model, seeded)
    after = served(exe, model, SAMPLED, dict(SAMPLED, temperature=0.3), seeded)
    assert text(alone[0]) == text(after[2])


def test_a_seeded_request_moves_the_stream_it_leaves_behind(exe, model):
    """Only the RNG state carries over, so an unseeded request after a seeded
    one continues from where the seeded one stopped: deterministic, and not
    the fresh server's first draw."""
    a = served(exe, model, dict(SAMPLED, seed=5), SAMPLED)
    b = served(exe, model, dict(SAMPLED, seed=5), SAMPLED)
    fresh = served(exe, model, SAMPLED)
    assert text(a[1]) == text(b[1])
    assert text(a[1]) != text(fresh[0])


def test_a_greedy_request_draws_nothing_from_the_stream(exe, model):
    """Temperature 0 is argmax: the unseeded stream after it is the fresh
    server's first draw."""
    after = served(exe, model, dict(SAMPLED, temperature=0), SAMPLED)
    fresh = served(exe, model, SAMPLED)
    assert text(after[1]) == text(fresh[0])


# --------------------------------------------- request knobs do not carry over

GREEDY = {"prompt": "one two three", "max_tokens": 8, "temperature": 0}


def _choice(resp):
    return resp[1]["choices"][0]


# Each knob is built from the fresh answer T and checked to act on its own
# request first, so a knob that leaked would be visible on the next one.
KNOBS = {
    "stop": (lambda t: dict(GREEDY, stop=[t[1:3]]),
             lambda a, t: _choice(a)["finish_reason"] == "stop" and len(text(a)) < len(t)),
    "max_tokens": (lambda t: dict(GREEDY, max_tokens=1),
                   lambda a, t: a[1]["usage"]["completion_tokens"] == 1),
    "logprobs": (lambda t: dict(GREEDY, logprobs=3),
                 lambda a, t: bool(_choice(a).get("logprobs"))),
    "json_mode": (lambda t: dict(GREEDY, response_format={"type": "json_object"}),
                  lambda a, t: a[1]["runner_telemetry"]["json_mode"] is True),
    "echo": (lambda t: dict(GREEDY, echo=True, logprobs=1),
             lambda a, t: text(a).startswith(GREEDY["prompt"])),
    "prompt_logprobs": (lambda t: dict(GREEDY, prompt_logprobs=2),
                        lambda a, t: _choice(a).get("prompt_logprobs") is not None),
    "sampling": (lambda t: dict(GREEDY, temperature=0.9, seed=3, repeat_penalty=1.8, top_k=2),
                 lambda a, t: a[1]["runner_telemetry"]["sampling"]["repeat_penalty"] == pytest.approx(1.8)),
    # no visible effect on this fixture; still must not carry over
    "stop_token_ids": (lambda t: dict(GREEDY, stop_token_ids=[1, 2]), None),
    "cache_prompt": (lambda t: dict(GREEDY, cache_prompt=False), None),
    "special_tokens": (lambda t: dict(GREEDY, special_tokens=True), None),
}


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_a_request_knob_is_gone_for_the_next_request(exe, model, knob):
    build, acts = KNOBS[knob]
    with server(exe, model) as srv:
        fresh = post(srv, "/v1/completions", GREEDY)
        t = text(fresh)
        assert len(t) >= 3, t
        first = post(srv, "/v1/completions", build(t))
        assert first[0] == 200, first
        if acts:
            assert acts(first, t), (knob, first[1])
        after = post(srv, "/v1/completions", GREEDY)
    assert text(after) == t
    a, f = after[1], fresh[1]
    for key in ("logprobs", "prompt_logprobs", "finish_reason"):
        assert a["choices"][0].get(key) == f["choices"][0].get(key), key
    assert a["usage"]["completion_tokens"] == f["usage"]["completion_tokens"]
    for key in ("json_mode", "schema", "sampling", "tool_protocol"):
        assert a["runner_telemetry"][key] == f["runner_telemetry"][key], key


def test_a_chat_after_a_constrained_chat_is_unconstrained(exe, model):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"]}
    constrained = dict(CHAT, temperature=0, response_format={
        "type": "json_schema", "json_schema": {"name": "x", "schema": schema}})
    plain = dict(CHAT, temperature=0)
    after = served(exe, model, constrained, plain, path="/v1/chat/completions")
    fresh = served(exe, model, plain, path="/v1/chat/completions")
    json.loads(text(after[0]))   # the constraint held for its own request
    assert text(after[1]) == text(fresh[0])
    t_after = after[1][1]["runner_telemetry"]
    t_fresh = fresh[0][1]["runner_telemetry"]
    assert t_after["sampling"] == t_fresh["sampling"]
    assert t_after["tool_protocol"] == t_fresh["tool_protocol"]


def test_a_refused_request_does_not_change_the_next_greedy_answer(exe, model):
    """Refusals arrive at many depths of validation; whatever a refused
    request set before it was refused is not seen by the next one."""
    refused = [
        dict(GREEDY, max_tokens="x"),
        dict(GREEDY, stop=""),
        dict(GREEDY, stream="yes"),
        dict(GREEDY, response_format={"type": "bogus"}),
        dict(GREEDY, stop=["a"], logprobs=99),
        dict(GREEDY, top_k=2.5),
        dict(GREEDY, prompt="x" * 4000),
    ]
    with server(exe, model) as srv:
        fresh = text(post(srv, "/v1/completions", GREEDY))
        for bad in refused:
            status, body = post(srv, "/v1/completions", bad)
            assert status == 400, (bad, body)
            assert text(post(srv, "/v1/completions", GREEDY)) == fresh, bad


# ------------------------------------------------- a client that goes away

def test_a_vanished_stream_leaves_the_slot_serving(exe, model):
    """A client that closes a stream mid-generation does not wedge the slot
    or bend the next request's answer."""
    long_stream = {"prompt": "one two three", "max_tokens": 200, "temperature": 0,
                   "stream": True}
    with server(exe, model) as srv:
        fresh = text(post(srv, "/v1/completions", GREEDY))
        host, port = "127.0.0.1", int(srv.base_url.rsplit(":", 1)[1])
        body = json.dumps(long_stream).encode()
        s = socket.create_connection((host, port), timeout=30)
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                  % len(body) + body)
        s.recv(256)          # the stream has started
        s.close()
        assert text(post(srv, "/v1/completions", GREEDY)) == fresh
