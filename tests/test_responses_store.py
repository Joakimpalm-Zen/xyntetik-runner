"""Responses persistence (R10.6): store:true, previous_response_id, and
GET / DELETE /v1/responses/{id}; plus parallel tool calls on Messages.

The store is in memory, bounded and never written to disk. What it must get
right is equivalence: a request that continues a stored response with
previous_response_id is the SAME request as one that sends the whole history
itself -- the same prompt (its input_tokens) and, greedy, the same output. The
explicit-history request is the anchor; it goes through the stateless path the
conformance suite already pins.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("rs") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _serve(runner_bin, model, env=None, extra=(), ctx=512):
    e = dict(os.environ)
    e.update(env or {})
    return RunnerServer(runner_bin, model, ctx=ctx, env=e,
                        extra_args=["--gpu", "off", "-t", "2", *extra])


def _req(srv, method, path, payload=None, stream=False):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(srv.base_url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)
    if stream:
        return status, [json.loads(l[6:]) for l in raw.decode().splitlines()
                        if l.startswith("data: ") and l[6:] != "[DONE]"]
    return status, json.loads(raw)


def _resp(srv, **payload):
    return _req(srv, "POST", "/v1/responses",
                {"max_output_tokens": 6, "temperature": 0, **payload})


def _text(body):
    return "".join(c["text"] for o in body["output"] if o["type"] == "message"
                   for c in o["content"] if c["type"] == "output_text")


def test_a_stored_response_is_retrievable(runner_bin, model):
    with _serve(runner_bin, model) as srv:
        st, a = _resp(srv, input="Name a colour.", store=True)
        assert st == 200, a
        assert a["store"] is True and a["previous_response_id"] is None
        st, got = _req(srv, "GET", f"/v1/responses/{a['id']}")
        assert st == 200, got
        assert got["id"] == a["id"] and got["output"] == a["output"]
        st, items = _req(srv, "GET", f"/v1/responses/{a['id']}/input_items")
        assert st == 200, items
        assert items["object"] == "list"
        assert items["data"][0]["role"] == "user"
        # store:false (and absent) keeps nothing
        st, b = _resp(srv, input="Name a colour.", store=False)
        assert st == 200 and b["store"] is False
        assert _req(srv, "GET", f"/v1/responses/{b['id']}")[0] == 404
        st, c = _resp(srv, input="Name a colour.")
        assert _req(srv, "GET", f"/v1/responses/{c['id']}")[0] == 404


def test_previous_response_id_is_the_explicit_history(runner_bin, model):
    with _serve(runner_bin, model) as srv:
        st, a = _resp(srv, input="Name a colour.", store=True)
        assert st == 200, a
        st, b = _resp(srv, input="And another?", previous_response_id=a["id"],
                      store=True)
        assert st == 200, b
        assert b["previous_response_id"] == a["id"]
        # the anchor: the same conversation sent statelessly
        history = [{"role": "user", "content": "Name a colour."}]
        history += a["output"]
        history += [{"role": "user", "content": "And another?"}]
        st, ref = _resp(srv, input=history)
        assert st == 200, ref
        assert b["usage"]["input_tokens"] == ref["usage"]["input_tokens"]
        assert _text(b) == _text(ref)
        # a chain of three: the third sees both earlier turns
        st, c = _resp(srv, input="One more.", previous_response_id=b["id"])
        assert st == 200, c
        history2 = history + b["output"] + [{"role": "user",
                                             "content": "One more."}]
        st, ref2 = _resp(srv, input=history2)
        assert c["usage"]["input_tokens"] == ref2["usage"]["input_tokens"]
        st, items = _req(srv, "GET", f"/v1/responses/{b['id']}/input_items")
        assert [i.get("role") for i in items["data"]
                if i.get("type", "message") == "message"][0] == "user"
        assert len(items["data"]) == len(history)


def test_unknown_and_deleted_responses(runner_bin, model):
    with _serve(runner_bin, model) as srv:
        st, b = _resp(srv, input="x", previous_response_id="resp_nope")
        assert st == 404 and b["error"]["code"] == "previous_response_not_found"
        st, a = _resp(srv, input="Name a colour.", store=True)
        st, d = _req(srv, "DELETE", f"/v1/responses/{a['id']}")
        assert st == 200 and d == {"id": a["id"], "object": "response.deleted",
                                   "deleted": True}
        assert _req(srv, "GET", f"/v1/responses/{a['id']}")[0] == 404
        assert _req(srv, "DELETE", f"/v1/responses/{a['id']}")[0] == 404
        st, b = _resp(srv, input="x", previous_response_id=a["id"])
        assert st == 404, b
        for bad in ({"store": "yes"}, {"previous_response_id": 7}):
            st, b = _resp(srv, input="x", **bad)
            assert st == 400, (bad, b)


def test_the_store_forgets_after_its_ttl(runner_bin, model):
    with _serve(runner_bin, model, {"RUNNER_RESPONSES_STORE_TTL": "0.3"}) as srv:
        st, a = _resp(srv, input="Name a colour.", store=True)
        assert _req(srv, "GET", f"/v1/responses/{a['id']}")[0] == 200
        time.sleep(0.8)
        assert _req(srv, "GET", f"/v1/responses/{a['id']}")[0] == 404


def test_a_streamed_response_is_stored(runner_bin, model):
    with _serve(runner_bin, model) as srv:
        st, events = _req(srv, "POST", "/v1/responses", {
            "input": "Name a colour.", "store": True, "stream": True,
            "max_output_tokens": 6, "temperature": 0}, stream=True)
        assert st == 200
        # six tokens usually cut the turn short: either terminal event
        # carries the whole response, and either is stored
        done = [e for e in events if e["type"] in ("response.completed",
                                                   "response.incomplete")]
        assert len(done) == 1, events
        rid = done[0]["response"]["id"]
        assert done[0]["response"]["store"] is True
        st, got = _req(srv, "GET", f"/v1/responses/{rid}")
        assert st == 200 and got["output"] == done[0]["response"]["output"]


WEATHER = {"name": "get_weather", "description": "Weather for a city",
           "input_schema": {"type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]}}
# compact: the envelope grammar admits no whitespace the model need not spend
TWO_CALLS = json.dumps({"calls": [
    {"tool": "get_weather", "args": {"city": "Osaka"}},
    {"tool": "get_weather", "args": {"city": "Athens"}}]},
    separators=(",", ":"))


def test_messages_parallel_tool_use(runner_bin, model):
    with _serve(runner_bin, model, {"RUNNER_TEST_SCRIPTED_REPLY": "1"},
                ctx=4096) as srv:
        st, d = _req(srv, "POST", "/v1/messages", {
            "max_tokens": 200, "tools": [WEATHER],
            "tool_choice": {"type": "any",
                            "disable_parallel_tool_use": False},
            "messages": [{"role": "user", "content": "Osaka and Athens?"}],
            "runner_test_reply": TWO_CALLS})
        assert st == 200, d
        uses = [b for b in d["content"] if b["type"] == "tool_use"]
        assert [u["input"] for u in uses] == [{"city": "Osaka"},
                                              {"city": "Athens"}]
        assert len({u["id"] for u in uses}) == 2
        assert d["stop_reason"] == "tool_use"
        # disable_parallel_tool_use:true keeps one call per turn: the same
        # dictated two-call document is not admitted by that grammar
        st, d = _req(srv, "POST", "/v1/messages", {
            "max_tokens": 200, "tools": [WEATHER],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": "Osaka and Athens?"}],
            "runner_test_reply": TWO_CALLS})
        uses = [b for b in d.get("content", []) if b["type"] == "tool_use"]
        assert len(uses) <= 1, d


def test_a_wrong_typed_input_is_refused_with_a_previous_response(runner_bin, model):
    """previous_response_id replaced `input` with the merged history before
    its type was checked, so `"input": 5` answered 200 on a continuation and
    400 everywhere else."""
    with _serve(runner_bin, model) as srv:
        st, a = _resp(srv, input="Name a colour.", store=True)
        assert st == 200, a
        assert _resp(srv, input=5)[0] == 400
        st, b = _resp(srv, input=5, previous_response_id=a["id"])
        assert st == 400, b
