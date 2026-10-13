"""How each surface reports a generation fault, before and after its headers.

A generation fault is a turn that produced nothing anybody can use, for the
server's reason rather than the model's or the caller's (completion.c,
generation_faulted: an allocation failure mid-generation, or a constrained
document that could not be mapped back). A buffered reply can still say so in
its status code; a streamed one has already sent its 200 and headers when the
fault happens, so it has to end on the surface's own terminator for it.

The fault is reached through the scripted-reply test hook: `runner_test_fault`
ends `runner_test_reply` in an allocation failure instead of a clean stop
(sample.h), admitted only under RUNNER_TEST_SCRIPTED_REPLY like the reply.
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

FAULT = {"runner_test_reply": "Hi there", "runner_test_fault": True}
CHAT = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "temperature": 0}


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("fault") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def srv(exe, model):
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", "-s", "42", "--chat-template", "chatml"],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as s:
        yield s


def post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def sse(raw):
    """[(event name or None, data)] of an SSE body."""
    out, event = [], None
    for line in raw.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            out.append((event, line[5:].strip()))
            event = None
    return out


def model_id(srv):
    return json.load(urllib.request.urlopen(srv.base_url + "/v1/models"))["data"][0]["id"]


def test_chat_reports_the_fault_as_its_finish_reason(srv):
    status, raw = post(srv, "/v1/chat/completions", dict(CHAT, **FAULT))
    assert status == 200, raw
    choice = json.loads(raw)["choices"][0]
    assert choice["finish_reason"] == "error", choice


def test_streamed_chat_ends_on_the_fault_then_done(srv):
    status, raw = post(srv, "/v1/chat/completions", dict(CHAT, stream=True, **FAULT))
    assert status == 200, raw
    events = sse(raw)
    assert events[-1] == (None, "[DONE]"), events[-3:]
    finishes = [c.get("finish_reason") for _, d in events[:-1]
                for c in json.loads(d).get("choices") or [] if c.get("finish_reason")]
    assert finishes == ["error"], finishes


def test_messages_reports_the_fault_as_an_error_not_a_stop_reason(srv):
    status, raw = post(srv, "/v1/messages", {"model": model_id(srv), "max_tokens": 32, "temperature": 0,
                                             "messages": [{"role": "user", "content": "Hello"}], **FAULT})
    assert status == 500, raw
    doc = json.loads(raw)
    assert doc["type"] == "error" and doc["error"]["type"] == "api_error", doc


def test_streamed_messages_end_on_an_error_event_after_the_headers(srv):
    status, raw = post(srv, "/v1/messages", {"model": model_id(srv), "max_tokens": 32, "temperature": 0,
                                             "stream": True,
                                             "messages": [{"role": "user", "content": "Hello"}], **FAULT})
    assert status == 200, raw   # the headers went out before the fault
    events = sse(raw)
    names = [e for e, _ in events]
    assert names[0] == "message_start" and "content_block_delta" in names, names
    assert names[-1] == "error", names
    assert "message_delta" not in names and "message_stop" not in names, names
    assert json.loads(events[-1][1])["error"]["type"] == "api_error", events[-1]


def test_the_fault_hook_is_refused_without_a_reply_or_as_a_non_boolean(srv):
    status, raw = post(srv, "/v1/chat/completions", dict(CHAT, runner_test_fault=True))
    assert status == 400 and "runner_test_reply" in raw, raw
    status, raw = post(srv, "/v1/chat/completions", dict(CHAT, runner_test_reply="Hi", runner_test_fault="yes"))
    assert status == 400 and "boolean" in raw, raw
    status, raw = post(srv, "/v1/chat/completions", dict(CHAT, runner_test_reply="Hi", runner_test_fault=False))
    assert status == 200 and json.loads(raw)["choices"][0]["finish_reason"] == "stop", raw


def test_the_fault_hook_is_refused_on_a_server_without_the_hook(exe, model):
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", "--chat-template", "chatml"]) as s:
        status, raw = post(s, "/v1/chat/completions", dict(CHAT, **FAULT))
        assert status == 400 and "test hook" in raw, raw
