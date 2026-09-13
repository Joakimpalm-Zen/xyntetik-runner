"""Qwen3-Coder's function/parameter XML, end to end over HTTP.

The fixture model knows nothing about tools; constrained decoding makes it
produce a grammatical native call anyway, so a forced `qwen3-coder` template
exercises the whole path a real checkpoint takes: the declarations rendered
into the system turn, the XML call grammar, the parser that turns the XML
back into the OpenAI `tool_calls` JSON (declared types: a string argument
stays a string), on the Chat, Responses and Anthropic surfaces, buffered
and streamed. The model's choice of argument bytes is its own; the shape of
what the client receives is the contract. A raw string value is unbounded
for a model that never chooses to close it, so the fixture often runs into
the token limit: the documented truncation contract then applies (the call
is closed to a legal document and the turn reports `length`), and both
finish reasons are accepted here as long as the call parses.
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402

TOOL = {"type": "function", "function": {
    "name": "save_note",
    "description": "Save a note",
    "parameters": {"type": "object",
                   "properties": {"text": {"type": "string", "description": "the note"},
                                  "count": {"type": "integer"}},
                   "required": ["text"]}}}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("coder") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def server(model):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    with RunnerServer(exe, model, ctx=8192, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", "qwen3-coder"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _post(server, path, payload, stream=False):
    req = urllib.request.Request(server.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not stream:
        return json.loads(body)
    events = []
    for line in body.decode().splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
        elif line.startswith("data: "):
            events.append("[DONE]")
    return events


def _check_call(name, arguments):
    assert name == "save_note", name
    args = json.loads(arguments)
    assert isinstance(args, dict) and isinstance(args.get("text"), str), args
    if "count" in args:
        assert isinstance(args["count"], int), args


def test_chat_required_call_round_trips(server):
    d = _post(server, "/v1/chat/completions", {
        "model": server.model_id, "messages": [{"role": "user", "content": "save the note"}],
        "tools": [TOOL], "tool_choice": "required", "max_tokens": 256, "temperature": 0})
    msg = d["choices"][0]["message"]
    assert d["choices"][0]["finish_reason"] in ("tool_calls", "length"), d
    assert msg["tool_calls"], d
    _check_call(msg["tool_calls"][0]["function"]["name"],
                msg["tool_calls"][0]["function"]["arguments"])
    # the native framing never leaks into content
    assert not (msg.get("content") or "").strip(), msg


def test_chat_streams_the_same_call(server):
    events = _post(server, "/v1/chat/completions", {
        "model": server.model_id, "messages": [{"role": "user", "content": "save the note"}],
        "tools": [TOOL], "tool_choice": "required", "max_tokens": 256,
        "temperature": 0, "stream": True}, stream=True)
    assert events and events[-1] == "[DONE]"
    name, args, content, finish = None, "", "", None
    for e in events[:-1]:
        for ch in e.get("choices", []):
            delta = ch.get("delta", {})
            for tc in delta.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                if fn.get("name"):
                    name = fn["name"]
                args += fn.get("arguments", "") or ""
            content += delta.get("content", "") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    assert finish in ("tool_calls", "length"), events
    _check_call(name, args)
    assert "<tool_call>" not in content and "<function=" not in content, content


def test_responses_required_call_round_trips(server):
    d = _post(server, "/v1/responses", {
        "model": server.model_id, "input": "save the note",
        "tools": [{"type": "function", "name": "save_note",
                   "description": "Save a note",
                   "parameters": TOOL["function"]["parameters"]}],
        "tool_choice": "required", "max_output_tokens": 256, "temperature": 0})
    calls = [o for o in d["output"] if o.get("type") == "function_call"]
    assert calls, d
    _check_call(calls[0]["name"], calls[0]["arguments"])


def test_messages_any_call_round_trips(server):
    d = _post(server, "/v1/messages", {
        "model": server.model_id, "max_tokens": 256, "temperature": 0,
        "messages": [{"role": "user", "content": "save the note"}],
        "tools": [{"name": "save_note", "description": "Save a note",
                   "input_schema": TOOL["function"]["parameters"]}],
        "tool_choice": {"type": "any"}})
    uses = [b for b in d["content"] if b.get("type") == "tool_use"]
    assert uses, d
    assert d["stop_reason"] in ("tool_use", "max_tokens"), d
    assert uses[0]["name"] == "save_note"
    assert isinstance(uses[0]["input"], dict) and isinstance(uses[0]["input"].get("text"), str)


def test_auto_mode_is_well_formed_either_way(server):
    """tool_choice auto lets the model answer in prose or call; whichever the
    fixture does, the client gets one well-formed shape and no native
    framing in the content."""
    d = _post(server, "/v1/chat/completions", {
        "model": server.model_id, "messages": [{"role": "user", "content": "hello"}],
        "tools": [TOOL], "max_tokens": 48, "temperature": 0})
    ch = d["choices"][0]
    msg = ch["message"]
    assert ch["finish_reason"] in ("stop", "tool_calls", "length"), d
    if msg.get("tool_calls"):
        _check_call(msg["tool_calls"][0]["function"]["name"],
                    msg["tool_calls"][0]["function"]["arguments"])
    content = msg.get("content") or ""
    assert "<tool_call>" not in content and "</function>" not in content, content
