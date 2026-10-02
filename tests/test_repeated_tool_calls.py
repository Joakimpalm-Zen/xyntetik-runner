"""A tool call that repeats one already made in the conversation is reported,
not refused (R4.12.19).

The lab's sixth runaway: the model called get_weather(Osaka), then
get_weather(Athens), stated both answers in its reasoning, and called
get_weather(Athens) again, turn after turn. Every request was clean on its
own; the loop exists only ACROSS requests, where the runner is stateless. But
the chat-shaped surfaces receive the earlier calls as structure (Chat
`tool_calls`, Responses `function_call` items, Messages `tool_use` blocks), so
a call this turn emits can be matched against them without parsing prose.
A legitimate agent re-calls a tool too (polling, a retry after an error), so
the server says so in runner_telemetry and changes nothing about the turn.

The expected flags are fixed by the test's own construction: the earlier calls
are in the request, the new one is dictated through the scripted-reply hook.
Arguments are compared as JSON values, so key order and whitespace do not
hide a repeat and a different value is never one.
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

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"},
                                  "unit": {"type": "string"}},
                   "required": ["city"]}}}
RESP_WEATHER = {"type": "function", "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": WEATHER["function"]["parameters"]}
ANTH_WEATHER = {"name": "get_weather",
                "description": "Current weather for a city",
                "input_schema": WEATHER["function"]["parameters"]}
NO_THINK = {"enable_thinking": False}


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    m = tmp_path_factory.mktemp("rep") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, m, ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", "qwen38"],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as s:
        yield s


def _post(srv, path, payload, stream=False):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not stream:
        return json.loads(body)
    return [json.loads(line[6:]) for line in body.decode().splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"]


def _chat_history():
    """Osaka, then Athens: the lab transcript's first two calls."""
    def call(i, city):
        return {"id": f"call_{i}", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": city})}}
    return [
        {"role": "user", "content": "Which is warmer, Osaka or Athens?"},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "Osaka")]},
        {"role": "tool", "tool_call_id": "call_0", "content": "-1C"},
        {"role": "assistant", "content": "", "tool_calls": [call(1, "Athens")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "28C"},
    ]


def _chat(srv, reply):
    return _post(srv, "/v1/chat/completions", {
        "max_tokens": 400, "messages": _chat_history(), "tools": [WEATHER],
        "chat_template_kwargs": NO_THINK, "runner_test_reply": reply})


def test_chat_flags_the_repeat_and_serves_the_turn_unchanged(srv):
    d = _chat(srv, _call("get_weather", city="Athens"))
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", d
    calls = ch["message"]["tool_calls"]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == \
        [{"city": "Athens"}]
    assert d["runner_telemetry"]["repeated_tool_calls"] == [
        {"index": 0, "name": "get_weather", "prior_calls": 1,
         "last_message_index": 3}]


def test_a_new_call_is_not_flagged(srv):
    for reply in (_call("get_weather", city="Tokyo"),
                  _call("get_weather", city="Athens", unit="F")):
        d = _chat(srv, reply)
        assert d["choices"][0]["finish_reason"] == "tool_calls", d
        assert "repeated_tool_calls" not in d["runner_telemetry"], d


def test_only_the_repeated_call_of_several_is_flagged(srv):
    d = _chat(srv, _call("get_weather", city="Tokyo") + "\n" +
              _call("get_weather", city="Osaka"))
    assert len(d["choices"][0]["message"]["tool_calls"]) == 2, d
    assert d["runner_telemetry"]["repeated_tool_calls"] == [
        {"index": 1, "name": "get_weather", "prior_calls": 1,
         "last_message_index": 1}]


def test_argument_spelling_does_not_hide_a_repeat(srv):
    msgs = _chat_history()
    # the earlier call written with extra whitespace and another key order
    msgs[3]["tool_calls"][0]["function"]["arguments"] = \
        '{ "unit" : "C",  "city":"Athens" }'
    d = _post(srv, "/v1/chat/completions", {
        "max_tokens": 400, "messages": msgs, "tools": [WEATHER],
        "chat_template_kwargs": NO_THINK,
        "runner_test_reply": _call("get_weather", city="Athens", unit="C")})
    assert d["runner_telemetry"]["repeated_tool_calls"][0]["prior_calls"] == 1


def _responses_input():
    return [
        {"role": "user", "content": "Which is warmer, Osaka or Athens?"},
        {"type": "function_call", "call_id": "c0", "name": "get_weather",
         "arguments": json.dumps({"city": "Osaka"})},
        {"type": "function_call_output", "call_id": "c0", "output": "-1C"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather",
         "arguments": json.dumps({"city": "Athens"})},
        {"type": "function_call_output", "call_id": "c1", "output": "28C"},
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_responses_flags_the_repeat(srv, stream):
    payload = {"input": _responses_input(), "tools": [RESP_WEATHER],
               "max_output_tokens": 400, "stream": stream,
               "chat_template_kwargs": NO_THINK,
               "runner_test_reply": _call("get_weather", city="Osaka")}
    if stream:
        events = _post(srv, "/v1/responses", payload, stream=True)
        done = [e for e in events if e.get("type") == "response.completed"]
        assert len(done) == 1, events
        body = done[0]["response"]
    else:
        body = _post(srv, "/v1/responses", payload)
    calls = [o for o in body["output"] if o["type"] == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "get_weather", body
    assert body["runner_telemetry"]["repeated_tool_calls"] == [
        {"index": 0, "name": "get_weather", "prior_calls": 1,
         "last_message_index": 1}]


def test_messages_flags_the_repeat(srv):
    msgs = [
        {"role": "user", "content": "Which is warmer, Osaka or Athens?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t0", "name": "get_weather",
             "input": {"city": "Osaka"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t0", "content": "-1C"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "get_weather",
             "input": {"city": "Athens"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "28C"}]},
    ]
    d = _post(srv, "/v1/messages", {
        "max_tokens": 400, "messages": msgs, "tools": [ANTH_WEATHER],
        "chat_template_kwargs": NO_THINK,
        "runner_test_reply": _call("get_weather", city="Athens")})
    assert d["stop_reason"] == "tool_use", d
    assert d["runner_telemetry"]["repeated_tool_calls"] == [
        {"index": 0, "name": "get_weather", "prior_calls": 1,
         "last_message_index": 3}]


def test_streamed_chat_flags_the_repeat_on_the_closing_chunk(srv):
    chunks = _post(srv, "/v1/chat/completions", {
        "max_tokens": 400, "messages": _chat_history(), "tools": [WEATHER],
        "chat_template_kwargs": NO_THINK, "stream": True,
        "runner_test_reply": _call("get_weather", city="Athens")}, stream=True)
    last = [c for c in chunks if c.get("choices") and
            c["choices"][0].get("finish_reason")]
    assert len(last) == 1 and \
        last[0]["choices"][0]["finish_reason"] == "tool_calls", chunks
    assert last[0]["runner_telemetry"]["repeated_tool_calls"] == [
        {"index": 0, "name": "get_weather", "prior_calls": 1,
         "last_message_index": 3}]


def test_streamed_chat_says_nothing_about_a_new_call(srv):
    chunks = _post(srv, "/v1/chat/completions", {
        "max_tokens": 400, "messages": _chat_history(), "tools": [WEATHER],
        "chat_template_kwargs": NO_THINK, "stream": True,
        "runner_test_reply": _call("get_weather", city="Tokyo")}, stream=True)
    # every streamed finish chunk carries the turn's telemetry since RI-4;
    # a call that repeats nothing is simply not listed in it
    assert all("repeated_tool_calls" not in c.get("runner_telemetry", {})
               for c in chunks), chunks


def test_streamed_messages_flags_the_repeat_on_message_delta(srv):
    msgs = [
        {"role": "user", "content": "Which is warmer, Osaka or Athens?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t0", "name": "get_weather",
             "input": {"city": "Osaka"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t0", "content": "-1C"}]},
    ]
    events = _post(srv, "/v1/messages", {
        "max_tokens": 400, "messages": msgs, "tools": [ANTH_WEATHER],
        "chat_template_kwargs": NO_THINK, "stream": True,
        "runner_test_reply": _call("get_weather", city="Osaka")}, stream=True)
    delta = [e for e in events if e.get("type") == "message_delta"]
    assert len(delta) == 1 and \
        delta[0]["delta"]["stop_reason"] == "tool_use", events
    assert delta[0]["runner_telemetry"]["repeated_tool_calls"] == [
        {"index": 0, "name": "get_weather", "prior_calls": 1,
         "last_message_index": 1}]
