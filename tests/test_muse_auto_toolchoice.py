"""Muse ATEM tool calling under tool_choice auto, on the model's own stream.

The lab's finding (2026-09-16): on Muse-Glimmer-30B, `tool_choice: "required"`
produces the call, but the default auto turn ends after the recipient header
with no call, and with `enable_thinking: true` the turn errors at the
`<|eom|><|start|>assistant to=get_weather` transition after the self turn.
The model's own raw stream for that prompt (greedy /v1/completions) is:

    ` to=self<|message|>{reasoning}<|eom|><|start|>assistant to=get_weather<|message|>`
    `<atem:function_calls>...</atem:function_calls><|eot|>`

The fixture is the CI Muse fixture carrying the family's four turn markers
as CONTROL tokens (they decode to no bytes, which is what makes this hard),
and the scripted-reply hook dictates that exact stream. Under test is the
served path: the grammar under auto, the eom/start transition, the header
strip, the reasoning split and the parsed call. Model quality is not.
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

TOOLS = [{"type": "function", "function": {
    "name": name, "description": f"{name} for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
    for name in ("get_weather", "get_forecast", "get_time", "get_news")]

CALL = ("<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n"
        "<atem:parameter name=\"city\">Sydney</atem:parameter>\n</atem:invoke>\n"
        "</atem:function_calls>")
REASON = "Compare the two cities, weather first."
# the stream shapes the model produces, spelled exactly
DIRECT_CALL = " to=get_weather<|message|>" + CALL + "<|eot|>"
SELF_THEN_CALL = (" to=self<|message|>" + REASON + "<|eom|>"
                  "<|start|>assistant to=get_weather<|message|>" + CALL + "<|eot|>")
AFTER_PRIMED_SELF = REASON + "<|eom|><|start|>assistant to=get_weather<|message|>" + CALL + "<|eot|>"
DIRECT_ANSWER = " to=user<|message|>Sydney is warmer today.<|eot|>"
SELF_THEN_ANSWER = (" to=self<|message|>" + REASON + "<|eom|>"
                    "<|start|>assistant to=user<|message|>Sydney is warmer today.<|eot|>")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("muse") / "muse.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--muse-glimmer",
                    "--control", "<|start|>,<|message|>,<|eom|>,<|eot|>", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.fixture(scope="module")
def muse(model):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    with RunnerServer(exe, model, ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _chat(server, reply, stream=False, **extra):
    payload = {"model": server.model_id, "max_tokens": 400, "temperature": 0,
               "messages": [{"role": "user", "content": "Which is warmer today, Sydney or Oslo?"}],
               "tools": TOOLS, "runner_test_reply": reply}
    payload.update(extra)
    if stream:
        payload["stream"] = True
    req = urllib.request.Request(server.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not stream:
        return json.loads(body)
    # fold the SSE stream into one buffered-shaped response
    content, reasoning, finish, calls = "", "", None, {}
    for line in body.decode().splitlines():
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices", []):
            delta = ch.get("delta", {})
            content += delta.get("content", "") or ""
            reasoning += delta.get("reasoning_content", "") or ""
            for tc in delta.get("tool_calls", []) or []:
                c = calls.setdefault(tc["index"], {"id": tc.get("id"), "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                fn = tc.get("function", {})
                if fn.get("name"):
                    c["function"]["name"] = fn["name"]
                c["function"]["arguments"] += fn.get("arguments", "") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": content,
           "tool_calls": [calls[i] for i in sorted(calls)] or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}], "streamed": True}


def _assert_call(d, reasoning=None):
    ch = d["choices"][0]
    msg = ch["message"]
    assert ch["finish_reason"] == "tool_calls", d
    calls = msg.get("tool_calls") or []
    assert [c["function"]["name"] for c in calls] == ["get_weather"], d
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Sydney"}, d
    assert not (msg.get("content") or ""), d
    if reasoning is not None:
        assert msg.get("reasoning_content") == reasoning, d
    for marker in ("to=", "<|message|>", "<atem:", "<|eom|>", "<|start|>"):
        assert marker not in (msg.get("content") or ""), d
        assert marker not in (msg.get("reasoning_content") or ""), d


def test_required_direct_call_is_the_control(muse):
    _assert_call(_chat(muse, DIRECT_CALL, tool_choice="required"))


def test_auto_direct_call(muse):
    """The lab's default case: auto, thinking unspecified, the model picks
    the tool as its recipient and writes the block."""
    _assert_call(_chat(muse, DIRECT_CALL))


def test_auto_self_turn_then_call(muse):
    """The model's own stream on that prompt: a self-addressed reasoning
    turn closed by <|eom|>, then the call in a new assistant turn."""
    _assert_call(_chat(muse, SELF_THEN_CALL), reasoning=REASON)


def test_thinking_on_primed_self_turn_then_call(muse):
    """enable_thinking true: the prompt already opened the self turn, the
    model closes it with <|eom|> and calls in the next assistant turn."""
    _assert_call(_chat(muse, AFTER_PRIMED_SELF, enable_thinking=True), reasoning=REASON)


def test_auto_direct_answer_to_user(muse):
    d = _chat(muse, DIRECT_ANSWER)
    ch = d["choices"][0]
    assert ch["finish_reason"] == "stop" and not ch["message"].get("tool_calls"), d
    assert ch["message"]["content"] == "Sydney is warmer today.", d


def test_auto_self_turn_then_answer(muse):
    d = _chat(muse, SELF_THEN_ANSWER)
    ch = d["choices"][0]
    assert ch["finish_reason"] == "stop" and not ch["message"].get("tool_calls"), d
    assert ch["message"]["content"] == "Sydney is warmer today.", d
    assert ch["message"].get("reasoning_content") == REASON, d


# every agent client streams: the same four shapes through the SSE path
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_auto_shapes_agree_across_transports(muse, stream):
    _assert_call(_chat(muse, DIRECT_CALL, stream=stream))
    _assert_call(_chat(muse, SELF_THEN_CALL, stream=stream), reasoning=REASON)
    _assert_call(_chat(muse, AFTER_PRIMED_SELF, stream=stream, enable_thinking=True), reasoning=REASON)
    d = _chat(muse, SELF_THEN_ANSWER, stream=stream)
    ch = d["choices"][0]
    assert ch["finish_reason"] == "stop" and not ch["message"].get("tool_calls"), d
    assert ch["message"]["content"] == "Sydney is warmer today.", d
    assert ch["message"].get("reasoning_content") == REASON, d


def test_required_with_thinking_calls_the_tool_the_reasoning_named(muse):
    """The lab's fifth row: required plus enable_thinking true used to return
    the FIRST declared tool with an argument lifted from the reasoning, the
    grammar having rejected `assistant` after <|eom|> and the sampler taking
    the first admissible name. Declared order puts get_forecast first here."""
    tools = [TOOLS[1], TOOLS[0], TOOLS[2], TOOLS[3]]
    _assert_call(_chat(muse, AFTER_PRIMED_SELF, tool_choice="required", enable_thinking=True,
                       tools=tools), reasoning=REASON)
