"""Gemma 4 auto turns preserve parallel calls through the serving grammar.

The scripted reply supplies native call blocks on a tiny fixture, including
control tokens. This tests grammar and response framing, not model quality.
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": name, "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
    for name in ("add", "multiply")]
ONE = "<|tool_call>call:add{a:1,b:2}<tool_call|>"
TWO = ONE + "<|tool_call>call:multiply{a:3,b:4}<tool_call|>"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("gemma4-auto") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--control", '<|tool_call>,<tool_call|>,<turn|>,<|tool_response>',
                    str(path)], check=True, stdout=subprocess.DEVNULL)
    return path


@pytest.fixture(scope="module", params=["gemma4", "gemma4-mainline"])
def server(request, runner_bin, model):
    with RunnerServer(runner_bin, model, ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", request.param],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as srv:
        yield srv


def _chat(server, stream, parallel):
    payload = {"messages": [{"role": "user", "content": "Add and multiply."}],
               "tools": TOOLS, "tool_choice": "auto", "max_tokens": 256,
               "chat_template_kwargs": {"enable_thinking": False},
               "runner_test_reply": (ONE if parallel is False else TWO) + "<|tool_response>",
               "stream": stream}
    if parallel is not None:
        payload["parallel_tool_calls"] = parallel
    req = urllib.request.Request(server.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        body = response.read()
    if not stream:
        return json.loads(body)["choices"][0]
    calls, finish, content = {}, None, ""
    assert body.rstrip().endswith(b"data: [DONE]")
    for line in body.decode().splitlines():
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices", []):
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call["index"], {
                    "id": call.get("id"), "function": {"name": "", "arguments": ""}})
                for key, value in call.get("function", {}).items():
                    if key in entry["function"]:
                        entry["function"][key] += value or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {"finish_reason": finish, "message": {
        "content": content, "tool_calls": [calls[i] for i in sorted(calls)]}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [None, True, False])
def test_auto_turn_honors_the_parallel_envelope(server, stream, parallel):
    choice = _chat(server, stream, parallel)
    calls = choice["message"].get("tool_calls") or []
    expected = [("add", {"a": 1, "b": 2})]
    if parallel is not False:
        expected.append(("multiply", {"a": 3, "b": 4}))
    assert [(call["function"]["name"],
             json.loads(call["function"]["arguments"]))
            for call in calls] == expected, choice
    assert len({call["id"] for call in calls}) == len(expected)
    assert all(call["id"] for call in calls)
    assert choice["finish_reason"] == "tool_calls", choice
    assert not (choice["message"].get("content") or ""), choice
