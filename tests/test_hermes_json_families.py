"""Hermes 4 and Granite 4 speak the Hermes JSON tool call the runner already
parsed for Qwen2.5: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.

The renderers are held to the publishers' templates by the conformance gate.
This holds the other half: a call written in that syntax by a model served
under either family comes back as a structured tool call, buffered and
streamed, with the same arguments, and a plain answer stays an answer.
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
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
# The scripted reply has to be a sequence the turn's grammar admits token for
# token (the hook dictates tokens, it does not sample), so the arguments are
# written without optional whitespace.
CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city":"Oslo"}}\n</tool_call>'


@pytest.fixture(scope="module", params=["hermes4", "granite4"])
def srv(request, tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    m = tmp_path_factory.mktemp(request.param) / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, m, ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", request.param],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as s:
        yield s


def _post(srv, payload):
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    if not payload.get("stream"):
        return json.loads(body)
    return [json.loads(line[6:]) for line in body.decode().splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"]


def _payload(reply, **extra):
    return {"max_tokens": 200, "tools": [WEATHER],
            "messages": [{"role": "user", "content": "Weather in Oslo?"}],
            "runner_test_reply": reply, **extra}


def test_a_native_call_is_a_structured_call(srv):
    d = _post(srv, _payload(CALL))
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", d
    calls = ch["message"]["tool_calls"]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Oslo"}
    assert not (ch["message"].get("content") or "").strip()


def test_the_streamed_call_carries_the_same_arguments(srv):
    chunks = _post(srv, _payload(CALL, stream=True))
    name, args, finish = "", "", None
    for c in chunks:
        for ch in c.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                f = tc.get("function") or {}
                name += f.get("name") or ""
                args += f.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    assert finish == "tool_calls", chunks
    assert name == "get_weather" and json.loads(args) == {"city": "Oslo"}


def test_a_plain_answer_stays_content(srv):
    d = _post(srv, _payload("It is cold in Oslo."))
    ch = d["choices"][0]
    assert ch["finish_reason"] == "stop", d
    assert "cold" in ch["message"]["content"]
    assert not ch["message"].get("tool_calls")
