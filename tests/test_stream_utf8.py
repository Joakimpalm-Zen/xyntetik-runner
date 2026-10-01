"""A character is never split across stream deltas.

Characters do not respect token boundaries: a model can emit the UTF-8 bytes
of one character as separate tokens, and the fixture's vocabulary is bytes, so
here every non-ASCII character is split exactly that way. A delta that carries
a lone byte escapes it to U+FFFD, and the client assembles well-formed JSON
holding the wrong text. Content deltas have held an unfinished tail since
v0.3; the streamed tool-call ARGUMENTS of the generic JSON envelope did not
(v0.5.7, found by a byte capture of granite-4.1-3b writing "Åsa": the wire
carried two U+FFFD then "sa").

The anchor is the buffered response of the same scripted turn, and the text
itself.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

TEXT = "Åsa gick ut på isen, 日本 🙂 naïve"
PARAMS = {"type": "object", "properties": {"city": {"type": "string"}},
          "required": ["city"]}
# the generic envelope, compact: the grammar admits no insignificant space
CALL = json.dumps({"tool": "get_weather", "args": {"city": TEXT}},
                  ensure_ascii=False, separators=(",", ":"))


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    m = tmp_path_factory.mktemp("u8") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    env = dict(os.environ, RUNNER_TEST_SCRIPTED_REPLY="1")
    # granite: a family whose tool calls are the generic JSON envelope
    with RunnerServer(exe, m, ctx=4096, env=env, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "granite"]) as s:
        yield s


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if not payload.get("stream"):
        return json.loads(raw)
    # the wire bytes must be UTF-8 as they are, before any JSON decoding
    return [json.loads(l[6:]) for l in raw.decode("utf-8").splitlines()
            if l.startswith("data: ") and l[6:] != "[DONE]"]


def _chat(srv, stream, tools):
    p = {"max_tokens": 400, "stream": stream,
         "messages": [{"role": "user", "content": "Weather?"}],
         "runner_test_reply": CALL if tools else TEXT}
    if tools:
        p["tools"] = [{"type": "function", "function": {
            "name": "get_weather", "parameters": PARAMS}}]
        p["tool_choice"] = "required"
    d = _post(srv, "/v1/chat/completions", p)
    if not stream:
        msg = d["choices"][0]["message"]
        return msg["tool_calls"][0]["function"]["arguments"] if tools \
            else msg["content"]
    out = ""
    for e in d:
        for c in e.get("choices", []):
            if tools:
                for t in c["delta"].get("tool_calls") or []:
                    out += t["function"].get("arguments", "")
            else:
                out += c["delta"].get("content") or ""
    return out


def _responses(srv, stream, tools):
    p = {"max_output_tokens": 400, "stream": stream, "input": "Weather?",
         "runner_test_reply": CALL if tools else TEXT}
    if tools:
        p["tools"] = [{"type": "function", "name": "get_weather",
                       "parameters": PARAMS}]
        p["tool_choice"] = "required"
    d = _post(srv, "/v1/responses", p)
    if not stream:
        for o in d["output"]:
            if tools and o["type"] == "function_call":
                return o["arguments"]
            if not tools and o["type"] == "message":
                return "".join(c["text"] for c in o["content"])
        raise AssertionError(d)
    kind = "response.function_call_arguments.delta" if tools \
        else "response.output_text.delta"
    return "".join(e["delta"] for e in d if e.get("type") == kind)


def _messages(srv, stream, tools):
    p = {"max_tokens": 400, "stream": stream,
         "messages": [{"role": "user", "content": "Weather?"}],
         "runner_test_reply": CALL if tools else TEXT}
    if tools:
        p["tools"] = [{"name": "get_weather", "input_schema": PARAMS}]
        p["tool_choice"] = {"type": "any"}
    d = _post(srv, "/v1/messages", p)
    if not stream:
        for b in d["content"]:
            if tools and b["type"] == "tool_use":
                return json.dumps(b["input"], ensure_ascii=False,
                                  separators=(",", ":"))
            if not tools and b["type"] == "text":
                return b["text"]
        raise AssertionError(d)
    key = ("input_json_delta", "partial_json") if tools else ("text_delta", "text")
    return "".join(e["delta"][key[1]] for e in d
                   if e.get("type") == "content_block_delta"
                   and e["delta"].get("type") == key[0])


SURFACES = {"chat": _chat, "responses": _responses, "messages": _messages}


@pytest.mark.parametrize("tools", [True, False], ids=["tool-arguments", "content"])
@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_a_streamed_character_is_whole(srv, surface, tools):
    fn = SURFACES[surface]
    buffered, streamed = fn(srv, False, tools), fn(srv, True, tools)
    want = {"city": TEXT} if tools else TEXT
    got_b = json.loads(buffered) if tools else buffered
    got_s = json.loads(streamed) if tools else streamed
    assert got_b == want, buffered
    assert "�" not in streamed, streamed
    assert got_s == want, streamed


def test_a_native_protocol_call_streams_whole_characters(tmp_path):
    """The same hole on a native tool protocol (gemma4's own call syntax,
    v0.5.7: "f\\ufffd\\ufffdljde" for "följde" in a streamed argument): the
    argument sink is shared, so the hold covers every protocol."""
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    m = tmp_path / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    env = dict(os.environ, RUNNER_TEST_SCRIPTED_REPLY="1")
    reply = '<|tool_call>call:get_weather{city:<|"|>' + TEXT + '<|"|>}<tool_call|>'
    with RunnerServer(exe, m, ctx=4096, env=env, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", "gemma4"]) as s:
        p = {"max_tokens": 400, "messages": [{"role": "user", "content": "Weather?"}],
             "tools": [{"type": "function", "function": {
                 "name": "get_weather", "parameters": PARAMS}}],
             "runner_test_reply": reply}
        d = _post(s, "/v1/chat/completions", p)
        buffered = d["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        ev = _post(s, "/v1/chat/completions", {**p, "stream": True})
        streamed = "".join(t["function"].get("arguments", "") for e in ev
                           for c in e.get("choices", [])
                           for t in (c["delta"].get("tool_calls") or []))
    assert json.loads(buffered) == {"city": TEXT}, buffered
    assert "�" not in streamed and json.loads(streamed) == {"city": TEXT}, streamed


