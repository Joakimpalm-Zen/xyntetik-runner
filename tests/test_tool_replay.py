"""A served tool grammar is rebuilt from its receipt, not its output (R1.1.3.1)."""
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "pick", "parameters": {
    "type": "object", "properties": {"value": {"type": "integer", "enum": [7]}},
    "required": ["value"], "additionalProperties": False}}}]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("tool-replay") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _verify(runner_bin, model, path, *extra):
    p = subprocess.run([runner_bin, "-m", model, "--verify", path, "--gpu", "off",
                        "-t", "2", "-c", "4096", *extra], capture_output=True, timeout=120)
    return p.returncode, p.stderr.decode(errors="replace")


def _serve_turn(runner_bin, model, d, template="qwen38", surface="chat", **extra):
    with RunnerServer(runner_bin, model, ctx=4096, extra_args=[
            "--gpu", "off", "-t", "2", "--chat-template", template,
            "--receipts", str(d)]) as srv:
        payload = {"messages": [{"role": "user", "content": "Pick a value."}],
                   "tools": TOOLS, "tool_choice": "required", "parallel_tool_calls": False,
                   "max_tokens": 160, "temperature": 0, "seed": 42, "cache_prompt": False,
                   "chat_template_kwargs": {"enable_thinking": False}, **extra}
        if surface == "responses":
            payload["input"] = payload.pop("messages")
            payload["max_output_tokens"] = payload.pop("max_tokens")
            payload["tools"] = [dict(type="function", **t["function"]) for t in TOOLS]
        elif surface == "messages":
            payload["tools"] = [{"name": t["function"]["name"],
                                 "input_schema": t["function"]["parameters"]} for t in TOOLS]
            payload["tool_choice"] = {"type": "any"}
        route = {"chat": "/v1/chat/completions", "responses": "/v1/responses",
                 "messages": "/v1/messages"}[surface]
        req = urllib.request.Request(srv.base_url + route, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if payload.get("stream"):
                calls = {}
                for line in r:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    for choice in json.loads(line[6:]).get("choices", []):
                        for call in choice.get("delta", {}).get("tool_calls", []):
                            f = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                            for k in f:
                                f[k] += call.get("function", {}).get(k, "") or ""
                body = {"choices": [{"message": {"tool_calls": [
                    {"function": calls[i]} for i in sorted(calls)]}}]}
            else:
                body = json.load(r)
        if surface != "chat":
            # Normalize the response's calls for the same public assertion.
            if surface == "responses":
                calls = [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                         for c in body["output"] if c["type"] == "function_call"]
            else:
                calls = [{"function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                         for c in body["content"] if c["type"] == "tool_use"]
            body["choices"] = [{"message": {"tool_calls": calls}}]
    path = sorted(d.iterdir())[-1]
    return path, json.loads(path.read_text()), body


@pytest.fixture(scope="module")
def turn(runner_bin, model, tmp_path_factory):
    return _serve_turn(runner_bin, model, tmp_path_factory.mktemp("tool-receipts"))


def _rewrite(path, out, change):
    """An unsigned receipt from an older or altered writer, with a valid chain."""
    raw = path.read_text()
    cut = raw.rindex(',"chain":')
    body = json.loads(raw[:cut] + "}")
    change(body["constraints"][0])
    head = json.dumps(body, separators=(",", ":"))[:-1]
    out.write_text(head + ',"chain":' + json.dumps(
        {"algo": "sha256", "prev": json.loads(raw)["chain"]["prev"],
         "hash": hashlib.sha256(head.encode()).hexdigest()}, separators=(",", ":")) + "}")
    return out


def test_a_served_tool_turn_replays(runner_bin, model, turn):
    path, rec, body = turn
    # Absolute anchor: the caller's singleton enum must be obeyed by an actual
    # generated call. No scripted reply, and no trust in the receipt's output.
    call, = body["choices"][0]["message"]["tool_calls"]
    assert call["function"]["name"] == "pick"
    assert json.loads(call["function"]["arguments"]) == {"value": 7}
    assert rec["tool_calls"] == [{"name": "pick", "arguments": call["function"]["arguments"]}]
    rc, out = _verify(runner_bin, model, path)
    assert rc == 0 and "VERIFIED" in out, out[-1000:]

    constraint, = rec["constraints"]
    assert constraint == {"kind": "tools", "sha256": hashlib.sha256(
        json.dumps(TOOLS, separators=(",", ":")).encode()).hexdigest(),
        "choice": "required", "declarations": TOOLS, "parallel": False,
        "protocol": "qwen3_xml", "thinking": False, "reasoning_open": False}


@pytest.mark.parametrize("template,extra,protocol", [
    ("llama3", {}, "generic"),
    ("chatml", {}, "qwen_json"),
    ("qwen38", {"parallel_tool_calls": True}, "qwen3_xml"),
    ("chatml", {"tool_choice": {"type": "function", "function": {"name": "pick"}}},
     "qwen_json"),
    ("harmony", {}, "harmony"),
    ("gemma4", {}, "gemma4"),
    ("gemma4", {"chat_template_kwargs": {"enable_thinking": True}}, "gemma4"),
])
def test_effective_tool_grammar_replays(runner_bin, model, tmp_path, template, extra, protocol):
    path, rec, body = _serve_turn(runner_bin, model, tmp_path / "r", template, **extra)
    assert rec["constraints"][0]["protocol"] == protocol
    calls = body["choices"][0]["message"]["tool_calls"]
    assert calls
    for call in calls:
        assert call["function"]["name"] == "pick"
        assert json.loads(call["function"]["arguments"]) == {"value": 7}
    rc, out = _verify(runner_bin, model, path)
    assert rc == 0 and "VERIFIED" in out, out[-1000:]


@pytest.mark.parametrize("field", ["declarations", "parallel", "protocol", "thinking", "reasoning_open"])
def test_older_tool_receipts_name_what_is_missing(runner_bin, model, turn, tmp_path, field):
    path = _rewrite(turn[0], tmp_path / "old.json", lambda c: c.pop(field))
    rc, out = _verify(runner_bin, model, path)
    assert rc == 3 and "UNVERIFIABLE" in out and field in out, out[-1000:]
    assert "malformed" not in out and "DIVERGED" not in out


def test_altered_tool_declarations_are_not_trusted(runner_bin, model, turn, tmp_path):
    def change(c, rehash):
        c["declarations"][0]["function"]["parameters"]["properties"]["value"]["enum"] = [8]
        if rehash:
            c["sha256"] = hashlib.sha256(json.dumps(
                c["declarations"], separators=(",", ":")).encode()).hexdigest()
    stale = _rewrite(turn[0], tmp_path / "stale.json", lambda c: change(c, False))
    rc, out = _verify(runner_bin, model, stale)
    assert rc == 3 and "UNVERIFIABLE" in out and "sha256" in out, out[-1000:]
    altered = _rewrite(turn[0], tmp_path / "altered.json", lambda c: change(c, True))
    rc, out = _verify(runner_bin, model, altered)
    assert rc == 2 and "DIVERGED" in out, out[-1000:]


@pytest.mark.parametrize("field,value,reason", [
    ("choice", "none", "choice"), ("name", "absent", "name"),
    ("protocol", "unknown", "protocol"), ("parallel", "false", "malformed"),
    ("thinking", 1, "malformed"), ("reasoning_open", 1, "malformed"),
])
def test_invalid_tool_settings_are_refused(runner_bin, model, turn, tmp_path, field, value, reason):
    def change(c):
        if field == "name":
            c["choice"] = "named"
        c[field] = value
    path = _rewrite(turn[0], tmp_path / "bad.json", change)
    rc, out = _verify(runner_bin, model, path)
    assert rc == 3 and "UNVERIFIABLE" in out and reason in out, out[-1000:]


@pytest.mark.parametrize("surface", ["responses", "messages"])
def test_translated_declarations_replay_on_generic_surface(runner_bin, model, tmp_path, surface):
    path, rec, body = _serve_turn(runner_bin, model, tmp_path / "r", "llama3", surface)
    assert rec["constraints"][0]["protocol"] == "generic"
    calls = body["choices"][0]["message"]["tool_calls"]
    assert calls
    for call in calls:
        assert call["function"]["name"] == "pick"
        assert json.loads(call["function"]["arguments"]) == {"value": 7}
    rc, out = _verify(runner_bin, model, path)
    assert rc == 0 and "VERIFIED" in out, out[-1000:]


def test_streamed_tool_turn_replays(runner_bin, model, tmp_path):
    path, rec, body = _serve_turn(runner_bin, model, tmp_path / "r", "llama3", stream=True)
    call, = body["choices"][0]["message"]["tool_calls"]
    assert call["function"]["name"] == "pick"
    assert json.loads(call["function"]["arguments"]) == {"value": 7}
    rc, out = _verify(runner_bin, model, path)
    assert rc == 0 and "VERIFIED" in out, out[-1000:]


@pytest.mark.parametrize("template", ["chatml", "gemma4"])
def test_auto_tools_with_a_response_schema_replay(runner_bin, model, tmp_path, template):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean", "enum": [True]}},
              "required": ["ok"], "additionalProperties": False}
    path, rec, body = _serve_turn(runner_bin, model, tmp_path / "r", template,
        tool_choice="auto", chat_template_kwargs={"enable_thinking": True}, response_format={"type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema}})
    assert [c["kind"] for c in rec["constraints"]] == ["json_schema", "tools"]
    msg = body["choices"][0]["message"]
    if msg.get("tool_calls"):
        for call in msg["tool_calls"]:
            assert call["function"]["name"] == "pick"
            assert json.loads(call["function"]["arguments"]) == {"value": 7}
    else:
        assert json.loads(msg["content"]) == {"ok": True}
    rc, out = _verify(runner_bin, model, path)
    assert rc == 3 and "json_schema" in out, out[-1000:]
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    rc, out = _verify(runner_bin, model, path, "--json", str(schema_path))
    assert rc == 0 and "VERIFIED" in out, out[-1000:]


def test_tool_result_continuation_replays_its_grammar_start(runner_bin, model, tmp_path):
    messages = [{"role": "user", "content": "Pick a value."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1",
                    "type": "function", "function": {"name": "pick", "arguments": '{"value":7}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "7"}]
    path, rec, body = _serve_turn(runner_bin, model, tmp_path / "r", "gemma4",
        messages=messages, chat_template_kwargs={"enable_thinking": True})
    assert rec["serve"]["reasoning_open"] is False  # the native demux owns the channel
    assert rec["constraints"][0]["reasoning_open"] is True  # grammar starts after the opener
    calls = body["choices"][0]["message"]["tool_calls"]
    assert calls
    for call in calls:
        assert call["function"]["name"] == "pick"
        assert json.loads(call["function"]["arguments"]) == {"value": 7}
    rc, out = _verify(runner_bin, model, path)
    assert rc == 0 and "VERIFIED" in out, out[-1000:]
