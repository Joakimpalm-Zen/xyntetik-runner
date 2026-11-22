"""Inkling 2 Mini native protocol, reasoning budget, and loop guard over HTTP.

Tests the full runtime stack with an Inkling control-token vocabulary
(`TT_CONTROL`, decoded-empty unless spelled for unconstrained chat):
  - tool calls (auto, required, named, parallel; buffered & streamed;
    truncation survival with valid JSON arguments)
  - reasoning channel demux (<|content_thinking|> .. <|end_message|>) and
    text channel (<|content_text|> .. <|end_message|>) without control-tag leaks
  - reasoning budget (--reasoning-budget / reasoning_max_tokens) closing via
    single control token <|end_message|> and continuing into the answer, plus
    HTTP 400 negative control when <|end_message|> is absent
  - loop guard (--loop-guard / loop_guard) closing repeating thinking turns
    via <|end_message|> while keeping normal answers untouched
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

INKLING_CONTROLS = (
    "<|message_system|>,<|message_user|>,<|message_model|>,<|message_tool|>,"
    "<|content_text|>,<|content_image|>,<|content_model_end_sampling|>,"
    "<|content_thinking|>,<|end_message|>,<|content_tool_error|>,"
    "<|content_xml|>,<|begin_of_text|>,<|content_invoke_tool_json|>,"
    "<|content_invoke_tool_text|>"
)

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Ping the service",
        "parameters": {"type": "object", "properties": {}},
    },
}

LOOP = "step 1 + 1 = 2; " * 12


@pytest.fixture(scope="module")
def inkling_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("ink") / "inkling.gguf"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/make-test-model.py",
            "--control",
            INKLING_CONTROLS,
            "--eos-token",
            "<|content_model_end_sampling|>",
            str(m),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return m


@pytest.fixture(scope="module")
def inkling_no_end_msg_model(tmp_path_factory):
    m = tmp_path_factory.mktemp("ink_noend") / "noend.gguf"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/make-test-model.py",
            "--control",
            "<|message_model|>,<|content_thinking|>,<|content_text|>,<|content_model_end_sampling|>",
            "--eos-token",
            "<|content_model_end_sampling|>",
            str(m),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return m


def _serve(model, extra=()):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    srv = RunnerServer(
        exe,
        model,
        ctx=4096,
        parallel=1,
        extra_args=["--gpu", "off", "-t", "2", "--chat-template", "inkling2", *extra],
        env={"RUNNER_TEST_SCRIPTED_REPLY": "1"},
    )
    return srv


@pytest.fixture(scope="module")
def server(inkling_model):
    with _serve(inkling_model) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def server_budget(inkling_model):
    with _serve(inkling_model, ["--reasoning-budget", "6"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def server_loop(inkling_model):
    with _serve(inkling_model, ["--loop-guard"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


@pytest.fixture(scope="module")
def server_no_end(inkling_no_end_msg_model):
    with _serve(inkling_no_end_msg_model) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _post(server, path, payload):
    req = urllib.request.Request(
        server.base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post_stream(server, path, payload):
    req = urllib.request.Request(
        server.base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read().decode()
    events = []
    for line in body.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_scripted_chat_and_tool_channels_round_trip(server):
    reply = (
        "<|message_model|><|content_thinking|>need sum<|end_message|>"
        "<|message_model|><|content_text|>42<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    for stream in (False, True):
        req = {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0,
            "runner_test_reply": reply,
            "stream": stream,
        }
        if not stream:
            code, d = _post(server, "/v1/chat/completions", req)
            assert code == 200, d
            msg = d["choices"][0]["message"]
            assert d["choices"][0]["finish_reason"] == "stop", d
            assert msg["reasoning_content"] == "need sum", msg
            assert msg["content"] == "42", msg
        else:
            ev = _post_stream(server, "/v1/chat/completions", req)
            rc = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in ev if c.get("choices"))
            ct = "".join(c["choices"][0]["delta"].get("content", "") for c in ev if c.get("choices"))
            assert rc == "need sum", ev
            assert ct == "42", ev

    call_reply = (
        "<|message_model|><|content_thinking|>plan<|end_message|>"
        "<|message_model|>add<|content_invoke_tool_json|>"
        '{"name":"add","args":{"a":2,"b":3}}<|end_message|>'
        "<|message_model|>ping<|content_invoke_tool_json|>"
        '{"name":"ping","args":{}}<|end_message|>'
        "<|content_model_end_sampling|>"
    )
    code, d = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "run tools"}],
            "tools": [ADD_TOOL, PING_TOOL],
            "parallel_tool_calls": True,
            "max_tokens": 128,
            "temperature": 0,
            "runner_test_reply": call_reply,
        },
    )
    assert code == 200, d
    msg = d["choices"][0]["message"]
    assert d["choices"][0]["finish_reason"] == "tool_calls", d
    assert d.get("runner_telemetry", {}).get("tool_protocol", {}).get("family") == "inkling2", d
    assert msg["reasoning_content"] == "plan", msg
    assert len(msg["tool_calls"]) == 2, msg
    assert msg["tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 2, "b": 3}
    assert msg["tool_calls"][1]["function"]["name"] == "ping"
    assert json.loads(msg["tool_calls"][1]["function"]["arguments"]) == {}


def test_constrained_tool_choices_and_truncation(server):
    # Required & named choices force valid calls even with random weights.
    code, d = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "add"}],
            "tools": [ADD_TOOL, PING_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "add"}},
            "max_tokens": 128,
            "temperature": 0,
        },
    )
    assert code == 200, d
    assert d.get("runner_telemetry", {}).get("tool_protocol", {}).get("family") == "inkling2", d
    tc = d["choices"][0]["message"].get("tool_calls") or []
    assert "<|content_thinking|>" not in (d["choices"][0]["message"].get("reasoning_content") or ""), d
    assert tc and tc[0]["function"]["name"] == "add", d
    args = json.loads(tc[0]["function"]["arguments"])
    assert isinstance(args["a"], int) and isinstance(args["b"], int), args

    # Very tight max_tokens mid-call closes cleanly with schema-valid JSON args.
    code, d_cut = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "add"}],
            "tools": [ADD_TOOL],
            "tool_choice": "required",
            "max_tokens": 12,
            "temperature": 0,
        },
    )
    assert code == 200, d_cut
    assert d_cut["choices"][0]["finish_reason"] in ("length", "tool_calls"), d_cut
    for item in d_cut["choices"][0]["message"].get("tool_calls") or []:
        parsed = json.loads(item["function"]["arguments"])
        assert isinstance(parsed["a"], int) and isinstance(parsed["b"], int), parsed

    # Streamed constrained tool call reproduces a parseable call.
    ev = _post_stream(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "add"}],
            "tools": [ADD_TOOL],
            "tool_choice": "required",
            "max_tokens": 128,
            "temperature": 0,
            "stream": True,
        },
    )
    chunks = [c["choices"][0] for c in ev if c.get("choices")]
    names = "".join(
        tc_part.get("function", {}).get("name", "")
        for ch in chunks
        for tc_part in ch.get("delta", {}).get("tool_calls", [])
    )
    arg_str = "".join(
        tc_part.get("function", {}).get("arguments", "")
        for ch in chunks
        for tc_part in ch.get("delta", {}).get("tool_calls", [])
    )
    assert names == "add", chunks
    parsed = json.loads(arg_str)
    assert isinstance(parsed["a"], int) and isinstance(parsed["b"], int), parsed


def test_reasoning_budget_forces_close_and_negative_control(
    server, server_budget, server_no_end
):
    # Unconstrained /v1/completions resuming <|content_thinking|>: default sentence,
    # custom sentence, bare "", and --reasoning-budget server default.
    code, d = _post(
        server,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_thinking|>",
            "max_tokens": 96,
            "temperature": 0,
            "reasoning_max_tokens": 6,
        },
    )
    assert code == 200, d
    rb = d.get("runner_telemetry", {}).get("reasoning_budget")
    assert rb is not None and rb["forced_close"] is True, d
    text = d["choices"][0]["text"].replace("\u2581", " ")
    assert "stopping deliberation here" in text, d
    assert "<|end_message|>" not in text, d
    assert d["usage"]["completion_tokens"] > 6, d

    code, d_custom = _post(
        server,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_thinking|>",
            "max_tokens": 96,
            "temperature": 0,
            "reasoning_max_tokens": 6,
            "reasoning_budget_message": "\nEnough. Answering now.\n",
        },
    )
    assert code == 200, d_custom
    assert d_custom["runner_telemetry"]["reasoning_budget"]["forced_close"] is True
    assert "Enough. Answering now." in d_custom["choices"][0]["text"].replace("\u2581", " ")

    # Bare empty transition sentence closes with single <|end_message|> token only.
    code, d2 = _post(
        server,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_thinking|>",
            "max_tokens": 64,
            "temperature": 0,
            "reasoning_max_tokens": 6,
            "reasoning_budget_message": "",
        },
    )
    assert code == 200, d2
    assert d2["runner_telemetry"]["reasoning_budget"]["forced_close"] is True
    assert "stopping deliberation" not in d2["choices"][0]["text"]

    # Server started with --reasoning-budget.
    code, d3 = _post(
        server_budget,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_thinking|>",
            "max_tokens": 96,
            "temperature": 0,
        },
    )
    assert code == 200, d3
    assert d3["runner_telemetry"]["reasoning_budget"]["forced_close"] is True
    assert "stopping deliberation here" in d3["choices"][0]["text"].replace("\u2581", " ")

    # Constrained chat turn that enters <|content_thinking|> closes via <|end_message|>
    # and keeps the transition sentence inside reasoning_content, not content.
    code, dc = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "add"}],
            "tools": [ADD_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "add"}},
            "max_tokens": 128,
            "temperature": 0,
            "reasoning_max_tokens": 6,
        },
    )
    assert code == 200, dc
    rbc = dc.get("runner_telemetry", {}).get("reasoning_budget")
    assert rbc is not None and rbc["forced_close"] is True, dc
    rc = (dc["choices"][0]["message"].get("reasoning_content") or "").replace("\u2581", " ")
    assert "stopping deliberation here" in rc, dc
    assert "<|end_message|>" not in rc and "<|content_thinking|>" not in rc, dc

    # Negative control: missing <|end_message|> control token refuses with 400.
    code, err = _post(
        server_no_end,
        "/v1/chat/completions",
        {
            "model": server_no_end.model_id,
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 64,
            "temperature": 0,
            "reasoning_max_tokens": 6,
        },
    )
    assert code == 400, err
    assert "single token" in json.dumps(err).lower(), err


def test_loop_guard_closes_thinking_and_scopes_to_reasoning(server, server_loop):
    # Guarded raw completion inside <|content_thinking|> closes early.
    code, d = _post(
        server,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_thinking|>",
            "max_tokens": 120,
            "temperature": 0,
            "loop_guard": True,
            "loop_guard_span": 4,
            "loop_guard_repeats": 3,
            "runner_test_reply": LOOP,
        },
    )
    assert code == 200, d
    lg = d.get("runner_telemetry", {}).get("loop_guard")
    assert lg is not None and lg["interventions"] == 1 and lg["ended_turn"] is False, d
    assert len(d["choices"][0]["text"]) < 80, d

    # Chat turn with server-wide --loop-guard closes thinking loop and continues.
    code, dc = _post(
        server_loop,
        "/v1/chat/completions",
        {
            "model": server_loop.model_id,
            "messages": [{"role": "user", "content": "solve"}],
            "max_tokens": 120,
            "temperature": 0,
            "loop_guard_span": 4,
            "loop_guard_repeats": 3,
            "runner_test_reply": "<|message_model|><|content_thinking|>" + LOOP,
        },
    )
    assert code == 200, dc
    assert dc["runner_telemetry"]["loop_guard"]["interventions"] == 1, dc
    assert "<|end_message|>" not in (dc["choices"][0]["message"].get("reasoning_content") or "")

    # Scoped to reasoning by default: loop inside <|content_text|> is untouched.
    code, d_ans = _post(
        server,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_text|>",
            "max_tokens": 120,
            "temperature": 0,
            "loop_guard": True,
            "loop_guard_span": 4,
            "loop_guard_repeats": 3,
            "runner_test_reply": LOOP,
        },
    )
    assert code == 200, d_ans
    assert d_ans["runner_telemetry"]["loop_guard"]["interventions"] == 0, d_ans

    # Non-repeating thinking negative control triggers zero interventions.
    code, d_clean = _post(
        server_loop,
        "/v1/chat/completions",
        {
            "model": server_loop.model_id,
            "messages": [{"role": "user", "content": "solve"}],
            "max_tokens": 80,
            "temperature": 0,
            "runner_test_reply": (
                "<|message_model|><|content_thinking|>one distinct thought<|end_message|>"
                "<|message_model|><|content_text|>seven<|end_message|>"
                "<|content_model_end_sampling|>"
            ),
        },
    )
    assert code == 200, d_clean
    assert d_clean["runner_telemetry"]["loop_guard"]["interventions"] == 0, d_clean
    assert d_clean["choices"][0]["message"]["content"] == "seven", d_clean
