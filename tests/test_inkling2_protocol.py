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
import os
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

    # T1: Tight max_tokens mid-call closes cleanly with a non-empty, schema-valid call.
    partial_source = (
        '<|message_model|>add<|content_invoke_tool_json|>'
        '{"name":"add","args":{"a":42,"b":99}}<|end_message|>'
        '<|content_model_end_sampling|>'
    )
    code, d_cut = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "add"}],
            "tools": [ADD_TOOL],
            "tool_choice": "required",
            "max_tokens": 34,
            "temperature": 0,
            "runner_test_reply": partial_source,
        },
    )
    assert code == 200, d_cut
    assert d_cut["choices"][0]["finish_reason"] == "length", d_cut
    cut_calls = d_cut["choices"][0]["message"].get("tool_calls") or []
    assert len(cut_calls) == 1 and cut_calls[0]["function"]["name"] == "add", d_cut
    assert d_cut.get("runner_telemetry", {}).get("closure", {}).get("closed") is True, d_cut
    cut_args = json.loads(cut_calls[0]["function"]["arguments"])
    assert cut_args["a"] == 42 and isinstance(cut_args["b"], int), cut_args

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


def test_f3_buffered_parse_only_other_family_keeps_no_reasoning(tmp_path):
    """F3: non-Inkling buffered parse_only and auto turns preserve 2fc588b9 behaviour:
    - qwen3-coder auto (parse_only=True) splits <think> via think_init without
      tool_stream_map_channels_ex touching reasoning;
    - gemma4 auto (!parse_only) maps via tool_envelope_map_turn (as at 2fc588b9);
    while C tests in tests/test_tools.c verify NULL reasoning on parse_only
    TP_GEMMA4 / TP_HARMONY."""
    m = tmp_path / "q3c_po.gguf"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/make-test-model.py",
            "--control",
            "<|im_start|>,<|im_end|>,<|tool_call>,<tool_call|>,<turn|>,<|tool_response>",
            str(m),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    exe = find_runner(ROOT)
    with RunnerServer(
        exe,
        m,
        ctx=2048,
        parallel=1,
        extra_args=["--gpu", "off", "-t", "2", "--chat-template", "qwen3-coder"],
        env={"RUNNER_TEST_SCRIPTED_REPLY": "1"},
    ) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            model_id = json.load(r)["data"][0]["id"]
        code, d = _post(
            srv,
            "/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "call"}],
                "tools": [ADD_TOOL],
                "tool_choice": "auto",
                "max_tokens": 200,
                "temperature": 0,
                "stream": False,
                "runner_test_reply": (
                    "<think>plan</think>\n"
                    "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n"
                    "<parameter=b>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>"
                ),
            },
        )
        assert code == 200, d
        assert d["runner_telemetry"]["tool_protocol"]["parse_only"] is True, d
        msg = d["choices"][0]["message"]
        assert len(msg.get("tool_calls") or []) == 1, d
        assert not msg.get("reasoning_content"), d
        assert msg.get("content") == "<think>plan</think>\n", d

    with RunnerServer(
        exe,
        m,
        ctx=2048,
        parallel=1,
        extra_args=["--gpu", "off", "-t", "2", "--chat-template", "gemma4"],
        env={"RUNNER_TEST_SCRIPTED_REPLY": "1"},
    ) as gsrv:
        with urllib.request.urlopen(gsrv.base_url + "/v1/models", timeout=30) as r:
            gmodel = json.load(r)["data"][0]["id"]
        code, gd = _post(
            gsrv,
            "/v1/chat/completions",
            {
                "model": gmodel,
                "messages": [{"role": "user", "content": "call"}],
                "tools": [ADD_TOOL],
                "tool_choice": "auto",
                "max_tokens": 200,
                "temperature": 0,
                "stream": False,
                "runner_test_reply": (
                    "<|channel>thought\nplan<channel|>"
                    "<|tool_call>call:add{a:1,b:2}<tool_call|><|tool_response>"
                ),
            },
        )
        assert code == 200, gd
        assert gd["runner_telemetry"]["tool_protocol"]["parse_only"] is False, gd
        gmsg = gd["choices"][0]["message"]
        assert len(gmsg.get("tool_calls") or []) == 1, gd
        assert gmsg.get("reasoning_content") == "plan", gd


def test_d1_plain_chat_without_tools_emits_no_tool_calls(server):
    """D1: Plain chat with NO tools declared (and no tool_calls contract) must
    never populate tool_calls or finish with 'tool_calls', buffered or streamed."""
    call_reply = (
        '<|message_model|>add<|content_invoke_tool_json|>'
        '{"name":"add","args":{"a":1,"b":2}}'
        '<|end_message|><|content_model_end_sampling|>'
    )
    code, d = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
            "runner_test_reply": call_reply,
        },
    )
    assert code == 200, d
    msg = d["choices"][0]["message"]
    assert not msg.get("tool_calls"), d
    assert d["choices"][0]["finish_reason"] != "tool_calls", d

    ev = _post_stream(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
            "runner_test_reply": call_reply,
        },
    )
    chunks = [c["choices"][0] for c in ev if c.get("choices")]
    streamed_calls = [
        tc_part for ch in chunks for tc_part in ch.get("delta", {}).get("tool_calls", [])
    ]
    assert not streamed_calls, chunks
    assert all(ch.get("finish_reason") != "tool_calls" for ch in chunks), chunks

    # tool_choice="none" with tools declared also emits no tool_calls.
    code, d_none = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [ADD_TOOL],
            "tool_choice": "none",
            "max_tokens": 64,
            "temperature": 0,
            "runner_test_reply": call_reply,
        },
    )
    assert code == 200, d_none
    assert not d_none["choices"][0]["message"].get("tool_calls"), d_none
    assert d_none["choices"][0]["finish_reason"] != "tool_calls", d_none


@pytest.mark.parametrize(
    "bad_reply",
    [
        "<|message_model|>add<|end_message|><|content_model_end_sampling|>",
        '<|message_model|>add<|content_invoke_tool_json|>{"name":"other","args":{}}<|end_message|><|content_model_end_sampling|>',
        "<|message_model|>add<|content_invoke_tool_json|>{bad<|end_message|><|content_model_end_sampling|>",
    ],
)
def test_d2_plain_chat_fault_matches_tools_branch(server, bad_reply):
    """D2: Buffered plain chat reports malformed frames / bad calls with
    finish 'envelope_error' consistently with streamed plain chat & the tools branch."""
    code, d_buf = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
            "runner_test_reply": bad_reply,
        },
    )
    assert code == 200, d_buf
    assert not d_buf["choices"][0]["message"].get("tool_calls"), d_buf
    assert d_buf["choices"][0]["finish_reason"] == "error", d_buf
    assert d_buf.get("runner_telemetry", {}).get("finish_detail") == "envelope_unmapped", d_buf

    ev = _post_stream(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
            "runner_test_reply": bad_reply,
        },
    )
    fin_chunks = [c for c in ev if c.get("choices") and c["choices"][0].get("finish_reason")]
    assert len(fin_chunks) == 1, ev
    assert fin_chunks[0]["choices"][0]["finish_reason"] == "error", fin_chunks
    assert fin_chunks[0].get("runner_telemetry", {}).get("finish_detail") == "envelope_unmapped", fin_chunks


@pytest.mark.parametrize("hdr", ["<|message_user|>", "<|message_system|>", "<|message_tool|>"])
def test_r2_turn_header_stops_chat_before_tail_buffered_and_streamed(server, hdr):
    """R2: Unconstrained Inkling chat stops decoding immediately at another turn
    header, keeps the header out of content/reasoning, names it in telemetry,
    while raw /v1/completions keeps publisher EOS-only stops."""
    # Prefix is 3 tokens: <|message_model|>, <|content_text|>, ok, <|end_message|> (4 tok),
    # followed immediately by hdr, then a long tail that must NEVER be decoded.
    reply_text = (
        "<|message_model|><|content_thinking|>t<|end_message|>"
        "<|message_model|><|content_text|>ok<|end_message|>"
        f"{hdr}TRAILING_NEVER_DECODED_1_2_3_4_5<|content_model_end_sampling|>"
    )
    code, d = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "temperature": 0,
            "stream": False,
            "runner_test_reply": reply_text,
        },
    )
    assert code == 200, d
    msg = d["choices"][0]["message"]
    assert msg["content"] == "ok", d
    assert msg.get("reasoning_content") == "t", d
    assert hdr not in (msg.get("content") or "") and hdr not in (msg.get("reasoning_content") or ""), d
    assert "TRAILING" not in (msg.get("content") or ""), d
    # Header halted decode immediately: far fewer completion tokens than full tail.
    assert d["usage"]["completion_tokens"] < 15, d
    assert d["choices"][0]["finish_reason"] == "stop", d
    assert d["choices"][0].get("stop_token") == hdr, d
    tel = d.get("runner_telemetry", {})
    assert tel.get("ended_at") == "turn_header", tel
    assert tel.get("turn_header") == hdr, tel

    ev = _post_stream(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "runner_test_reply": reply_text,
        },
    )
    chunks = [c["choices"][0] for c in ev if c.get("choices")]
    txt = "".join(ch.get("delta", {}).get("content", "") for ch in chunks)
    rsn = "".join(ch.get("delta", {}).get("reasoning_content", "") for ch in chunks)
    assert txt == "ok" and rsn == "t", chunks
    assert hdr not in txt and hdr not in rsn, chunks
    fin = [c for c in ev if c.get("choices") and c["choices"][0].get("finish_reason")][0]
    assert fin["choices"][0]["finish_reason"] == "stop", fin
    assert fin["choices"][0].get("stop_token") == hdr, fin
    assert fin.get("runner_telemetry", {}).get("ended_at") == "turn_header", fin
    assert fin.get("runner_telemetry", {}).get("turn_header") == hdr, fin
    usage_ev = [c for c in ev if c.get("usage")]
    assert usage_ev and usage_ev[-1]["usage"]["completion_tokens"] == d["usage"]["completion_tokens"], ev

    # Raw /v1/completions keeps publisher EOS-only stops (does not stop at <|message_user|>).
    code, raw = _post(
        server,
        "/v1/completions",
        {
            "prompt": "<|message_model|><|content_text|>",
            "max_tokens": 128,
            "temperature": 0,
            "runner_test_reply": f"prefix{hdr}AFTER_HDR<|content_model_end_sampling|>",
        },
    )
    assert code == 200, raw
    assert "AFTER_HDR" in raw["choices"][0]["text"], raw
    assert raw.get("runner_telemetry", {}).get("ended_at") is None, raw


def _patch_zero_branch_embd(gguf_path, plan_fn):
    """Patch selected token_embd.weight rows in a --zero-branches GGUF."""
    import struct

    data = bytearray(gguf_path.read_bytes())
    pos = 0

    def rd(fmt):
        nonlocal pos
        sz = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, data, pos)
        pos += sz
        return vals

    def rd_str():
        nonlocal pos
        (n,) = rd("<Q")
        s = data[pos : pos + n].decode("utf-8", errors="replace")
        pos += n
        return s

    def skip_val(vtype):
        nonlocal pos
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype in fixed:
            pos += fixed[vtype]
            return None
        if vtype == 8:
            return rd_str()
        if vtype == 9:
            (et, n) = rd("<IQ")
            return [skip_val(et) for _ in range(n)]
        raise ValueError(f"unknown gguf type {vtype}")

    magic, ver, n_tensors, n_kv = rd("<IIQQ")
    assert magic == 0x46554747 and ver == 3
    tokens = None
    for _ in range(n_kv):
        k = rd_str()
        (vt,) = rd("<I")
        v = skip_val(vt)
        if k == "tokenizer.ggml.tokens":
            tokens = v
    emb_off = None
    n_embd = None
    for _ in range(n_tensors):
        name = rd_str()
        (nd,) = rd("<I")
        dims = rd(f"<{nd}Q")
        ttype, off = rd("<IQ")
        if name == "token_embd.weight":
            assert ttype == 0
            n_embd = dims[0]
            emb_off = off
    data_start = (pos + 31) & ~31
    assert tokens is not None and n_embd is not None and emb_off is not None
    rows, result = plan_fn(tokens)
    for tid, (c0, c1) in rows.items():
        row = [c0, c1] + [0.0] * (n_embd - 2)
        off = data_start + emb_off + tid * n_embd * 4
        struct.pack_into(f"<{n_embd}f", data, off, *row)
    gguf_path.write_bytes(data)
    return result


def _patch_embd_to_emit_then_header(gguf_path, first_piece, header_piece):
    """Patch token_embd.weight so greedy decode emits `first_piece` then `header_piece`."""
    def _plan(tokens):
        tid_end = tokens.index("<|end_message|>")
        tid_first = tokens.index(first_piece) if first_piece in tokens else tokens.index("<0x61>")
        tid_hdr = tokens.index(header_piece)
        return (
            {tid_end: (1.0, 0.0), tid_first: (20.0, 1.0), tid_hdr: (0.0, 1000.0)},
            (tid_first, tid_hdr),
        )

    return _patch_zero_branch_embd(gguf_path, _plan)


def test_r2_turn_header_stop_replays_with_verify(tmp_path):
    """R2 addition 3: a turn ended by an Inkling turn header records the stop in
    the receipt and replays VERIFIED under `runner --verify`, while tampering the
    stop constraint fails verification."""
    m = tmp_path / "ink_verify.gguf"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/make-test-model.py",
            "--zero-branches",
            "--control",
            INKLING_CONTROLS,
            "--eos-token",
            "<|content_model_end_sampling|>",
            str(m),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    tid_first, tid_hdr = _patch_embd_to_emit_then_header(m, "a", "<|message_user|>")
    rdir = tmp_path / "receipts"
    exe = find_runner(ROOT)
    with RunnerServer(
        exe,
        m,
        ctx=1024,
        parallel=1,
        extra_args=[
            "--gpu",
            "off",
            "-t",
            "2",
            "--chat-template",
            "inkling2",
            "--receipts",
            str(rdir),
        ],
    ) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            model_id = json.load(r)["data"][0]["id"]
        code, d = _post(
            srv,
            "/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "temperature": 0,
                "enable_thinking": False,
                "cache_prompt": False,
            },
        )
    assert code == 200, d
    assert d["choices"][0]["message"]["content"] == "a", d
    assert d["usage"]["completion_tokens"] == 1, d
    assert d["choices"][0]["finish_reason"] == "stop", d
    assert d["choices"][0].get("stop_token") == "<|message_user|>", d
    tel = d.get("runner_telemetry", {})
    assert tel.get("ended_at") == "turn_header", tel
    assert tel.get("turn_header") == "<|message_user|>", tel

    rec_path = next(rdir.iterdir())
    rec = json.loads(rec_path.read_text())
    assert rec["output"]["tokens"] == [tid_first], rec
    assert rec["serve"].get("ended_at") == "turn_header", rec["serve"]
    assert rec["serve"].get("turn_header") == "<|message_user|>", rec["serve"]
    stop_cons = [c for c in rec.get("constraints", []) if c.get("kind") == "stop"]
    assert len(stop_cons) == 1 and tid_hdr in stop_cons[0].get("token_ids", []), rec

    v = subprocess.run(
        [exe, "-m", m, "--verify", rec_path, "--gpu", "off", "-t", "2", "-c", "1024"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    err = v.stderr.decode(errors="replace")
    assert v.returncode == 0 and "VERIFIED" in err, err[-600:]

    # Dropping the stop constraint makes the record fail --verify.
    import hashlib

    raw = rec_path.read_text()
    cut = raw.rindex(',"chain":')
    body = json.loads(raw[:cut] + "}")
    body["constraints"] = [c for c in body["constraints"] if c.get("kind") != "stop"]
    if not body["constraints"]:
        del body["constraints"]
        body["serve"]["shaped_by"] = []
    head = json.dumps(body, separators=(",", ":"))[:-1]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        head
        + ',"chain":'
        + json.dumps(
            {
                "algo": "sha256",
                "prev": rec["chain"]["prev"],
                "hash": hashlib.sha256(head.encode()).hexdigest(),
            },
            separators=(",", ":"),
        )
        + "}"
    )
    vt = subprocess.run(
        [exe, "-m", m, "--verify", tampered, "--gpu", "off", "-t", "2", "-c", "1024"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    vt_err = vt.stderr.decode(errors="replace")
    assert vt.returncode != 0 and ("DIVERGED" in vt_err or "UNVERIFIABLE" in vt_err), vt_err[-600:]


def test_t2_loop_guard_unscripted_emits_end_message_and_continues(tmp_path):
    """T2: On an unscripted loop inside <|content_thinking|>, loop_guard fires
    think_arm_close, emits <|end_message|>, leaves ended_turn=False, and
    continues generating tokens past the forced close."""
    m = tmp_path / "ink_loop_real.gguf"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/make-test-model.py",
            "--zero-branches",
            "--control",
            INKLING_CONTROLS,
            "--eos-token",
            "<|content_model_end_sampling|>",
            str(m),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    def _plan(tokens):
        tid_think = tokens.index("<|content_thinking|>")
        tid_loop = next(tokens.index(t) for t in ("x", "<0x78>", "<0x78>".upper()) if t in tokens)
        tid_end = tokens.index("<|end_message|>")
        tid_after = next(tokens.index(t) for t in ("z", "<0x7A>", "<0x7a>") if t in tokens)
        # <|content_thinking|> -> self-repeating 'x'; once loop_guard forces
        # <|end_message|>, <|end_message|> selects post-close 'z'.
        return (
            {
                tid_think: (1.0, 0.0),
                tid_loop: (50.0, 0.0),
                tid_end: (0.0, 1.0),
                tid_after: (0.0, 50.0),
            },
            (tid_loop, tid_end, tid_after),
        )

    tid_loop, tid_end, tid_after = _patch_zero_branch_embd(m, _plan)
    rdir = tmp_path / "loop_receipts"
    exe = find_runner(ROOT)
    with RunnerServer(
        exe,
        m,
        ctx=1024,
        parallel=1,
        extra_args=["--gpu", "off", "-t", "2", "--chat-template", "inkling2", "--receipts", str(rdir)],
    ) as srv:
        code, d = _post(
            srv,
            "/v1/completions",
            {
                "prompt": "<|message_model|><|content_thinking|>",
                "max_tokens": 12,
                "temperature": 0,
                "loop_guard": True,
                "loop_guard_span": 2,
                "loop_guard_repeats": 2,
                "cache_prompt": False,
            },
        )
    assert code == 200, d
    lg = d.get("runner_telemetry", {}).get("loop_guard")
    assert lg is not None and lg["interventions"] == 1 and lg["ended_turn"] is False, d
    assert d["usage"]["completion_tokens"] == 12, d

    toks = json.loads(next(rdir.iterdir()).read_text())["output"]["tokens"]
    assert tid_loop in toks and tid_end in toks and tid_after in toks, toks
    close_at = toks.index(tid_end)
    assert toks[:close_at] == [tid_loop] * close_at and close_at >= 3, toks
    assert all(t == tid_after for t in toks[close_at + 1 :]) and len(toks) > close_at + 1, toks


def test_r2_cli_chat_stops_at_turn_header(tmp_path):
    """R2 residue: CLI -i chat on Inkling adds turn-header stops and halts at
    <|message_user|> without decoding further tokens."""
    m = tmp_path / "ink_cli.gguf"
    subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/make-test-model.py",
            "--zero-branches",
            "--control",
            INKLING_CONTROLS,
            "--eos-token",
            "<|content_model_end_sampling|>",
            str(m),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    _patch_embd_to_emit_then_header(m, "a", "<|message_user|>")
    exe = find_runner(ROOT)
    p = subprocess.run(
        [
            str(exe),
            "-m",
            str(m),
            "--chat-template",
            "inkling2",
            "-i",
            "-n",
            "16",
            "--temp",
            "0",
            "--no-tray",
            "-c",
            "1024",
        ],
        input=b"hi\n/exit\n",
        capture_output=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    out = p.stdout.decode(errors="replace")
    err = p.stderr.decode(errors="replace")
    assert "a" in out and "<|message_user|>" not in out, (out, err)
    assert "[1 tok," in err and "ended_at=turn_header <|message_user|>" in err, err


def test_plain_chat_strips_spelled_end_message_outside_channel(server):
    """Spelled <|end_message|> in bare TS_INK_TEXT outside any channel does
    not leak into content in buffered or streamed plain chat."""
    reply = "hello<|end_message|><|content_model_end_sampling|>"
    code, d = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
            "runner_test_reply": reply,
        },
    )
    assert code == 200, d
    assert d["choices"][0]["finish_reason"] == "stop", d
    assert d["choices"][0]["message"]["content"] == "hello", d
    assert not d["choices"][0]["message"].get("tool_calls"), d

    ev = _post_stream(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": True,
            "runner_test_reply": reply,
        },
    )
    chunks = [c["choices"][0] for c in ev if c.get("choices")]
    txt = "".join(ch.get("delta", {}).get("content", "") for ch in chunks)
    assert txt == "hello", chunks


def test_t3_end_message_inside_json_string_argument_http(server):
    """T3 HTTP checks:
    (a) JSON Unicode escape '\\u003c|end_message|>' preserves '<|end_message|>'
        across JSON serialization in buffered and streamed calls.
    (b) Scripted control token '<|end_message|>' inside a constrained JSON string
        argument is masked inside SN_STR by constraint_spelling_ok and closes
        cleanly instead of leaking the control token or corrupting the frame.
    (Note: raw ASCII '<|end_message|>' retry in inkling_parse_call is covered
    in tests/test_tools.c:test_inkling2_t3_edge_cases because runner_test_reply
    encodes '<|end_message|>' as control token TT_CONTROL.)"""
    echo_tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo a string",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    esc_reply = (
        '<|message_model|>echo<|content_invoke_tool_json|>'
        '{"name":"echo","args":{"text":"alpha \\u003c|end_message|> omega"}}'
        '<|end_message|><|content_model_end_sampling|>'
    )
    for stream_mode in (False, True):
        if not stream_mode:
            code, d = _post(
                server,
                "/v1/chat/completions",
                {
                    "model": server.model_id,
                    "messages": [{"role": "user", "content": "echo"}],
                    "tools": [echo_tool],
                    "tool_choice": "required",
                    "max_tokens": 128,
                    "temperature": 0,
                    "stream": False,
                    "runner_test_reply": esc_reply,
                },
            )
            assert code == 200, d
            assert d["choices"][0]["finish_reason"] == "tool_calls", d
            tc = d["choices"][0]["message"]["tool_calls"]
            assert len(tc) == 1 and tc[0]["function"]["name"] == "echo", tc
            assert json.loads(tc[0]["function"]["arguments"]) == {"text": "alpha <|end_message|> omega"}, tc
        else:
            ev = _post_stream(
                server,
                "/v1/chat/completions",
                {
                    "model": server.model_id,
                    "messages": [{"role": "user", "content": "echo"}],
                    "tools": [echo_tool],
                    "tool_choice": "required",
                    "max_tokens": 128,
                    "temperature": 0,
                    "stream": True,
                    "runner_test_reply": esc_reply,
                },
            )
            chunks = [c["choices"][0] for c in ev if c.get("choices")]
            arg_str = "".join(
                tc_part.get("function", {}).get("arguments", "")
                for ch in chunks
                for tc_part in ch.get("delta", {}).get("tool_calls", [])
            )
            assert json.loads(arg_str) == {"text": "alpha <|end_message|> omega"}, chunks

    # Real control token <|end_message|> inside constrained SN_STR is blocked:
    raw_ctrl_reply = (
        '<|message_model|>echo<|content_invoke_tool_json|>'
        '{"name":"echo","args":{"text":"alpha <|end_message|> omega"}}'
        '<|end_message|><|content_model_end_sampling|>'
    )
    code, d_blk = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "echo"}],
            "tools": [echo_tool],
            "tool_choice": "required",
            "max_tokens": 128,
            "temperature": 0,
            "runner_test_reply": raw_ctrl_reply,
        },
    )
    assert code == 200, d_blk
    tc_blk = d_blk["choices"][0]["message"]["tool_calls"]
    assert len(tc_blk) == 1 and tc_blk[0]["function"]["name"] == "echo", tc_blk
    assert json.loads(tc_blk[0]["function"]["arguments"]) == {"text": "alpha "}, tc_blk


def _render_cases(cases):
    exe = ROOT / ("template-conformance-render.exe" if sys.platform == "win32" else "template-conformance-render")
    if not exe.exists():
        pytest.skip("make template-conformance-render first")
    p = subprocess.run([str(exe)], input=json.dumps({"cases": cases}), capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert p.returncode == 0, p.stderr
    return {r["id"]: r for r in json.loads(p.stdout)["results"]}


def test_r4416_null_effort_in_kwargs_falls_through_to_top_level(server):
    """Condition 1 (R4.48.16): JSON null for chat_template_kwargs.reasoning_effort must
    fall through to top-level reasoning_effort instead of shadowing it."""
    res = _render_cases([{
        "id": "null_low",
        "template": "inkling2",
        "chat_template": "<|message_model|>Thinking effort level:",
        "request": {
            "model": "x",
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": None},
            "messages": [{"role": "user", "content": "hi"}],
        },
    }])["null_low"]
    assert res.get("error") is None, res
    assert "Thinking effort level: 0.2<|end_message|>" in res["prompt"], res["prompt"]

    def _ptok(body):
        c, d = _post(server, "/v1/chat/completions", {"model": server.model_id, "max_tokens": 2, "temperature": 0,
                                                      "messages": [{"role": "user", "content": "hi"}], **body})
        assert c == 200, d
        return d["usage"]["prompt_tokens"]

    assert _ptok({"reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": None}}) == _ptok(
        {"reasoning_effort": "none"})
    assert _ptok({"reasoning_effort": "none"}) + 2 == _ptok({})


def test_r4416_enable_thinking_false_maps_to_effort_none(server, tmp_path):
    """R4.48.16: enable_thinking=false with no reasoning_effort maps to effort 'none' (0)
    on inkling2; explicit reasoning_effort always wins; null falls through."""
    base_tmpl = "<|message_model|>Thinking effort level:"
    reads_tmpl = "{% if enable_thinking %}{% endif %}<|message_model|>Thinking effort level:"

    def _case(cid, tmpl_txt, extra):
        return {"id": cid, "template": "inkling2", "chat_template": tmpl_txt,
                "request": {"model": "x", "messages": [{"role": "user", "content": "hi"}], **extra}}

    res = _render_cases([
        _case("top_false", base_tmpl, {"enable_thinking": False}),
        _case("kw_false_plain", base_tmpl, {"chat_template_kwargs": {"enable_thinking": False}}),
        _case("kw_false_reads", reads_tmpl, {"chat_template_kwargs": {"enable_thinking": False}}),
        _case("null_eff_false", base_tmpl, {"enable_thinking": False, "chat_template_kwargs": {"reasoning_effort": None}}),
        _case("win_top_low", base_tmpl, {"enable_thinking": False, "reasoning_effort": "low"}),
        _case("win_kw_low", base_tmpl, {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}),
        _case("win_high", base_tmpl, {"enable_thinking": False, "reasoning_effort": "high"}),
        _case("win_num", base_tmpl, {"enable_thinking": False, "reasoning_effort": 0.37}),
        _case("win_none", base_tmpl, {"enable_thinking": False, "reasoning_effort": "none"}),
        _case("win_bad", base_tmpl, {"enable_thinking": False, "reasoning_effort": "bogus"}),
        _case("top_true", base_tmpl, {"enable_thinking": True}),
        _case("kw_true", base_tmpl, {"chat_template_kwargs": {"enable_thinking": True}}),
        _case("absent", base_tmpl, {}),
        _case("kw_true_over_top_false", base_tmpl, {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True}}),
        _case("kw_false_over_top_true", base_tmpl, {"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": False}}),
    ])

    for cid in ("top_false", "kw_false_plain", "kw_false_reads", "null_eff_false", "win_none", "kw_false_over_top_true"):
        assert res[cid].get("error") is None, (cid, res[cid])
        assert "Thinking effort level: 0<|end_message|>" in res[cid]["prompt"], (cid, res[cid]["prompt"])

    for cid, want in (("win_top_low", "0.2"), ("win_kw_low", "0.2"), ("win_high", "0.9"), ("win_num", "0.37"),
                      ("top_true", "0.9"), ("kw_true", "0.9"), ("absent", "0.9"), ("kw_true_over_top_false", "0.9")):
        assert res[cid].get("error") is None, (cid, res[cid])
        assert f"Thinking effort level: {want}<|end_message|>" in res[cid]["prompt"], (cid, res[cid]["prompt"])

    assert res["win_bad"].get("error") and "reasoning_effort must be" in res["win_bad"]["error"], res["win_bad"]

    # Live HTTP checks on /v1/chat/completions (plain + template that reads enable_thinking).
    def _ptok(srv, extra):
        c, d = _post(srv, "/v1/chat/completions", {"model": srv.model_id, "max_tokens": 2, "temperature": 0,
                                                   "messages": [{"role": "user", "content": "hi"}], **extra})
        return c, d

    c_none, d_none = _ptok(server, {"reasoning_effort": "none"})
    c_def, d_def = _ptok(server, {})
    assert c_none == 200 and c_def == 200
    n0 = d_none["usage"]["prompt_tokens"]
    assert d_def["usage"]["prompt_tokens"] == n0 + 2

    for extra in ({"enable_thinking": False}, {"chat_template_kwargs": {"enable_thinking": False}},
                  {"enable_thinking": False, "reasoning_effort": "none"}):
        c, d = _ptok(server, extra)
        assert c == 200 and d["usage"]["prompt_tokens"] == n0, (extra, c, d)

    # Embedded Jinja reading enable_thinking: on d4ec380f chat_template_kwargs.enable_thinking returned 400;
    # now renderer_honours includes inkling2 so it accepts 200 and renders effort 0.
    tpl = tmp_path / "ink_et.jinja"
    tpl.write_text(reads_tmpl, encoding="utf-8")
    m_et = tmp_path / "ink_et.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--control", INKLING_CONTROLS,
                    "--eos-token", "<|content_model_end_sampling|>", "--chat-template-file", str(tpl), str(m_et)],
                   check=True, stdout=subprocess.DEVNULL)
    with _serve(m_et) as srv_et:
        with urllib.request.urlopen(srv_et.base_url + "/v1/models", timeout=30) as r:
            srv_et.model_id = json.load(r)["data"][0]["id"]
        c, d = _ptok(srv_et, {"chat_template_kwargs": {"enable_thinking": False}})
        assert c == 200 and d["usage"]["prompt_tokens"] == n0, (c, d)


def test_r4416_cli_interactive_think_flags(inkling_model):
    """Condition 3 (R4.48.16): CLI -i --think off and --no-think give effort 0 on inkling2,
    while --think on, --think auto, and absent keep 0.9."""
    import re
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    env = {**os.environ, "RUNNER_DEBUG_ACT": "1"}

    def _ntok(flag_args):
        p = subprocess.run([str(exe), "-m", str(inkling_model), "--chat-template", "inkling2",
                            "-i", "-n", "1", "--temp", "0", "--no-tray", "-c", "512", "--gpu", "off",
                            *flag_args],
                           input="hi\n/exit\n", capture_output=True, text=True, env=env, timeout=60)
        assert p.returncode == 0, p.stderr
        m = re.search(r"ACT ==== forward n=(\d+)", p.stderr)
        assert m, p.stderr[-1000:]
        return int(m.group(1))

    n_def = _ntok([])
    assert _ntok(["--think", "on"]) == n_def
    assert _ntok(["--think", "auto"]) == n_def
    assert _ntok(["--think", "off"]) == n_def - 2
    assert _ntok(["--no-think"]) == n_def - 2


def test_r4416_other_families_enable_thinking_unchanged():
    """Condition 2/gate: non-inkling2 families do not rewrite enable_thinking=false to none."""
    res = _render_cases([
        {"id": "qwen38_off", "template": "qwen38", "chat_template": "",
         "request": {"model": "x", "enable_thinking": False, "messages": [{"role": "user", "content": "hi"}]}},
        {"id": "granite42_off", "template": "granite42", "chat_template": "",
         "request": {"model": "x", "enable_thinking": False, "messages": [{"role": "user", "content": "hi"}]}},
        {"id": "muse_off", "template": "muse", "chat_template": "",
         "request": {"model": "x", "enable_thinking": False, "messages": [{"role": "user", "content": "hi"}]}},
    ])
    for cid in res:
        assert res[cid].get("error") is None, res[cid]
        assert "Thinking effort level: 0" not in res[cid]["prompt"]
    assert "<think>\n\n</think>\n\n" in res["qwen38_off"]["prompt"]
    assert "thinking: false" in res["granite42_off"]["prompt"] or "none" not in res["granite42_off"]["prompt"]
    assert "reasoning strength: none" not in res["muse_off"]["prompt"]


def test_r4423_inkling2_tool_args_free_key_order(server):
    """R4.48.23: Inkling 2 constrained tool calls accept argument keys in any
    order (e.g. optional/alphabetical key before required declared key) and
    close truncated partial reordered arguments cleanly with required keys."""
    star_tool = {
        "type": "function",
        "function": {
            "name": "github_star",
            "description": "Track GitHub stars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repos": {"type": "string"},
                    "aligned": {"type": "boolean"},
                },
                "required": ["repos"],
            },
        },
    }
    # Constrained tool call emits optional `aligned` before required `repos`.
    reordered_reply = (
        '<|message_model|>github_star<|content_invoke_tool_json|>'
        '{"name":"github_star","args":{"aligned":true,"repos":"octocat/Hello-World"}}<|end_message|>'
        '<|content_model_end_sampling|>'
    )
    code, d = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "star octocat/Hello-World"}],
            "tools": [star_tool],
            "tool_choice": "required",
            "max_tokens": 128,
            "temperature": 0,
            "runner_test_reply": reordered_reply,
        },
    )
    assert code == 200, d
    assert d["choices"][0]["finish_reason"] == "tool_calls", d
    assert d.get("runner_telemetry", {}).get("tool_protocol", {}).get("family") == "inkling2", d
    tc = d["choices"][0]["message"].get("tool_calls") or []
    assert len(tc) == 1 and tc[0]["function"]["name"] == "github_star", d
    raw_args = tc[0]["function"]["arguments"]
    assert raw_args.index('"aligned"') < raw_args.index('"repos"'), raw_args
    assert json.loads(raw_args) == {"aligned": True, "repos": "octocat/Hello-World"}

    # Truncated mid-call after emitting `aligned` first closes cleanly with required `repos`.
    code, d_cut = _post(
        server,
        "/v1/chat/completions",
        {
            "model": server.model_id,
            "messages": [{"role": "user", "content": "star octocat/Hello-World"}],
            "tools": [star_tool],
            "tool_choice": "required",
            "max_tokens": 16,
            "temperature": 0,
            "runner_test_reply": reordered_reply,
        },
    )
    assert code == 200, d_cut
    assert d_cut["choices"][0]["finish_reason"] == "length", d_cut
    cut_tc = d_cut["choices"][0]["message"].get("tool_calls") or []
    assert len(cut_tc) == 1 and cut_tc[0]["function"]["name"] == "github_star", d_cut
    cut_args = json.loads(cut_tc[0]["function"]["arguments"])
    assert "repos" in cut_args, cut_args





