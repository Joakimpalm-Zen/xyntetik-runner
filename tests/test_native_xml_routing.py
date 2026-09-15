"""Native function/parameter XML through the server, on known bytes.

The 2026-09-14 Windows report: Qwen 3.8 (and by the same routing Granite 4.2
and Ornith) emitted a well-formed `<tool_call><function=...>` block and the
STREAMED turn delivered it as assistant content, `finish_reason: stop`, no
`tool_calls[]`; the buffered turn parsed it. Every agent client streams.

The model here is the CI fixture, which cannot be made to say anything in
particular, so the server is started with the scripted-reply hook
(RUNNER_TEST_SCRIPTED_REPLY=1, `runner_test_reply` on the request) and the
reply is dictated. What is under test is everything between the sampler and
the wire: the demultiplexer routing, the parser, the finish reason, the three
API surfaces, buffered and streamed. Model quality is a different suite.

The fixture's byte-fallback vocabulary spells the XML in many short tokens,
so the demultiplexer sees the framing arrive in pieces, as it does from a
real checkpoint.
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

BASH = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}
GLOB = {"type": "function", "function": {
    "name": "glob", "description": "Find files matching a glob pattern",
    "parameters": {"type": "object",
                   "properties": {"pattern": {"type": "string"}},
                   "required": ["pattern"]}}}

# the report's case A reply, byte for byte
ONE_CALL = ("<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n"
            "</parameter>\n</function>\n</tool_call>")


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# the report's case B reply: prose, then one block per pattern
THREE_CALLS = ("I'll check for each file.\n" +
               _call("glob", pattern="**/boss.json") + "\n" +
               _call("glob", pattern="**/boss.toml") + "\n" +
               _call("glob", pattern="**/boss.yaml"))
NO_THINK = {"enable_thinking": False}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("xml") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _server(model, template):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    return RunnerServer(exe, model, ctx=4096, parallel=1,
                        extra_args=["--gpu", "off", "-t", "2",
                                    "--chat-template", template],
                        env={"RUNNER_TEST_SCRIPTED_REPLY": "1"})


@pytest.fixture(scope="module")
def qwen38(model):
    with _server(model, "qwen38") as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def _post(server, path, payload, stream=False):
    req = urllib.request.Request(server.base_url + path,
                                 data=json.dumps(payload).encode(),
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


def _chat_stream(events):
    """Fold a chat stream into (content, calls-by-index, finish_reason)."""
    assert events and events[-1] == "[DONE]", events
    content, finish, calls = "", None, {}
    for e in events[:-1]:
        for ch in e.get("choices", []):
            delta = ch.get("delta", {})
            content += delta.get("content", "") or ""
            for tc in delta.get("tool_calls", []) or []:
                c = calls.setdefault(tc["index"], {"name": None, "args": "",
                                                   "id": None})
                fn = tc.get("function", {})
                if fn.get("name"):
                    c["name"] = fn["name"]
                if tc.get("id"):
                    c["id"] = tc["id"]
                c["args"] += fn.get("arguments", "") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return content, calls, finish


def _no_framing(text):
    for marker in ("<tool_call", "<function=", "<parameter="):
        assert marker not in text, text


def test_qwen38_streams_one_native_call_as_tool_calls(qwen38):
    """The report's case A, streamed: a single bare XML block at the family's
    shipped defaults (no tool_choice, no template kwargs) must arrive as a
    tool_calls delta, never as content."""
    events = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "stream": True, "max_tokens": 400,
        "messages": [{"role": "user",
                      "content": "List the files in the current directory."}],
        "tools": [BASH],
        "chat_template_kwargs": {"enable_thinking": False},
        "runner_test_reply": ONE_CALL}, stream=True)
    content, calls, finish = _chat_stream(events)
    assert finish == "tool_calls", (finish, content, calls)
    assert list(calls) == [0], calls
    assert calls[0]["name"] == "bash", calls
    assert json.loads(calls[0]["args"]) == {"command": "ls -la"}, calls
    _no_framing(content)
    assert content == "", content


def test_qwen38_buffered_one_native_call(qwen38):
    """The buffered turn already parsed this; it stays the regression control
    for the streamed one."""
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 400,
        "messages": [{"role": "user",
                      "content": "List the files in the current directory."}],
        "tools": [BASH],
        "chat_template_kwargs": {"enable_thinking": False},
        "runner_test_reply": ONE_CALL})
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", d
    calls = ch["message"]["tool_calls"]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "bash", d
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}
    _no_framing(ch["message"].get("content") or "")


def test_qwen38_streams_prose_and_three_calls(qwen38):
    """The report's case B, streamed: the prose stays content, each block is
    its own tool_calls index, and the turn ends with tool_calls."""
    events = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "stream": True, "max_tokens": 600,
        "messages": [{"role": "user", "content": "Check whether boss.json, "
                      "boss.toml and boss.yaml exist anywhere in this project."}],
        "tools": [GLOB, BASH], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": THREE_CALLS}, stream=True)
    content, calls, finish = _chat_stream(events)
    assert finish == "tool_calls", (finish, content)
    assert content == "I'll check for each file.\n", content
    assert sorted(calls) == [0, 1, 2], calls
    assert [json.loads(calls[i]["args"])["pattern"] for i in (0, 1, 2)] == \
        ["**/boss.json", "**/boss.toml", "**/boss.yaml"], calls
    assert all(calls[i]["name"] == "glob" for i in calls)
    ids = [calls[i]["id"] for i in (0, 1, 2)]
    assert len(set(ids)) == 3 and all(ids), ids


def test_qwen38_buffered_prose_and_three_calls(qwen38):
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 600,
        "messages": [{"role": "user", "content": "Check the three files."}],
        "tools": [GLOB, BASH], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": THREE_CALLS})
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", d
    assert ch["message"]["content"] == "I'll check for each file.\n", d
    calls = ch["message"]["tool_calls"]
    assert [json.loads(c["function"]["arguments"])["pattern"] for c in calls] == \
        ["**/boss.json", "**/boss.toml", "**/boss.yaml"], d
    assert len({c["id"] for c in calls}) == 3


def test_qwen38_default_thinking_reasons_then_calls(qwen38):
    """At the family's default (thinking on, xhigh) the generation prompt
    opens the reasoning block, so the stream starts inside it: the reasoning
    goes to reasoning_content, the prose to content, the call to
    tool_calls, streamed and buffered alike."""
    reply = ("The user wants a listing.\n</think>\n\nListing now.\n" +
             _call("bash", command="ls"))
    body = {"model": qwen38.model_id, "max_tokens": 600,
            "messages": [{"role": "user", "content": "List the files."}],
            "tools": [BASH], "runner_test_reply": reply}
    events = _post(qwen38, "/v1/chat/completions", dict(body, stream=True),
                   stream=True)
    content, calls, finish = _chat_stream(events)
    reasoning = "".join((e.get("choices") or [{}])[0].get("delta", {})
                        .get("reasoning_content") or ""
                        for e in events[:-1] if isinstance(e, dict))
    assert finish == "tool_calls", (finish, content, reasoning)
    assert "listing" in reasoning.lower(), reasoning
    assert content.strip() == "Listing now.", content
    _no_framing(content); _no_framing(reasoning)
    assert json.loads(calls[0]["args"]) == {"command": "ls"}, calls
    d = _post(qwen38, "/v1/chat/completions", body)
    msg = d["choices"][0]["message"]
    assert d["choices"][0]["finish_reason"] == "tool_calls", d
    assert "listing" in (msg.get("reasoning_content") or "").lower(), d
    assert (msg.get("content") or "").strip() == "Listing now.", d
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}


def test_qwen38_budget_cut_inside_a_call_keeps_length(qwen38):
    """A token budget that cuts the third block: the two complete calls
    arrive, the partial one is dropped (never completed), and the turn says
    length on both paths."""
    body = {"model": qwen38.model_id,
            "messages": [{"role": "user", "content": "Check the three files."}],
            "tools": [GLOB], "chat_template_kwargs": NO_THINK,
            "runner_test_reply": THREE_CALLS}
    # find a budget that lands inside the third block: the fixture spells the
    # reply a byte per token (it has spaces), so the count is the byte offset
    cut = len(THREE_CALLS) - len(_call("glob", pattern="**/boss.yaml")) + 25
    body["max_tokens"] = cut
    d = _post(qwen38, "/v1/chat/completions", body)
    ch = d["choices"][0]
    assert ch["finish_reason"] == "length", d
    calls = ch["message"].get("tool_calls") or []
    assert [json.loads(c["function"]["arguments"])["pattern"] for c in calls] == \
        ["**/boss.json", "**/boss.toml"], d
    _no_framing(ch["message"].get("content") or "")
    events = _post(qwen38, "/v1/chat/completions", dict(body, stream=True),
                   stream=True)
    content, scalls, finish = _chat_stream(events)
    assert finish == "length", (finish, content)
    assert sorted(scalls) == [0, 1], scalls
    _no_framing(content)


def test_qwen38_invalid_block_is_reported_not_served(qwen38):
    """A block that is not a valid call (an undeclared function) is neither
    content nor a call: the valid call still arrives, nothing is invented for
    the bad one, and the finish reason names the protocol fault."""
    reply = _call("nosuch", x="1") + "\n" + _call("bash", command="ls")
    body = {"model": qwen38.model_id, "max_tokens": 600,
            "messages": [{"role": "user", "content": "Do it."}],
            "tools": [BASH], "chat_template_kwargs": NO_THINK,
            "runner_test_reply": reply}
    d = _post(qwen38, "/v1/chat/completions", body)
    ch = d["choices"][0]
    assert ch["finish_reason"] == "error", d
    assert d["runner_telemetry"]["finish_detail"] == "envelope_unmapped", d
    calls = ch["message"].get("tool_calls") or []
    assert len(calls) == 1 and calls[0]["function"]["name"] == "bash", d
    _no_framing(ch["message"].get("content") or "")
    events = _post(qwen38, "/v1/chat/completions", dict(body, stream=True),
                   stream=True)
    content, scalls, finish = _chat_stream(events)
    assert finish == "error", (finish, content)
    assert list(scalls) == [0] and scalls[0]["name"] == "bash", scalls
    _no_framing(content)


def test_qwen38_prose_mentioning_the_tag_is_content(qwen38):
    reply = "Wrap calls in a <tool_call> block, as the docs say."
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 600,
        "messages": [{"role": "user", "content": "How do I call tools?"}],
        "tools": [BASH], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": reply})
    ch = d["choices"][0]
    assert ch["finish_reason"] == "stop", d
    assert ch["message"]["content"] == reply, d
    assert not ch["message"].get("tool_calls"), d
    events = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 600, "stream": True,
        "messages": [{"role": "user", "content": "How do I call tools?"}],
        "tools": [BASH], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": reply}, stream=True)
    content, calls, finish = _chat_stream(events)
    assert (finish, content, calls) == ("stop", reply, {})


TYPED = {"type": "function", "function": {
    "name": "edit", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "line": {"type": "integer"},
        "dry_run": {"type": "boolean"}, "tags": {"type": "array",
                                                 "items": {"type": "string"}},
        "opts": {"type": "object", "properties": {"indent": {"type": "integer"}}},
        "note": {"type": "string"}},
        "required": ["path", "line"]}}}


def test_qwen38_typed_arguments_keep_their_types(qwen38):
    """Declared types drive the mapping: integers and booleans are typed,
    arrays and objects are JSON, strings stay strings even when they look
    like numbers, and Unicode plus quotes survive."""
    reply = _call("edit", path="src/m\u00fcll.c", line="42", dry_run="true",
                  tags='["a", "b"]', opts='{"indent": 4}',
                  note='say "hi" \\ 123')
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 600,
        "messages": [{"role": "user", "content": "Edit it."}],
        "tools": [TYPED], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": reply})
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", d
    args = json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "src/m\u00fcll.c", "line": 42, "dry_run": True,
                    "tags": ["a", "b"], "opts": {"indent": 4},
                    "note": 'say "hi" \\ 123'}, args


def test_qwen38_responses_surface_streams_the_call(qwen38):
    body = {"model": qwen38.model_id, "max_output_tokens": 600,
            "input": [{"role": "user", "content": "List the files."}],
            "tools": [{"type": "function", "name": "bash",
                       "parameters": BASH["function"]["parameters"]}],
            # the family's default opens the reasoning block; close it first
            "runner_test_reply": "Plan.\n</think>\n\nListing.\n" + _call("bash", command="ls")}
    d = _post(qwen38, "/v1/responses", body)
    kinds = [o["type"] for o in d["output"]]
    assert "function_call" in kinds, d
    fc = [o for o in d["output"] if o["type"] == "function_call"][0]
    assert fc["name"] == "bash" and json.loads(fc["arguments"]) == {"command": "ls"}
    assert d["status"] == "completed", d
    raw = _post(qwen38, "/v1/responses", dict(body, stream=True), stream=True)
    names = [e.get("type") for e in raw if isinstance(e, dict)]
    assert "response.output_item.added" in names, names
    args = "".join(e.get("delta", "") for e in raw if isinstance(e, dict) and
                   e.get("type") == "response.function_call_arguments.delta")
    assert json.loads(args) == {"command": "ls"}, raw
    done = [e for e in raw if isinstance(e, dict) and e.get("type") == "response.completed"]
    assert done and done[0]["response"]["status"] == "completed", raw
    text = "".join(e.get("delta", "") for e in raw if isinstance(e, dict) and
                   e.get("type") == "response.output_text.delta")
    _no_framing(text)


def test_qwen38_messages_surface_streams_the_call(qwen38):
    body = {"model": qwen38.model_id, "max_tokens": 600,
            "messages": [{"role": "user", "content": "List the files."}],
            "tools": [{"name": "bash", "description": "Run a shell command",
                       "input_schema": BASH["function"]["parameters"]}],
            "runner_test_reply": "Plan.\n</think>\n\nListing.\n" + _call("bash", command="ls")}
    d = _post(qwen38, "/v1/messages", body)
    assert d["stop_reason"] == "tool_use", d
    uses = [b for b in d["content"] if b["type"] == "tool_use"]
    assert uses and uses[0]["name"] == "bash" and uses[0]["input"] == {"command": "ls"}, d
    for b in d["content"]:
        if b["type"] == "text":
            _no_framing(b["text"])
    raw = _post(qwen38, "/v1/messages", dict(body, stream=True), stream=True)
    starts = [e for e in raw if isinstance(e, dict) and
              e.get("type") == "content_block_start" and
              e["content_block"]["type"] == "tool_use"]
    assert starts and starts[0]["content_block"]["name"] == "bash", raw
    partial = "".join(e["delta"].get("partial_json", "") for e in raw
                      if isinstance(e, dict) and e.get("type") == "content_block_delta"
                      and e["delta"].get("type") == "input_json_delta")
    assert json.loads(partial) == {"command": "ls"}, raw
    stops = [e for e in raw if isinstance(e, dict) and e.get("type") == "message_delta"]
    assert stops and stops[-1]["delta"]["stop_reason"] == "tool_use", raw


def test_qwen38_tool_result_round_trip(qwen38):
    """Call, client execution, result, answer: the result turn is attributed by
    tool_call_id and the next scripted turn is plain prose."""
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 600,
        "messages": [{"role": "user", "content": "List the files."}],
        "tools": [BASH], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": _call("bash", command="ls")})
    msg = d["choices"][0]["message"]
    call = msg["tool_calls"][0]
    d2 = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 600,
        "messages": [{"role": "user", "content": "List the files."},
                     {"role": "assistant", "content": msg.get("content") or "",
                      "tool_calls": msg["tool_calls"]},
                     {"role": "tool", "tool_call_id": call["id"],
                      "content": "README.md\nsrc\n"}],
        "tools": [BASH], "chat_template_kwargs": NO_THINK,
        "runner_test_reply": "Two entries: README.md and src."})
    ch = d2["choices"][0]
    assert ch["finish_reason"] == "stop", d2
    assert ch["message"]["content"] == "Two entries: README.md and src.", d2
    assert not ch["message"].get("tool_calls"), d2


@pytest.mark.parametrize("template", ["granite42", "ornith"])
def test_other_xml_families_stream_the_call(model, template):
    with _server(model, template) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            mid = json.load(r)["data"][0]["id"]
        events = _post(srv, "/v1/chat/completions", {
            "model": mid, "stream": True, "max_tokens": 600,
            "messages": [{"role": "user", "content": "Check the three files."}],
            "tools": [GLOB, BASH], "chat_template_kwargs": NO_THINK,
            "runner_test_reply": THREE_CALLS}, stream=True)
        content, calls, finish = _chat_stream(events)
        assert finish == "tool_calls", (template, finish, content)
        assert sorted(calls) == [0, 1, 2], (template, calls)
        assert content == "I'll check for each file.\n", (template, content)


@pytest.mark.parametrize("template", ["granite42", "ornith", "qwen38"])
def test_three_surfaces_render_the_same_declared_prompt(model, template):
    """Chat, Responses and Messages teach a declared tool identically: the
    same system text, tools and user turn tokenize to the same prompt on all
    three. Ornith and Granite 4.2 render their declarations outside the
    template and fold the caller's system text into that turn the way their
    references do (ornith: declarations then the text, granite 4.2: the text
    then the declarations); the Chat surface did, the typed surfaces rendered
    a second system turn, so the same conversation was a different prompt
    depending on the door it came through."""
    system = "You are a careful assistant. Answer briefly."
    user = "Check the three files."
    with _server(model, template) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            mid = json.load(r)["data"][0]["id"]
        chat = _post(srv, "/v1/chat/completions", {
            "model": mid, "max_tokens": 4,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "tools": [GLOB, BASH], "chat_template_kwargs": NO_THINK,
            "runner_test_reply": "Ok."})
        responses = _post(srv, "/v1/responses", {
            "model": mid, "max_output_tokens": 4,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "tools": [{"type": "function", "name": t["function"]["name"],
                       "description": t["function"]["description"],
                       "parameters": t["function"]["parameters"]}
                      for t in (GLOB, BASH)],
            "reasoning": {"effort": "low"}, "runner_test_reply": "Ok.",
            **({"chat_template_kwargs": NO_THINK})})
        messages = _post(srv, "/v1/messages", {
            "model": mid, "max_tokens": 4, "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": t["function"]["name"],
                       "description": t["function"]["description"],
                       "input_schema": t["function"]["parameters"]}
                      for t in (GLOB, BASH)],
            "chat_template_kwargs": NO_THINK, "runner_test_reply": "Ok."})
        n_chat = chat["usage"]["prompt_tokens"]
        n_resp = responses["usage"]["input_tokens"]
        n_msg = messages["usage"]["input_tokens"]
        assert n_chat == n_resp == n_msg, (template, n_chat, n_resp, n_msg)


def test_reasoning_only_turn_reports_no_first_visible_byte(qwen38):
    """A turn that never leaves its thought block produced nothing visible:
    the stage is null, not a number that pretends a byte reached the client."""
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Think about it."}],
        "runner_test_reply": "Still thinking about the files"})
    tm = d["runner_telemetry"]["timing"]
    assert tm["first_visible_seconds"] is None, tm
    assert d["choices"][0]["message"].get("reasoning_content"), d
    d = _post(qwen38, "/v1/chat/completions", {
        "model": qwen38.model_id, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Think about it."}],
        "runner_test_reply": "Plan.\n</think>\n\nDone."})
    tm = d["runner_telemetry"]["timing"]
    assert tm["first_visible_seconds"] is not None and tm["first_visible_seconds"] >= tm["prefill_seconds"], tm


def test_scripted_reply_is_refused_without_the_hook(model):
    exe = find_runner(ROOT)
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2",
                                  "--chat-template", "qwen38"]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            mid = json.load(r)["data"][0]["id"]
        req = urllib.request.Request(
            srv.base_url + "/v1/chat/completions",
            data=json.dumps({"model": mid, "max_tokens": 4,
                             "messages": [{"role": "user", "content": "hi"}],
                             "runner_test_reply": "x"}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60)
            assert False, "the hook must be refused when not enabled"
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code
            assert "RUNNER_TEST_SCRIPTED_REPLY" in e.read().decode()



@pytest.fixture(scope="module")
def gemma4(model):
    with _server(model, "gemma4-mainline") as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            srv.model_id = json.load(r)["data"][0]["id"]
        yield srv


def test_gemma4_prose_then_call_is_a_call(gemma4):
    """Gemma 4's own format is call-first, but the report's case C (a user
    asking for a word before the call) had the 12B QAT model write prose and
    then its native call, and the turn served the whole thing as content with
    the framing in it, on both presets. The prose branch of the turn grammar
    now hands off to the call at the family's opener, and the demultiplexer
    and the buffered map read a call after prose."""
    reply = ("I will list the files.\n\n"
             "<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>")
    body = {"model": gemma4.model_id, "max_tokens": 200,
            "messages": [{"role": "user", "content": "Say what you will do, then list."}],
            "tools": [BASH], "runner_test_reply": reply}
    d = _post(gemma4, "/v1/chat/completions", body)
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls", d
    calls = ch["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "bash", d
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}, d
    assert (ch["message"].get("content") or "").strip() == "I will list the files.", d
    _no_framing(ch["message"].get("content") or "")
    events = _post(gemma4, "/v1/chat/completions", dict(body, stream=True), stream=True)
    content, scalls, finish = _chat_stream(events)
    assert finish == "tool_calls", (finish, content)
    assert list(scalls) == [0] and json.loads(scalls[0]["args"]) == {"command": "ls"}, scalls
    assert content.strip() == "I will list the files.", content
    for m in ("<|tool_call", "<tool_call|>"):
        assert m not in content, content
