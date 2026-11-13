"""Apertus thinking is split from the answer (R2.15).

Apertus deliberates between two control tokens, <|inner_prefix|> and
<|inner_suffix|>. A control token decodes to no bytes, and the reasoning
split reads bytes, so with `enable_thinking` on the whole deliberation came
back as the answer, unmarked, and `reasoning_content` was empty (measured on
the real checkpoint: a reply that opened "We need to compute ..." and ended
with the answer). A chat turn now spells those two tokens for the split.

The fixture is a small apertus-architecture model whose vocabulary carries
the family's control tokens; the reply is scripted (the server's test hook),
because a fixture cannot be made to deliberate on purpose.
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

TURN = ("<|system_start|>,<|system_end|>,<|developer_start|>,<|developer_end|>,"
        "<|user_start|>,<|user_end|>,<|assistant_start|>,<|assistant_end|>,"
        "<|tools_prefix|>,<|tools_suffix|>")
# Both releases name the pair <|inner_prefix|> and <|inner_suffix|> (ids 69
# and 70 in the 2509 and v1.5 files). <think> and </think> are other control
# tokens there; a file that carries them and no inner pair binds those, which
# is the second fixture. Its key is kept from when 2509 was believed to be
# such a file.
TAGS = {"v1.5": ("<|inner_prefix|>", "<|inner_suffix|>"),
        "2509": ("<think>", "</think>"),
        # v1.5's vocabulary also holds <|tool_output_start|> and
        # <|tool_output_end|>, which select the apertus15 renderer when a file
        # carries no template text (2509's has neither)
        "v1.5-tool-outputs": ("<|inner_prefix|>", "<|inner_suffix|>")}
EXTRA = {"v1.5-tool-outputs": ",<|tool_output_start|>,<|tool_output_end|>"}
TEMPLATE = {"v1.5": "apertus", "2509": "apertus", "v1.5-tool-outputs": "apertus15"}
REPLY = None   # set per server: the deliberation between that release's tags


@pytest.fixture(scope="module", params=sorted(TAGS))
def server(request, tmp_path_factory):
    global REPLY
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    opened, closed = TAGS[request.param]
    m = tmp_path_factory.mktemp("apthink") / f"apertus-{request.param}.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), "--apertus", "REAL",
                    "--control", f"{TURN},{opened},{closed}{EXTRA.get(request.param, '')}", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    REPLY = f"{opened}think it over{closed}the answer"
    with RunnerServer(exe, m, ctx=1024, parallel=1, extra_args=["--gpu", "off", "-t", "2"],
                      env=dict(os.environ, RUNNER_TEST_SCRIPTED_REPLY="1")) as srv:
        srv.release = request.param
        yield srv


def _post(srv, path, body, stream=False):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    if not stream:
        return 200, json.loads(raw)
    events = [json.loads(line[6:]) for line in raw.decode().splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    return 200, events


def _chat(**extra):
    body = {"messages": [{"role": "user", "content": "How long is the trip?"}],
            "max_tokens": 64, "temperature": 0, "enable_thinking": True,
            "runner_test_reply": REPLY}
    body.update(extra)
    return body


def test_the_fixture_tags_are_control_tokens(server):
    """The premise: spelled in a raw completion's reply they decode to nothing."""
    code, body = _post(server, "/v1/completions",
                       {"prompt": "x", "max_tokens": 8, "temperature": 0,
                        "runner_test_reply": REPLY.replace("think it over", "").replace("the answer", "!")})
    assert code == 200, body
    assert body["choices"][0]["text"] == "!", body


def test_the_template_is_apertus(server):
    code, body = _post(server, "/v1/chat/completions", _chat(runner_test_reply="plain"))
    assert code == 200, body
    assert body["choices"][0]["message"]["content"] == "plain"
    assert body["runner_telemetry"]["tool_protocol"]["template"] == TEMPLATE[server.release], body


def test_the_deliberation_is_reasoning_and_the_answer_is_the_answer(server):
    code, body = _post(server, "/v1/chat/completions", _chat())
    assert code == 200, body
    msg = body["choices"][0]["message"]
    assert msg["content"] == "the answer", msg
    assert msg.get("reasoning_content") == "think it over", msg


def test_a_streamed_turn_splits_the_same_way(server):
    code, events = _post(server, "/v1/chat/completions", _chat(stream=True), stream=True)
    assert code == 200, events
    reasoning = "".join((e["choices"][0]["delta"].get("reasoning_content") or "")
                        for e in events if e.get("choices"))
    content = "".join((e["choices"][0]["delta"].get("content") or "")
                      for e in events if e.get("choices"))
    assert (reasoning, content) == ("think it over", "the answer"), events


def test_a_raw_completion_keeps_the_decoded_text(server):
    """A raw completion's text is the tokens decoded, and a control token
    decodes to nothing: the two tags are not spelled into it."""
    code, body = _post(server, "/v1/completions",
                       {"prompt": "x", "max_tokens": 64, "temperature": 0,
                        "runner_test_reply": REPLY})
    assert code == 200, body
    assert body["choices"][0]["text"] == "think it overthe answer", body


def test_the_responses_api_returns_a_reasoning_item(server):
    code, body = _post(server, "/v1/responses",
                       {"input": "How long is the trip?", "max_output_tokens": 64,
                        "temperature": 0, "reasoning": {"effort": "medium"},
                        "runner_test_reply": REPLY})
    assert code == 200, body
    kinds = [o["type"] for o in body["output"]]
    assert kinds == ["reasoning", "message"], body["output"]
    assert body["output"][0]["summary"][0]["text"] == "think it over"
    assert body["output"][1]["content"][0]["text"] == "the answer"


def test_the_messages_api_accepts_thinking_and_returns_a_thinking_block(server):
    """Before, `thinking: enabled` was refused for this family: "it has no
    reasoning channel to separate"."""
    with urllib.request.urlopen(server.base_url + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    code, body = _post(server, "/v1/messages",
                       {"model": model, "max_tokens": 64, "temperature": 0,
                        "thinking": {"type": "enabled", "budget_tokens": 1024},
                        "messages": [{"role": "user", "content": "How long is the trip?"}],
                        "runner_test_reply": REPLY})
    assert code == 200, body
    assert [(c["type"], c.get("thinking") or c.get("text")) for c in body["content"]] == [
        ("thinking", "think it over"), ("text", "the answer")], body["content"]


def test_a_turn_ends_at_the_token_that_closes_a_tool_call(server):
    """The publisher's generation_config ends generation at <|tools_suffix|>
    (id 72) as well as at <|assistant_end|>; the files declare only the
    second. Without the first as a stop, a reply that wrote a call on a turn
    with no tools declared ran on and invented the call's result (measured on
    the 2509 checkpoint: `["Answers displayed"]` after a display_answers
    call)."""
    reply = ('the answer<|tools_prefix|>[{"display_answers": {}}]<|tools_suffix|>'
             '["Answers displayed"]')
    code, body = _post(server, "/v1/chat/completions",
                       _chat(enable_thinking=False, runner_test_reply=reply))
    assert code == 200, body
    ch = body["choices"][0]
    assert "Answers displayed" not in ch["message"]["content"], ch
    assert ch["finish_reason"] == "stop", ch
    assert body["usage"]["completion_tokens"] < 64, body["usage"]


TOOLS = [{"type": "function", "function": {
    "name": "trip_duration",
    "parameters": {"type": "object",
                   "properties": {"depart": {"type": "string"}, "arrive": {"type": "string"}},
                   "required": ["depart", "arrive"]}}}]
ENVELOPE = json.dumps({"calls": [{"tool": "trip_duration",
                                  "args": {"depart": "14:10", "arrive": "16:45"}}]},
                      separators=(",", ":"))


def _tools_turn(srv, reply, thinking, choice):
    return _post(srv, "/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "How long is the trip?"}],
                  "max_tokens": 96, "temperature": 0, "enable_thinking": thinking,
                  "tools": TOOLS, "tool_choice": choice, "runner_test_reply": reply})


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_a_tool_call_is_a_tool_call_with_thinking_on_or_off(server, thinking, choice):
    """The first version of this change gave the family a text think pair, and
    every constrained turn then probed for it as a prelude: on the real
    checkpoints every tools request ended in an error with the call inside
    reasoning_content (found on the Linux CUDA host before the merge). The
    family has no text pair; its control-token pair is the chat split's alone."""
    code, body = _tools_turn(server, ENVELOPE, thinking, choice)
    assert code == 200, body
    ch = body["choices"][0]
    assert ch["finish_reason"] == "tool_calls", ch
    call = ch["message"]["tool_calls"][0]["function"]
    assert call["name"] == "trip_duration"
    assert json.loads(call["arguments"]) == {"depart": "14:10", "arrive": "16:45"}
    assert not ch["message"].get("reasoning_content"), ch["message"]


@pytest.fixture(scope="module")
def no_tag_server(tmp_path_factory):
    """The family's turn markers only: the think tags are not in the vocabulary,
    so a reply can spell one as plain characters, as the real model did under
    the constraint."""
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    m = tmp_path_factory.mktemp("apthink") / "apertus-notags.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), "--apertus", "REAL",
                    "--control", TURN, str(m)], check=True, stdout=subprocess.DEVNULL)
    with RunnerServer(exe, m, ctx=1024, parallel=1, extra_args=["--gpu", "off", "-t", "2"],
                      env=dict(os.environ, RUNNER_TEST_SCRIPTED_REPLY="1")) as srv:
        yield srv


@pytest.mark.parametrize("thinking", [False, True])
def test_a_constrained_turn_has_no_think_prelude_to_spell(no_tag_server, thinking):
    """Under a tool constraint the family's turn is the envelope from its
    first byte. A reply that spells the open tag as characters is not a
    prelude: nothing of it becomes reasoning (with the text pair bound, the
    spelled tag opened a reasoning block that swallowed the call)."""
    code, body = _tools_turn(no_tag_server, "<|inner_prefix|>" + ENVELOPE, thinking, "required")
    assert code == 200, body
    assert not body["choices"][0]["message"].get("reasoning_content"), body["choices"][0]
    code, body = _tools_turn(no_tag_server, ENVELOPE, thinking, "required")
    assert code == 200 and body["choices"][0]["finish_reason"] == "tool_calls", body
