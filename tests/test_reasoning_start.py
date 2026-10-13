"""Where a chat turn's generation starts: inside a thought block or in content.

Several families open the thought block in the generation prompt itself, so
the model's first token is reasoning, and the server has to split the stream
from that state rather than wait for a `<think>` it will never see. The
families and modes, as their templates render them (src/template.c) and the
completion path reads them (run_completion):

  granite42, nemotron35, qwen38, qwen35, ornith: open in every mode but off
  qwen35-nothink: opens only when thinking is explicitly on
  nemotron (Nano): from the conversation's own /think and /no_think, with an
                   explicit request setting winning
  chatml, qwen3-coder: never open (controls)

Observed with the fixture model and greedy decoding: its output never closes
a thought block, so a primed start shows as reasoning and no content, an
unprimed one as content and no reasoning, on every surface: chat buffered and
streamed, Responses (a reasoning item) and Messages (a thinking block). The
cases pin the start state the server derives today, so moving that decision
into the renderer cannot change it unnoticed.
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

PRIMED = {  # family -> modes whose generation starts in reasoning
    "granite42": {"default", "on"},
    "nemotron35": {"default", "on"},
    "qwen38": {"default", "on"},
    "qwen35": {"default", "on"},
    "ornith": {"default", "on"},
    "qwen35-nothink": {"on"},
    "nemotron": {"default", "on"},
    "chatml": set(),
    "qwen3-coder": set(),
}
MODES = {"default": {}, "on": {"enable_thinking": True}, "off": {"enable_thinking": False}}


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("rs") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def stream_chat(srv, body):
    """Concatenated content and reasoning deltas of a streamed chat turn."""
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                 data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    content, reasoning = "", ""
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices") or []:
                d = ch.get("delta") or {}
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
    return content, reasoning


def chat(srv, messages, mode):
    body = {"messages": messages, "max_tokens": 12, "temperature": 0, **MODES[mode]}
    status, doc = post(srv, "/v1/chat/completions", body)
    assert status == 200, doc
    msg = doc["choices"][0]["message"]
    return body, msg.get("content") or "", msg.get("reasoning_content") or ""


@pytest.mark.parametrize("family", sorted(PRIMED))
def test_generation_starts_where_the_prompt_left_it(exe, model, family):
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", "-s", "42",
                                  "--chat-template", family]) as srv:
        mid = json.load(urllib.request.urlopen(srv.base_url + "/v1/models"))["data"][0]["id"]
        for mode in MODES:
            primed = mode in PRIMED[family]
            body, content, reasoning = chat(srv, [{"role": "user", "content": "Hello"}], mode)
            where = "%s, thinking %s" % (family, mode)
            if primed:
                assert reasoning and not content, (where, content, reasoning)
            else:
                assert content and not reasoning, (where, content, reasoning)
            # streamed: the same split, delta by delta
            assert stream_chat(srv, body) == (content, reasoning), where

            status, doc = post(srv, "/v1/responses", {"input": "Hello", "max_output_tokens": 12,
                                                      "temperature": 0, **MODES[mode]})
            assert status == 200, doc
            kinds = [o.get("type") for o in doc.get("output") or []]
            assert (kinds == ["reasoning", "message"]) if primed else (kinds == ["message"]), (where, kinds)

            status, doc = post(srv, "/v1/messages", {"model": mid, "max_tokens": 12, "temperature": 0,
                                                     "messages": [{"role": "user", "content": "Hello"}],
                                                     **MODES[mode]})
            assert status == 200, doc
            blocks = [b.get("type") for b in doc.get("content") or []]
            assert blocks == (["thinking"] if primed else ["text"]), (where, blocks)


def test_nemotron_nano_reads_its_own_control_strings(exe, model):
    """The conversation's last /think or /no_think decides; an explicit
    request setting wins over both."""
    cases = [  # (user text, mode, starts in reasoning)
        ("Hello /think", "default", True),
        ("Hello /no_think", "default", False),
        ("Hello /no_think", "on", True),
        ("Hello /think", "off", False),
    ]
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", "-s", "42",
                                  "--chat-template", "nemotron"]) as srv:
        for text, mode, primed in cases:
            _, content, reasoning = chat(srv, [{"role": "user", "content": text}], mode)
            if primed:
                assert reasoning and not content, (text, mode, content, reasoning)
            else:
                assert content and not reasoning, (text, mode, content, reasoning)


ANTHROPIC_THINKING = {"on": {"thinking": {"type": "enabled", "budget_tokens": 1024}},
                      "off": {"thinking": {"type": "disabled"}}}


def stream_block_types(srv, body):
    """content_block_start types of a streamed Messages turn."""
    req = urllib.request.Request(srv.base_url + "/v1/messages",
                                 data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    kinds = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data:"):
                ev = json.loads(line[5:])
                if ev.get("type") == "content_block_start":
                    kinds.append(ev["content_block"]["type"])
    return kinds


@pytest.mark.parametrize("family", ["granite42", "nemotron35", "qwen38", "qwen35", "qwen35-nothink"])
def test_messages_thinking_object_decides_where_generation_starts(exe, model, family):
    """Anthropic's own spelling of the mode, `thinking: {"type": ...}`, is the
    one a Messages client sends. The prompt is rendered under it, so the turn
    has to be split from where that prompt left the model -- not from where
    `enable_thinking`, absent here, would have left it."""
    with RunnerServer(exe, model, ctx=1024, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", "-s", "42",
                                  "--chat-template", family]) as srv:
        mid = json.load(urllib.request.urlopen(srv.base_url + "/v1/models"))["data"][0]["id"]
        for mode, extra in ANTHROPIC_THINKING.items():
            primed = mode in PRIMED[family]
            body = {"model": mid, "max_tokens": 12, "temperature": 0,
                    "messages": [{"role": "user", "content": "Hello"}], **extra}
            status, doc = post(srv, "/v1/messages", body)
            assert status == 200, doc
            blocks = [b.get("type") for b in doc.get("content") or []]
            want = ["thinking"] if primed else ["text"]
            assert blocks == want, (family, extra, blocks)
            assert stream_block_types(srv, body) == want, (family, extra, "streamed")


G4_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
G4_CONTINUATION = [
    {"role": "user", "content": "Weather in Oslo, then Bergen?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":12}"}]
# what the model writes after a prompt that opened its thought channel: the
# thought, the channel's close, then the next call
G4_REPLY = "Oslo is done; now Bergen.<channel|><|tool_call>call:get_weather{city:<|\"|>Bergen<|\"|>}<tool_call|>"


@pytest.mark.parametrize("family", ["gemma4", "gemma4-mainline"])
def test_gemma4_continuation_reads_the_thought_its_prompt_opened(exe, model, family):
    """With thinking on, a Gemma 4 prompt that continues after a tool result
    ends by opening the thought channel. The grammar, the stream
    demultiplexer and the buffered mapper all start inside it, so the model's
    thought is reasoning and the call after it is a call, buffered and
    streamed alike."""
    with RunnerServer(exe, model, ctx=4096, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", "-s", "42", "--chat-template", family],
                      env={"RUNNER_TEST_SCRIPTED_REPLY": "1"}) as srv:
        body = {"messages": G4_CONTINUATION, "tools": G4_TOOLS, "enable_thinking": True,
                "max_tokens": 96, "temperature": 0, "runner_test_reply": G4_REPLY}
        status, doc = post(srv, "/v1/chat/completions", body)
        assert status == 200, doc
        msg = doc["choices"][0]["message"]
        assert msg.get("reasoning_content") == "Oslo is done; now Bergen.", msg
        calls = msg.get("tool_calls") or []
        assert [c["function"]["name"] for c in calls] == ["get_weather"], msg
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Bergen"}, calls
        assert not (msg.get("content") or "").strip(), msg
        # streamed: the same reasoning and the same call
        req = urllib.request.Request(srv.base_url + "/v1/chat/completions",
                                     data=json.dumps(dict(body, stream=True)).encode(),
                                     headers={"Content-Type": "application/json"})
        reasoning, names, args = "", [], ""
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                for ch in json.loads(line[5:]).get("choices") or []:
                    d = ch.get("delta") or {}
                    reasoning += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if fn.get("name"): names.append(fn["name"])
                        args += fn.get("arguments") or ""
        assert reasoning == "Oslo is done; now Bergen.", reasoning
        assert names == ["get_weather"] and json.loads(args) == {"city": "Bergen"}, (names, args)


@pytest.fixture(scope="module")
def qwen3_model(tmp_path_factory):
    """A qwen3-architecture fixture: the architecture gives it `<think>` tags,
    which the CLI chat splits on (model.c); the default fixture has none."""
    m = tmp_path_factory.mktemp("rs-q3") / "q3.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--arch", "qwen3",
                    "--qk-norm", str(m)], check=True, stdout=subprocess.DEVNULL)
    return m


@pytest.mark.parametrize("family,flag,primed", [
    ("qwen38", None, True), ("qwen38", "--no-think", False), ("qwen38", "--think", True),
    ("qwen35-nothink", None, False), ("qwen35-nothink", "--think", True),
    ("granite42", None, True), ("granite42", "--no-think", False),
    ("chatml", None, False), ("chatml", "--think", False),
])
def test_cli_chat_starts_where_its_prompt_left_the_turn(exe, qwen3_model, family, flag, primed):
    """`runner -i` reads the same render_state as the server: a prompt that
    opened the thought block shows its first tokens as [thinking], one that
    did not shows them as the answer."""
    argv = [str(exe), "-m", str(qwen3_model), "--chat-template", family, "-i", "-n", "8",
            "--temp", "0", "--no-tray", "-c", "2048"] + ([flag] if flag else [])
    p = subprocess.run(argv, input=b"Hello\n", capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr[-400:]
    assert (b"[thinking]" in p.stdout) == primed, (family, flag, p.stdout[-200:])
