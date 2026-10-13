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
