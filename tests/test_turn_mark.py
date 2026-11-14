"""A recurrent slot resumes the next agent turn at the prompt boundary.

The 2026-09-14 OpenCode loop on Qwen 3.8 (a qwen35 hybrid: recurrent
DeltaNet layers between full-attention ones) re-folded its whole 7.5K-token
prompt on every turn, "0 cached" each time. An agent client's next request
replays the previous prompt, then the reply re-rendered by the template
(reasoning stripped, tool calls re-serialised), then the tool result or a
new user turn. The kept run of the slot's history therefore ends past the
prompt but short of the position the fold sits at, and a recurrent fold
cannot be sliced to an arbitrary position: it is restorable only where a
checkpoint was taken. The turn mark is that checkpoint, taken at the end
of every prefill; engine_rewind resumes from it whenever the kept run
reaches it.

Gate, on the qwen35 CI fixture served on the CPU: request 2 replays request
1's prompt with a different assistant reply and a new user turn. Its
telemetry (`prompt_cached_tokens`, and `prompt_reuse` saying how) must say
the slot resumed from the turn mark with request 1's whole prompt cached, and its greedy answer must equal the answer the same
request gets on a fresh slot (cache_prompt:false resets the slot). The
bit-for-bit logits identity is pinned one level down in
tests/test_recurrent_rewind.c; this is the served contract.

R10.27: a chat request takes that checkpoint at the last message boundary,
where the generation prompt begins, and the telemetry calls it `turn_anchor`.
The 2026-11-09 agent loop on Qwen3.5-4B re-folded the whole conversation on
every turn all the same: the mark sat one token short of the prompt end,
inside the generation prompt, and Qwen3.5's generation prompt opens a think
block that the next request's history does not replay. The kept run ended a
token or two before the mark, so the fold reset. The message boundary is the
last position a rewritten history still replays. A raw completion has no
message boundary and keeps the mark one token short of its prompt.
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


@pytest.fixture(scope="module")
def ornith(tmp_path_factory):
    p = tmp_path_factory.mktemp("turnmark") / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", str(p)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return p


def _post(server, body, path="/v1/chat/completions"):
    req = urllib.request.Request(server.base_url + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _chat(server, messages, **fields):
    with urllib.request.urlopen(server.base_url + "/v1/models", timeout=30) as r:
        mid = json.load(r)["data"][0]["id"]
    # prefix_cache:false keeps the SHARED tier out of the way: on the CPU an
    # exact full-prefix hit there forks KV + fold and would serve request 2
    # first (see the last request); the mark is the slot's OWN tier, the one
    # an offloaded recurrent model has, and it is what this gate is about.
    body = {"model": mid, "messages": messages, "max_tokens": 6,
            "temperature": 0, "seed": 1, "prefix_cache": False}
    body.update(fields)
    return _post(server, body)


def _boundary(exe, model, messages, extra=()):
    """Where the generation prompt begins, counted by the server that pins a
    context: it renders the messages without the generation prompt. A server
    of its own, so the pin's prefill is not in the slot under test."""
    with RunnerServer(exe, model, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", *extra]) as srv:
        return _post(srv, {"id": "boundary", "messages": messages},
                     "/v1/runner/contexts")["tokens"]


def _reuse(d):
    t = d["runner_telemetry"]
    return t["prompt_reuse"], t["prompt_cached_tokens"], t["timing"]["prefill_tokens"]


def test_second_turn_resumes_from_the_turn_mark(ornith):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    first = [{"role": "user", "content": "List the files in this directory."}]
    second = first + [
        {"role": "assistant", "content": "I will run ls now."},
        {"role": "user", "content": "Now show me the biggest one."}]
    with RunnerServer(exe, ornith, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        def reuse(d):
            t = d["runner_telemetry"]
            return t["prompt_reuse"], t["prompt_cached_tokens"], t["timing"]["prefill_tokens"]

        d1 = _chat(srv, first)
        assert reuse(d1)[:2] == ("none", 0), reuse(d1)
        n1 = d1["usage"]["prompt_tokens"]
        assert reuse(d1)[2] == n1, (reuse(d1), n1)

        # the mark sits one token short of the prompt end (that token is
        # always fed, it is what produces the logits), so everything before
        # it is behind the mark and nothing before it is fed again. The
        # fixture's own template writes no generation prompt, so there is no
        # message boundary short of the prompt end to take instead (R10.27).
        assert _boundary(exe, ornith, first) == n1
        d2 = _chat(srv, second)
        n2 = d2["usage"]["prompt_tokens"]
        assert reuse(d2) == ("turn_mark", n1 - 1, n2 - (n1 - 1)), (reuse(d2), n1, n2)

        # an identical replay resumes at the new mark and feeds one token
        d4 = _chat(srv, second)
        assert reuse(d4) == ("turn_mark", n2 - 1, 1), reuse(d4)
        assert d4["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]

        # the same request on a fresh slot gives the same greedy answer;
        # cache_prompt:false resets the slot and marks nothing
        d3 = _chat(srv, second, cache_prompt=False)
        assert reuse(d3)[:2] == ("none", 0), reuse(d3)
        assert d3["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]
        d5 = _chat(srv, second)
        assert reuse(d5)[:2] == ("recurrent_reset", 0), reuse(d5)
        assert d5["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]

        # with the shared tier on, a CPU slot forks the exact published
        # prefix (KV + fold blob, tracer 5) and the telemetry names that tier
        _chat(srv, first, prefix_cache=True)
        d7 = _chat(srv, second, prefix_cache=True)
        assert reuse(d7)[:2] == ("prefix_cache", n1), reuse(d7)
        assert d7["choices"][0]["message"]["content"] == d2["choices"][0]["message"]["content"]


# R10.27, the rewrite an agent loop makes on a Qwen3.5-class hybrid: the
# generation prompt opens a think block (or, with thinking off, an empty
# one), and the next request's history renders the reply without it.
@pytest.mark.parametrize("thinking", [None, False])
def test_a_reply_rendered_without_its_think_block_resumes_at_the_message_boundary(ornith, thinking):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    tmpl = ("--chat-template", "qwen35")
    kw = {} if thinking is None else {"enable_thinking": thinking}
    first = [{"role": "user", "content": "List the files in this directory."}]
    second = first + [
        {"role": "assistant", "content": "I will run ls now."},
        {"role": "user", "content": "Now show me the biggest one."}]
    third = second + [
        {"role": "assistant", "content": "It is a.txt."},
        {"role": "user", "content": "Thanks."}]
    # an earlier turn edited: nothing after the first message replays
    edited = [{"role": "user", "content": "List the files in that directory."}] + third[1:]
    a1, a2, a3 = (_boundary(exe, ornith, m, tmpl) for m in (first, second, third))
    with RunnerServer(exe, ornith, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", *tmpl]) as srv:
        d1 = _chat(srv, first, **kw)
        n1 = d1["usage"]["prompt_tokens"]
        assert _reuse(d1) == ("none", 0, n1), _reuse(d1)
        # the generation prompt is more than the one token the old mark
        # stood back from the prompt end
        assert a1 < n1 - 1, (a1, n1)

        d2 = _chat(srv, second, **kw)
        n2 = d2["usage"]["prompt_tokens"]
        assert _reuse(d2) == ("turn_anchor", a1, n2 - a1), (_reuse(d2), a1, n2)
        d3 = _chat(srv, third, **kw)
        n3 = d3["usage"]["prompt_tokens"]
        assert _reuse(d3) == ("turn_anchor", a2, n3 - a2), (_reuse(d3), a2, n3)
        # a verbatim replay feeds the generation prompt alone
        d4 = _chat(srv, third, **kw)
        assert _reuse(d4) == ("turn_anchor", a3, n3 - a3), (_reuse(d4), a3, n3)

        # the served greedy answers are a fresh slot's
        for d, msgs in ((d2, second), (d3, third), (d4, third)):
            cold = _chat(srv, msgs, cache_prompt=False, **kw)
            assert _reuse(cold)[:2] == ("none", 0), _reuse(cold)
            assert cold["choices"][0]["message"] == d["choices"][0]["message"], msgs

        # a history edited before the boundary resets the fold, and says so
        _chat(srv, third, **kw)
        d5 = _chat(srv, edited, **kw)
        assert _reuse(d5)[:2] == ("recurrent_reset", 0), _reuse(d5)


def test_a_raw_completion_keeps_the_mark_one_token_short_of_its_prompt(ornith):
    """No messages, no boundary: /v1/completions marks at n - 1 as before."""
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    with RunnerServer(exe, ornith, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        def raw(prompt):
            return _post(srv, {"prompt": prompt, "max_tokens": 4, "temperature": 0,
                               "seed": 1, "prefix_cache": False}, "/v1/completions")
        d1 = raw("the river runs past the mill")
        n1 = d1["usage"]["prompt_tokens"]
        d2 = raw("the river runs past the mill and the old bridge")
        n2 = d2["usage"]["prompt_tokens"]
        assert _reuse(d2) == ("turn_mark", n1 - 1, n2 - (n1 - 1)), (_reuse(d2), n1, n2)


# The Responses and Messages surfaces render their own prompts, and an agent
# reaches a hybrid model through them as often as through chat: the boundary
# is found where every surface's prompt is rendered, not on the chat route.
@pytest.mark.parametrize("path, field, cap", [("/v1/responses", "input", "max_output_tokens"),
                                              ("/v1/messages", "messages", "max_tokens")])
def test_the_responses_and_messages_surfaces_resume_at_the_message_boundary(ornith, path, field, cap):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    tmpl = ("--chat-template", "qwen35")
    first = [{"role": "user", "content": "List the files in this directory."}]
    second = first + [
        {"role": "assistant", "content": "I will run ls now."},
        {"role": "user", "content": "Now show me the biggest one."}]
    a1 = _boundary(exe, ornith, first, tmpl)
    with RunnerServer(exe, ornith, ctx=512, parallel=1,
                      extra_args=["--gpu", "off", "-t", "2", *tmpl]) as srv:
        with urllib.request.urlopen(srv.base_url + "/v1/models", timeout=30) as r:
            mid = json.load(r)["data"][0]["id"]
        def ask(msgs):
            return _post(srv, {"model": mid, field: msgs, cap: 6, "temperature": 0,
                               "seed": 1, "prefix_cache": False}, path)
        d1 = ask(first)
        assert _reuse(d1)[:2] == ("none", 0), _reuse(d1)
        d2 = ask(second)
        n2 = d2["usage"]["input_tokens"]
        assert _reuse(d2) == ("turn_anchor", a1, n2 - a1), (_reuse(d2), a1, n2)
