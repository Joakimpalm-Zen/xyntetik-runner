"""Cached prompt tokens in the standard `usage` object, seen from the wire.

Runner has always counted, per request, how many prompt rows it did not have
to prefill: `runner_telemetry.prompt_cached_tokens`. That is a Runner
extension, and a client that reads OpenAI's schema never looks at it. OpenAI
carries the same fact as `usage.prompt_tokens_details.cached_tokens`, and the
cost dashboards, SDK helper properties and agent frameworks that report
"cache hit rate" all read it from there.

Two invariants hold everywhere it appears:

* `prompt_tokens` keeps its meaning. Cached tokens are INCLUDED in it, as
  they are at OpenAI, so no caller's total moves because this field arrived.
* `cached_tokens <= prompt_tokens`. Reuse cannot cover rows the prompt does
  not have.

The cold case is expressed as `cache_prompt: false` rather than "a prompt
nobody sent before". A conformance session sends hundreds of requests through
the same chat template, so every later prompt shares its opening tokens with
an earlier one and a genuinely cold prompt is not something this suite can
construct. `cache_prompt: false` is the documented full opt-out and makes the
zero a fact about the server rather than about the ordering of the suite.

The Anthropic surface is deliberately excluded, and stays excluded: see
test_messages.test_cache_usage_fields_are_not_claimed. `cache_read_input_tokens`
means something specific about Anthropic's product (their `input_tokens`
EXCLUDES it), so reporting Runner's unrelated figure there would misstate a
client's accounting. Messages reports the same number under `runner_telemetry`,
as every other surface does.
"""

import json

from _errors import ProtocolError

# Comfortably past the 16-token floor the shared prefix tier requires, so the
# warm case is reusable whether the repeat lands on the slot that already
# holds it or has to fork the shared snapshot into the other one.
SYSTEM = ("You are a usage-accounting fixture. " +
          " ".join(f"clause {i}: stay brief." for i in range(8)))


def _chat_payload(user, **kw):
    body = {"model": "test",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": 8}
    body.update(kw)
    return body


def _completion_payload(prompt, **kw):
    body = {"model": "test", "prompt": SYSTEM + "\n" + prompt,
            "temperature": 0, "max_tokens": 8}
    body.update(kw)
    return body


def _cached(usage, request, details="prompt_tokens_details"):
    """The cached-token figure, checked for shape before it is believed."""
    d = usage.get(details)
    if not isinstance(d, dict):
        raise ProtocolError("usage carries no cached-token details object",
                            request=request, field=details, usage=usage)
    c = d.get("cached_tokens")
    if not isinstance(c, int) or isinstance(c, bool):
        raise ProtocolError("cached_tokens is not an integer",
                            request=request, got=repr(c), usage=usage)
    if c < 0 or c > usage.get("prompt_tokens", usage.get("input_tokens", 0)):
        raise ProtocolError("cached_tokens outside [0, prompt_tokens]",
                            request=request, usage=usage)
    return c


def _stream_usage(stream, request):
    """The one chunk `stream_options.include_usage` adds, and only it."""
    stream.expect_sse()
    carrying = [c for c in stream.chunks if c.get("usage") is not None]
    if len(carrying) != 1:
        raise ProtocolError("expected exactly one usage chunk",
                            request=request, got=len(carrying))
    return carrying[0]["usage"]


# ------------------------------------------------------------------ buffered
def test_chat_reports_cached_prompt_tokens(client):
    cold = client.chat(_chat_payload("first question", cache_prompt=False),
                       name="cached-chat-cold").expect_status(200)
    u = cold.usage
    if _cached(u, "cached-chat-cold") != 0:
        raise ProtocolError("cache_prompt:false still reported reuse", usage=u)

    warm_body = _chat_payload("second question")
    client.chat(warm_body, name="cached-chat-warm-1").expect_status(200)
    warm = client.chat(warm_body, name="cached-chat-warm-2").expect_status(200)
    u = warm.usage
    n = _cached(u, "cached-chat-warm-2")
    if n <= 0:
        raise ProtocolError("a repeated prompt reported no cached tokens",
                            usage=u, telemetry=warm.telemetry)
    # The extension field and the standard field are the same measurement,
    # so a client reading either gets the same answer.
    if warm.telemetry.get("prompt_cached_tokens") != n:
        raise ProtocolError("usage and runner_telemetry disagree about reuse",
                            usage=u, telemetry=warm.telemetry)


def test_chat_prompt_tokens_still_include_the_cached_ones(client):
    """OpenAI's semantics: the detail breaks `prompt_tokens` down, it does not
    subtract from it. A client that bills on `prompt_tokens` must not see its
    total move because the prompt was cheap to serve."""
    body = _chat_payload("billing semantics")
    a = client.chat(body, name="cached-chat-sum-1").expect_status(200)
    b = client.chat(body, name="cached-chat-sum-2").expect_status(200)
    if a.usage["prompt_tokens"] != b.usage["prompt_tokens"]:
        raise ProtocolError("prompt_tokens changed when the prompt was reused",
                            cold=a.usage, warm=b.usage)
    for r, name in ((a, "cached-chat-sum-1"), (b, "cached-chat-sum-2")):
        u = r.usage
        _cached(u, name)
        if u["prompt_tokens"] + u["completion_tokens"] != u["total_tokens"]:
            raise ProtocolError("usage does not add up", usage=u)


def test_completions_reports_cached_prompt_tokens(client):
    cold = client.completion(
        _completion_payload("first", cache_prompt=False),
        name="cached-completion-cold").expect_status(200)
    if _cached(cold.usage, "cached-completion-cold") != 0:
        raise ProtocolError("cache_prompt:false still reported reuse",
                            usage=cold.usage)

    body = _completion_payload("second")
    client.completion(body, name="cached-completion-warm-1").expect_status(200)
    warm = client.completion(body,
                             name="cached-completion-warm-2").expect_status(200)
    if _cached(warm.usage, "cached-completion-warm-2") <= 0:
        raise ProtocolError("a repeated prompt reported no cached tokens",
                            usage=warm.usage, telemetry=warm.telemetry)


def test_responses_reports_cached_input_tokens(client):
    """The Responses surface spells the same fact `input_tokens_details`."""
    cold = client.responses({"model": "test", "input": "first question",
                             "max_output_tokens": 8, "temperature": 0,
                             "cache_prompt": False},
                            name="cached-responses-cold").expect_status(200)
    if _cached(cold.json["usage"], "cached-responses-cold",
               "input_tokens_details") != 0:
        raise ProtocolError("cache_prompt:false still reported reuse",
                            usage=cold.json["usage"])

    body = {"model": "test", "input": SYSTEM + " repeated input",
            "max_output_tokens": 8, "temperature": 0}
    client.responses(body, name="cached-responses-warm-1").expect_status(200)
    warm = client.responses(body,
                            name="cached-responses-warm-2").expect_status(200)
    if _cached(warm.json["usage"], "cached-responses-warm-2",
               "input_tokens_details") <= 0:
        raise ProtocolError("a repeated input reported no cached tokens",
                            usage=warm.json["usage"])


# ----------------------------------------------------------------- streamed
def test_chat_stream_usage_chunk_reports_cached_tokens(client):
    """`stream_options.include_usage` must carry the same object the buffered
    reply does: a client that flips `stream` on and off gets one shape."""
    body = _chat_payload("streamed question",
                         stream_options={"include_usage": True})
    client.chat(dict(body), name="cached-stream-warm-1").expect_status(200)
    st = client.chat_stream(body, name="cached-stream-usage")
    u = _stream_usage(st, "cached-stream-usage")
    if _cached(u, "cached-stream-usage") <= 0:
        raise ProtocolError("streamed usage chunk reported no cached tokens",
                            usage=u)
    if u["prompt_tokens"] + u["completion_tokens"] != u["total_tokens"]:
        raise ProtocolError("streamed usage does not add up", usage=u)


def test_completion_stream_usage_chunk_reports_cached_tokens(client):
    body = _completion_payload("streamed",
                               stream_options={"include_usage": True})
    client.completion(dict(body),
                      name="cached-cstream-warm-1").expect_status(200)
    st = client.completion_stream(body, name="cached-cstream-usage")
    u = _stream_usage(st, "cached-cstream-usage")
    if _cached(u, "cached-cstream-usage") <= 0:
        raise ProtocolError("streamed usage chunk reported no cached tokens",
                            usage=u)


# ------------------------------------------------------------------ excluded
def test_messages_usage_keeps_the_anthropic_vocabulary(client):
    """Messages does NOT gain the OpenAI detail object.

    Its `usage` is Anthropic-shaped, and the cache decision there is pinned by
    test_cache_usage_fields_are_not_claimed. Adding an OpenAI-spelled field to
    an Anthropic body would give a typed SDK a member its schema does not have
    while still not answering the question it would ask (`cache_read_input_tokens`).
    """
    r = client.messages({"model": "local", "max_tokens": 8, "temperature": 0,
                         "messages": [{"role": "user", "content": "hello"}]},
                        name="cached-messages-vocabulary").expect_status(200)
    usage = r.json["usage"]
    if "prompt_tokens_details" in usage or "input_tokens_details" in usage:
        raise ProtocolError(
            "the Anthropic body gained an OpenAI-spelled usage detail; the "
            "prefix-cache figure belongs in runner_telemetry on this surface",
            usage=usage)
    if not isinstance(r.telemetry.get("prompt_cached_tokens"), int):
        raise ProtocolError("Messages lost its runner_telemetry reuse figure",
                            telemetry=r.telemetry)


def test_streamed_usage_chunk_is_valid_json_for_the_sdk(client):
    """The chunk is parsed by the official SDK elsewhere; this pins the bytes.

    An extra nested object inside `usage` is exactly the kind of change that
    can be emitted with a comma in the wrong place and still look right in a
    log, because the SSE framing hides it until something parses the payload.
    """
    body = _chat_payload("json shape", stream_options={"include_usage": True})
    st = client.chat_stream(body, name="cached-stream-json")
    st.expect_sse()
    for data in st.events:
        if data and data != "[DONE]":
            json.loads(data)
