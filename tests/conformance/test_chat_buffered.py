"""Buffered chat completions: response shape, truncation, stop sequences."""

import pytest

from harness import ProtocolError

BASE = {"messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8, "temperature": 0}


def test_health_and_models(client, report):
    client.get("/health").expect_status(200)
    models = client.get("/v1/models").expect_status(200).json
    if models.get("object") != "list" or not models.get("data"):
        raise ProtocolError("/v1/models is not an OpenAI model list", body=models)
    if not models["data"][0].get("id"):
        raise ProtocolError("model entry has no id", entry=models["data"][0])
    report.check_fixture("models", models)


def test_capabilities(client, report, server):
    caps = client.get("/v1/capabilities").expect_status(200).json
    if caps.get("pid") != server.proc.pid:
        raise ProtocolError("capabilities identifies the wrong process",
                            got=caps.get("pid"), expected=server.proc.pid)
    for feature in ("json_schema", "stop_sequences"):
        if not caps.get("features", {}).get(feature):
            raise ProtocolError("capability not advertised", feature=feature,
                                features=caps.get("features"))
    report.check_fixture("capabilities", caps)


def test_capabilities_reports_effective_slots_and_draft(client, server):
    """RI-2: the EFFECTIVE execution mode has to be machine-readable.

    Several requested configurations legitimately resolve to a different one:
    `--draft` is dropped in swap mode and can be refused at load (vocab
    mismatch, a fully GPU-offloaded target, out of memory), and `-m name=path`
    with `--parallel N` gives up the registry to keep the slots. Each of those
    prints a note to stderr and continues, which a benchmark harness cannot
    see -- so it can measure the fallback and report it as the mode it asked
    for. These fields are how it finds out instead."""
    caps = client.get("/v1/capabilities").expect_status(200).json
    slots = caps.get("slots")
    if not isinstance(slots, int) or slots < 1:
        raise ProtocolError("capabilities does not report effective slots",
                            slots=slots)
    draft = caps.get("draft")
    if not isinstance(draft, dict):
        raise ProtocolError("capabilities does not report draft state",
                            draft=draft)
    for key in ("requested", "active"):
        if not isinstance(draft.get(key), bool):
            raise ProtocolError("draft state field missing or not a boolean",
                                key=key, draft=draft)
    # this server is started without --draft, so both are false and the
    # reason is absent rather than invented
    if draft["requested"] or draft["active"]:
        raise ProtocolError("draft reported active on a server started "
                            "without --draft", draft=draft)


def test_request_telemetry_capability_is_qualified(client):
    """RI-4: the telemetry claim has to say WHICH surface carries it.

    Buffered replies carry the full runner_telemetry object; ordinary streamed
    chat and legacy completions carry none, because a stream's only extra
    terminal chunk is the opt-in include_usage one. Advertising
    `request_telemetry: true` unqualified is the same accepted-then-ignored
    shape the project refuses everywhere else: a claimed capability a caller
    cannot actually get on the surface they are using.

    Widening the streamed wire is a separate change, deliberately deferred by
    an owner decision on 2026-08-08 and still deferred. What is fixed here is
    the CLAIM."""
    caps = client.get("/v1/capabilities").expect_status(200).json
    rt = caps.get("features", {}).get("request_telemetry")
    if not isinstance(rt, dict):
        raise ProtocolError("request_telemetry is advertised unqualified; it "
                            "differs by surface and must say so",
                            request_telemetry=rt)
    if rt.get("buffered") is not True:
        raise ProtocolError("buffered telemetry should be advertised", rt=rt)
    if rt.get("streamed") is not False:
        raise ProtocolError("streamed telemetry is not carried today, so it "
                            "must not be advertised as available", rt=rt)


def test_buffered_chat_shape(client, report):
    r = client.chat(dict(BASE), name="buffered-chat").expect_status(200)
    d = r.json
    if d.get("object") != "chat.completion":
        raise ProtocolError("wrong object type", got=d.get("object"))
    for field in ("id", "created", "model", "choices", "usage"):
        if field not in d:
            raise ProtocolError("buffered response missing required field",
                                field=field, keys=sorted(d))
    msg = r.choice["message"]
    if msg.get("role") != "assistant":
        raise ProtocolError("buffered message role is not assistant",
                            got=msg.get("role"))
    if r.finish_reason not in ("stop", "length", "tool_calls"):
        raise ProtocolError("unknown finish_reason", got=r.finish_reason)
    u = r.usage
    if u["prompt_tokens"] + u["completion_tokens"] != u["total_tokens"]:
        raise ProtocolError("usage does not add up", usage=u)
    report.check_fixture("chat_buffered", d)


def test_completions_endpoint_shape(client, report):
    r = client.completion({"prompt": "hello", "max_tokens": 8, "temperature": 0},
                          name="buffered-completion").expect_status(200)
    d = r.json
    if d.get("object") != "text_completion":
        raise ProtocolError("wrong object type", got=d.get("object"))
    if not isinstance(r.choice.get("text"), str):
        raise ProtocolError("completion choice has no text string",
                            choice=r.choice)
    report.check_fixture("completion_buffered", d)


def test_max_tokens_truncates(client):
    """Truncation: the cap is honoured exactly and reported as finish_reason
    'length'. A cap that silently over-runs would blow a caller's budget."""
    for cap in (1, 4, 16):
        r = client.chat(dict(BASE, max_tokens=cap), name=f"truncate-{cap}")
        r.expect_status(200)
        got = r.usage["completion_tokens"]
        if got > cap:
            raise ProtocolError("max_tokens exceeded", cap=cap, generated=got)
        if got == cap and r.finish_reason != "length":
            raise ProtocolError("hit the cap but did not report length",
                                cap=cap, finish_reason=r.finish_reason)


def test_max_tokens_clamped_to_context(client):
    """An absurd max_tokens is clamped to the remaining context, not rejected
    and not obeyed: agent clients routinely send a huge cap."""
    r = client.chat(dict(BASE, max_tokens=999999), name="truncate-clamp")
    r.expect_status(200)
    if r.usage["completion_tokens"] >= 1024:
        raise ProtocolError("max_tokens not clamped to context window",
                            generated=r.usage["completion_tokens"])


@pytest.mark.parametrize("field", [
    "max_tokens", "max_completion_tokens", "max_output_tokens",
])
def test_negative_max_tokens_is_rejected(client, field):
    payload = dict(BASE)
    payload.pop("max_tokens")
    payload[field] = -1
    client.expect_400(payload, field)


def test_prefix_cache_is_deterministic(client):
    """Identical greedy requests must produce identical text whether or not the
    KV prefix cache served them. This is the cheapest correctness canary the
    harness has and it is why temp=0 is used throughout."""
    a = client.chat(dict(BASE, max_tokens=12), name="determinism-1")
    b = client.chat(dict(BASE, max_tokens=12), name="determinism-2")
    if a.content != b.content:
        raise ProtocolError("identical greedy requests diverged",
                            first=a.content, second=b.content)
    if b.telemetry.get("prompt_cached_tokens", 0) <= 0:
        raise ProtocolError("second identical request did not hit the prefix cache",
                            telemetry=b.telemetry)


def _stub_text(client):
    """A prefix of the stub model's greedy output that is safe to slice.

    test.gguf emits near-random token ids, so the decoded bytes can contain
    invalid UTF-8; everything after the first replacement char is dropped."""
    text = client.chat(dict(BASE, max_tokens=24), name="stop-baseline").content
    clean = text.split("�")[0]
    if len(clean) < 3:
        pytest.skip(f"stub model produced too little clean text: {text!r}")
    return text, clean


def test_stop_sequence_truncates_before_match(client):
    text, clean = _stub_text(client)
    stop = clean[:3]
    expected = text[:text.find(stop)]
    for payload_stop, label in ((stop, "string"), ([stop], "array")):
        r = client.chat(dict(BASE, max_tokens=24, stop=payload_stop),
                        name=f"stop-{label}").expect_status(200)
        if r.content != expected:
            raise ProtocolError("stop sequence did not truncate before the match",
                                form=label, expected=expected, got=r.content)
        if r.finish_reason != "stop":
            raise ProtocolError("stop match did not report finish_reason stop",
                                got=r.finish_reason)


def test_unmatched_stop_flushes_held_bytes(client):
    """A partial stop match that never completes must be flushed, not eaten."""
    text, clean = _stub_text(client)
    r = client.chat(dict(BASE, max_tokens=24, stop=[clean + "NEVERMATCHES"]),
                    name="stop-unmatched").expect_status(200)
    if r.content != text:
        raise ProtocolError("unmatched stop sequence swallowed output",
                            expected=text, got=r.content)
    if r.finish_reason == "stop":
        raise ProtocolError("reported a stop match that never happened")
