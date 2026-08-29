"""Malformed and unsupported request fields must be rejected, never ignored.

This is a project invariant, not a style preference. A request that names a
field the server does not implement, or names one it does implement with a
value it cannot use, must fail with 400. Answering 200 while quietly dropping
the field is the worst outcome available: the caller asked for guaranteed
structure, or a stop sequence, or SSE framing, and got a response that looks
successful without it.

Every case below is verified current behaviour.
"""

import pytest

from harness import ProtocolError

CHAT = {"messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4, "temperature": 0}


# ------------------------------------------------------- response_format
@pytest.mark.parametrize("rf,label,contains", [
    ({"type": "json_schema"}, "json_schema-without-member", "json_schema"),
    ({"type": "json_schema", "json_schema": "nope"}, "json_schema-not-object", "json_schema"),
    ({"type": "json_schema", "json_schema": {"schema": []}}, "schema-not-object", "schema"),
    ({"type": "nonsense"}, "unknown-type", "response_format.type"),
    ({"type": ""}, "empty-type", "response_format.type"),
    ({}, "missing-type", "response_format.type"),
    ({"type": 7}, "non-string-type", "response_format.type"),
    ("json_object", "not-an-object", "response_format"),
    ([], "array", "response_format"),
])
def test_bad_response_format_is_rejected(client, rf, label, contains):
    """A response_format the server cannot honour must 400.

    ``{"type":"json_schema"}`` with no ``json_schema`` member is the headline
    case: it reads as "give me schema-constrained output" and there is no
    schema, so unconstrained 200 would be a silent lie."""
    client.expect_400(dict(CHAT, response_format=rf),
                      name=f"bad-response-format-{label}", contains=contains)


@pytest.mark.parametrize("rf", [{"type": "text"}, {"type": "json_object"}])
def test_supported_response_format_is_accepted(client, rf):
    client.chat(dict(CHAT, max_tokens=16, response_format=rf),
                name=f"good-response-format-{rf['type']}").expect_status(200)


def test_unsupported_schema_construct_is_rejected(client):
    """schema_compile rejects what it cannot enforce rather than approximating
    it — an approximated schema is an unenforced schema."""
    for schema, label in (
            ({"type": "array", "items": {"type": "string"},
              "minItems": 2, "maxItems": 1}, "impossible-bounds"),
            ({"type": []}, "type-is-array"),
            ({"type": "object", "properties": {'bad"key': {"type": "string"}}},
             "escaped-property-key"),
    ):
        client.expect_400(
            dict(CHAT, response_format={"type": "json_schema",
                                        "json_schema": {"schema": schema}}),
            name=f"bad-schema-{label}", contains="schema")


# ----------------------------------------------------- scalar parameters
@pytest.mark.parametrize("field,value,contains", [
    ("stream", "true", "stream"),
    ("stream", 1, "stream"),
    ("stream", [], "stream"),
    ("top_k", 1e300, "sampling"),
    ("temperature", 1e300, "sampling"),
    ("top_p", 2.0, "sampling"),
    ("top_p", -1, "sampling"),
    ("min_p", 5, "sampling"),
    ("keep_alive", 1e300, "keep_alive"),
    ("top_logprobs", 999, "top_logprobs"),
    # Wrong TYPE is rejected for the same reason wrong RANGE is: a value the
    # server cannot use must not be replaced by a default the caller never
    # asked for. Several HTTP layers stringify numbers out of form or
    # env-derived config, so this is the common shape of the mistake.
    ("temperature", "hot", "sampling"),
    ("temperature", "0.7", "sampling"),
    ("top_p", "0.9", "sampling"),
    ("top_k", True, "sampling"),
    ("min_p", [], "sampling"),
    ("repeat_penalty", "1.1", "sampling"),
    ("seed", "1234", "sampling"),
    ("max_tokens", "eight", "max_tokens"),
    ("max_completion_tokens", "eight", "max_tokens"),
    ("keep_alive", "300", "keep_alive"),
    ("top_logprobs", "2", "top_logprobs"),
    ("logprobs", "true", "logprobs"),
    # A fractional value is rejected for being fractional, not for being out
    # of range -- 2.5 is inside every bound top_k has. The message has to say
    # which, or a caller goes looking for a limit they never exceeded, so the
    # substring asserted here is the rule and not just the field name.
    ("top_k", 2.5, "whole number"),
    ("seed", 1.5, "whole number"),
    # This one IS a range rejection: the seed bound is the largest double below
    # 2^64, so 2^64 itself never reaches the uint64_t cast.
    ("seed", 18446744073709551616, "sampling"),
    ("max_tokens", 1.5, "max_tokens"),
    ("keep_alive", 1.5, "keep_alive"),
    ("top_logprobs", 1.5, "top_logprobs"),
])
def test_out_of_range_scalar_is_rejected(client, field, value, contains):
    payload = dict(CHAT, **{field: value})
    if field == "top_logprobs":
        payload["logprobs"] = True
    if field == "max_completion_tokens":
        # max_tokens wins when both are present, so the alias is only
        # reachable once the primary name is gone
        payload.pop("max_tokens")
    client.expect_400(payload, name=f"bad-{field}-{value!r}", contains=contains)


@pytest.mark.parametrize("field", ["temperature", "top_p", "min_p", "top_k",
                                   "seed", "repeat_penalty", "max_tokens",
                                   "max_completion_tokens", "keep_alive",
                                   "stop", "logprobs"])
def test_explicit_null_scalar_reads_as_absent(client, field):
    """The boundary of the rule above. Every mainstream OpenAI SDK serialises
    an unset optional field as ``null`` rather than omitting it, so ``null``
    must mean "absent" and take the default. Treating it as a wrong type
    would 400 on ordinary traffic from an unmodified client."""
    r = client.chat(dict(CHAT, **{field: None}), name=f"null-{field}")
    r.expect_status(200)


@pytest.mark.parametrize(("field", "value"), [
    ("n", 2),
    ("frequency_penalty", 0.5),
    ("presence_penalty", -0.5),
    ("logit_bias", {"1": 10}),
    ("user", 7),
])
def test_unsupported_completion_field_semantics_are_rejected(client, field, value):
    client.expect_400(dict(CHAT, **{field: value}), name=f"unsupported-{field}",
                      contains=field)


@pytest.mark.parametrize("field", ["cache_prompt", "prefix_cache"])
def test_cache_controls_require_booleans(client, field):
    client.expect_400(dict(CHAT, **{field: "false"}),
                      name=f"bad-{field}-type", contains=field)


def test_stream_include_usage_requires_a_boolean(client):
    client.expect_400(
        dict(CHAT, stream=True, stream_options={"include_usage": "true"}),
        name="bad-stream-include-usage-type", contains="include_usage")


@pytest.mark.parametrize("value", [-1, 21])
def test_top_logprobs_range_is_rejected(client, value):
    client.expect_400(dict(CHAT, logprobs=True, top_logprobs=value),
                      name=f"top-logprobs-{value}", contains="top_logprobs")


@pytest.mark.parametrize("stop,label", [
    ([1], "non-string-item"),
    ([""], "empty-item"),
    ("", "empty-string"),
    (["a", "b", "c", "d", "e"], "too-many"),
    ({"a": 1}, "object"),
    (7, "number"),
])
def test_malformed_stop_is_rejected(client, stop, label):
    client.expect_400(dict(CHAT, stop=stop), name=f"bad-stop-{label}",
                      contains="stop")


# ----------------------------------------------------------- body/routing
def test_invalid_json_body_is_rejected(client):
    for body, label in (
        (b"{", "truncated"),
        (b"not json", "garbage"),
        (b"[]", "array-not-object"),
        (b"", "empty"),
        (b'{"messages":[],"max_tokens":4,"max_tokens":4096}',
         "duplicate-key"),
        (b'{"messages":[],"model":1,"m\\u006fdel":2}',
         "escaped-duplicate-key"),
    ):
        r = client.post_bytes(f"bad-body-{label}", "/v1/chat/completions", body)
        if r.status == 200:
            raise ProtocolError("invalid JSON body accepted", case=label,
                                body=r.text[:200])
        r.expect_status(400)
        r.expect_error_envelope()


def test_missing_required_fields_are_rejected(client):
    client.expect_400({"max_tokens": 4}, name="chat-no-messages",
                      contains="messages")
    client.expect_400({"messages": []}, name="chat-empty-messages")
    client.expect_400({"messages": [{"role": "user"}]}, name="chat-no-content")
    client.expect_400({"max_tokens": 4}, name="completion-no-prompt",
                      contains="prompt", path="/v1/completions")
    client.expect_400({}, name="embeddings-no-input", contains="input",
                      path="/v1/embeddings")


@pytest.mark.parametrize("message,label,contains", [
    ("plain text", "item-not-object", "messages[0]"),
    ({"content": "hello"}, "missing-role", "role"),
    ({"role": 7, "content": "hello"}, "role-not-string", "role"),
    ({"role": "invented", "content": "hello"}, "unknown-role", "role"),
    ({"role": "user", "content": 12345}, "numeric-content", "content"),
    ({"role": "user", "content": {"text": "hello"}},
     "object-content", "content"),
])
def test_malformed_chat_messages_are_rejected(client, message, label, contains):
    """A malformed history turn must not be defaulted or dropped while the
    remaining conversation is answered. That changes the caller's prompt under
    a successful HTTP status."""
    client.expect_400(
        {"messages": [message, {"role": "user", "content": "retained"}],
         "max_tokens": 4, "temperature": 0},
        name=f"bad-message-{label}", contains=contains)


def test_chat_rejects_image_parts_instead_of_answering_text_only(client):
    client.expect_400({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "describe this image"},
        ]}],
        "max_tokens": 4,
    }, name="chat-image-part", contains="image_url")


def _chat_body(extra):
    """A chat body with a raw JSON fragment spliced in, for values json.dumps
    cannot emit (bare Infinity, overflowing exponents)."""
    return (b'{"messages":[{"role":"user","content":"hi"}],' + extra + b"}")


def test_overflowing_exponent_in_a_sampling_param_is_rejected(client):
    """``1e400`` overflows to infinity. It must never become a sampling
    parameter, whichever guard catches it."""
    r = client.post_bytes("nonfinite-temperature", "/v1/chat/completions",
                          _chat_body(b'"temperature":1e400,"max_tokens":4'))
    if r.status == 200:
        raise ProtocolError("infinite temperature accepted", body=r.text[:200])
    r.expect_status(400)
    r.expect_error_envelope()


def test_overflowing_max_tokens_cannot_run_away(client):
    """FINDING (build-flag dependent, deliberately asserted loosely).

    json.c's number parser rejects non-finite values with ``isfinite``, and
    request_max_tokens re-checks — but the shipping CFLAGS include
    ``-ffast-math``, under which the compiler is licensed to fold ``isfinite``
    to true. So on a stock build ``{"max_tokens": 1e400}`` parses, survives
    both guards as +inf, and is clamped by the ``v > INT_MAX`` branch instead.

    Asserting "400" here would pass on a -fno-fast-math build and fail on the
    shipped one. What actually matters — and holds either way — is that an
    infinite cap cannot produce unbounded generation: either it is rejected, or
    it is clamped to the context window.
    """
    r = client.post_bytes("nonfinite-max-tokens", "/v1/chat/completions",
                          _chat_body(b'"temperature":0,"max_tokens":1e400'))
    if r.status == 400:
        r.expect_error_envelope()
        return
    r.expect_status(200)
    if r.usage["completion_tokens"] >= 1024:
        raise ProtocolError("infinite max_tokens produced unbounded generation",
                            generated=r.usage["completion_tokens"])


def test_max_tokens_above_int_max_is_clamped_not_rejected(client):
    """Finite-but-huge is clamped to the context window (agent clients send
    absurd caps routinely); only non-finite is an error. Pinned because the
    boundary between "clamp" and "reject" is easy to move by accident."""
    r = client.chat(dict(CHAT, max_tokens=1e300), name="max-tokens-huge")
    r.expect_status(200)
    if r.usage["completion_tokens"] >= 1024:
        raise ProtocolError("huge max_tokens was not clamped to the context",
                            generated=r.usage["completion_tokens"])


def test_unknown_route_is_404_not_400(client):
    """Routing errors and validation errors must not be confused: a 400 on an
    unknown path would tell a client its request was malformed."""
    r = client.get("/v1/does-not-exist", name="unknown-route")
    r.expect_status(404)
    r.expect_error_envelope()


def test_error_bodies_are_always_json_envelopes(client, report):
    """Every rejection carries {"error":{"message","type"}}. Clients branch on
    this shape; an HTML or bare-text error page breaks them."""
    r = client.chat(dict(CHAT, response_format={"type": "nope"}),
                    name="error-envelope-fixture")
    r.expect_status(400)
    err = r.expect_error_envelope()
    if not err["message"].strip():
        raise ProtocolError("rejection carried an empty message")
    report.check_fixture("error_envelope", r.json)


def test_extra_unknown_top_level_fields_are_tolerated(client):
    """The other half of the invariant: fields OpenAI clients routinely send
    that runner does not act on (frequency_penalty, user, ...) must NOT 400.
    Rejecting those would break Cline/OpenCode-shaped traffic outright.

    The line is: a field whose *semantics runner cannot honour* is rejected; a
    field that is merely advisory is accepted."""
    r = client.chat(dict(CHAT, frequency_penalty=0, presence_penalty=0,
                         user="conformance", n=1, logit_bias={},
                         parallel_tool_calls=False, tool_choice="auto"),
                    name="advisory-fields")
    r.expect_status(200)


def test_unknown_model_is_rejected_on_single_model_server(client):
    """A single-model server still has a model contract.

    Accepting any request ``model`` value makes client routing mistakes look
    successful. Registry mode already rejects unknown names; single-model mode
    must do the same instead of silently serving the one loaded model.
    """
    r = client.chat(dict(CHAT, model="definitely-not-a-real-model"),
                    name="unknown-model-single")
    r.expect_status(404)
    err = r.expect_error_envelope("unknown model")
    assert err["param"] == "model"
    assert err["code"] == "model_not_found"


@pytest.mark.parametrize(("path", "payload"), [
    ("/v1/chat/completions", dict(CHAT, model=7)),
    ("/v1/completions", {"model": True, "prompt": "hi", "max_tokens": 4}),
    ("/v1/embeddings", {"model": {"name": "local"}, "input": "hi"}),
    ("/v1/responses", {"model": ["local"], "input": "hi",
                        "max_output_tokens": 4}),
    ("/v1/messages", {"model": 7, "max_tokens": 4,
                       "messages": [{"role": "user", "content": "hi"}]}),
])
def test_model_selector_must_be_a_string(client, path, payload):
    """A present model value must never turn into the default model merely
    because its JSON type cannot be read as a string."""
    client.expect_400(payload, name=f"bad-model-type-{path.rsplit('/', 1)[-1]}",
                      contains="model", path=path)


def test_context_overflow_is_a_typed_request_error(client):
    payload = dict(CHAT, messages=[{"role": "user", "content": "hello " * 2000}])
    r = client.chat(payload, name="context-overflow")
    r.expect_status(400)
    err = r.expect_error_envelope("context")
    assert err["param"] == "messages"
    assert err["code"] == "context_length_exceeded"


# --------------------------------------------------- accepted-then-ignored
#
# The class this file exists for, found again by a full-tree sweep on
# 2026-08-29. Each of these was answered 200 while the field the caller
# reached for did nothing.

def test_explicit_seed_zero_is_rejected(client):
    """seed:0 asks for a reproducible run and did not get one.

    The sampler's xorshift64 has a fixed point at state 0, so the engine only
    adopts a seed above zero -- an explicit 0 left the inherited state and two
    identical requests diverged. The CLI has refused -s 0 by name for exactly
    this reason since the reasoning was first written down. Absent stays
    absent: only a seed the caller actually sent is refused."""
    client.expect_400(dict(CHAT, seed=0), name="seed-zero", contains="seed")


def test_absent_seed_is_still_accepted(client):
    r = client.raw("seed-absent", "POST", "/v1/chat/completions", CHAT)
    r.expect_status(200)


@pytest.mark.parametrize("path,payload", [
    ("/v1/chat/completions",
     {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
      "tools": [{"type": 7, "function": {"name": "f", "parameters":
                                         {"type": "object", "properties": {}}}}]}),
    ("/v1/responses",
     {"input": "hi", "max_output_tokens": 4,
      "tools": [{"type": 7, "name": "f",
                 "parameters": {"type": "object", "properties": {}}}]}),
    ("/v1/messages",
     {"max_tokens": 4, "messages": [{"role": "user", "content": "hi"}],
      "tools": [{"type": 7, "name": "f",
                 "input_schema": {"type": "object", "properties": {}}}]}),
])
def test_wrong_typed_tool_type_is_rejected(client, path, payload):
    """A non-string tools[].type was read as if it were absent.

    jv_str hands back the default for a wrong type as well as a missing one,
    so `"type": 7` was normalised to "function" and the declaration accepted
    on all three surfaces. Null still reads as absent, as everywhere else."""
    client.expect_400(payload, name=f"tool-type-number{path.replace('/', '-')}",
                      contains="type", path=path)


def test_wrong_typed_tool_description_is_rejected(client):
    """Same normalisation, one field over: the renderers read description with
    a "" default, so a wrong-typed one silently vanished from the prompt."""
    client.expect_400(
        dict(CHAT, tools=[{"type": "function",
                           "function": {"name": "f", "description": 7,
                                        "parameters": {"type": "object",
                                                       "properties": {}}}}]),
        name="tool-description-number", contains="description")


# ------------------------------------------- Responses tool-history shapes
@pytest.mark.parametrize("item,label,contains", [
    ({"type": "function_call_output", "call_id": "c1", "output": 7},
     "output-number", "output"),
    ({"type": "function_call_output", "call_id": "c1", "output": {"a": 1}},
     "output-object", "output"),
    ({"type": "function_call_output", "output": "ok"},
     "missing-call-id", "call_id"),
    ({"type": "function_call_output", "call_id": 7, "output": "ok"},
     "numeric-call-id", "call_id"),
])
def test_malformed_function_call_output_is_rejected(client, item, label, contains):
    """A tool result is only attributable through its call_id, and the item
    schema types `output` as a string. Both were unchecked: a number, a bool
    or an object was JSON-dumped straight into the prompt, and an absent or
    wrongly typed call_id reported for whatever the fallback found."""
    client.expect_400({"input": [{"type": "message", "role": "user",
                                  "content": "hi"}, item],
                       "max_output_tokens": 4},
                      name=f"fco-{label}", contains=contains,
                      path="/v1/responses")


def test_wrong_typed_function_call_name_is_rejected(client):
    """An absent name is deduced from a sole declared tool -- that is the
    documented attribution rule. A wrong-typed one took the same path, so a
    caller's typo came back as a successful call to whatever the one tool
    was."""
    client.expect_400(
        {"input": [{"type": "message", "role": "user", "content": "hi"},
                   {"type": "function_call", "call_id": "c1", "name": 7,
                    "arguments": "{}"}],
         "max_output_tokens": 4,
         "tools": [{"type": "function", "name": "only_one",
                    "parameters": {"type": "object", "properties": {}}}]},
        name="fc-numeric-name", contains="name", path="/v1/responses")


def test_tool_result_after_text_is_rejected(client):
    """Anthropic puts tool_result blocks first in a user message. Because a
    result becomes its own turn here, a message that put text ahead of one was
    silently REORDERED -- result first, text deferred -- and answered 200
    having been rewritten into a different conversation."""
    client.expect_400(
        {"max_tokens": 4,
         "messages": [{"role": "user", "content": "go"},
                      {"role": "assistant", "content": [
                          {"type": "tool_use", "id": "toolu_1",
                           "name": "f", "input": {}}]},
                      {"role": "user", "content": [
                          {"type": "text", "text": "before"},
                          {"type": "tool_result", "tool_use_id": "toolu_1",
                           "content": "done"}]}],
         "tools": [{"name": "f", "input_schema": {"type": "object",
                                                  "properties": {}}}]},
        name="tool-result-after-text", contains="tool_result",
        path="/v1/messages")


@pytest.mark.parametrize("budget", [512, 1.5, "1024"])
def test_thinking_budget_tokens_is_validated(client, budget):
    """budget_tokens was neither checked nor honoured. Runner cannot enforce
    it as a hard cap -- nothing makes a model stop reasoning at a token count
    -- so it stays advisory, and usage.reasoning_tokens reports what the turn
    actually spent. What it must not do is accept a malformed one."""
    client.expect_400(
        {"max_tokens": 4, "messages": [{"role": "user", "content": "hi"}],
         "thinking": {"type": "disabled", "budget_tokens": budget}},
        name=f"thinking-budget-{budget!r}", contains="budget_tokens",
        path="/v1/messages")
