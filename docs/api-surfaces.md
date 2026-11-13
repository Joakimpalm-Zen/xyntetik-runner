# API surfaces: why they say what they say

Runner speaks three API vocabularies over one generation path: Chat
Completions (and the legacy completions endpoint), Responses, and Anthropic
Messages. The comments in `src/completion.c` state each rule; this page keeps
the reasons and the evidence behind the ones that were argued for.

## Finish reasons on Chat Completions

OpenAI's `finish_reason` is a closed set: `stop`, `length`, `tool_calls`,
`content_filter`, `function_call`. Runner's own `reasoning_limit` goes out as
`length` and `envelope_error` as `error`; `runner_telemetry.finish_detail`
keeps the exact reason, on streamed turns too (the finish chunk carries
`runner_telemetry`).

A strictly typed client whose finish reason is a closed union has no case for
a value outside it; the official OpenAI SDK happens to deserialise one, which
is how the raw value went unnoticed. The field evidence behind the change
(owner decision, 2026-08-08): a reasoning model truncated inside its think
block returns an empty answer, and a harness that saw the non-standard reason
recorded incapability instead of truncation, so every agent case scored 0.00
for every model in the Syntetik-MoE baseline. `length` is accurate: the turn
did run out of budget; only where the budget went is lost, and that is what
`finish_detail` is for. Responses (`max_output_tokens`) and Messages
(`max_tokens`) already reported it this way.

## A generation fault on the Messages surface

Anthropic's `stop_reason` values (`end_turn`, `max_tokens`, `stop_sequence`,
`tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded`) all
describe a turn that completed: "Unlike errors, which indicate failures in
processing your request, `stop_reason` tells you why Claude completed its
response generation." A fault (an allocation failure, or a strict-envelope
document that could not be mapped) is therefore an error object, never a
`stop_reason`. `end_turn`, which was sent before, told the caller the model
finished normally over an empty `content[]`.

The error type is `api_error` with a 500, because each other type in
Anthropic's set says something false here:

| type | why not |
|---|---|
| `invalid_request_error` (400) | blames the caller for a well-formed request; nothing they can edit changes the outcome, and SDKs do not retry a 4xx |
| `overloaded_error` (529) | means capacity; clients shed load or fail over, and nothing is loaded |
| `timeout_error` (504) | nothing timed out |

`api_error` is documented as an unexpected internal error to retry with
exponential backoff, which is what both faults are: an unmappable envelope
depends on sampling, so a retry can succeed, and an allocation failure can
clear. Every official SDK already retries a 500 with backoff. Which fault it
was is in `runner_telemetry`, not in `error.type`, where a typed client has no
case for it.

No partial content is shipped. Truncation (`max_tokens` with the partial in
the body) is a completion with a known boundary; an unmapped envelope is
protocol framing, not a partial answer, and the allocation fault's real text
would need a `stop_reason` that is false about why generation stopped. That
text stays in the server log and on the OpenAI surfaces, whose `error` finish
reason can carry it beside the content.

Streamed, the 200 and the headers are already sent, so the stream ends on
Anthropic's `error` event (documented beside `overloaded_error`), never
`message_delta` and `message_stop`, which assert a completed turn. It must
end: a stream left hanging is the fault fixed on the OpenAI surfaces. No
`content_block_stop` is written for an interrupted block: an SDK accumulator
discards the message on `error`, and closing the block would suggest the
content before it was a usable partial answer.

## The prompt cache in usage

Chat Completions reports cache reuse as
`usage.prompt_tokens_details.cached_tokens`, included in `prompt_tokens`: a
caller billing on `prompt_tokens` must not see its total move because the
prompt happened to be cheap to serve, so it is a breakdown of that number, not
a fourth one beside it. Responses spells it
`input_tokens_details.cached_tokens`. Messages spells it nowhere, on purpose:
`cache_read_input_tokens` describes Anthropic's product with Anthropic's
semantics, and an unrelated figure there would misstate a client's
accounting. Every surface carries `runner_telemetry.prompt_cached_tokens`.

## Under a strict tool envelope

The generated document is protocol, not text (Harmony channel markers and
recipient headers, Muse's `<atem:invoke>`, gemma4's `<|tool_call>` blocks, the
generic envelope's own JSON), and the client receives only the demultiplexed
result.

- **Stop sequences are refused, not ignored.** Matched against the raw
  document they fire on framing the caller never wrote (`["\n\n"]`, `["}"]`
  and `["<|"]` are ordinary agent defaults and every one hits), and they
  corrupt the turn: `stop_feed` drops the matched bytes, but the constraint
  validator consumed them first, so `constraint_close` continues a state the
  client's copy never reached. Measured: the generic envelope with stop `"`
  returned `"content":"{tool"`, and the Harmony envelope failed to map at all.
  The Harmony demultiplexer cannot tell visible text from framing until the
  document is complete, so honouring the field is not available on this path;
  an ignored field is undetectable for the caller, the same reason `server.c`
  refuses a `parallel_tool_calls` it cannot honour.
- **A document that cannot be parsed is dropped**, and the turn reports the
  fault. The earlier fallback served it verbatim: clients got
  `<|channel|>analysis<|message|>...to=functions.unknown...` as assistant
  content, and the generic envelope leaked `{tool`.
- **A truncated call keeps `length`.** It still parses (`sval_close` closed
  it), but only a cleanly finished document reports `tool_calls`, so the
  caller knows the arguments are minimal closures (finding A of the
  2026-08-11 evaluation).

## Smaller records

- Top logprobs: the Mellum2.1 parity record (2026-10-29) found five positions
  where two candidates escaped to the same key.
- Sampling provenance in `runner_telemetry`: a 2026-09-14 Windows report could
  only infer, from an A/B against explicit request fields, that a family preset
  had applied a repeat penalty the vendor never published.
- Closure provenance (owner decision, 2026-10-02): before it nothing said
  which values of a closed document the model wrote. XML-style native calls
  (Qwen3-Coder, Muse) reported `"fields": null` until 2026-10-14, found by an
  outside review of the runner 1.2.0 tool-calling study.
- Repeated tool calls are reported, not refused: the lab saw
  `get_weather(Athens)` re-called turn after turn with the answer already in
  the model's reasoning.
- The forced-close sentence: llama.cpp's reasoning-budget measurement gave
  HumanEval 93% uncapped, about 89% capped with a message and 79% with a bare
  end tag; s1 (arXiv 2501.19393) and Qwen3's "Considering the limited time..."
  line do the same.
- `think_on` is reset per request: on the muse fixture a request whose forced
  close did not fit left the slot's reasoning turn open, and the next request,
  with no reasoning in its prompt, was capped from its first token.
- The request start line: a 1,200-token prompt on a Mac returned nothing in
  300 s with an empty server log.
- Unseeded sampling takes a place in one server-wide order: per-slot streams
  sampled differently from run to run under the same `-s` seed at
  `--parallel 2` or more.
