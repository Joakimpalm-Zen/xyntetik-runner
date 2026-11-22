# Inkling 2 Mini protocol

This records protocol checks, not model admission or numerical fidelity.
The native chat renderer and tokenizer are checked here. Generated tool-call
parsing, truncation closure and reasoning-channel transitions are still pending. No weights are needed for the tokenizer check.

## Tokenizer

The publisher's [tokenizer.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/b85c3cfa615fc413cc9af3379f491b963f3b586a/tokenizer.json)
uses the o200k pre-tokenizer and `ignore_merges=true`. The converter names
that rule `tokenizer.ggml.pre=inkling2`, with `tokenizer.ggml.model=gpt2`.
The validated GGUF metadata has 200,058 tokens, EOS 200006
(`<|content_model_end_sampling|>`), padding 199999, and `add_bos_token=false`.
Runner now recognizes that name instead of assuming the GPT-2 split.

On 2026-11-22, using only metadata from
`unsloth/Inkling-2-Mini-GGUF` revision
`45fd08764f2b0bf16b03a5483a80e5c7aaec8772`, file
`Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`:

- Before the fix: 21 of the fixed corpus's 721 strings differed.
- After the fix: 0 of 721 differed against the publisher revision above.
- Negative control: remove the single `H` + `E` merge from the publisher
  tokenizer. The same differential fails, 1 of 721 strings differs:
  `HELLO WORLD`. Its reference IDs are `[146066,148284,199464,146001]`;
  the control gives `[39,146981,46,199464,146001]`.
- `test-tokenizer-classes` also fails before the fix on the new name, and
  passes after it, including the ignore-merges and no-BOS settings.

The captured reference is
`tests/compatibility/tokenizer-references/inkling-2-mini.tokenizer-json.json`.
With a metadata-only GGUF from that file, reproduce without downloading
weights:

```sh
make STRICT=1 difftok test-tokenizer-classes
./test-tokenizer-classes
python3 scripts/difftok.py --gguf /path/to/tokenizer-only.gguf \
  --ref-ids tests/compatibility/tokenizer-references/inkling-2-mini.tokenizer-json.json
```

README impact: none at this step. This tokenizer correction does not add a
model to the support list or publish a model artifact; the admission, site
and model-card updates remain dependent on their engine and evidence gates.

## Native chat renderer

The oracle is the publisher's [chat_template.jinja](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/b85c3cfa615fc413cc9af3379f491b963f3b586a/chat_template.jinja),
SHA-256 prefix `53a9035f32559457`, pinned in the conformance baseline.
It uses message/content-channel tokens, emits a thinking-effort system turn
before the first non-system message, and appends no assistant generation
prefix. Historical reasoning, text parts, calls and tool results retain their
separate segments. Tool declarations sort all keys; call arguments sort only
the outer keys, as the template does.

On 2026-11-22, `scripts/template-conformance.py --family inkling2
--require-tokens` compared 35 cases, all byte-identical and token-identical,
with no allowlist or backlog entries. They include all named levels (`none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, `max`) and numeric 0, 0.37, 0.99.
Changing the output's single colon in `Thinking effort level:` to a question
mark makes the same gate fail; restoring and rebuilding makes it pass.
`test-template-families` checks this renderer's capacity and mode contract.

This step renders tool histories; it does not yet certify generated calls,
forced reasoning closes or the complete CPU suite. The chat renderer's
structured source fields preserve content-part and reasoning boundaries;
other API adapters still need their protocol checks before admission.

## Review checks

The capacity sweep includes zero, and Inkling also permits a `NULL, 0` size
query. Structured family fixtures now carry real source messages, content
arrays, reasoning, calls and effort values. An exact publisher-framing anchor
checks their output; removing content, reasoning, calls or effort changes that
anchor. It also pins case-sensitive outer argument sorting and preserves the
nested argument order. The model's template deliberately ignores
`add_generation_prompt` and `enable_thinking`; it does not pre-open reasoning.

The widened template matrix compares 37 text/history cases byte for byte and
token for token, including an empty argument string (rendered as `{}`) and an
unattributed tool result (a bare tool header). Two developer-role cases are
refused by both sides, as the publisher requires. Two image cases are
intentionally refused by the HTTP text path, each with a cited allowlist entry.
The low-level renderer's user/tool image placeholders match the publisher, but
this does not encode images or claim image understanding. A refusal is never
counted as a byte or token comparison. Its gate requires a successful oracle
render and the expected API error; replacing that expected error is a failing
negative control.

The vocabulary differential visits all 199,998 base entries. It compares the
197,979 that decode to whole UTF-8 text: zero differences. The existing corpus
builder explicitly excludes 2,019 partial-UTF-8 pieces, which are not valid
standalone text inputs; this is not a claim of 199,998 successful encodings.
A reference with `ignore_merges=false` differs on 337 of the 197,979 strings,
confirming this check reaches the whole-vocabulary rule missed by 721 texts.

The real metadata-only fixture also checks stop handling: `end_message` is
200010 and is not a generation stop; `content_model_end_sampling` is 200006
and is the stop. The generic spelling list had also stopped on padding ID
199999 (`endoftext`); Inkling now uses only declared EOS/EOG IDs. The real-header
test failed before that correction and passes after it:

```sh
./test-tokenizer-classes /path/to/tokenizer-only.gguf
python3 scripts/difftok.py --gguf /path/to/tokenizer-only.gguf   --ref thinkingmachines/Inkling-2-Mini   --ref-revision b85c3cfa615fc413cc9af3379f491b963f3b586a --vocab
```

## Caller audit and remaining scope

`server.c:handle_chat` supplies structured source and effort fields.
`api_responses.c:handle_responses` and `api_anthropic.c:handle_messages` had
flattened them away and are reachable with a forced Inkling template. Those
surfaces now return 400 for Inkling and direct callers to
`/v1/chat/completions`, rather than silently rendering incomplete history.
Their native structured adapters remain work before those surfaces can be
claimed for this model.

The CLI in `main.c` builds flat text turns, and `decide.c` builds its two flat
policy turns. Neither supplies structured reasoning or tool-call objects or
numeric template kwargs; the flat renderer uses the publisher's default effort.
The CLI's generated-tool history and reasoning transitions still need their
protocol gate. The public rendering wrappers in `template.c` forward the
fields they receive and do not invent missing structured data.

## Generated protocol, grammar closure, and reasoning transitions

- `TP_INKLING2` (`src/schema_tools.c:schema_compile_inkling2_turn`,
  `src/template.c:ts_inkling`, `inkling2_tool_calls_parse`) handles both
  named (`<|message_model|><name><|content_invoke_tool_json|>`) and bare
  (`<|content_invoke_tool_json|>`) calls, `<|content_thinking|>`,
  `<|content_text|>`, `<|content_tool_error|>`, and
  `<|content_invoke_tool_text|>` blocks, and parallel call sequences up to
  `<|content_model_end_sampling|>`.
- `src/schema.c:sval_raw_marker_opens` and `src/engine.c:constraint_spelling_ok`
  (`has_control_sentinel`) admit single control-token closers (`<|end_message|>`)
  in Inkling `SN_RAW` frames while rejecting arbitrary control tokens before or
  inside Inkling raw content without altering other families;
  `sval_close` closes truncated turns cut at any token or byte boundary to a
  schema-valid turn with parseable arguments (`tests/test_tools.c:test_inkling2_native_protocol`,
  `tests/test_inkling2_protocol.py`, `tests/test_choice_mass.c`).
- `src/engine.c:engine_bind_think_tokens` binds `<|content_thinking|>` and
  `<|end_message|>` as single control tokens so `--reasoning-budget`
  (`reasoning_max_tokens`) and `--loop-guard` terminate deliberation cleanly via
  `<|end_message|>` without stopping generation, and
  `src/completion.c` spells the six channel control tokens (`INK_TOOL_JSON` only
  when tools are declared) on unconstrained chat turns for `INKLING_PLAIN_ENV`
  (`TP_INKLING_PLAIN`) demultiplexing. Models lacking a single-token
  `<|end_message|>` return HTTP 400 on `reasoning_max_tokens`.
- Plain chat without declared tools (`TP_INKLING_PLAIN`) never emits `tool_calls`
  or `finish_reason="tool_calls"`, and reports frame or call faults with
  `envelope_error` in both streamed and buffered paths.
- **Deliberate chat-serving deviation (`--verify` covered)**: publisher EOS-only
  `tok_stop_ids` is preserved verbatim for tokenizer fidelity and raw
  `/v1/completions`; unconstrained Inkling `/v1/chat/completions` turns and
  interactive CLI `-i` chat (`src/main.c:run_chat`) add `<|message_user|>`,
  `<|message_system|>`, and `<|message_tool|>` as per-request `req_stop_ids` so
  generation halts at the turn header before decoding another token, reports
  `ended_at="turn_header"` and `turn_header` (`runner_telemetry`, receipt `serve`,
  and CLI stderr), and replays `VERIFIED` under `--verify`
  (`tests/test_inkling2_protocol.py`).

This work completes protocol parsing and reasoning transitions; weights
download, CUDA/Metal inference fidelity runs, and model admission remain
outside this branch.
