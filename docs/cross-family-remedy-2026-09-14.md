# Cross-family agent reliability: the remedy record

Started 2026-09-14 against main at `53dc280`, from the outside review of the
v0.5.3 Windows report (Qwen 3.8 streaming loses native calls; Gemma 4 served
under Gemma 3's repeat penalty; Qwen3-Coder untested on 0.5.3; CUDA prefill
and WDDM pressure on a 12 GB card). This file is the durable progress record
the plan asked for: what landed, what each gate proved, what is still open
and why. It is updated in the same pull request as the work it describes.

Legend: DONE (merged, gate green), PARTIAL (merged, scope remaining named),
NOT RUN (gate exists or is planned, not executed on the required hardware or
artifact), BLOCKED (cannot run here; the blocker is named), OPEN (not started).

## 1. Tool contract

| Item | State | Evidence |
|---|---|---|
| Qwen 3.8 / Granite 4.2 / Ornith native XML parsed when streamed | DONE (PR: xml-native-routing) | `tests/test_native_xml_routing.py`: the report's case A and case B replies dictated through the scripted-reply hook, streamed and buffered, on Chat, Responses and Messages; `tests/test_tools.c` parse-only demultiplexer at every byte split. Red first: the streamed case A reproduced the report byte for byte (`finish_reason: stop`, content = the XML). |
| Native parsing independent of constrained generation | DONE | `tool_envelope.parse_only` (template.h): no grammar is compiled, the demultiplexer runs; the buffered turn is mapped by the same demultiplexer (`tool_stream_map`). |
| Same declarations, grammar, parser, history on all three surfaces | DONE for the XML families | `tool_decl_native` selects the protocol once; Chat, Responses and Messages render `tools_render_for` (the family's declaration block) and never the generic teaching turn when the envelope is native. Responses used to pair Qwen 3.8's native declarations with the GENERIC grammar. History serialization was already template-keyed (`tool_history_render_for`). Remaining difference: the Chat surface merges the caller's system text into ornith/granite42's declaration turn, Responses and Messages render two system turns (OPEN, rendering parity, not a routing fault). |
| Prose before / between / after calls | DONE | Kept as content; the whitespace the template writes after a call is framing and dropped. |
| Multiple calls, stable ids, contiguous indices | DONE | Case B: indices 0,1,2, three distinct ids. |
| Typed arguments | DONE | integer, boolean, array, nested object, Unicode, escaped quotes, a string that looks like a number stays a string (`test_qwen38_typed_arguments_keep_their_types`). |
| Arbitrary stream boundaries | DONE | Every split of the case-B document in `tests/test_tools.c`; the fixture spells a scripted reply with spaces one byte per token, so the HTTP tests run the hardest split. |
| Tool-result replay | DONE (Chat) | `test_qwen38_tool_result_round_trip`; Responses/Messages replay goldens already in `tests/test_tool_attribution.c`. |
| Truncation semantics | DONE | A budget cut inside a block: complete calls kept, partial block dropped and never completed, `length`. Constrained (Qwen3-Coder) keeps the documented closer contract. |
| Malformed / invalid block | DONE | Undeclared function, typed parameter that does not parse, missing required parameter: the block is neither content nor a call, `finish_reason: error` + `finish_detail: envelope_unmapped`, valid calls kept. Nothing is invented. |
| Literal tool syntax in prose | DONE | `<tool_call>` mentioned in a sentence that never opens a function is content. |
| Reasoning framing | DONE | Granite 4.2 had NO reasoning splitter (think tags were keyed on architecture; `granitehybrid` sets none): `template_think_tags` now supplies the template's pair where the architecture has none. Ornith's splitter primed only when thinking is on. |
| Real artifact: Qwen 3.8 27B GSQ-RCO IQ3_S, Blackwell MIG 1g.24gb, shipped defaults (`qwen3` preset) | DONE | `docs/cross-family-remedy-evidence/tool-protocol-qwen38-gsq-iq3s-blackwell-branch.json`: cases A/B/C buffered+streamed 6/6 PASS, case B three `glob` calls in both modes, C prose beside the call. The baseline binary (main `53dc280`, v0.5.3 routing) on the same file and script: `...-baseline-53dc280.json`, 4/6 FAIL exactly as the report (streamed turns `stop` with the XML in content; buffered C erased the leading prose). Recorded on the way: decode 1.8 tok/s with 61/64 layers on the device and `q35_hist_prev` refused by the device allocator after the fit (the fit does not budget it), `prompt_cached_tokens` 0 on repeated prompts (GPU-backed recurrent instances decline shared-prefix restore). Both are section 5 items. |
| Qwen3-Coder retest on 0.5.3 | NOT RUN | Needs the real 30B-A3B Q4_K_M artifact on a box with the memory for it (Blackwell); the fixture gate `tests/test_qwen3_coder_tools.py` stays green and is the regression control. |
| Real Windows / OpenCode tracer | NOT RUN | Planned on ZEN-GAMING (RTX 3070 8 GB): Granite 4.2 3B (fits) under OpenCode latest; Qwen 3.8 27B IQ3_S on the Blackwell. |
| `tool_choice: required` / named on the parse-only families | OPEN | Without a grammar the choice is taught by the prompt only; enforcing it means the constrained XML turn for these families, a separate measured decision (the grammar has no branch for their reasoning block). |

## 2. Sampling

| Item | State | Evidence |
|---|---|---|
| Gemma 4 publisher-backed preset (repeat penalty 1.0) | OPEN | |
| Preset audit across the inventory | OPEN | |
| Effective sampling / protocol diagnostics per request | OPEN | |
| Positive-temperature default-sampling regression | OPEN | |

## 3. Coverage inventory

OPEN. `tests/compatibility/models.json` (29 artifacts) to be reconciled with
`model_supported_archs()`, the 22 template ids, README/site claims and the
three reported artifacts.

## 4. Layered gates

| Layer | State |
|---|---|
| Deterministic protocol/HTTP contract suite | PARTIAL: the XML families on all three surfaces (this PR); the scripted-reply hook is the enabler for the rest of the matrix. |
| Publisher-anchored template/tokenizer/sampling | existing `scripts/template-conformance.py --require-tokens`; sampling anchors OPEN. |
| Actual artifact suite at shipped defaults | OPEN |
| OpenCode / Continue real client loops incl. Windows | OPEN (see the addendum: all client rows re-validated at latest versions, Blackwell first) |
| Pressure / cancellation / recovery / cache / performance | OPEN |

## 5. Resources and performance

OPEN: WDDM budget provider, fault telemetry naming, stage timings, cache-miss
reasons, RTX 5070 profile, TC dispatch engagement on Gemma Q4_0.

## Test hook: scripted replies

`RUNNER_TEST_SCRIPTED_REPLY=1` at server start admits `runner_test_reply`
(a string) on a completion request; the sampler returns it token for token
instead of sampling, then stops. Without the variable the field is refused
with 400, so nothing served in production can rely on it. The fixture's
byte-fallback vocabulary cannot spell a space, so a reply containing one is
spelled a byte per token (the hardest split for a demultiplexer); a reply
without spaces arrives in the vocabulary's pieces. Specials spelled in the
reply are the model's own tokens.

## Blockers

- None yet. Hardware used so far: the M1 development box only.
