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
| Real client loop: OpenCode 1.18.31 (latest) against Qwen 3.8 27B GSQ-RCO IQ3_S, Blackwell | DONE | `docs/cross-family-remedy-evidence/opencode-1.18.31-qwen38-gsq-iq3s-blackwell.txt`: `opencode run` in a fixture directory, `→ Read sentinel.txt`, answer "The code word in sentinel.txt is TANGERINE-4471." (the sentinel is independently known), 586 s wall for three requests of 614/7491/7639 prompt tokens at 3.7 tok/s decode, `0 cached` on the 7.5K-token shared prefix (section 5). |
| Real Windows / OpenCode tracer | NOT RUN | Planned on ZEN-GAMING (RTX 3070 8 GB): Granite 4.2 3B (fits) under OpenCode latest for Windows. |
| `tool_choice: required` / named on the parse-only families | OPEN | Without a grammar the choice is taught by the prompt only; enforcing it means the constrained XML turn for these families, a separate measured decision (the grammar has no branch for their reasoning block). |

## 2. Sampling

| Item | State | Evidence |
|---|---|---|
| Gemma 4 publisher-backed preset (repeat penalty 1.0) | DONE (PR: sampling-presets) | `gemma4` preset from google/gemma-4-{12B,E4B,26B-A4B,31B}-it `generation_config.json` (all four: temperature 1.0, top_k 64, top_p 0.95, no repetition_penalty). Red first in `tests/test_sampler.c`. |
| Preset audit across the inventory | DONE, two owner decisions open | Table below. New: `qwen38`, `qwen3-coder`, `granite42`; corrected: `gemma3`; labelled as runner's calibration (unchanged): llama3, mistral, mistral-nemo, smollm2, lucie, teuken (1.10), phi3 (1.03). |
| Effective sampling / protocol diagnostics per request | DONE | `runner_telemetry.sampling` (preset, five values, seed, per-field source preset/cli/request) and `runner_telemetry.tool_protocol` (template, family, tools, constrained, parse_only) on every buffered body and the opt-in usage chunk; `/v1/capabilities` `template` + `tool_protocol`. |
| Positive-temperature default-sampling regression | DONE | `tests/test_sampling_defaults.py`: family default with a fixed seed equals explicit penalty 1.0 and differs from 1.1; greedy control agrees at both; request isolation; CLI precedence named as the source. |
| Real artifact: the report's Gemma 4 12B QAT Q4_0 (sha256 `93567e57…`), Blackwell, shipped defaults | DONE, one open case | `docs/cross-family-remedy-evidence/tool-protocol-gemma4-12b-qat-q4_0-blackwell-gemma4-preset.json`: under the `gemma4` preset the report's cases A and B pass 4/4 (buffered and streamed, arguments clean); under the previous `gemma3` preset (penalty 1.10, `...-gemma3-preset-penalty-1.10.json`) the same binary reads 0/6, the report's finding exactly, with the 1,300-character argument soup the gate now refuses (`framing in arguments`). Case C (prose then call) fails under both presets: Gemma 4's own format is call-first (the reference renders an assistant turn's calls before its text), its turn grammar's prose branch has no handoff to a call and admits the text-spelled framing, and the demultiplexer's text state only looks for the turn end. OPEN item: a Gemma 4 prose-then-call handoff (grammar raw-branch marker, control-token spelling admission, demux text state, buffered map). |
| Real artifact: Granite 4.2 8B Q4_K_M, ZEN-GAMING RTX 3070 (Windows 11, CUDA 39/40 layers), shipped defaults | DONE, one open case | Before (`...-baseline-v0.5.3.json`, `...-routing-only-generic-preset.json`): 0/6 both, the model reasoning to the 400-token limit under the generic preset's 1.10 penalty, reasoning served as content by v0.5.3. After (`...-granite42-preset-400.json`, `...-1500.json`): A 4/4, B 3/4 (one buffered turn `envelope_unmapped`; eight traced repeats 8/8 after the stray-opener tolerance), C 0/4 because the model puts its "what I am about to do" inside the reasoning block and calls with empty content, which the case requires as content. Model behaviour at defaults, recorded as FAIL. |
| Stop / suppression exemptions audit | OPEN | `tests/test_server_penalty_exemptions.py` pins the turn-terminator exemption; per-slot reset and model reload paths not yet re-read. |

### Preset audit, 2026-09-14 (publisher `generation_config.json`; a missing key is transformers' default)

| Preset | Pinned artifact | Publisher values | Runner before | Runner now |
|---|---|---|---|---|
| gemma4 | google/gemma-4-12B-it, -E4B-it, -26B-A4B-it, -31B-it | temp 1.0, top_k 64, top_p 0.95, no penalty | inherited gemma3 (1.0/0.95/64, penalty 1.10) | 1.0/0.95/0/1.0/64 |
| gemma3 | google/gemma-3-4b-it (read through the unsloth mirror, the Google repo is gated) | top_k 64, top_p 0.95, do_sample; no temperature or penalty key | 1.0/0.95/0/1.10/64 | 1.0/0.95/0/1.0/64 |
| qwen38 | Qwen/Qwen3.8-27B (config + model card, thinking mode) | temp 1.0, top_p 0.95, top_k 20, min_p 0, repetition 1.0 | qwen3 (0.6/0.95/20) by name | 1.0/0.95/0/1.0/20 |
| qwen3-coder | Qwen/Qwen3-Coder-30B-A3B-Instruct | temp 0.7, top_p 0.8, top_k 20, repetition 1.05 | qwen3 (0.6/0.95/20/1.0) by name | 0.7/0.8/0/1.05/20 |
| granite42 | ibm-granite/granite-4.2-3b, -8b | temp 1.0, top_p 0.95, do_sample; nothing else | generic (0.8/0.95/0.05/1.10/40) | 1.0/0.95/0/1.0/0 |
| qwen3 | Qwen/Qwen3-4B | 0.6/0.95/20 | same | unchanged |
| qwen2.5 | Qwen/Qwen2.5-7B-Instruct | 0.7/0.8/20/1.05 | same | unchanged |
| llama3 | meta-llama/Llama-3.2-3B-Instruct (unsloth mirror) | 0.6/0.9, no penalty | 0.6/0.9/0/1.10/0 | unchanged, penalty labelled a calibration (owner decision) |
| gpt-oss | openai/gpt-oss-20b | no sampling keys (card: 1.0/1.0) | 1.0/1.0/0/1.0/0 | unchanged |
| phi3 | microsoft/Phi-4-mini-instruct | no sampling keys (Phi-3.5 card: greedy) | 0.0/1.0/0/1.03/0 | unchanged, penalty labelled |
| smollm2 | HuggingFaceTB/SmolLM2-1.7B-Instruct | no keys (card: 0.2/0.9) | 0.2/0.9/0/1.10/0 | unchanged, penalty labelled (owner decision) |
| mistral, mistral-nemo, lucie, teuken | as cited in `src/sample.c` | no penalty stated | 1.10 | unchanged, penalty labelled (owner decision) |
| salamandra | BSC-LT salamandra-7b-instruct | 0.6, repetition 1.2 | same | unchanged |

Owner decisions: (a) drop the 1.10 calibration from the llama3/mistral/
smollm2/lucie/teuken presets, which the publishers do not state; (b) Qwen 3.8
instruct mode's `presence_penalty 1.5`, a knob this sampler does not have.

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
