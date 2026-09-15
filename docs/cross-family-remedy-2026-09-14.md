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
| Qwen3-Coder retest: the report's 30B-A3B Q4_K_M (sha256 `79ad15a5…`), Blackwell | DONE, two defects found and fixed | Under the XML grammar at the family's own temperature 0.7 the report's cases read 2/6 (`tool-protocol-qwen3-coder-30b-blackwell-constrained.json`): the raw-string parameter's sentinel required `\n</parameter>` and the model closes with ` </parameter>` as often as not, so the value ran on through the closers. Unconstrained (parse-only auto, `...-parseonly-penalty-1.05.json`) still 3/6: the publisher's `repetition_penalty 1.05` under this sampler corrupts the protocol (`<function=bash}`, then an OpenAI-shaped JSON array as prose); the A/B at temperature 0.7 read 0/8 at 1.05 and 8/8 at 1.0, and temperature 0 (penalty bypassed) is clean on GPU and CPU. The `qwen3-coder` preset carries 1.0 by measurement; the final row at shipped defaults is `...-final.json`. OPEN: why the publisher's stack tolerates 1.05 (a logit-margin comparison against the reference implementation). |
| `tool_choice: required` keeps the grammar on all four XML families | DONE | Qwen 3.8 GSQ IQ3_S with `--tool-choice required`, case A: 2/2 (`tool-protocol-qwen38-gsq-blackwell-required.json`), the constrained call after the reasoning block. Qwen3-Coder required (`...-required.json`) still shows the penalty corruption inside arguments; re-measured at the corrected preset below. |
| Real client loop: OpenCode 1.18.31 (latest) against Qwen 3.8 27B GSQ-RCO IQ3_S, Blackwell | DONE | `docs/cross-family-remedy-evidence/opencode-1.18.31-qwen38-gsq-iq3s-blackwell.txt`: `opencode run` in a fixture directory, `→ Read sentinel.txt`, answer "The code word in sentinel.txt is TANGERINE-4471." (the sentinel is independently known), 586 s wall for three requests of 614/7491/7639 prompt tokens at 3.7 tok/s decode, `0 cached` on the 7.5K-token shared prefix (section 5). |
| Real client loop on Windows: OpenCode 1.18.31 for Windows against Granite 4.2 8B Q4_K_M, ZEN-GAMING RTX 3070 | DONE | `docs/cross-family-remedy-evidence/opencode-1.18.31-granite42-8b-q4_k_m-windows-rtx3070.txt`: `→ Read sentinel.txt`, the sentinel answered, 276 s for five requests of up to 8.3K prompt tokens (prefix cache 7916/7990 hits on the second turn). |
| The README's coding-agent rows at the clients' latest versions (owner's addendum), Blackwell, Qwen3-4B Q4_K_M | DONE | `agent-client-sweep-qwen3-4b-blackwell-2026-09-15.json` (`scripts/agent-client-sweep.py`): opencode 1.18.31, claude 2.1.272, codex 0.154.0 (hosted web search disabled, the README's recipe), continue 1.5.47, cline 3.0.61, pi 0.85.1, aider 0.86.2 (dry-run) all PASS. Two versions could not make a single request against v0.5.3 and are fixed in this PR: Claude Code 2.1.272 (`allOf` of two patterns on `SendMessage.to`), Codex 0.154.0 (`include:["reasoning.encrypted_content"]`, `parallel_tool_calls:true`, replayed `reasoning` items). |
| One contract for the four XML families | DONE | `auto` (the shipped default) is the model's free turn, parsed; `required`/named keep the XML grammar (a prompt alone cannot enforce a choice); the grammar runs after the reasoning block through the constraint prelude (`engine_think_started`). |

## 2. Sampling

| Item | State | Evidence |
|---|---|---|
| Gemma 4 publisher-backed preset (repeat penalty 1.0) | DONE (PR: sampling-presets) | `gemma4` preset from google/gemma-4-{12B,E4B,26B-A4B,31B}-it `generation_config.json` (all four: temperature 1.0, top_k 64, top_p 0.95, no repetition_penalty). Red first in `tests/test_sampler.c`. |
| Preset audit across the inventory | DONE, two owner decisions open | Table below. New: `qwen38`, `qwen3-coder`, `granite42`; corrected: `gemma3`; labelled as runner's calibration (unchanged): llama3, mistral, mistral-nemo, smollm2, lucie, teuken (1.10), phi3 (1.03). |
| Effective sampling / protocol diagnostics per request | DONE | `runner_telemetry.sampling` (preset, five values, seed, per-field source preset/cli/request) and `runner_telemetry.tool_protocol` (template, family, tools, constrained, parse_only) on every buffered body and the opt-in usage chunk; `/v1/capabilities` `template` + `tool_protocol`. |
| Positive-temperature default-sampling regression | DONE | `tests/test_sampling_defaults.py`: family default with a fixed seed equals explicit penalty 1.0 and differs from 1.1; greedy control agrees at both; request isolation; CLI precedence named as the source. |
| Real artifact: the report's Gemma 4 12B QAT Q4_0 (sha256 `93567e57…`), Blackwell, shipped defaults | DONE | `docs/cross-family-remedy-evidence/tool-protocol-gemma4-12b-qat-q4_0-blackwell-gemma4-preset.json`: under the `gemma4` preset the report's cases A and B pass 4/4 (buffered and streamed, arguments clean); under the previous `gemma3` preset (penalty 1.10, `...-gemma3-preset-penalty-1.10.json`) the same binary reads 0/6, the report's finding exactly, with the 1,300-character argument soup the gate now refuses (`framing in arguments`). Case C (prose then call) failed under both presets: Gemma 4's own format is call-first, its turn grammar's prose branch had no handoff to a call, and the demultiplexer's text state only looked for the turn end. FIXED (PR: gemma4-prose-call): the prose branch hands off at `<|tool_call>call:`, the engine admits the control token there, the demux and the buffered map read a call after prose; `...-handoff.json` reads 6/6 at shipped defaults, case C 4/4 on repeats. |
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

DONE as a roster, mostly NOT RUN as rows: `tests/compatibility/coverage.json`
gives every admitted architecture label (19, read from `runner --caps`),
every chat-template id (22, read from `template_name`) and every pinned
artifact (32 rows in `models.json`, the three reported artifacts added by
hash with upstream, template, preset and quant composition) a disposition
for the tool-protocol gate, the client loop, the sampling anchor and each
backend: PASS with evidence, FAIL with a reason, NOT RUN, UNSUPPORTED.
`tests/test_coverage_inventory.py` refuses a label without a disposition,
a PASS without its evidence file, and a NOT RUN that carries evidence
(mutation-proven: dropping a template id and an architecture fails two of
its four checks). The rows that are PASS today: qwen38, granite42,
qwen3-coder, gemma4-mainline (A/B), chatml-think (client loop), and the
architectures behind them. Everything else is NOT RUN for the new gates,
which is the honest state, not a claim.

## 4. Layered gates

| Layer | State |
|---|---|
| Deterministic protocol/HTTP contract suite | PARTIAL: the XML families on all three surfaces (`tests/test_native_xml_routing.py`), sampling defaults (`tests/test_sampling_defaults.py`), the Codex Responses shape and replayed reasoning (`tests/conformance/test_responses.py`), Claude Code's schemas (`tests/test_json_schema.c`); the scripted-reply hook is the enabler for the remaining families (Gemma 4, Harmony, Muse, Qwen JSON, generic). |
| Publisher-anchored template/tokenizer/sampling | existing `scripts/template-conformance.py --require-tokens`; sampling anchors OPEN. |
| Actual artifact suite at shipped defaults | OPEN |
| OpenCode / Continue real client loops incl. Windows | DONE for the first pass: all seven rows on the Blackwell (Qwen3-4B), OpenCode on Qwen 3.8 (Linux) and Granite 4.2 (Windows). OPEN: Continue on Windows; the loops as a CI-adjacent gate (`--require` rows in release certification). |
| Pressure / cancellation / recovery / cache / performance | OPEN |

## 5. Resources and performance

| Item | State | Evidence |
|---|---|---|
| Windows-aware budgeting | DONE (PR: wddm-budget) | `plat_gpu_os_budget` (DXGI `IDXGIAdapter3::QueryVideoMemoryInfo`, adapter matched to the CUDA device by LUID) bounds the driver's free view; `model_gpu_budget` (gated in `tests/test_autofit.c`) applies it, `--reserve-vram`, and a headroom of max(512 MiB, budget/16). ZEN-GAMING RTX 3070: `OS video memory budget 7.60 GB for this process, 0.14 GB in use; offload budget 7.46 GB, headroom 0.54 GB`. The report's 12 GB WDDM paging is not reproducible here (no 12 GB WDDM card); the mechanism it needs is in place and logged. Recheck after allocations: the existing `VRAM ... free after init` line. |
| Peak allocations and serving slots in the budget | PARTIAL | The recurrent state (qwen35, granitehybrid, nemotron_h) was already in `act_bytes`; the headroom now scales. Per-slot KV is budgeted by the fit's own accounting. Not yet: a draft model's bytes and the prefix-cache snapshots. |
| Fault telemetry naming | DONE | `page_fault_counter` beside `major_page_faults` (`major` POSIX, `all` Windows). |
| Stage timings | PARTIAL | `timing.prefill_seconds/prefill_tokens/prefill_tok_s` measured; decode was `generation_seconds`. OPEN: queue wait, tokenization, first visible token, tool execution. |
| Cache-miss reasons | OPEN | Observed: Qwen 3.8 (qwen35, GPU-backed recurrent) `0 cached` on a 7.5K shared prefix under OpenCode; Granite 4.2 (dense) on Windows 7916/7990 hits. The README states GPU-backed recurrent instances decline shared-prefix restore; a per-request `cache_miss_reason` is not reported yet. |
| RTX 5070 profile, TC engagement on Gemma Q4_0, IQ3_S split throughput | OPEN | Observed on the way: Qwen 3.8 GSQ 61/64 layers on the Blackwell slice decodes at 1.8 tok/s and prefills at ~25 tok/s (the qwen35 recurrent CUDA path); Gemma 4 12B QAT Q4_0 full residency answered the report cases in 1.1-2.8 s. No profile taken yet. |

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

- None. Hardware used: M1 (development), Blackwell MIG 1g.24gb (real artifacts, client sweep), ZEN-GAMING RTX 3070 (Windows rows). Codex's bubblewrap sandbox cannot start in the Blackwell container (no unprivileged user namespaces), so its row ran with `--dangerously-bypass-approvals-and-sandbox` inside that container; the runner side is unaffected.
