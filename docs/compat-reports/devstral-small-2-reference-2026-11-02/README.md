# Devstral Small 2: two third-party files against the publisher's checkpoint, with the tool-calling axis (2026-11-02)

**Limits first.** Mistral AI publishes Devstral Small 2 24B Instruct 2512 as an FP8 (E4M3) checkpoint and serves it
with static FP8 activations. Every GGUF of the model, the two measured here included, requantizes that FP8
checkpoint. The reference below dequantizes it exactly to float32: that is the reference arithmetic, not the
publisher's serving arithmetic, so these numbers say how close a file is to the checkpoint, not to the publisher's
own deployment. The text path only (the vision tower is not loaded), the CPU only, one corpus, no task quality.

## Result

| File | Mean KLD | Max KLD | Top-1 | Margin-qualified top-1 | Bar (0.05 / 97%) |
|---|---|---|---|---|---|
| unsloth Q4_K_M | 0.0223 | 0.178 | 92.33% | 99.53% | inside |
| unsloth Q8_0 | 0.00126 | 0.024 | 99.33% | 100% | inside |
| ggml-org Q8_0 | 0.00126 | 0.024 | 99.33% | 100% | inside |

- The Q8_0 is closer than the Q4_K_M by the per-position KLD (Wilcoxon p 1e-50); margin-qualified top-1 is the same
  but for one position.
- **ggml-org's Q8_0 scores byte-identically to unsloth's.** Every logprob and every top-20 entry of the two score files
  is equal at every position. The files are 25,055,308,352 and 25,055,317,152 bytes: the same Q8_0 tensors (Q8_0
  quantization is deterministic from the same parent) with 8,800 bytes of metadata between them.
- **Agent torture, Q4_K_M, CPU:** 100 of 100 cases pass (`agent-torture-runner-q4_k_m.json`), across tool selection,
  nested arguments, large enums, forced truncation, structured finals, reasoning then a tool, and streamed and
  buffered normalization.

## Setup

- **Reference:** `mistralai/Devstral-Small-2-24B-Instruct-2512` at `55c5b41e98c2`, transformers'
  `Mistral3ForConditionalGeneration` driven with text only, on the CPU. The FP8 linears dequantize exactly to float32
  (weight times its per-tensor scale, checked over 167,772,160 elements); the 239 tensors the checkpoint stores in BF16
  (embedding, norms, head, vision tower) are cast to float32 after the load, which is exact (`scripts/gold-logits.py`,
  #545). Whole blocks beyond a 24 GiB budget are read from an offload directory at use.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, matched by token id at every position; the
  reference tokenizer's ids equal Runner's at all 8,104 corpus tokens.
- **Runner sides:** `runner --score --score-top 20` with `RUNNER_SCORE_CHUNKED=1` on the CPU (x86-64, AVX-512).
- **Files:** `unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF` at `6e458b8a`: Q4_K_M (sha256 `d14ba9ed...c380aee496`)
  and Q8_0 (`0760502e...cccc80866`); `ggml-org/Devstral-Small-2-24B-Instruct-2512-GGUF` at `f7fe27c2`: Q8_0
  (`3c191981...cd41f50759`).

## Another engine on the same files

llama.cpp build 12032 (commit `dde879bc4`, clang 22, CPU, `-fa off --jinja`), served with `llama-server` and read
through the same reference with the reference's own token ids (`--endpoint-ids`):

| File | Runner mean KLD | llama.cpp mean KLD | Runner margin-qualified top-1 | llama.cpp margin-qualified top-1 | Paired, same positions |
|---|---|---|---|---|---|
| unsloth Q4_K_M | 0.0223 | 0.0259 | 99.53% | 99.53% | Runner closer (Wilcoxon p 7e-16); no discordant qualified position |
| unsloth Q8_0 | 0.00126 | 0.00207 | 100% | 100% | Runner closer (Wilcoxon p 4e-28); no discordant qualified position |

Agent torture on the same Q4_K_M: llama.cpp passes 74 of 100 (`agent-torture-llamacpp-q4_k_m.json`). Every one of
its 26 failures falls in two categories, and each has one cause:

1. **Tool selection, 13 of 13:** with a named `tool_choice`, llama.cpp `dde879bc4` does not hold the turn to that
   function under Devstral's template, and the model answers in prose ("Understood. I am ready to proceed...",
   `finish_reason` `stop`).
2. **Forced truncation, 13 of 13:** with `tool_choice` `required` and `max_tokens` 1, llama.cpp `dde879bc4` returns the
   raw `[TOOL_CALLS]` token as content (`finish_reason` `length`) and no call, where Runner closes the truncated call
   into a valid one.

## Files here

- `gold-logits-runner-q4_k_m-q8_0.json`: the two unsloth files, every row.
- `gold-logits-runner-ggml-org-q8_0-llamacpp-q4_k_m.json`: ggml-org's Q8_0 on Runner and the Q4_K_M on llama.cpp.
- `gold-logits-llamacpp-q8_0.json`: the Q8_0 on llama.cpp.
- `agent-torture-runner-q4_k_m.json`, `agent-torture-llamacpp-q4_k_m.json`: the two torture reports.
