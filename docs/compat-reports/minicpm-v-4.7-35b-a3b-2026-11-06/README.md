# MiniCPM-V-4.7-35B-A3B: openbmb's files against their BF16 parent in float32, text path (2026-11-06)

**What is not flattering, first:**
- **The row's file is the Q8_0 (36.9 GB).** openbmb's Q4_K_M misses the bar (0.0594 mean KLD, 95.44%
  margin-qualified top-1).
- **The reference is not MiniCPM-V's own code.** The parent declares model_type minicpmv4_7, which transformers 5.16.1
  has no class for, and the repository ships no modeling code. Its text model is a qwen3_5_moe_text model by its own
  config, so the reference is transformers' Qwen3_5MoeForCausalLM on the checkpoint's language-model weights. What
  MiniCPM-V's code might do differently on text alone is not measured.
- **Text path only:** the vision tower and merger are not loaded; image and video content is refused.
- **Thinking is fixed off in this file:** the GGUF repository's root files carry the repository's template with the
  generation prompt's thinking test replaced, so enable_thinking has no effect, as in llama.cpp. openbmb recommends
  these files ("not trained as a reasoning model"). The thinking/ files are another template and not this row.
- **On a 24 GB CUDA card** the file runs split, 26 of 40 blocks on the device.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%); the Q4_K_M first, the Q8_0 if it missed.

## The files and the reference

- **Parent:** `openbmb/MiniCPM-V-4.7-35B-A3B` at `98e79ba972b586ac6146d3505771ce1467e199ba` (apache-2.0), held whole
  from the Hub (28 files, 70.45 GB, the vision tower and merger among them).
- **Reference path:** transformers 5.16.1. A view of the held checkpoint whose config.json is the parent's own
  text_config (model_type qwen3_5_moe_text, architectures Qwen3_5MoeForCausalLM); transformers maps
  model.language_model.* to model.*, and the model's 693 parameters match the checkpoint's 693 text tensors exactly.
  `scripts/gold-logits.py --upcast-at-use`: float32 arithmetic on the stored bfloat16, cast at use per decoder block
  (40 blocks).
- **Window:** 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, token-id keyed, every position
  matched by id, no spelling mismatches, no dropped reference mass.
- **Files:** openbmb's own GGUF repository, `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` at `cf943fbe`, the root files.
  - Q8_0: 36,902,368,928 bytes, sha256 `851fdec9d87d689342c3df04e509de4565ac3a96a226f06e7b12af651311b5ef`.
  - Q4_K_M: 21,166,255,008 bytes, sha256 `9ecf91e814d7e9ad...` (LFS record).
  - Each file's sha256 was checked against the Hub's LFS record as it was held.
- **Runner:** the score files from R4.27.6's tip (a10cb822, which main carries since #644), `--score --score-top 20
  --score-tokens 4510 -c 4510 --gpu off` on the host's CPUs; the cells from this branch on that main. Both are the same
  binary (runner 1.14.0, sha256 `6483dd008745...`).

## Against the reference

| File | Bytes | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|---|
| **Q8_0** | 36,902,368,928 | **0.0088** | 0.0027 | 0.222 | 96.00% | **100.00%** | **pass** |
| Q4_K_M | 21,166,255,008 | 0.0594 | 0.0339 | 0.679 | 88.33% | 95.44% | fail |

Per position: `gold-q8_0.json`, `gold-q4_k_m.json` (FILES.sha256). The Q8_0's distance also bears on the reference's
loading path: a mismatched text model could not land this close to the file.

## The tokenizer and the template

- **Tokenizer:** difftok against openbmb's tokenizer.json at 98e79ba9: 0 of 721 corpus strings and 0 of 247,091
  vocabulary entries (before R4.27.5: 3 and 5,503). The file labels its pre-tokenizer qwen35; openbmb's tokenizer.json
  uses the qwen2 split, which R4.27.5's rule reads by the file's general.name. The reference ids are captured in
  `tests/compatibility/tokenizer-references/minicpm-v-4.7-35b-a3b.tokenizer-json.json`.
- **Template:** the file's own (7,249 bytes, sha256 `5aaa30dc`), rendered by `minicpmv-fixed` (R4.27.6).
  template-conformance `--require-tokens`, family `minicpmv47-fixed` on this file: conformant in text, one case not
  expressible (a system message mid-history, which the reference refuses), and the reasoning-history case accounted
  as a token-only deviation (caller-spelled `<think>` tags Runner reads as text, R2.4.11).

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work.
- **CPU and CUDA:** tier-smoke pinned each, then a compare run matched (fact found). The two 48-token replies are
  byte-identical.
- **gpu-identity:** ok, the fitted split at 26 of 40 blocks. Pinned routing 2.01e-06 of the mean range (limit
  0.001), worst 0.000661; free routing 8.88e-04 (limit 0.002), 1 of 30,400 expert slots off the device's own choice.
- **compat:** load, tokenizer and chat pass; cpu_cuda pass, 9 of 9 prompts exact.
