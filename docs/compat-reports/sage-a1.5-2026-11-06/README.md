# Sage-A1.5: InternScience's files against their BF16 parent in float32 (2026-11-06)

Sage-A1.5 is InternScience's model. Its publisher built it on Qwen3.6-35B-A3B and on InternScience's Agents-A1, both
apache-2.0. It is not a Qwen release. The row names it as InternScience's throughout and states the lineage as the
publisher gives it. Agents-A1's own base is not given by its publisher, and the row does not guess it.

**What is not flattering, first:**
- **The row's file is the Q6_K, not a 4-bit file.** InternScience's Q4_K_M misses the bar, at 0.1445 mean KLD and 92.5%
  margin-qualified top-1. The publisher ships no Q5_K_M. Why the Q4_K_M reads 3x the base's Q4_K_M by the same method
  (bartowski's, 0.0475) is not examined here.
- **The template's verification switch is not served.** Sage's template adds a `verification` switch to Qwen3.6's: with
  `chat_template_kwargs: {"verification": true}` it appends "Verification protocol: enabled" to the system message,
  and without a system message it raises. Runner does not honour it: such a request does not get the trigger.
  - Measured on this file: the prompt is 26 tokens with the switch absent and 26 with it true. A true switch with no
    system message is answered 200, where the template would raise.
  - With the switch absent, the rendering is the template's (conformance below).
- **The name inside the file differs:** the GGUF's `general.name` is "Agents A1.5". The row uses the repository's name.
- **The file's size:** 29.2 GB, so on a 24 GB CUDA card it runs split, 34 of 40 blocks on the device.
- **Scope:** one 300-position window over one corpus. Text only: the vision projector is not loaded. Task quality and
  the tool protocol on the real checkpoint are not measured.

The kills were declared before each read: the licence and lineage, then the tokenizer, then the template, then the
owner's bar (mean KLD at most 0.05 and margin-qualified top-1 at least 97%). The smallest passing file is the row's.

## The files and the reference

- **Parent:** `InternScience/Sage-A1.5` at `0e3205cf441cf01df0c0d238cc4738d0e43b4ca7` (apache-2.0, no base_model in
  the card's metadata), held whole from the Hub (15 safetensors shards, 71.9 GB).
- **Reference path:** transformers 5.16.1, the text model of the composite checkpoint (Qwen3_5MoeForCausalLM), with
  `scripts/gold-logits.py --upcast-at-use`: float32 arithmetic on the stored bfloat16, cast at use per decoder block
  (40 blocks).
- **Window:** 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, token-id keyed, every position
  matched by id, no token spelling mismatches, no dropped reference mass.
  - The ids agree although Runner reads this file with the qwen35 split and transformers 5.16.1 tokenizes with the
    qwen2 split in memory. The gold corpus's combining marks fall where the two splits agree, as on the base row.
- **Files:** InternScience's own GGUF repositories. Each file's sha256 was checked against the Hub's LFS record as it
  was held.
  - Q6_K: `InternScience/Sage-A1.5-Q6_K-GGUF` at `8dfbf093`, 29,208,731,872 bytes, sha256
    `6862a4df34d209fc83d454f491460a4ebac23d1778c563b3266acf81bcd2397e`.
  - Q4_K_M: `InternScience/Sage-A1.5-Q4_K_M-GGUF` at `10b179b5`, 21,713,463,520 bytes, sha256 `a64fe0bbf936db3e...`.
  - The Q8_0 (`Sage-A1.5-Q8_0-GGUF`) was not read: the plan stops at the smallest passing file.
- **The NextN block:** the files carry Qwen's NextN predictor (block 40, `nextn_predict_layers` 1). Runner binds the
  40-block trunk and leaves the predictor out unless `--mtp`: "mtp: 1 predictor block(s) declared, excluded from the
  backbone".
- **Runner:** main e11ffca (runner 1.14.0), with
  `--score --score-top 20 --score-tokens 4510 -c 4510 --gpu off` on the host's CPUs.

## Against the reference

| File | Bytes | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|---|
| **Q6_K** | 29,208,731,872 | **0.0395** | 0.0109 | 2.134 | 95.33% | **97.91%** | **pass** |
| Q4_K_M | 21,713,463,520 | 0.1445 | 0.0429 | 7.507 | 84.33% | 92.47% | fail |

For scale, by the same method on the base: bartowski's Q4_K_M of Qwen3.6-35B-A3B reads 0.0475 / 97.2%, and his Q8_0
0.0349 / 99.1%.

Per position: `gold-q6_k.json`, `gold-q4_k_m.json` (FILES.sha256).

## The tokenizer

- **difftok against InternScience's tokenizer.json at 0e3205cf:** 0 of 721 corpus strings and 0 of 247,091 vocabulary
  entries differ (953 partial-UTF-8 entries left out).
- **That file uses the qwen35 split, as the GGUF header labels it,** so no per-file rule is involved. It has the
  vocabulary and merges of Qwen3.6-35B-A3B's tokenizer, plus 7 added tokens (248070-248076, audio and TTS).
- **Against Qwen3.6-35B-A3B's tokenizer.json:** also 0 and 0.
- The reference ids are captured in `tests/compatibility/tokenizer-references/sage-a1.5.tokenizer-json.json`.

## The template

- The GGUF's template (9,025 bytes, sha256 `0080393e`) is byte-identical to InternScience's chat_template.jinja at
  0e3205cf. It is Qwen3.6-35B-A3B's template with the verification switch added. Runner picks its `qwen36` renderer
  with qwen3_xml tools.
- `scripts/template-conformance.py --require-tokens`, family `sage-a1.5-qwen35moe` (the file's own template as the
  oracle, the switch absent): 25 cases, 25 identical in text and tokens. One case is not expressible:
  system-mid-history, which the reference refuses ("System message must be at the beginning."). The same holds on the
  Q4_K_M's header.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, with the host's CPUs shared with other work.
- **CPU:** tier-smoke pinned, then a compare run matched (fact found).
- **CUDA:**
  - test-gpu-identity ok, the fitted split at 34 of 40 blocks;
  - pinned routing (the CPU run's expert choices): 1 of 30,400 expert slots differed from the device's own choice;
  - tier-smoke pinned, then a compare run matched (fact found).
  - The 48-token reply is byte-identical to the CPU reply (sha256 `0341fe4d`).
- **compat:** load pass, tokenizer pass, chat pass, and cpu_cuda pass (9 of 9 prompts exact, with routing pinned).
