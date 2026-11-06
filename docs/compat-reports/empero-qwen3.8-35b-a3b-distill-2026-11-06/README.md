# Qwen3.8-35B-A3B-Distill: empero-ai's files against their BF16 parent in float32 (2026-11-06)

Qwen3.8-35B-A3B-Distill is empero-ai's finetune of Qwen3.6-35B-A3B (SFT, off-policy distillation on teacher traces,
per its card). It is not a Qwen release; Qwen has published no model of this name. The row names it as empero-ai's
finetune throughout, and replaces the Upcoming `qwen3.8-35b-a3b`, which named a Qwen model.

**What is not flattering, first:**
- **The row's file is the Q6_K, not a 4-bit file.** empero-ai's Q4_K_M (0.1278) and Q5_K_M (0.0508) miss the bar,
  the Q5_K_M by 0.0008 mean KLD and 1.5 points of margin-qualified top-1. The publisher's Q4_K_M reads 2.7x the base
  model's Q4_K_M by the same method (bartowski's of Qwen3.6-35B-A3B, 0.0475). Why is not examined here: the
  repository's README states no importance matrix or converter.
- **The tokenizer depends on a rule keyed on header fields.** The file labels its pre-tokenizer qwen35, but empero-ai's
  tokenizer.json, saved by transformers 5.17.0, uses the qwen2 split. Runner reads this file with the qwen2 split by
  R4.27.5's rule, keyed on `general.name` "Ours", the `empero-ai` tag and the base model's URL. A requant that rewrites
  those fields falls outside the rule and reads the qwen35 split.
- **The file's size:** 29.2 GB, so on a 24 GB CUDA card it runs split, 34 of 40 blocks on the device.
- **Scope:** one 300-position window over one corpus. Text only: the vision projector is not loaded. Task quality and
  the tool protocol on the real checkpoint are not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), and the smallest passing file is the row's.

## The files and the reference

- **Parent:** `empero-ai/Qwen3.8-35B-A3B-Distill` at `bcc2dbe2f21b213625df2dc1a5a690212373af07` (apache-2.0,
  base_model Qwen/Qwen3.6-35B-A3B), the publisher's checkpoint, held whole from the Hub (34 files, 71.9 GB).
- **Reference path:** transformers 5.16.1, the text model of the composite checkpoint (Qwen3_5MoeForCausalLM),
  `scripts/gold-logits.py --upcast-at-use`: float32 arithmetic on the stored bfloat16, cast at use per decoder block
  (40 blocks).
- **Window:** 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, token-id keyed, every position
  matched by id, no token spelling mismatches, no dropped reference mass. 222 positions are margin-qualified.
- **Files:** empero-ai's own GGUF repository, `empero-ai/Qwen3.8-35B-A3B-Distill-GGUF` at `b1f9d1dc`.
  - Q6_K: 29,208,731,296 bytes, sha256 `0bb743311ee5d58eeeeff67d42d0e62a52347539e825ceb320d1a2178003f47b`.
  - Q5_K_M: 25,347,532,448 bytes, sha256 `f1903bac4ee3eec1...` (LFS record).
  - Q4_K_M: 21,713,462,944 bytes, sha256 `196103269085bc54...` (LFS record).
  - Each file's sha256 was checked against the Hub's LFS record as it was held. The Q8_0 was not read: the plan stops
    at the smallest passing file.
- **Runner:** the score files from 11c57228 (R4.27.5's tip, whose tokenizer main carries since e28991fa), with
  `--score --score-top 20 --score-tokens 4510 -c 4510 --gpu off` on the host's CPUs; the cells from this branch after
  main e28991fa was merged. Both builds are the same binary (runner 1.14.0, sha256 `bce3e25e51c0...`).

## Against the reference

| File | Bytes | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|---|
| **Q6_K** | 29,208,731,296 | **0.0336** | 0.0082 | 2.088 | 91.67% | **98.65%** | **pass** |
| Q5_K_M | 25,347,532,448 | 0.0508 | 0.0211 | 1.795 | 89.33% | 95.50% | fail |
| Q4_K_M | 21,713,462,944 | 0.1278 | 0.0395 | 6.009 | 85.00% | 94.59% | fail |

For scale, by the same method on the base model: bartowski's Q4_K_M of Qwen3.6-35B-A3B reads 0.0475 / 97.2%, and its
Q8_0 0.0349 / 99.1%. empero-ai's Q6_K sits about where the base's Q8_0 does.

Per position: `gold-q6_k.json`, `gold-q5_k_m.json`, `gold-q4_k_m.json` (FILES.sha256).

## The tokenizer

- difftok against empero-ai's tokenizer.json at bcc2dbe2: 0 of 721 corpus strings and 0 of 247,091 vocabulary entries
  differ (953 partial-UTF-8 entries left out). Before R4.27.5 it was 3 and 5,503, Thai and Devanagari strings where
  Runner kept a combining mark in its letter run and the reference splits it off.
- The file still tokenizes as its base, Qwen3.6-35B-A3B, did not: against Qwen's tokenizer.json at 995ad96e the same
  harness gives 3 and 5,503, by design.
- The reference ids are captured in
  `tests/compatibility/tokenizer-references/empero-qwen3.8-35b-a3b-distill.tokenizer-json.json`.

## The template

- The GGUF's template (7,764 bytes, sha256 `e84f32a2`) is byte-identical to empero-ai's chat_template.jinja at
  bcc2dbe2 and to Qwen3.6-35B-A3B's at 995ad96e. Runner picks its `qwen36` renderer with qwen3_xml tools.
- `scripts/template-conformance.py --require-tokens`, family `empero-qwen38-35b-a3b-distill` (the file's own
  template as the oracle): 23 cases, 23 identical in text and tokens. One case is not expressible:
  system-mid-history, which the reference refuses ("System message must be at the beginning."). That is the
  qwen36-35b-a3b family's result.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work.
- **CPU:** tier-smoke pinned, then a compare run matched (fact found). The 48-token reply is byte-identical to the
  base Qwen3.6-35B-A3B Q4_K_M's CPU pin (sha256 `f3a25e63`).
- **CUDA:** test-gpu-identity ok, the fitted split at 34 of 40 blocks. Pinned routing (the CPU run's expert choices):
  mean |dlogit| 3.61e-06 of the mean range (limit 0.001), worst 0.00331. Free routing reads 3.59e-06 (limit 0.002);
  6 of 30,400 expert slots differed from the device's own choice. tier-smoke pinned, then a compare run matched (fact
  found). The CUDA reply differs from the CPU reply from the second planet's sentence on.
- **compat:** load pass, tokenizer pass, chat pass, cpu_cuda pass_margin_qualified (8 of 9 prompts exact, one near-tie
  flip at margins of 0.012 and 0.004 nats, with routing pinned).
