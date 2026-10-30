# Clef-Flash, 2026-10-30

Roster rows `cloudflare-clef-flash` (Q4_K_M) and `cloudflare-clef-flash-q8_0`, Tier 2. **The float32
reference is owed.** Nothing here measures fidelity against Cloudflare's parent.

## The file

Cloudflare publishes no GGUF of `Cloudflare/clef-flash`. The pinned files are ggml-org's conversion:

| | |
|---|---|
| Upstream checkpoint | `Cloudflare/clef-flash` at `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` |
| Converter | ggml-org/convert (the repository's `.src_sha` names `f0b7d3cd` as its source) |
| GGUF repository | `ggml-org/Clef-Flash-GGUF` at `6e2a164d70aae2af7fce86005da3f91805565b1d` |
| Q4_K_M | `Clef-Flash-Q4_K_M.gguf`, 6,732,781,824 bytes, sha256 `eef2968c238a80d49d340430f22410ea3db9934ad1c6fdb233a503cbf9a9b54e` |
| Q8_0 | `Clef-Flash-Q8_0.gguf`, 9,657,260,288 bytes, sha256 `25f801e2a27610b0ccc0cb1558d7eda3d7a304e34e2c4a04e3079357ff7d906e` |

Cloudflare's weights have not changed since they were uploaded (`89c693f9`, 2026-09-30). Every later commit to
the checkpoint repository changes only the card, the eval results or the tags.

**What the file contains.** It has `general.architecture` `clef`, which is the admitted family
(docs/clef-admission-2026-10-16.md):
- the Qwen3.5-9B hybrid backbone (32 layers, full attention every fourth layer);
- the joint schema head as `dec.blk.*` (2 routing and 4 decoder blocks, 16 heads), with `clef.decision.type`
  `clef`;
- tokenizer pre `qwen35`.

Two keys are new against the 27B Clef files, and Runner ignores both: `tokenizer.ggml.normalizer.nfc` (Runner
already normalizes NFC on this tokenizer) and `tokenizer.ggml.clean_spaces`. The file also embeds the systemone
prompt as `tokenizer.chat_template.systemone`. Runner builds that prompt in `src/clef.c` instead, and the
15 fixture records below encode identically on both backends. Comparing the record encoding against the
publisher's `encode_record` is part of the owed float32 run.

**The quantization mix.** The Q4_K_M is ggml-org's mix and not bartowski's file of the same name:
- Q8_0: attention, the SSM projections, the output matrix and the decision head;
- Q4_K: the feed-forward and the token embedding.

**The two files' headers.** All 38 `tokenizer.*` and `clef.*` metadata values of the Q8_0 are byte-identical
to the Q4_K_M's, including the vocabulary, the merges and both templates. The Q8_0 was read by an anonymous
range request; it has not been downloaded.

## Tokenizer

`scripts/difftok.py --gguf models/Clef-Flash-Q4_K_M.gguf --ref <Cloudflare/clef-flash tokenizer.json at f0b7d3cd>`:

| Corpus | Strings that differ |
|---|---|
| tests/fixtures/tokenizer-corpus.txt | 0 of 721 |
| tests/fixtures/tokenizer-nfc-corpus.txt | 0 of 13,264 |

Cloudflare's `tokenizer.json` carries the qwen2 split (`\p{L}+`), and Runner reads `qwen35` on a clef file as
that split. Its `tokenizer_config.json` names a different `pretokenize_regex` (`[\p{L}\p{M}]+`), but
`tokenizers` reads the `tokenizer.json`. The capture is
tests/compatibility/tokenizer-references/clef-flash.tokenizer-json.json.

## Template

**The template.** The file's `tokenizer.chat_template` is 7,756 bytes, sha256 `a4aee8afcf2e0711...`. It is
byte-identical to `chat_template.jinja` in `Cloudflare/clef-flash` at f0b7d3cd, and to Qwen3.5-9B's and
Qwen3.5-4B's. Runner detects it as `qwen35`, with tool family `qwen3_xml`.

**Conformance.** `scripts/template-conformance.py --family clef-flash --require-tokens` reads the oracle from
the file:
- 22 of 22 cases are conformant in text, and 22 of 22 in tokens.
- `system-mid-history` cannot be expressed, because the template raises "System message must be at the
  beginning".

## Smoke cells (the Windows CUDA host, RTX 3070, Runner 1.7.1 built at b9ad4e9d)

| Row | Backend | Result | Placement | Reply |
|---|---|---|---|---|
| cloudflare-clef-flash | cpu | pass, 16.1 s | `--gpu off` | sha256 `0b9c7366...` |
| cloudflare-clef-flash | cuda | pass, 4.7 s | resident, 32 of 32 layers and the head | sha256 `0b9c7366...`, byte-identical to the CPU |
| cloudflare-clef-flash-q8_0 | cpu, cuda | not run yet | | |

The cells were run by `tier-smoke.py --row ... --pin`, then `--record`, never typed. The pins are in
tests/compatibility/tier-smoke/.

## Decisions, CPU against CUDA (Q4_K_M)

The 15 records of tests/fixtures/clef-records.jsonl went through `clef-gold.py requests`, then
`runner --systemone` with `--gpu off` and resident on CUDA (ctx 4096; the longest state is 3,789 tokens). The
answers are in `answers-q4_k_m-cpu.jsonl` and `answers-q4_k_m-cuda.jsonl`.

| Measure | Result |
|---|---|
| input ids equal | 15 / 15 |
| question and option spans equal | 15 / 15 |
| max \|dlogit\| | 1.8e-4 |
| max \|dprob\| | 3.0e-5 |
| answers byte-equal after rounding | 27 / 29 |

The two answers that differ (`choice-keys/route` and `long-state/worst`) differ by one unit in the fourth
decimal. The model card's two examples decide as the card describes:
- **The invoice:** `overdue` at 0.9716, and the total above 1000 USD at 0.9723.
- **The checkout message:** `technical` at 0.9606, urgency `Today` at 0.85 (score 1.7742), service down at
  0.8465.

That shows the head binds and decides on this checkpoint. It is not a fidelity measurement.

## Owed

1. **The float32 gold run**, on the lab CPU lane after MiMo and Spark:
   - `scripts/clef-gold.py reference` with Cloudflare's `joint_schema_model.py` at f0b7d3cd, in float32;
   - compared against ggml-org's BF16 GGUF of the same revision (`Clef-Flash-BF16.gguf`, sha256
     `34635743...`), and against these two files.
2. **The Q4_K_M compat report**, from `compat_matrix.py --execute-checks --verify-files`. The run was stopped
   for low memory on the host, before any check finished.
3. **The Q8_0:**
   - the CUDA cell, as a layer split with `--gpu N`;
   - the CPU cell, or its skip reason if the file does not fit free RAM with headroom;
   - the decisions, CPU against CUDA;
   - the compat report.
4. **Metal** (a declared limit: the qwen35 backbone has no Metal path) and **a llama.cpp column**.
