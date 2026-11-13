# gpt-oss-20b: ggml-org's MXFP4 file against OpenAI's float32 reference, measured and not admitted (2026-11-13)

gpt-oss-20b is OpenAI's 21B mixture-of-experts model: 24 layers, 32 experts with 4 active, sliding-window attention
(128) on alternate layers, attention sinks, YaRN rope (factor 32 over an original 4,096, base 150,000), vocabulary
201,088. The gpt-oss family's only roster row is the lab's own gpt-oss-20b-keep30-mxfp4, a 32-to-30-expert derivative
published as a disclosed near-miss, still without cells. This read asked whether ggml-org's MXFP4 conversion could be
the family's first file with cells. **It does not pass the fidelity bar, and no row joins.**

**The result, first:**
- **Mean KLD 0.0709 ± 0.0143 over 544 positions, against a bar of 0.05: fails.** The bootstrap 95% interval is 0.0468
  to 0.1019, and 95% of the resampled means are over 0.05. Margin-qualified top-1 is 98.02% (496/506), which passes.
- **The gap is not Runner's.** On window 1, llama.cpp 73a43d1 reads the same file at 0.0672 and 98.01%, against
  Runner's 0.0634 and 97.61% on the same positions. The two engines agree with each other closely: their top-1 is the
  same token at 260 of 272 positions (95.6%; 243 of 251 margin-qualified), and their per-position KLDs track (Pearson
  r 0.78, median absolute difference 0.009).
- **The file is not the publisher's precision outside the experts.** Its experts are MXFP4, as the publisher's are,
  but its attention projections (q, k, v, o), token embedding and output head are Q8_0 where the publisher stores
  bf16 (norms, sinks, router and all biases are F32). Runner's route reports the mix as MXFP4 84%, Q8_0 16%. Those
  Q8_0 tensors carry a loss of their own, which both engines read alike.
- **What the evidence allows:** the gap belongs to the file, or to a reading of it that both engines share. It does
  not separate those two, since no file of this model with bf16 attention and embeddings was measured.
- **The text is outside this model's usual input.** gpt-oss is trained on harmony-framed conversations, and the
  corpus is bare text with no special tokens. Runner's top-1 matches the corpus token at 1.2% of positions (a score
  perplexity of about 8.2 million). The worst positions are diffuse, low-margin distributions over punctuation-like
  tokens (position 2865: reference "unch", Runner "…", KLD 5.2). The median KLD is 0.0127, so most of the mean comes
  from that tail, but the median itself is well above the other families' Q8_0 rows. A harmony-framed read would be a
  method change. It is parked for the owner and was not made here.
- **Scope:** two windows over one corpus, CPU score files, inside 4,096 tokens. Task quality is not measured.

The kills were declared before the read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%).

## The file and the reference

- **Reference:** `openai/gpt-oss-20b` at `6cee5e81ee83917806bbde320786a8fb61efebee` (not gated; licence apache-2.0),
  the three safetensors shards, each equal to the Hub's LFS record. The experts are stored in MXFP4, everything else
  in bf16.
- **Reference path:** transformers 5.16.1, the native GptOssForCausalLM (no remote code), on the CPU. With no Triton
  and no kernels it dequantizes the MXFP4 experts to bf16 on load, which is exact: every MXFP4 value times its
  power-of-two scale is representable in bf16. `scripts/gold-logits.py --upcast-at-use` then runs float32 arithmetic
  per decoder block. Load peak 51.6 GB, the read's peak resident set 116 GB including the lab's held files.
- **So both sides carry the publisher's own 4-bit experts.** The measurement covers everything else in the file,
  plus each engine's MXFP4 arithmetic, not a quantization of the experts.
- **File:** `ggml-org/gpt-oss-20b-GGUF` at `41e8a94d15d62070b36cf4553a0250717ecf46b8`, `gpt-oss-20b-MXFP4.gguf`,
  12,109,566,624 bytes, sha256 `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` (checked against
  the Hub's LFS record; the file the existing manifest entry `gpt-oss-20b-mxfp4` pins).
- **Windows:** two windows of 272 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, from position 8 and
  from 15. Token-id keyed: every position matched by id, no token spelling mismatches, no dropped reference mass.
- **Runner:** a CPU score file (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill
  route, 4,096 tokens, `-c 4096`) from the clean release-recipe STRICT build of main 92ef2e92 (runner 1.18.0, gcc 13.3.0,
  sha256 `50cca98c…`, provenance 92ef2e92 dirty false).
- **YaRN is native to this model on both sides.** Runner's load line reads `rope: YaRN scaling x32.00 (model
  metadata)`. The GGUF carries factor 32, original context 4,096, beta_fast 32, beta_slow 1, base 150,000, which equals
  the reference config's `rope_scaling` (`yarn`, factor 32.0, original_max_position_embeddings 4096, beta_fast 32.0,
  beta_slow 1.0, truncate false) and `rope_theta` 150000. Runner's ramp follows OpenAI's reference, with the correction
  range not rounded (MANUAL's architecture table; `src/model.c`).

## Against the reference

| Engine | Window | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Runner | 1 | 0.0634 | 0.0127 | 2.102 | 93.75% | 97.61% |
| | 2 | 0.0783 | 0.0127 | 5.211 | 95.96% | 98.43% |
| | **pooled 544** | **0.0709 ± 0.0143** | 0.0127 | 5.211 | 94.85% | **98.02% (496/506)** |
| llama.cpp 73a43d1 | 1 | 0.0672 | 0.0150 | 1.661 | 94.85% | 98.01% |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0468 to 0.1019. The llama.cpp row is llama-server at `-c 4096` on the CPU, sent
the reference's token ids, beside Runner's score file on the same positions. Positions over 0.05 on window 1: 36 on
both engines, 12 on Runner only, 29 on llama.cpp only.

Per position: `gold-mxfp4-w1.json`, `gold-mxfp4-w2.json`, `gold-mxfp4-w1-llama.json` (FILES.sha256).

## The tokenizer

- difftok against OpenAI's tokenizer.json at 6cee5e81: 0 of 721 corpus strings and 0 of 198,414 vocabulary entries
  differ. Tokenizer pre `gpt-4o`.

## The template

- Runner picks its `harmony` renderer. The `harmony` conformance family against the openai-harmony 0.0.8 library
  (the protocol's reference implementation): 21 cases compared, 21 identical in text and tokens.

## What the file resolves to on load

Architecture gpt-oss, template harmony, the native harmony tool protocol, sampling preset gpt-oss. `--adapt-info`
refuses training ("FFN activation not in the backward (gpt-oss/apertus)"); adapter serving is supported.

## What this changes

Nothing on the roster: no row, no cells, and the gpt-oss family line stays as it was. The manifest entry
`gpt-oss-20b-mxfp4` keeps its checks; its notes gain one sentence naming this read.
