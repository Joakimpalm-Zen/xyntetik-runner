# Gemma 4 26B-A4B against its publisher's float32 model, 2026-11-10

## Correction (2026-11-10, the same day)

**The rows this page first published were measured against a reference whose
embedding was scaled by 53.0, where float32 gives 53.066.**
- **The bug:** gold-logits' `--upcast-at-use` built the model at its stored
  dtype, so gemma's embedding scale (the square root of the hidden size,
  computed when the model is built) was born rounded to BF16. The check this
  page relied on could not see it: its tiny model's scale, the square root
  of 64, is exact in BF16.
- **The fix:** gold-logits now rebuilds such constants in float32 and refuses
  to score when one still differs (R6.36). Every row below is measured again
  against the fixed reference, with the same score files: Runner's side did
  not change.

**The chat number got better and the raw number got worse:**

| file | backend | corpus | first published | measured again |
|---|---|---|---|---|
| A, bf16 conversion | CPU | chat | 0.0060 (99.79%) | **0.0007 (100.00%)** |
| A, bf16 conversion | CPU | raw | 0.3226 (87.68%) | **0.4279 (83.53%)** |
| B, Google QAT Q4_0 | CUDA | chat | 0.2183 (90.59%) | **0.2140 (90.90%)** |
| B, Google QAT Q4_0 | CUDA | raw | 1.4900 (40.93%) | **1.4648 (41.46%)** |

Mean KL, with margin-qualified top-1 in brackets.

**The raw-text gap is now explained:** it is the f16 KV cache's rounding,
amplified by a model that is out of format on raw text.
- With `--kv f32` (R6.32), A reads 0.00003 on chat (100.00%) and 0.0055 on
  raw (99.75%).
- In the first publication the gap looked unexplained, because the
  reference's own embedding was off.

The sections below are the page as first published, with its rows replaced
by the measured-again ones. The text that called the raw gap unexplained,
and the equivalence check that missed the scale, are kept and marked.

The gemma-4-26B-A4B entry in [model support](model-support.md) listed a
publisher float32 comparison as owed. This page is that comparison: Runner
on two files, each scored against the transformers forward of
`google/gemma-4-26B-A4B-it` in float32 on the same token ids.

- **A, a fresh bf16 conversion of the checkpoint.** This measures the
  engine: its weights are the reference's own.
- **B, the file Runner lists for this model.** That is Google's QAT Q4_0,
  `gemma-4-26B_q4_0-it.gguf`, and this measures the file.

## What does not read well

They come first, because they are the parts the claim has to survive.

- **The conversion's raw-text row is not small.** On raw web text, A reads
  mean KL 0.4279 (first published: 0.3226), against 0.0007 on chat text.
  Quantization is not the cause. *As first published, this page did not
  explain the gap. The correction above does: it is the f16 KV cache's
  rounding, amplified out of format.* On raw text this instruction-tuned model is in a
  near-flat regime (Runner's perplexity on the scored text is about 41,000),
  where the 2026-10-21 investigation found that a row has little power. So
  the verdict rests on the chat corpus, and the raw rows are reported as
  measured.
- **The reference ran at the stored footprint.** gold-logits'
  `--upcast-at-use` keeps the bf16 weights and casts each decoder block to
  float32 for its own forward. On a tiny random Gemma 4 with this model's
  text block (the routed experts beside the dense MLP, K=V full attention,
  sliding and full layers, softcapped logits), loaded through the same class
  path, its logits equal a full float32 load's bit for bit. The same model
  in bf16 arithmetic differs by up to 2.42, so the check can fail. *That check
  was blind to the embedding scale, because the square root of its hidden size
  64 is exact in BF16. At hidden size 80 the unfixed path differs. See the
  correction above.*
- **B's distance is the whole distance of the shipped file** from the
  publisher's bf16 model: Google's quantization-aware retraining and its Q4_0
  rounding together. This page does not split the two.
- **A ran on the CPU and B on CUDA.** The bf16 file (50.5 GB) does not fit
  the 24 GB device. A anchors the engine and B is the file's row; their
  difference is not a backend comparison.

## The rows

Mean KL is the divergence from the reference's next-token distribution,
averaged over 2,000 positions (lower is closer). Margin-qualified top-1 is
the share of agreements with the reference's top token at the positions
where that token leads its second by more than 0.5 nats. Every position on
every side matched the reference by token id, with no dropped mass.

**Chat text** (nine user turns and the model's replies, rendered by the
publisher's chat template; 3,669 tokens):

| file | backend | mean KL | max KL | top-1 | margin-qualified top-1 |
|---|---|---|---|---|---|
| A, bf16 conversion | CPU | **0.0007** | 0.30 | 99.70% | 100.00% |
| A, bf16 conversion, `--kv f32` | CPU | **0.00003** | 0.02 | 99.95% | 100.00% |
| B, Google QAT Q4_0 | CUDA | **0.2140** | 7.95 | 87.55% | 90.90% |

**Raw text** (`tests/fixtures/gold-corpus-2k.txt`):

| file | backend | mean KL | max KL | top-1 | margin-qualified top-1 |
|---|---|---|---|---|---|
| A, bf16 conversion | CPU | 0.4279 | 9.78 | 76.85% | 83.53% |
| A, bf16 conversion, `--kv f32` | CPU | 0.0055 | 1.38 | 99.45% | 99.75% |
| B, Google QAT Q4_0 | CUDA | 1.4648 | 10.17 | 36.25% | 41.46% |

**A against B:**

- chat: 1,868 against 1,698 margin-qualified hits of 1,868, McNemar p =
  1.3e-51;
- raw: 1,354 against 672 of 1,621, p = 2.8e-164.

**Reading:**

- On chat text, the conversion is within the bar the project's own
  quantized releases are held to (mean KL at most 0.05, margin-qualified
  top-1 at least 97%). So Runner's gemma-4 MoE forward tracks the publisher
  on a faithful file.
- With `--kv f32` the conversion reads 0.00003 on chat and 0.0055 on raw
  text: Runner's forward is the publisher's float32 forward to that, and the
  f16 cache's rounding is the rest of its distance.
- The shipped QAT file reads 0.2140. On the same chat corpus, the
  2026-10-21 investigation's plain quants of a conversion read 0.295 (Q4_0)
  and 0.2485 (Q4_K_M), on an earlier build.

## The files and the run

| item | value |
|---|---|
| reference | `google/gemma-4-26B-A4B-it` at revision `4d7ae4984b7d` (unchanged since 2026-07-20); transformers 5.16.1, torch 2.14.0, CPU; class `Gemma4ForConditionalGeneration`; the measured-again rows used `scripts/gold-logits.py` with its R6.36 fix (embedding scale 53.066) |
| A | llama.cpp `convert_hf_to_gguf.py` at `a2c942dcfe`, `--outtype bf16`: 50,505,137,152 bytes, sha256 `75b8ae12d7706cc8c07755b64dfcb5941eef3a9cbdfc7cafc2ccf1f8b37e1ca7` |
| B | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` at `d1c082be9cf3`, `gemma-4-26B_q4_0-it.gguf`, sha256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |
| Runner | main `536dddc0`, the default build |
| A's route | CPU, BF16 100%, KV f16 (and f32 with `--kv f32`, Runner a8bad0dd) |
| B's route | CUDA MIG slice, 1g.24gb, sm_120: 30 of 30 layers, Q4_0 96% and Q6_K 4%, KV f16 |

Runner scored each file with:

```sh
RUNNER_SCORE_CHUNKED=1 RUNNER_SPEC_BATCH=64 \
  runner -m FILE --score --score-top 20 -f CORPUS -c 4096 > score.json   # A adds --gpu off
```

On raw text, Runner scored the corpus's first 12,000 characters, which
cover the 2,000 positions. One reference pass per corpus scored both files:

```sh
python scripts/gold-logits.py --hf CHECKPOINT --corpus CORPUS \
  --score-a score-A.json --score-b score-B.json --force-bos \
  --max-positions 2000 --stride 1 --top-n 20 --upcast-at-use
```
