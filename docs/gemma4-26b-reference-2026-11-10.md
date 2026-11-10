# Gemma 4 26B-A4B against its publisher's float32 model, 2026-11-10

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
  mean KL 0.3226, against 0.0060 on chat text: 54 times larger, on weights
  that are the reference's own. Quantization is not the cause, and this page
  does not explain the gap. On raw text this instruction-tuned model is in a
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
  in bf16 arithmetic differs by up to 2.42, so the check can fail.
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
| A, bf16 conversion | CPU | **0.0060** | 2.51 | 99.35% | 99.79% |
| B, Google QAT Q4_0 | CUDA | **0.2183** | 8.03 | 87.4% | 90.59% |

**Raw text** (`tests/fixtures/gold-corpus-2k.txt`):

| file | backend | mean KL | max KL | top-1 | margin-qualified top-1 |
|---|---|---|---|---|---|
| A, bf16 conversion | CPU | 0.3226 | 8.96 | 81.4% | 87.68% |
| B, Google QAT Q4_0 | CUDA | 1.4900 | 9.74 | 36.2% | 40.93% |

**A against B:**

- chat: 1,867 against 1,695 margin-qualified hits of 1,871, McNemar p =
  1.5e-50;
- raw: 1,416 against 661 of 1,615, p = 2.3e-196.

**Reading:**

- On chat text, the conversion is within the bar the project's own
  quantized releases are held to (mean KL at most 0.05, margin-qualified
  top-1 at least 97%). So Runner's gemma-4 MoE forward tracks the publisher
  on a faithful file.
- The shipped QAT file reads 0.2183. On the same chat corpus, the
  2026-10-21 investigation's plain quants of a conversion read 0.295 (Q4_0)
  and 0.2485 (Q4_K_M), on an earlier build.

## The files and the run

| item | value |
|---|---|
| reference | `google/gemma-4-26B-A4B-it` at revision `4d7ae4984b7d` (unchanged since 2026-07-20); transformers 5.16.1, torch 2.14.0, CPU; class `Gemma4ForConditionalGeneration` |
| A | llama.cpp `convert_hf_to_gguf.py` at `a2c942dcfe`, `--outtype bf16`: 50,505,137,152 bytes, sha256 `75b8ae12d7706cc8c07755b64dfcb5941eef3a9cbdfc7cafc2ccf1f8b37e1ca7` |
| B | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` at `d1c082be9cf3`, `gemma-4-26B_q4_0-it.gguf`, sha256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |
| Runner | main `536dddc0`, the default build |
| A's route | CPU, BF16 100%, KV f16 |
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
