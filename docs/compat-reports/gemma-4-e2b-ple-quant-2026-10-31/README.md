# Gemma 4 E2B: the per-layer embedding table under quantization, against Google's float32 reference (2026-10-31)

**Unflattering first: a uniform 4-bit E2B fails the fidelity bar by a wide margin, whatever happens to the
per-layer embedding table.** Runner's own uniform Q4_K of E2B reads a mean KLD of **0.437** against a bar of 0.05
(81.1% margin-qualified top-1 against 97%), and unsloth's published Q4_K_M reads 0.173. The per-layer embedding
table itself is not where that error comes from: quantizing it to 4 bits on top of a Q8_0 file roughly doubles a
small KLD (0.0046 to 0.0088) and leaves top-1 where it was.

## The rule

Runner's rewriter quantizes `per_layer_token_embd.weight` like any other 2-D weight: to the `--quant` target, or
to whatever a `--type-plan` rule names for it. It has no special case. That table is 262144 x 8960 on E2B, 2.35
billion parameters, about half of the checkpoint (the BF16 file is 9.31 GB), so its type sets much of the file's size: at Q8_0 the E2B
file is 4.95 GB, and with only this table at Q4_K it is 3.78 GB. Measured here, the size saving costs 0.004 to
0.005 mean KLD and no top-1, so the rule stays as it is. A plan that wants the table at Q8_0 can name it:
`{"match": "per_layer_token_embd", "type": "q8_0"}`.

## The files

Every Runner file is written by `runner --rewrite` (main at d06249f8) from ggml-org's BF16 GGUF with one of the
type plans in `type-plans.json`, which differ only in the per-layer embedding table or in everything else.

| File | Everything else | Per-layer embedding | Size | sha256 |
|---|---|---|---|---|
| p8-r8 | Q8_0 | Q8_0 | 4.95 GB | `e7c230ff…` |
| p4k-r8 | Q8_0 | Q4_K | 3.78 GB | `78f6bc36…` |
| p40-r8 | Q8_0 | Q4_0 | 3.78 GB | `448f17ee…` |
| p8-r4k | Q4_K | Q8_0 | 3.81 GB | `c562ec3b…` |
| p4k-r4k | Q4_K | Q4_K | 2.63 GB | `338a70f8…` |
| unsloth Q4_K_M | unsloth's mix | Q5_K | 3.11 GB | `740185b2…` |
| unsloth Q5_K_M | unsloth's mix | Q5_K | 3.36 GB | `90293b8c…` |

Sources: `ggml-org/gemma-4-E2B-it-GGUF` at 6938c58e (`gemma-4-E2B-it-BF16.gguf`, `246c5fb1…`) and
`unsloth/gemma-4-E2B-it-GGUF` at 0314792d. Both unsloth files keep the per-layer embedding table at Q5_K, so
their pair says nothing about that table; it is here as the published reference point.

## Against the reference

Reference: `google/gemma-4-E2B-it` at 3e22461f, BF16 weights with float32 arithmetic, cast at use, on the CPU,
with BOS prepended as the GGUF declares (`--force-bos`). Corpus `tests/fixtures/gold-corpus-2k.txt`, 2,000
positions at stride 2, CPU scoring (`--score 20`, 16,384 context), 1,643 margin-qualified positions.

| File | Mean KLD | Median | Max KLD | Positions above 0.1 | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| p8-r8 | **0.0046** | 0.0012 | 0.52 | 5 | 97.40% | 99.82% |
| p4k-r8 | **0.0088** | 0.0021 | 1.21 | 17 | 96.00% | 99.57% |
| p40-r8 | **0.0094** | 0.0028 | 0.89 | 22 | 96.00% | 99.88% |
| p8-r4k | 0.4374 | 0.1811 | 7.59 | 1,185 | 74.40% | 81.13% |
| p4k-r4k | 0.4337 | 0.1767 | 6.40 | 1,175 | 74.65% | 81.19% |
| unsloth Q4_K_M | 0.1728 | 0.0592 | 5.39 | 794 | 83.85% | 90.63% |
| unsloth Q5_K_M | 0.0539 | 0.0193 | 3.03 | 253 | 90.80% | 96.84% |

Paired, on the same positions:

| Pair | Wilcoxon on per-position KLD | McNemar on margin-qualified top-1 |
|---|---|---|
| table at Q8_0 vs Q4_K, rest Q8_0 | Q8_0 closer, p 5e-118 | 4 against 0, p 0.125 |
| table at Q8_0 vs Q4_0, rest Q8_0 | Q8_0 closer, p 8e-148 | 0 against 1, p 1.0 |
| table at Q8_0 vs Q4_K, rest Q4_K | not distinguishable, p 0.67 | 11 against 12, p 1.0 |
| unsloth Q4_K_M vs Q5_K_M | Q5_K_M closer, p 5e-179 | 39 against 141, p 9e-15 |

## What this says

- The per-layer embedding table tolerates 4 bits. Alone, it moves the mean KLD by 0.004 to 0.005 and no
  margin-qualified top-1, while cutting nearly a quarter of the Q8_0 file.
- In a 4-bit file it does not matter at all: with everything else at Q4_K, the table at Q8_0 or Q4_K reads the
  same (p 0.67). The rest of the model's 4-bit error is a hundred times larger.
- A 4-bit E2B fails the bar either way: Runner's uniform Q4_K at 0.437, unsloth's Q4_K_M at 0.173. Of the
  files measured, only a Q8_0 body passes; unsloth's Q5_K_M misses narrowly (0.0539, 96.84%).

Runner's `q4_k` writer is not the cause of the gap to unsloth's Q4_K_M: on a real BF16 tensor it writes the same
bytes as ggml's `quantize_q4_K` in 3,422 of 3,456 super-blocks, and the other 34 differ by one step in a scale.
The gap is the recipe: a uniform Q4_K against unsloth's mix of types and calibration.

The records are the four `gold-logits-*.json` files beside this one.
