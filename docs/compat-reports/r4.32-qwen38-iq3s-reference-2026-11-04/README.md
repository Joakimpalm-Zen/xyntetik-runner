# R4.32: Qwen3.8 GSQ-RCO IQ3_S files against Qwen's float32 reference (drafts)

**Status: draft texts only. Nothing here is applied.** No Hub card and no
roster row changes until the lab's read of ISTA-DASLab's unmodified IQ3_S
is in and the owner has decided. The tier line in every roster draft is left
blank for the owner.

## What was found

The lab read the published Tier 1 file `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf`
(sha256 `873df48d...`) against `Qwen/Qwen3.8-27B` at revision `1d4bf0f2` in
float32, in the house window and at positions 4000-5999:

| | mean KLD | top-1 | margin-qualified top-1 | corpus perplexity |
|---|---:|---:|---:|---:|
| fresh BF16 GGUF of the parent | 0.00002 | 100% | 100% | 73.29 |
| the recovered IQ3_S | 0.41 | 77.7-80.4% | 85-86% | 21.54 |

The BF16 row shows no engine or reference gap. The recovered card claims
0.0450 / 97.80%, measured against the project's BF16 GGUF on 500 short
held-out Wikitext-2 prefixes with a 0.5-nat tie band (plain top-1 87.20%).
A faithful 3.4-bit quantization of this parent cannot predict the corpus
better than the parent does, which this file does (21.5 against 73.3).

## Provenance (settled)

Both files are quantizations of the public parents. Each parent has one
weight set on the Hub, and the files' unquantized tensors match it bit for
bit (norms equal to the parent's bf16 + 1; ssm, router and hyper-connection
tensors equal up to the converter's head reorder).

- 27B: `Qwen/Qwen3.8-27B@1d4bf0f2`. The 18 weight files are unchanged since
  the 2026-08-13 upload, and the later commits are card, license, tokenizer
  and template. The recovery changed only the fp16 `d`/`dmin` fields of
  399 tensors of ISTA's file `64b53b64...`. Its teacher was a BF16 GGUF
  converted from the same revision.
- Flash-Next: `Qwen/Qwen3.8-Flash-Next@de4b8e4d`, with one weight set across
  all three commits. The 4-part set carries ISTA's trunk bytes unchanged,
  and the head was built from `de4b8e4d`.

GSQ is a learned post-training quantization: codes and scales are fitted
jointly on calibration data (an imatrix of 1000 chunks of 4096 tokens).
ISTA reports Wikitext-2 perplexity 7.05 for BF16 and 7.07 for the IQ3_S.

## Which variant applies

The lab is reading ISTA's unmodified `64b53b64` file in both windows, with
its perplexity.

- **Variant A (GSQ-RCO):** the source file reads about as far from the
  reference as the recovered one. The divergence is the source
  quantization's. The recovered file stays, with the reference number first
  and the old anchor under it.
- **Variant B (recovery):** the source file reads close to the reference.
  The divergence is the scale recovery's. The recovered file is withdrawn
  from the roster and its card says why.

| Draft | Variant A | Variant B |
|---|---|---|
| 27B Hub card | [27b-card-A.md](27b-card-A.md) | [27b-card-B.md](27b-card-B.md) |
| 27B roster row | [27b-roster-A.md](27b-roster-A.md) | [27b-roster-B.md](27b-roster-B.md) |
| Flash-Next Hub card | [flash-next-card-A.md](flash-next-card-A.md) | [flash-next-card-B.md](flash-next-card-B.md) |

Values the lab's read will supply are marked `[lab: ...]`.
