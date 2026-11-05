# Mellum 2.1: four published Q5-band files against the float32 reference (2026-11-04)

**Unflattering first: the smallest passing file is a third party's, its plain top-1 is 93.0%, and its first 750
positions sit at the bar's edge.** bartowski's `Mellum2.1-12B-A2.5B-Thinking-Q5_K_S.gguf` (8,963,422,944 bytes) reads
a mean KLD of **0.0289** and a margin-qualified top-1 of **98.33%** against JetBrains' BF16 parent computed in
float32, inside the bar of 0.05 and 97%. It is the first Mellum 2.1 file under 9 GB measured to pass; the publisher's
Q4_K_M (8.07 GB) fails at 0.1172 / 92.89% in the same window, and Runner's own selective-precision attempt at 9.38 GB
failed at 0.0983 / 92.89%. The two Q4_K-expert files in the band miss. The engine cannot yet write the passing recipe
(experts at Q5_K, down projections at Q5_1, everything else at Q8_0): the rewrite has no q5_k type and the derive
planner has two rungs and no importance matrix (plan items R4.33 and R4.34).

## Setup

- **Reference:** `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at revision `92ddae9fc766`, transformers' own mellum code in
  float32 (`scripts/gold-logits.py`, transformers 5.16.1), on the Linux CUDA host's CPUs.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt` (sha256 `56e9fcba...`), positions 8 to
  4,493, token-id keyed, no special prefix, no dropped mass; 239 positions margin-qualified. The same window, corpus
  and reference as the 2026-11-03 derive study, so its parent, D1 and publisher Q4_K_M rows sit on this table.
- **Runner side:** `runner --score --score-top 20 --score-tokens 4510 -c 4510 --gpu 99` on the Linux CUDA host's MIG
  slice, Runner main `7567c948` (1.12.0), every layer on the device, no expert bank on the host.
- **Files** (sha256 checked against the Hub's LFS record as each landed, and over the held bytes; every header
  declares `general.license` apache-2.0, architecture mellum, 28 blocks, 64 experts, 8 used):

| Label | Repository, revision | File | Bytes | sha256 |
|---|---|---|---|---|
| N4 | `Ninnix96/Mellum2.1-12B-A2.5B-Thinking-gguf` `43bb23f9` | `Mellum2.1-12B-A2.5B-Thinking-Q4K-Q5_0.gguf` | 8,051,112,672 | `51882158...239d23` |
| B4 | `bartowski/Mellum2.1-12B-A2.5B-Thinking-GGUF` `355e9704` | `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf` | 8,819,358,432 | `7adfc5bc...7b8fea` |
| B5 | `bartowski/Mellum2.1-12B-A2.5B-Thinking-GGUF` `355e9704` | `Mellum2.1-12B-A2.5B-Thinking-Q5_K_S.gguf` | 8,963,422,944 | `9fe81c18e6628a918757a356614abb4590b5c415a8281272fae3298eafb678c1` |
| N5 | `Ninnix96/Mellum2.1-12B-A2.5B-Thinking-gguf` `43bb23f9` | `Mellum2.1-12B-A2.5B-Thinking-Q5K-Q5_0.gguf` | 8,975,956,704 | `38cd6b3f...172cbb` |

## Against the reference

| File | Bytes | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|---|
| N4 Q4K-Q5_0 | 8,051,112,672 | 0.0601 | 0.0192 | 1.579 | 88.67% | 95.82% | fail |
| B4 Q4_K_M (bartowski) | 8,819,358,432 | 0.0533 | 0.0176 | 2.102 | 92.33% | 97.07% | fail (KLD by 0.0033) |
| **B5 Q5_K_S (bartowski)** | **8,963,422,944** | **0.0289** | 0.0126 | 0.476 | 93.00% | **98.33%** | **pass** |
| N5 Q5K-Q5_0 | 8,975,956,704 | 0.0376 | 0.0150 | 1.400 | 90.67% | 97.49% | pass |
| Publisher Q4_K_M (2026-11-03) | 8,071,295,264 | 0.1172 | 0.0476 | 3.663 | 88.67% | 92.89% | fail |
| Runner derive D1 (2026-11-03) | 9,380,575,424 | 0.0983 | 0.0437 | 2.242 | 86.67% | 92.89% | fail |
| BF16 parent GGUF (2026-11-03) | 24,311,969,984 | 0.0025 | 0.0002 | 0.091 | 99.00% | 100.00% | |

By depth (mean KLD / margin-qualified top-1; 50, 83 and 167 positions):

| File | 8 to 749 | 750 to 1,999 | 2,000 to 4,493 |
|---|---|---|---|
| N4 | 0.0785 / 97.4% | 0.0984 / 90.8% | 0.0356 / 97.8% |
| B4 | 0.0987 / 97.4% | 0.0707 / 95.4% | 0.0310 / 97.8% |
| B5 | 0.0497 / 97.4% | 0.0282 / 100% | 0.0230 / 97.8% |
| N5 | 0.0756 / 97.4% | 0.0458 / 95.4% | 0.0222 / 98.5% |
| Publisher Q4_K_M | 0.1022 / 94.9% | 0.1666 / 89.2% | 0.0972 / 94.1% |

## What the bytes bought (per-tensor types read from each header)

| File | Attention | Expert gate / up | Expert down (896 wide) | `token_embd` / `output` | Routers |
|---|---|---|---|---|---|
| N4 | Q8_0 | Q4_K / Q4_K | Q5_0 | F16 / Q8_0 | F32 |
| B4 | Q8_0 | Q4_K (27) + Q6_K / Q4_K (21), Q6_K (5), Q5_K, Q8_0 | Q8_0 (14), Q5_0 (14) | Q8_0 / Q8_0 | F32 |
| B5 | q: Q8_0 (17), Q6_K (6), Q5_K (5); k, v, output Q8_0 | Q5_K / Q5_K | Q5_1 | Q8_0 / Q8_0 | F32 |
| N5 | Q8_0 | Q5_K / Q5_K | Q5_0 | F16 / Q8_0 | F32 |

Both passing files have every expert gate and up projection at Q5_K; both failing files have them at Q4_K. These are
readings from four files, not a controlled ablation (the files also differ in the importance matrix, the embedding type
and the down type).

## What this is not

Not a benchmark claim, not a speed row, not a claim about other text: one 300-position window over one corpus. The
8.96 GB file is not for an 8 GB machine. The Q8_0 (12.9 GB) remains the publisher's own passing file. The byte pins
of the Q5_K_S smoke cells, when recorded, are under `tests/compatibility/tier-smoke/`.

Since 2026-11-05 (R4.33) Runner writes this recipe itself: [mellum2.1-rewrite-q5-2026-11-05](../mellum2.1-rewrite-q5-2026-11-05/README.md).
