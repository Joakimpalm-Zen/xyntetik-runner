# Qwen3-30B-A3B: an importance matrix on a mixed q3_k/q4_0 plan, against Qwen's float32 reference (2026-10-31)

**Unflattering first: the importance matrix does not move top-1.** On 2,000 positions, margin-qualified top-1 reads
98.55% with the imatrix and 98.29% without, a difference no larger than chance (McNemar 15 against 11 discordant
positions, exact p 0.56). What it moves is the distribution: mean KLD falls from **0.0424 to 0.0355**, and the paired
per-position test puts that beyond chance (Wilcoxon p 8e-10). Both files pass the fidelity bar.

## The files

Both mixes are written by `runner --rewrite` from Qwen's own Q8_0 with one type plan (`n11-plan.json`): the expert
tensors of blocks 37 to 47 at q3_k, every other expert tensor at q4_0, everything else kept at Q8_0. The only
difference is `--imatrix`.

| Input | Source | sha256 |
|---|---|---|
| Q8_0 | `Qwen/Qwen3-30B-A3B-GGUF` at e4d4bafd, `Qwen3-30B-A3B-Q8_0.gguf` | `4ad960d1…` |
| imatrix | `bartowski/Qwen_Qwen3-30B-A3B-GGUF` at 46f17e07, `Qwen_Qwen3-30B-A3B.imatrix` (legacy `.dat`, 384 entries) | `6aa5b886…` |
| plan | `n11-plan.json` (here) | `dc6f4d5d…` |
| mix without imatrix | 17.1 GB | `1c5677e5…` |
| mix with imatrix | 17.1 GB, 144 tensors weighted, every row calibrated | `340701c4…` |

## Against the reference

Reference: `Qwen/Qwen3-30B-A3B` at ad44e777, BF16 weights with float32 arithmetic, cast at use, on the CPU. Corpus
`tests/fixtures/gold-corpus-2k.txt`, 2,000 positions at stride 2, CPU scoring (`--score 20`, 8,192 context).

| File | Mean KLD | Median | Max KLD | Positions above 0.1 | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 (the source) | 0.0049 | 0.0013 | 0.31 | 3 | 97.85% | 99.93% |
| mix, no imatrix | 0.0424 | 0.0225 | 2.04 | 191 | 90.90% | 98.29% |
| mix, imatrix | **0.0355** | 0.0201 | 1.50 | 141 | 91.55% | 98.55% |

Paired, the imatrix file is closer at 1,121 of 2,000 positions. One position differs by more than a nat between the
two files. The records are `gold-logits-n11-vs-imx.json` and `gold-logits-q8.json`.

## What this says

Measured from the Q8_0 source's own 0.0049, the plan adds 0.0375 mean KLD without the imatrix and 0.0306 with it,
about a fifth less, and the positions above 0.1 fall from 191 to 141. Top-1 is where it was. The imatrix makes the
same plan closer to the reference; it does not change what a 3-bit expert costs.
