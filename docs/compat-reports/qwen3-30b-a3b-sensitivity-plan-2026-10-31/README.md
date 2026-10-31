# Qwen3-30B-A3B: a type plan by measurement against the hand-picked one (2026-10-31)

**Unflattering first: the measured plan does not move top-1.** Against Qwen's float32 reference over 2,000
positions, margin-qualified top-1 reads 99.01% with the measured plan and 98.55% with the hand-picked one, a
difference no larger than chance (McNemar 16 against 9 discordant positions, p 0.23). It also has 2 positions more
than a nat from the reference where the hand plan has 1. What it moves is the distribution: mean KLD falls from
**0.0356 to 0.0292**, 18% less at the same file size, and the paired test puts that far beyond chance (Wilcoxon
p 4e-37).

## The two plans

Both demote 11 blocks' expert tensors (`ffn_*_exps`) from q4_0 to q3_k, keep the rest at Qwen's Q8_0, and are
written with bartowski's imatrix, so both files are 17,106,040,320 bytes.

- **Hand-picked** (2026-08-17): the eleven deepest blocks, 37 to 47 (the R4.7.2 reading's file,
  `docs/compat-reports/qwen3-30b-a3b-imatrix-2026-10-31/`).
- **Measured**: `runner --quant-error report.jsonl --quant q4_0,q3_k --imatrix IMX` on the Q8_0 (338 tensors, 2,142
  CPU-seconds), then `scripts/type-plan-by-sensitivity.py report.jsonl --match _exps --high q4_0 --low q3_k
  --groups 11`. It demoted blocks 0 to 10 (`plan-sens.json`).

The proxy ranks blocks by the imatrix-weighted squared error the q3_k writer adds to their experts per byte it
saves. It put the hand plan's eleven blocks last: 37 to 47 are the eleven costliest of 48, and block 47 costs 258
times what block 0 does per saved byte (`report.jsonl`, one line per tensor and type).

## Against the reference

Reference: `Qwen/Qwen3-30B-A3B` at ad44e777, BF16 weights with float32 arithmetic, cast at use, on the CPU. Corpus
`tests/fixtures/gold-corpus-2k.txt`, 2,000 positions at stride 2, CPU scoring (`--score 20`, 8,192 context). The
measured plan's file was written by this branch at ef3a0a0d.

| Plan | Mean KLD | Median | Max KLD | Positions above 0.1 | Above 1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| hand-picked, blocks 37-47 | 0.0356 | 0.0201 | 1.50 | 141 | 1 | 98.55% |
| measured, blocks 0-10 | **0.0292** | 0.0137 | 1.20 | 112 | 2 | 99.01% |

Paired, the measured plan is closer at 1,271 of 2,000 positions, and no position differs by more than a nat between
the two files. The record is `gold-logits-n11-vs-sens.json`.

## What this says

On this model, at this size, the measured order beats the rule of thumb it replaces, and the rule of thumb was not
just suboptimal but the worst choice the proxy could name. The proxy is a ranking, not a fidelity claim: a plan it
writes is measured against the reference before it ships, as this one was.
