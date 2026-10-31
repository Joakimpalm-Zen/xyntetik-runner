# The CPU K-quant one-accumulator change on the batched route (2026-10-31)

3a8499fe (in 1.5.0) changed the Q4_K, Q5_K and Q6_K dots to keep one accumulator per row: the single-column and
group-sum forms (AVX2 and NEON) and AVX2's multi-column form. Its D1 reading used solo `--score`, which runs only the
single-column dots; the multi-column form, which the batched prefill route runs, had no reading. This is that row.

Qwen3-4B Q4_K_M (`Qwen/Qwen3-4B-GGUF` at bc640142, sha256 `7485fe6f…`; 71% Q4_K, 29% Q6_K) scored with
`RUNNER_SCORE_CHUNKED=1` on an x86-64 CPU (16 threads, CUDA hidden) at 3a8499fe's parent (7d5f0435), at 3a8499fe and at
main (406fb163). The route line reads `path=keep rows=16`, the batched route. The host has AVX-512, and `src/quants.c`
undefines the AVX-512 macros, so the AVX2 multi-column forms are the ones that ran. Read against `Qwen/Qwen3-4B` at
1cfa9a72 in float32 (BF16 weights cast at use), 2,000 positions at stride 2.

| build | mean KLD | margin-qualified top-1 |
|---|---|---|
| 3a8499fe^ | 0.131241 | 91.95% |
| 3a8499fe | 0.131241 | 91.95% |

The two score files differ, so the change is live on this route, and the readings are level: mean KL equal to six
digits, Wilcoxon p 0.65, McNemar no discordant position, no position more than a nat apart. main's score file is
byte-identical to 3a8499fe's: nothing since has changed this route for this file. The records are
`gold-logits-pre-vs-commit.json` and `gold-logits-pre-vs-main.json`.
