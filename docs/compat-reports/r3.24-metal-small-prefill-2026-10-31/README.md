# Metal prompt chunks of 4 to 16 rows on the GEMM tiles (R3.24 (b)), the D1 rows, 2026-10-31

The change moves Metal forwards of 4 to 16 rows (a prompt's last chunk, a chunked score) from the multi-column
identity matvec to the GEMM tiles, which wider forwards already took. The new route is within tolerance, not
bit-identical, so the owner's D1 rule applies.

**Setup**
- Side a: main at eabb40bb, the matvec route. Side b: this branch, the tiles.
- Score files from the Metal test host: `RUNNER_SCORE_CHUNKED=1 runner --score 20 -c 8192 --gpu auto` over
  `tests/fixtures/gold-corpus-2k.txt`, at keep-chunk widths 16 (the default) and 8 (`RUNNER_SPEC_BATCH=8`, between
  the two crossovers).
- Every file's route record reads `keep` with `rows_max` 16 or 8, on both sides.
- Reference: transformers float32 on the CPU, 2,000 positions.
- Models:
  - Qwen3.5-0.8B Q8_0: ggml-org/Qwen3.5-0.8B-GGUF @807dfd9a, against Qwen/Qwen3.5-0.8B @2fc06364715b.
  - granite-4.0-h-micro Q4_K_M: ibm-granite/granite-4.0-h-micro-GGUF @dc1dd258, against
    ibm-granite/granite-4.0-h-micro @d5f01a3ea75f.

| model, width | mean KL a (matvec) | mean KL b (tiles) | b / a | Wilcoxon | McNemar (a only : b only, of qualified) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B, 16 | 0.0032237 | 0.0032221 | 0.9995 | z -5.4, b closer | 0 : 0 of 1,411 |
| Qwen3.5-0.8B, 8 | 0.0032237 | 0.0032221 | 0.9995 | z -5.4, b closer | 0 : 0 of 1,411 |
| granite-h-micro, 16 | 0.0917606 | 0.0918362 | 1.0008 | p 0.62 | 1 : 0 of 1,433 |
| granite-h-micro, 8 | 0.0917606 | 0.0918362 | 1.0008 | p 0.62 | 1 : 0 of 1,433 |

All four rows pass D1 (mean KL within 1.01x; the paired tests are not worse at p < 0.05).

**Why the widths agree**
- Side a's scores are identical at widths 8 and 16 on both models: the matvec gives each row a solo step's bits.
- Side b's are identical on granite. On Qwen they differ in the last bits and read the same against gold to 7
  digits.

**Speed.** GPU ms per forward:

| model | rows | matvec | tiles |
|---|---|---|---|
| Qwen3.5-0.8B Q8_0 | 4 | 43 | 57 |
| Qwen3.5-0.8B Q8_0 | 6 | 61 | 58 |
| Qwen3.5-0.8B Q8_0 | 16 | 148 | 63 |
| granite-h-micro Q4_K_M | 3 | 251 | 265 |
| granite-h-micro Q4_K_M | 4 | 335 | 267 |
| granite-h-micro Q4_K_M | 16 | 1,332 | 282 |

So 2 and 3 rows keep the matvec. A chunked score of the 2,000-position corpus on granite takes 225 s at width 16
(741 s on main) and 356 s at width 8 (749 s).
