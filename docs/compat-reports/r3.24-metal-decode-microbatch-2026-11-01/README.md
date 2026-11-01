# R3.24: the Metal decode microbatch by stage (2026-11-01)

**The weight products, not the weights, scale with the slots.** When N server slots decode together on Metal, a step
costs about 14 ms more for each added slot on K2-Horizon-1B Q8_0 (40 ms at one slot, 95 at four, 151 at eight, under
the profile). The QKV, O, FFN and head products are 89% of the eight-slot step, and they grow almost linearly in N,
although every slot reads the same weights. Attention, which is per slot by design, is about 10%.

The cause is in the matvec kernels. `MV_HEAD` loops over a threadgroup's columns outside the block loop, so each
column re-loads every block's scale and quants and converts them to float again. The weights come from cache after
the first column, but the dequantize-and-dot is repeated in full per column. The lever is a multi-column kernel that
loads and converts each block once and accumulates into every column. Kept in the same order per column, it stays
bit-identical to the one-column kernel. That is the second half of R3.24.

## Measurement

`RUNNER_METAL_TIMING=2` (this change) commits and waits at each stage boundary of the microbatch, so each stage's time
is its own command buffer's GPU span. The syncs add a fixed cost per stage, so absolute times are higher than in
service (one slot reads 40 ms here against 23 ms on the unprofiled solo path). The growth with N is the measurement.
The profile routes a lone slot through the microbatch encoder too, so the three widths share their boundaries.

Server: `--parallel 8 -c 4096`, N concurrent `/v1/completions` requests of 96 tokens each (`ignore_eos`), median
of the steady steps. Apple M1, 8 GB, at this branch's head. `micro.py` is the script.

| K2-Horizon-1B, GPU ms a step | N | total | qkv | attn | o | gate_up | down | head |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 1 | 40.3 | 5.2 | 3.5 | 3.9 | 16.4 | 6.9 | 3.0 |
| Q8_0 | 4 | 95.1 | 11.6 | 7.9 | 7.5 | 37.9 | 17.1 | 9.6 |
| Q8_0 | 8 | 151.5 | 18.8 | 15.0 | 11.9 | 59.4 | 28.5 | 16.1 |
| Q4_K_M | 1 | 50.2 | 7.1 | 3.3 | 3.9 | 15.7 | 9.8 | 7.1 |
| Q4_K_M | 4 | 104.3 | 12.0 | 7.8 | 5.7 | 32.6 | 28.2 | 17.4 |
| Q4_K_M | 8 | 202.0 | 22.3 | 15.1 | 11.4 | 63.2 | 56.3 | 33.6 |

- **Per slot added (Q8_0, N 4 to 8):** gate_up 5.4 ms, down 2.8, qkv 1.8, head 1.6, attn 1.8, o 1.1.
- **Q4_K_M's down and head are fully linear** (9.8, 28.2, 56.3; 7.1, 17.4, 33.6). The head runs one `enc_mv` per slot,
  so nothing is shared even through the cache.
- **Throughput without the profile**, the same requests: 12.4 tok/s at one slot, 29.4 at four, 36.4 at eight on the
  Q8_0.

## The column-inner twins (R3.24 (a), same day)

Each block's quants and scales are now loaded once per lane step for the whole column tile (`k_mvc_q8_0`,
`k_mvc_q6_K`, `k_mvfc_q4_K`), and the head goes through the same multi-column path. Every column keeps its one-column
kernel's arithmetic: each twin is admitted at load only if it gives the one-column kernel's bits on random blocks at
two shapes. The first Q6_K twin, written as `k_mv_q6_K`'s text, failed that test and was rewritten with explicit
`fma`, as `k_mv2_q6_K` is. `test-batch-identity` on both real files: byte-identical, 11 sequences x 12 steps, with the
microbatch engaged.

| K2-Horizon-1B, GPU ms a step (profile) | N | twins off | twins on |
|---|---|---|---|
| Q4_K_M | 4 | 99.5 | 86.8 |
| Q4_K_M | 8 | 198.5 | 154.8 |
| Q8_0 | 4 | 99.1 | 90.4 |
| Q8_0 | 8 | 139.8 | 115.9 |

The Q4_K_M's down (54.4 to 32.2 ms) and head (35.0 to 29.1) were the stages the earlier Q4_K-only twin could not
reach. What remains per column is the multiply-add itself, on scalar lanes: the opt-in tiled route (owner decision,
2026-11-01) is the next step.
