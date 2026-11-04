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
