# Metal decode profile: Qwen3-4B Q4_K_M against mlx-lm (2026-11-04)

Profile only: no code change. One decode step of Qwen3-4B on the 8 GB laptop
(Apple M1, macOS 27.0.1), split into buckets, to find where Runner's 11.3
tok/s stands against mlx-lm's 21.7 (the R3.6.2 rows in `docs/benchmarks.md`).

The kill number was committed before the read: the largest bucket a change
could address must be at least 15% of the step. **It is: the Q4_K matvecs
are 55% of the step**, at 37.6 GB/s against mlx-lm's 50 to 54 GB/s at the
same 4.5 bits a weight.

## Limits first

- **The buckets come from dropping one bucket at a time.** A profile-only
  lever (`profile-levers.diff`, never committed) skips that bucket's
  dispatches and the step is timed again; the bucket's time is the
  difference. Output under the lever is garbage, so it times and nothing
  else. The buckets add to 86.24 ms against the 85.40 ms measured, within 1%.
- **mlx-lm is profiled more coarsely.** Its matvecs are timed in isolation
  (`mlx_matvec.py`: `mx.quantized_matmul` at Qwen3-4B's shapes, 36 distinct
  weights a shape and one evaluation, the median of 20). Its attention and
  small kernels are the step's remainder (the step at 21.97 tok/s, minus the
  matvecs), an upper bound that includes its host time.
- **This compares two formats as well as two engines.** Runner's Q4_K_M keeps
  the LM head, attn_v and half of ffn_down at Q6_K (6.56 bits a weight),
  2,491 MB a token. The mlx-community file is 4-bit everywhere (4.5 bits with
  its bf16 scale and bias for each 64 weights).
- One machine, one model, about 600 tokens of context (the 512-token prompt
  plus the decode).

## Runner: one decode step

Runner `348c3420` (the R3.28.2 branch, #591), Metal, `--mlock`, the 512-token
prompt of the R3.6.2 rows, 128 decode tokens. GPU time is the median of
`RUNNER_METAL_TIMING=1` spans over steps 5 to 128, three interleaved rounds
(`runner-profile.log`, `runner_profile.py`), each under the swap guard.

| bucket | ms a step | share | how the bucket was read |
|---|---:|---:|---|
| Q4_K matvecs (`k_mvg_q4_K`; 1,765 MB) | 46.94 | 55.0% | all minus no-Q4_K |
| Q6_K matvecs (attn_v, half of ffn_down) | 16.13 | 18.9% | all minus no-Q6_K |
| LM head (Q6_K, 319 MB) | 11.85 | 13.9% | all minus no-head |
| attention, about 600 tokens of history | 10.43 | 12.2% | all minus no-attention |
| norms, rope, element-wise and the gaps between dispatches | 0.89 | 1.0% | everything else dropped |
| **the step** | **85.40** | | 85.82, 85.40, 85.36 |

The step is 507 dispatches (Metal's census, `RUNNER_METAL_STATS=1`): 253
matvecs, 145 norms, 36 rope, 36 attention, 37 element-wise. The CPU encodes a
step in 0.72 ms. Dispatch and synchronisation overhead is inside the 0.89 ms
bucket, which still holds 254 dispatches, so it is not a lever here.

## Against mlx-lm, bucket by bucket

| bucket | Runner ms | mlx-lm ms | Runner's bandwidth | mlx-lm's bandwidth |
|---|---:|---:|---|---|
| projection matvecs | 63.07 | 40.26 | Q4_K 37.6 GB/s; Q6_K about 26 GB/s | 49 to 54 GB/s (q, o, gate, up, down), 33 to 35 (k, v) |
| LM head | 11.85 | 4.19 | 26.9 GB/s over 319 MB (Q6_K) | 52.2 GB/s over 219 MB (4-bit) |
| attention and the rest | 11.32 | about 1.1 | attention reads about 88 MB of KV at about 8.5 GB/s | |
| **the step** | **85.40** | **about 45.5** | | |

## What a change could address, largest first

1. **The Q4_K matvecs, 55% of the step.** 37.6 GB/s against mlx-lm's 50 to 54
   at the same bits a weight. Reaching 50 GB/s would save about 12 ms a step.
2. **The Q6_K matvecs and the head, 32.8% together.** About 26 GB/s. Part of
   this is the format (Q6_K's 6.56 bits against 4.5), part the kernel: Q6_K
   has no narrow-slice kernel like R3.28.2's.
3. **Attention, 12.2%, under the 15% bar by itself.** The decode attention
   reads its KV at about 8.5 GB/s at this context, while mlx-lm's attention
   and everything else take about 1.1 ms.

Dispatch overhead and the small kernels (1%) are not a lever on this model.
