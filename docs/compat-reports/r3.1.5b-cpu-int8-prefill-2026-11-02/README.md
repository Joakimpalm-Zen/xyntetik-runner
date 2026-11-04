# R3.1.5 B: CPU prefill in int8 (2026-11-02)

**Unflattering first: the int8 route is measurably farther from the publisher's model than the f32 route, and it
stays opt-in.** On Qwen3-0.6B Q8_0 against Qwen's float32 model over 2,000 positions, its mean KLD is 0.00747 against
the f32 route's 0.00610 (1.22x), which the paired Wilcoxon test separates (p 5e-86). Margin-qualified top-1 is 100% on
both. It reaches 46% of llama.cpp's own int8 build on the same file, short of the plan's 50% for the Apple M1. There is no
x86 kernel yet.

## What the route does

`RUNNER_CPU_I8_PREFILL=1` sends CPU batches of 8 or more rows (prompt prefill, chunked scoring) to int8 tiles. The
activations are quantized once a matvec, as the decode route `RUNNER_CPU_I8` does (16-element blocks, one scale and a
quant sum each). Four weight rows are repacked once into int8 codes with a scale and an offset per 16 elements and row,
the form Q8_0, Q4_0, Q4_K and Q6_K all reduce to, with the four rows' codes interleaved four bytes at a time. `SDOT` by
lane then computes four rows against one activation block in four instructions, four columns a pass. The NEON kernel
and a scalar twin perform the same fused operations in the same order; `test_quants_simd` holds them equal bit for bit
and both to the same-quantization double-precision dot. Decode (one row a forward) is untouched.

## Speed

Apple M1, 8 GB, `runner --bench-json --gpu off -t 4` (512-token prompt), routes alternated, warm rounds:

| File | f32 route, tok/s | int8 tiles, tok/s | Gain |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | 187 | 315 | +68% |
| Qwen3-4B Q4_K_M | 29.7 | 44.4 | +49% |

The design note (#536) read llama.cpp c1d0e7a without BLAS at 690 tok/s on the Qwen3-0.6B Q8_0 on this machine, so the
route takes Runner from 27% to 46% of it. A profile of a long prompt puts the rest in attention and in the pack (about
a fifth of the tile time). Eight columns a pass measured within noise of four.

## Against the publisher's float32 model (D1)

`runner --score --score-top 20` with `RUNNER_SCORE_CHUNKED=1` (16-row batches, so every position comes from the
prefill route) over the first 4,100 tokens of `tests/fixtures/gold-corpus-2k.txt` on the Apple M1 CPU, read against
`Qwen/Qwen3-0.6B` at `c1899de2` in float32 (`scripts/gold-logits.py`), 2,000 positions at stride 2 from position 8,
every position matched by token id. The file is `Qwen3-0.6B-Q8_0.gguf` (sha256 `9465e63a...`).

| Route | Mean KLD | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|
| f32 | 0.00610 | 0.076 | 97.65% | 100% |
| int8 tiles | 0.00747 (1.22x) | 0.096 | 97.0% | 100% |

- Wilcoxon on the per-position KLD: z 19.7, p 5e-86, the f32 route closer. McNemar on the 1,402 margin-qualified
  positions: no discordant position. No position past 1 nat on either side.
- 38 of the 2,000 scored positions change their top token between the routes. Every one is a near-tie: the float32
  model's own margin there is at most 0.255 nats, and none is margin-qualified (above 0.5). The float32 model picks
  the f32 route's token at 25 of them and the int8 route's at 12.
- Over all 4,099 teacher-forced positions the routes differ in top-1 at 70, all within 0.166 nats of the f32 route's
  own runner-up (median 0.030).

`gold-logits-q8_0-f32-vs-int8-prefill.json` holds every row.

## Not in this change

- An x86 kernel (AVX2 / AVX-512): the flag is ignored on those builds, with a notice. That is the follow-up, measured
  on x86 hosts against the design note's 13% x86 reading.
- D1 rows for a K-quant file; the Q4_K_M row above is speed only.
- Table lookup for 2-3 bit types (R3.1.5 C).
