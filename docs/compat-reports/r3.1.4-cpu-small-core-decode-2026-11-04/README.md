# R3.1.4: CPU decode on a small AVX2 core (2026-11-04)

**Unflattering first.** _(Written when the D1 reads land.)_

## The host and the files

The Windows CUDA host's CPU, the owner's small-laptop case: an Intel i7-7700K (4 cores, 8 threads, AVX2 with
FMA and F16C, no AVX-512, no VNNI), 16 GB, `--gpu off` with the GPU hidden. Four files:

| File | Source | sha256 |
|---|---|---|
| Qwen3-0.6B Q4_K_M | bartowski/Qwen_Qwen3-0.6B-GGUF at `60b85c0e` | `9acfc1e001311f34...` |
| Qwen3-1.7B Q4_K_M | bartowski/Qwen_Qwen3-1.7B-GGUF at `dcb19155` | `72c5c3cb38fa32d5...` |
| Qwen3-0.6B Q8_0 | Qwen/Qwen3-0.6B-GGUF at `23749fef` | `9465e63a22add535...` |
| Qwen3-1.7B Q8_0 | Qwen/Qwen3-1.7B-GGUF at `90862c4b` | `061b54daade076b5...` |

## Kill numbers (stated before the first measurement, agreed with the coordinator)

1. Profile first. The step is split into buckets: the matvec dot workers by weight type, the serial activation
   quantization, attention, the other serial work, and the thread pool (the wall time a run takes beyond its
   slowest thread, plus the gap between the slowest and the mean thread). If the largest bucket a change could
   address is under 15% of the step, the item stops.
2. The first change must read at least 1.15x end to end on one of the four files, against its own D1 row (an
   int8 activation route is tolerance class, so its D1 row is measured before the end-to-end number is claimed).

## Step 0: the baseline

Runner main 78db276c against llama.cpp b11748 on the CPU (`llama-bench -p 0 -n 128 -ngl 0`, CUDA hidden):
decode, 128 tokens after a short prompt, `-t 4` and `-t 8`, a warm-up pair, then three rounds with the arm
order alternated. Medians:

| File | `-t` | runner tok/s | llama.cpp tok/s | runner / llama.cpp |
|---|---|---|---|---|
| 0.6B Q4_K_M | 4 | 39.5 | 69.2 | 0.57 |
| | 8 | 40.8 | 58.8 | 0.69 |
| 0.6B Q8_0 | 4 | 45.7 | 47.0 | 0.97 |
| | 8 | 45.3 | 48.1 | 0.94 |
| 1.7B Q4_K_M | 4 | 16.6 | 27.6 | 0.60 |
| | 8 | 23.1 | 30.1 | 0.77 |
| 1.7B Q8_0 | 4 | 17.7 | 18.3 | 0.97 |
| | 8 | 18.6 | 19.4 | 0.96 |

The gap is the Q4_K_M files'. Q8_0 decode reads the weights at 28 to 30 GB/s on both engines, near this host's
memory ceiling. The runner's Q4_K_M step reads about 19 GB/s against llama.cpp's 33, so the K-quant dots are
compute-limited where llama.cpp's are not. The 0.6B rounds spread widely (up to 33% on one arm); the conclusion
does not rest on them alone.

## The profile (kill 1: pass)

`RUNNER_CPU_PROF=1` (this change) brackets every one-token forward. Per pool run it times the calling thread's
wall and each thread's busy time inside the work function, keyed by the function and the weight types of the
matvec; the serial regions are timed by name. Shares of the step, 128 decode tokens:

| File | `-t` | Q4_K dot | QKV (Q4_K + Q6_K) | Q6_K LM head | Q6_K other | Q8_0 dot | attention | other serial | pool (handoff / imbalance) |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B Q4_K_M | 4 | 39% | 7% | 22% | 6% | | 7% | 4% | 14% (1 / 13) |
| | 8 | 39% | 7% | 23% | 6% | | 7% | 6% | 11% (6 / 5) |
| 1.7B Q4_K_M | 4 | 50% | 6% | 18% | 10% | | 3% | 2% | 11% (1 / 11) |
| | 8 | 51% | 6% | 18% | 10% | | 3% | 4% | 9% (4 / 5) |
| 0.6B Q8_0 | 4 | | | | | 60% (+21% head) | 7% | 4% | 7% (0 / 7) |
| | 8 | | | | | 58% (+21% head) | 6% | 5% | 10% (6 / 4) |
| 1.7B Q8_0 | 4 | | | | | 73% (+16% head) | 3% | 2% | 5% (0 / 5) |
| | 8 | | | | | 70% (+16% head) | 2% | 3% | 9% (4 / 5) |

The largest bucket is the Q4_K dot at 39 to 51% of the step. The Q6_K dots add 25 to 33%, most of it the tied
LM head (151,936 rows of 1,024 on 0.6B). Attention is under 8%, and the pool's handoff stays under 1% at four
threads (4 to 6% at eight, the SMT siblings). The existing int8 route (`RUNNER_CPU_I8=1`) moved the Q4_K
step little: 1.00x on 0.6B and 1.11x on 1.7B, one profiled run each. It quantizes activations into 16-element
blocks with a float scale each, so every 32 weights pay an int-to-float conversion and a multiply, and the
scales and mins unpack in scalar code per 64 weights.

## The change: a K-quant int8 decode dot (`RUNNER_CPU_I8K=1`, opt-in)

One-column matvecs on Q4_K and Q6_K rows quantize the activation column once, into 256-element blocks: one float
scale, 256 int8 values, and the sum of every 16 (llama.cpp's q8_K layout). Every row then dots against it in
integer arithmetic.

**The shape is taken from llama.cpp b11748's `ggml_vec_dot_q4_K_q8_K` and `ggml_vec_dot_q6_K_q8_K` (AVX2),**
read before the kernels were written:
- Q4_K: the twelve scale bytes unpack once per superblock into eight 6-bit scales and eight 6-bit mins, widened
  to int16 lanes. Each 64 weights take one 32-byte nibble load. The low and high nibbles go through `maddubs`
  against the activations (u8 x s8, pairs to int16), then through `madd` against their scale broadcast
  (pairs to int32). The mins come off the activation sums: the 16-element sums pair into 32-element sums, and
  one `madd` against the mins gives the correction.
- Q6_K: the two high bits OR into the low nibbles (four 32-weight groups from two 32-byte loads and one of high
  bits), then `maddubs` and `madd` against the sign-extended int8 scales. The weights' -32 offset is folded in
  once a superblock as 32 times the scales dotted with the activation sums.
- Everything through a superblock accumulates in int32, with no overflow at the formats' ranges: at most about
  31M on Q4_K and 260M on Q6_K, both under 2^31.

**One difference: the exact-reduction twin.** llama.cpp keeps eight float lanes across superblocks and reduces
them at the end. Here each superblock's int32 vector is reduced exactly to one integer, which takes one float
multiply-add with the two scales, in block order. The AVX2 kernel and a portable scalar twin
(`vec_dot_i8k_scalar`) therefore give the same bytes. The activation rounding is the runner's existing int8
rule (one rounding, half away from zero, `trunc(fma(x, 1/d, copysign(0.5, x)))`), clamped to +-127 by the
scale.

**Where it applies.**
- Only one-column matvecs (decode), and only Q4_K and Q6_K rows whose length is a whole number of 256-blocks.
- Fused projections that share an input share one quantized column.
- A verify round's columns, batches and prefill keep their routes. With the flag on, speculative decoding's
  rows are therefore not the solo step's bits; that is one reason the route is opt-in.

**The record.** The route record's `cpu_act` gains `+int8k-decode`, and `score_route` now carries `cpu_act`, so
a score file says which route made it.

**Gates.**
- `test-quants-simd`, for Q4_K and Q6_K at 256 and 4,096 elements, random and filled activations, a zero block
  and an outsized block:
  - the quantizer's 16-element sums are exact;
  - the AVX2 kernel equals its scalar twin byte for byte;
  - the dot is within 5e-5 of a double reference on the same quantized activations;
  - the dot is within half a quantization step per element of the true dot.
- Greedy text on Qwen3-0.6B Q4_K_M is the same as the float route's for the first 40 tokens of a short prompt.
  It is not identical in general (below).

## Against the publisher's float32 model (D1)

_(The lab's reads, from the score files at runner fc790a29: solo, 8,007 positions each, route off and on. Mean
NLL off / on: 0.6B 3.57381 / 3.58642, 1.7B 3.18179 / 3.18415.)_

## Speed

_(Interleaved A/B, the float route, this route and llama.cpp, Q4_K_M at `-t 4` and `-t 8`: held until the D1
reads land.)_
