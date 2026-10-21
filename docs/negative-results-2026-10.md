# Negative results of the 2026-10 performance work

The work that shipped in 1.5.0 is in the [CHANGELOG](../CHANGELOG.md) and the
[benchmark report](benchmarks.md). This page keeps what did not ship, with
the bar each idea had to clear and the number that stopped it, so that
nobody spends the same week again. Every numeric bar was written into the
plan before the measurement; the ideas found along the way had to beat the
code they would replace, with the same output bytes. Hosts: an RTX 3070
8 GB, a datacenter MIG 1g.24gb slice of an RTX PRO 6000 (compute capability
12.0), an Apple M1 8 GB and an x86 desktop CPU. "A/A" is the same build
measured against itself in the same session, the noise floor a result has
to clear.

## Rejected for fidelity

A route that adds in another order than the one it replaces becomes a
default only if, over 2,000 positions against the publisher's float32
model, its mean KL is within 1.01x of the current route's and neither a
paired Wilcoxon test on per-position KL nor McNemar's test on
margin-qualified top-1 finds it worse at p < 0.05.

- **Grouped expert prefill as gemma-4's MoE default.** The prompt's tokens
  grouped by expert and each expert's multiplied on the tensor cores, as
  Qwen3 MoE does by default. On gemma-4-26B-A4B-it (a Q4_0 quant of the
  publisher's checkpoint) it read 2.22047 against 2.22727 nats on raw
  corpus text, which passes, but the instruction-tuned model reads about 2.2
  nats from its own reference on raw text, where a row cannot separate two
  routes. On chat-formatted text (the model's own replies, top-1 agreement
  about 88%) it read 0.30527 against 0.29527: 1.034x, over the bound
  (Wilcoxon p 0.54, McNemar p 0.48). It stays opt-in
  (`RUNNER_MOE_GROUPED=1` or `--gpu-route moe-grouped`); the prefill it
  would have bought is 4.9x at 512 tokens and 2.4x at 8K.
- **Int8 activations as a CUDA decode default (`RUNNER_CUDA_Q8X`).** IQ3_S
  decoded 22% faster, but read 1.0009x the KL with Wilcoxon p 0.041 in the
  float route's favour; Q8_0 and Q4_K were slower (-1.8% and -3.1 to -6.6%).
  It stays opt-in.
- **Int8 activations on the CPU (`RUNNER_CPU_I8`).** Qwen2.5-1.5B Q4_K_M read
  0.4% more KL with Wilcoxon p 0.0005 against it, and Q8_0 67% more. Opt-in;
  the int8 prefill and repacking ideas built on it were dropped at their
  pre-check.
- **An int8 Q4_K prompt GEMM.** One top-1 flip in 64 on Llama-3.2-3B and a
  free-running divergence on Qwen2.5-7B, and 5 to 15% slower than the
  pipelined fp16 GEMM on compute capability 8.6. Dropped.
- **Tensor-core prefill attention with only Q split into two fp16 halves.**
  Mistral-7B v0.3 read 0.02070 against 0.02069 with Wilcoxon p 0.0088
  (worse), and the split cost 14 to 28% of the kernel's speed. Splitting
  the softmax weights as well passed and shipped.

## Measured, and the gain was not there

- **Pinned host memory for the logits copy:** +0.3 to +0.5% decode against
  a 3% bar (A/A -0.4 to 0.0%). Removed.
- **Two rows per warp in the dense decode GEMVs:** +1.7, +1.7 and -3.2%
  against a 5% bar. Removed. (Several rows a warp did pay later, inside
  the multi-column decode twins, where a staged activation is shared.)
- **A conditional branch for K/V in the decode graph:** +0.8 to +2.0% on
  both GPUs against a 4% bar.
- **Interleaving dequantization with the tensor-core multiplies in the
  pipelined GEMM:** -1.4 to -5.0% prefill on the RTX 3070.
- **Reusing staged activations across sibling projections:** +0.6 to +2.5%
  prefill against a 3% bar, and an address-keyed cache served stale
  staging after an input write (perplexity 14.5 became 57,517 in one test).
  Dropped.
- **A staged-activation Q6_K decode GEMV:** faster on every shape of the RTX
  3070 by staying with the existing kernel (staging -8 to -21%); mixed on
  compute capability 12.0 (+12% and -15% on two shapes). A win would need
  per-device, per-shape dispatch for under 2% of decode.
- **The CUDA activation staging carried over to Metal's Q4_K matvec:** the
  same bytes, Llama-3.2-1B unchanged and Qwen2.5-1.5B 10% slower. Apple's
  unified cache already absorbs the strided reads.
- **Skipping unused activation sums in the AVX2 K-quant dot:** -0.2 and -0.4%
  decode inside an A/A spread of 1.2%, bytes identical. The NEON half of
  that change (table-lookup widening) shipped on its own evidence.
- **Link-time and profile-guided optimisation of the CPU build:** every
  result inside the A/A spread on the Apple M1, and profile-guided builds
  change score bytes.
- **A fixed draft width for prompt lookup and the MTP head:** the best arm
  was +9% (Llama-3.2-3B lookup on code) against a 10% bar on complete-turn
  latency, prose and tool turns were slower, and MTP on Qwen3.5-4B was
  slower in every arm. No fixed winner; outputs identical throughout.

## Stopped at the pre-check

These were costed from timelines or arithmetic before any build, against a
bar written into the plan.

- **A persistent decode kernel:** at most 6.4% / 3.0% of a decode step was
  removable on the slice and 4.7% on the RTX 3070 after the one-copy KV
  gather, against 15%.
- **Device argmax for greedy decoding:** the logits copy and host argmax
  were 1.3 to 1.7% of a step, against 3%.
- **Overlapping the KV copy-back:** at most about 0.6% left to overlap after
  the single copy, against 2%.
- **A grid swizzle for L2 reuse in prefill:** at a 512-token window the
  pipelined GEMM re-reads each weight matrix once per 64-column tile, about
  0.09 ms of DRAM time for Llama-3.2-3B's query projection against about
  0.5 ms of tensor work on the RTX 3070, and the reads overlap the
  multiplies: even a perfect swizzle cannot approach the 15% bar.
- **A quantized copy of the same model as the draft:** 67.9% acceptance on
  Qwen2.5-7B with a Q2_K draft, against 80%.
- **One launch for sibling projections:** Q, K and V mix Q4_K and Q6_K in
  the measured files, so one grid needs a mixed-type scheduler; the cheaper
  variant (K and V as a parallel branch of the decode graph) was measured
  and is listed above.
- **Progressive-precision output head:** the head is 7.8% (Qwen2.5-7B) to
  10.4% (Llama-3.2-3B) of a decode step and already read at 83 to 85% of
  bandwidth. The cheapest sound per-row bound (Cauchy-Schwarz on stored
  residual norms) on a 4-bit base leaves tens to hundreds of rows to refine
  per token, and even with free refinement the saving is at most 2.4% and
  3.2% of a decode step: under the 3% bar on one file and 0.2 points over
  it on the other, before the cost of the bound and the refinement pass,
  which that estimate leaves out. Not built.
- **Expert streaming for host-resident MoE prompts, tile by tile or layer
  by layer:** the tile-major design failed its fixture gate (a layer's
  router and shared tensors were never resident), and the layer-major
  pre-check put the upload floor at 0.30 to 1.06 s against a device MoE
  prefill floor of 87 s on Qwen3-30B-A3B at the time (llama.cpp: 2.9 s with
  its experts on the CPU). The lever was the device MoE prefill itself,
  which the by-expert and grouped tensor-core routes then moved.

## Blocked

- **Rows on Qwen3.8-Flash-Next (78 GB) under `--cpu-moe`:** the only host
  that holds the file could not keep it in its page cache (two back-to-back
  reads at about 92 MB/s, about 8 GB cached afterwards), so three attempts
  ran disk-bound. One clean warm round stands as single-round evidence
  (prefill 1.71x the build before this work, at 16K). The new tensor-core
  types stay off for its architecture until it has a row.
- **Full-card measurement windows:** the shared datacenter host gives a MIG
  slice, not the whole card, so every promotion rests on the slice and the
  RTX 3070.
