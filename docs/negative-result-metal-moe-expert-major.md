# Negative result: expert-major MoE prefill kernels — M5 Max, 2026-09-01

The 120B validation left one conclusion standing: Runner's Metal MoE prefill
trails llama.cpp's non-tensor prefill by ~9.8x warm, and "batching/grouping
expert work is the next mechanism to investigate." This session built and
measured the two cheapest grouping shapes. Both lose. The numbers are kept
so the third shape starts from evidence, not intuition.

## What was built

The slot-major `k_moe_mv_*` family runs one threadgroup row per
(token, selected-expert) slot, so at prefill batch N an expert's weight rows
are fetched once per slot that selected it (~N·used/n_expert times). Two
inversions of that grid, both behind `RUNNER_METAL_MOE_EM=1`, both engaging
only when a dispatch carries at least as many slots as experts (decode always
stays slot-major):

1. **Flat expert-major** (`k_moe_mv_em_*`, all ten quant types): one
   threadgroup row per EXPERT; the threadgroup compacts its expert's slots
   into threadgroup memory, then walks the list with the same dot body —
   weight rows read once per threadgroup, reused from cache across tokens.
2. **Column-tiled expert-major** (`k_moe_mv_em8_*`, q8_0 and mxfp4): as
   above, but each weight block load feeds 8 gathered token columns with
   per-column accumulators — the reuse moved from cache into registers.

Both keep the per-(row, slot) arithmetic and reduction order of the
slot-major body (one shared `moe_dot_*` definition), so their outputs are
**byte-identical** to slot-major — `make test-metal-moe-em` gates that across
five fixture families (f32, q8_0-requantized, mxfp4, gemma4 dual-branch) with
an engagement check so a fallback cannot pass vacuously, and the gate is
mutation-proven (a dropped element in the q8_0 tiled dot fails it).

## What was measured

Interleaved warm A/B, 714-token prompt, batch 512, three reps per arm, same
binary, `RUNNER_METAL_MOE_EM` the only variable:

| model | slot-major prefill | flat expert-major | column-tiled |
|---|---|---|---|
| Qwen3-30B-A3B Q8_0 | 169–174 tok/s | 161–165 (−4%) | 166–167 (−3%) |
| gpt-oss-120b MXFP4 | 70–79 tok/s | 68–73 (−4%) | 69–73 (−3.5%) |
| Qwen3-235B Q2_K | 20–23 tok/s | 19–22 (−3%) | not built for q2_K |

Decode was unchanged in every pair, as designed.

## What this kills, and what survives

Killed: the assumption that slot-major prefill pays for its redundant weight
reads at DRAM. If it did, reading each expert's rows once (shape 1) had to
win; it lost 4%. The M5 Max's cache hierarchy is evidently already absorbing
the re-reads at these batch sizes. Also killed: 8-wide scalar register reuse
(shape 2) — the FMA structure, not the load count, is the constraint.

Survives, sharpened: llama.cpp's MoE prefill advantage comes from running
expert FFNs as **simdgroup-MMA matmul tiles** (`mul_mat_id`: gather a group's
tokens contiguous, one real GEMM per expert, scatter back) — the same
mechanism Runner's *dense* prefill already has in `k_mm_*` and its MoE path
has never used. That route reassociates sums, so unlike the two shapes here
it is not byte-identical and must be promoted through the tolerance gates
exactly as dense `RUNNER_METAL_MM` was. It is the next tracer, and it starts
with this file's numbers as the floor to beat: anything under +10% on the
120B is not worth the numerics change.

The expert-major kernels stay in the tree behind `RUNNER_METAL_MOE_EM`, off
by default — a lever that has not cleared its bar on a machine is not a
default anywhere, and the byte-identity gate they carry also locks the
shared-dot refactor the next shape will build on.
