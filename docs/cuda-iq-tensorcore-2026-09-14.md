# Tensor-core prefill for the codebook i-quants (started 2026-09-14)

The program that follows the v0.5.3 CUDA IQ review: repair the measuring
instruments first, then give each codebook format a tensor-core prefill
GEMM one at a time, each behind the forced-TC gate until it is promoted on
measured rows. This file is the evidence log; the entries below are
appended as formats land.

Ground rule, unchanged from the i-quant port: llama.cpp defines the
formats and serves as the interoperability oracle, the runner's own CPU
decoders are the blueprint, nothing is ported from ggml-cuda. The PTX is
built by CUDA 13.3 on the Windows box, the only machine that has it.

## Step 0, the instruments (PR #94, merged 2026-09-14)

Five review findings, each fixed red first where a mutation could show it:

| finding | repair | proof |
|---|---|---|
| P1: gpu-identity accepted a NaN logit (NaN > limit is false) | non-finite logits fail on either side, detected in a fast-math-free TU (`tests/finite_check.c`); Apple clang 21 folds even a copied-bits test to false under `-ffinite-math-only` | `test-gpu-identity --self-test`, 8 cases in `make test`; both NaN cases read as ok against the old verdict |
| P2: the logprob comparison truncated to the shorter list and passed a NaN after a finite maximum | `worst_deviation()` requires exact counts and finite values first | unit test on each malformed shape |
| P2: Windows tool discovery missed `llama-quantize.exe` | all three tools resolved once, bare or `.exe`; `RUNNER_REQUIRE_IQ_GATES` turns skips into failures; `RUNNER_IQ_FIXTURES`; `make test-cuda-iquants` | unit test on an `.exe`-only directory; the Windows ledger run uses the target |
| P2: the TC gate's hand-kept type list lacked Q6_K; engagement was one total | `gpu_tc_type_has_kernel()` and `gpu_tc_dispatches_type()` from the backend's own kernel table; per-format engagement required in the forced-on arm, none allowed in the forced-off arms | `test-tc-tol --types`; granite-4.2-3b Q4_K_M on the Blackwell: `Q4_K=480 Q6_K=80` |
| P3: the sign-expansion test never executed the device helper | the decode primitives moved to `src/iq_decode.h`, compiled by nvcc and by the C compiler into `tests/test_iq_decode.c` | the review's parity mutation fails 64 of 128 expansions; the PTX header's entries and tables are checked from Python |

## Step 1, IQ3_S (opt-in)

`k_gemm_iq3_s_tc`: the `TC_GEMM_BLK` shape (the Q8_0/Q4_0 macro generalised
over the block's element count, their PTX unchanged) with the 110-byte,
256-weight block. The 64-row x 128-K fp16 weight tile is staged two
64-element segments per row per K-step by `iq3s_stage64` in
`src/iq_decode.h`; the same source, compiled for the host, is held to
`dequant_row()` bit for bit on a hand-built block (the ninth index bit
through `qh`, two distinct odd scales in one scale byte, a negative sign
in each half of an octet), 500 random blocks and the scale corners (zero,
the smallest subnormal, the largest half). Registered in `TC_KERNELS`, not
promoted: `RUNNER_CUDA_TC=1` or the gate's `gpu_tc_force(1)` reaches it.
Single-token decode stays on the generic matvec.

Precision contract: fp16 operands, fp32 accumulation, the class the four
promoted formats already live in. No IQ3_S product overflows fp16 (the
block scale is a half, the sub-block scale is at most 31, the grid
magnitudes at most 15; the staged value is `d * scale * magnitude` and the
scalar kernel computes the same product in fp32), so the staging error is
the fp16 rounding of that product and of the activations, measured below
as the deviation from the scalar path.

### Measured (forced on against forced off, same binary and split)

| model | box | dispatches (forced-on arm) | teacher-forced (64 positions) | free-running |
|---|---|---|---|---|
| llama-quantize IQ3_S fixture (256-wide) | RTX 3070, Windows | IQ3_S=28 | 7e-5 of range, 0 flips | 128 prompt + 32 greedy tokens identical |
| same | Blackwell MIG slice | IQ3_S=28 | 7e-5 of range, 0 flips | identical |
| granite-4.2-3b requantized to IQ3_S (241 IQ3_S, 40 Q4_K tensors), all 40 layers on the device | RTX 3070 | Q4_K=80 IQ3_S=480 | 8e-5 of range (mean 0.0024 of 29.2), 0 flips | identical |
| same | Blackwell slice | Q4_K=80 IQ3_S=480 | 8e-5 of range, 0 flips | identical |
| Qwen3.8-27B GSQ-RCO (the motivating file), 40 of 64 layers on the slice, `tc_vs_scalar.py` | Blackwell slice | IQ3_S tensors of the resident layers | greedy at 64 tokens: 9 of 9 prompts identical | (the runs are free-running) |

Sanitizers, RTX 3070, `compute-sanitizer` 13.3 on the whole `test-tc-tol`
run of the fixture (IQ3_S=28 dispatches under each tool): memcheck 0
errors, racecheck 0 hazards, synccheck 0 errors, initcheck 0 errors.

### Prefill throughput, `pp_bench` (5 warmed runs, median, `-n 1`)

| prompt | RTX 3070 scalar | RTX 3070 TC | Blackwell slice scalar | Blackwell slice TC |
|---|---|---|---|---|
| 118 tokens | 24.6 tok/s | 495.4 tok/s (20x) | 13.8 | 221.7 (16x) |
| 475 tokens | 26.2 | 459.8 (18x) | 12.1 | 237.2 (20x) |
| 1892 tokens | 25.7 | 376.9 (15x) | 11.9 | 161.6 (14x) |

Run-to-run spread on the RTX 3070 is under 1%; on the Blackwell slice it is
6 to 53% because the slice is shared with a training job, so those rows are
order-of-magnitude evidence, not a number to quote. The scalar column is the
generic warp-per-row `k_mv_iq3_s_b` applied to 64 columns, which is why the
gap is an order of magnitude larger than the promoted formats' rows: the
codebook formats had no batched GEMM at all.

### Promotion status

Not promoted. The plan's gate (at least 10% median improvement over five
warmed runs above noise, no regression over 5% on the workloads assigned to
the path) is met by a wide margin on both device families for the one
architecture measured in full (granite), and the precision rows match the
promoted formats'. Promotion waits for the rest of the family so the
decision is made once per architecture with the per-type dispatch counts
in hand, as the plan orders.

Evidence files: `cuda-iq-tensorcore-evidence/` (the two benches, the two
gate logs, the racecheck log, the GSQ-RCO greedy comparison).
