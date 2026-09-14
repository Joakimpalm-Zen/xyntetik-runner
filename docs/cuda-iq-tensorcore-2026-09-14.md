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

## Step 2, group 2: IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS (opt-in)

The same `TC_GEMM_BLK` shape with the 98, 82, 74 and 66-byte blocks; the
stagers in `src/iq_decode.h` are held to `dequant_row()` bit for bit on
500 random blocks and the scale corners per format, and to the format's
definition on a hand-built block each, one per row of the plan's hazard
table: IQ3_XXS's scale nibble in the top of a 2-byte-aligned word and its
packed seven-bit signs over two four-magnitude grids; IQ2_S's ten-bit
index through the high-bit byte, direct sign bytes and two scales per
sub-block; IQ2_XS's nine-bit index and seven-bit sign index sharing one
16-bit word; IQ2_XXS's 32-bit field at byte 6 of a 66-byte block. PTX from
CUDA 13.3 on the Windows box: four kernels added, no other body changed.

Real models: granite-4.2-3b-bf16 requantized with llama-quantize b10353
and an imatrix from the runner's own docs (200 chunks of 512). The
llama-quantize recipes mix: the IQ3_XXS file carries 120 IQ3_XXS, 80
IQ2_S and 41 IQ3_S tensors, the IQ2_S recipe stores 195 IQ2_XS and 46
IQ3_S (no IQ2_S tensor at all), the IQ2_XS and IQ2_XXS files 235 of their
own type; each has 40 Q4_K tensors beside them. Every type present is
required to dispatch, and does.

| model (all 40 layers on the device) | box | forced-on dispatches | teacher-forced | free-running |
|---|---|---|---|---|
| fixtures m-IQ3_XXS / m-IQ2_S / m-IQ2_XS / m-IQ2_XXS | RTX 3070 and Blackwell slice | IQ3_XXS=16 IQ3_S=4 IQ2_S=8 / IQ2_XS=20 IQ3_S=8 / IQ2_XS=24 / IQ2_XXS=24 | 3e-5 to 7e-5 of range, 0 flips each | identical, each |
| granite-4.2-3b IQ3_XXS recipe | RTX 3070 | Q4_K=80 IQ3_XXS=240 IQ3_S=80 IQ2_S=160 | 6e-5 of range, 0/64 flips | identical |
| same | Blackwell slice | same counts | 7e-5, 0/64 | identical |
| granite-4.2-3b IQ2_S recipe | Blackwell slice | Q4_K=80 IQ2_XS=390 IQ3_S=90 | 1.1e-4, 0/64 | identical |
| granite-4.2-3b IQ2_XS | Blackwell slice | Q4_K=80 IQ2_XS=470 | 1.2e-4, 0/64 | identical |
| granite-4.2-3b IQ2_XXS | RTX 3070 | Q4_K=80 IQ2_XXS=470 | 1.3e-4, 0/64 | identical |
| same | Blackwell slice | same | 1.3e-4, 0/64 | identical |

Sanitizers, RTX 3070, the whole `test-tc-tol` run of each of the four
fixtures under memcheck, racecheck, synccheck and initcheck: 0 errors, 0
hazards, every run with its claimed types dispatching.

### Prefill throughput (RTX 3070, 5 warmed runs, median, spread under 1%)

| model | 118 tokens | 475 tokens | 1892 tokens |
|---|---|---|---|
| IQ3_XXS recipe (IQ3_XXS + IQ2_S + IQ3_S on the tensor cores) | 24.4 to 458.5 tok/s (19x) | 26.1 to 436.4 (17x) | 25.8 to 359.8 (14x) |
| IQ2_XXS | 25.3 to 326.8 (13x) | 26.9 to 322.6 (12x) | 26.5 to 276.3 (10x) |

The Blackwell slice was shared with another job's 14B load during this
group's benches (one run was refused VRAM outright, spreads up to 50%), so
its rows (`pp-bench-*-blackwell.json`) are kept as order-of-magnitude
evidence only: IQ3_XXS 17 to 324-378 tok/s, IQ2_S 11-16 to 181-280, IQ2_XS
11-18 to 100-127, IQ2_XXS 12 to 104-132.

Promotion status unchanged: opt-in, the whole family first.
