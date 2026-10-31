# CUDA speculative verify pays at full offload (R3.3.4), 2026-10-31

Measured on the Windows CUDA host (RTX 3070, 8 GB, driver 596.36) with runner 1.9.0 at main 26b2ebc6 against the
same tree with this change. Target Qwen2.5-Coder-7B-Instruct Q4_K_M, draft Qwen2.5-Coder-1.5B-Instruct Q4_K_M, both
fully offloaded; one prompt ("Write a Python function that parses an ISO 8601 date string and returns a datetime
object, with error handling and docstring."), 200 tokens, temperature 0, seed 1.

## The premise had moved

The plan item read "gpu_spec_keep_ok is false on CUDA". Since 9a42cac9 (2026-10-07) it is true at full offload, and
the walk is target-exact there (`make test-cuda-spec-identity`). What remained was cost: the verify forward was a
loss on a discrete card.

| Arm | main 26b2ebc6 | this change |
|---|---|---|
| plain | 70.9 tok/s | 71.1 tok/s |
| draft, k 2 | | 80.4 tok/s (2.86 tok/round) |
| draft, k 4 | 61.0 tok/s | 86.0 tok/s (4.17 tok/round) |
| draft, k 6 | | 84.3 tok/s (5.41 tok/round) |
| draft, k 8 | 52.8 tok/s | 74.5 tok/s (6.67 tok/round) |

All outputs are byte-identical to the plain run (sha256 3b007a75...). At k 4, `RUNNER_SPEC_PROF=1` put the verify
forwards at 2185 ms of 3246 on main (45.5 ms a round, 3.2 one-token decode steps for a five-row pass) and at
1246 ms with this change; drafting is unchanged at about 1.07 s.

## Where the verify time went

`RUNNER_CUDA_PROFILE=1` on main: 88% of the verify tiles' GPU time was matvec. A verify round's rows run the
width-classed GEMV twins (exact rows, each column the batch-1 kernel's bits). Timed alone on the 7B FFN shapes, the
Q4_K twin cost 3.5 to 4.7 batch-1 passes for five columns and the Q6_K twin 2.5 to 2.8. Nsight Compute on the 8-wide
Q4_K twin: 118 registers (two blocks an SM, 29% achieved occupancy), 18% of DRAM bandwidth, the warps stalled on
long scoreboard (the weight loads, issued and consumed in the same iteration, and the per-block x staging between
two barriers).

## The change

`GEMVB_KQ` (Q4_K, Q5_K) and the Q6_K twin now stage x double-buffered in shared memory with the next block
prefetched into registers, one barrier a block, and load each row's next block of weights into registers while
this one computes. Rows a warp and warps a block are set per kernel width (`KQ_DBR`, `KQ_DBW` in
`src/kernel_args.h`), and exact widths 5 to 7 join the existing `f_gemvx` slots for Q4_K, Q5_K and Q6_K, so a k 4
round (five rows) no longer runs the 8-wide kernel. Every (row, column) keeps the batch-1 kernel's expressions in
their order; the bits are unchanged.

Kernel time on the 7B shapes (18944 x 3584 and 3584 x 18944), in batch-1 passes:

| Kernel, columns | before | after |
|---|---|---|
| Q4_K, 4 | 2.9 to 3.2 | 1.5 to 1.7 |
| Q4_K, 5 | 3.5 to 4.7 | 1.7 to 2.0 |
| Q4_K, 8 | 4.8 to 6.2 | 2.3 to 2.8 |
| Q4_K, 9 (16-wide) | 6.4 to 7.9 | 3.4 to 3.9 |
| Q6_K, 4 | 2.4 to 2.5 | 1.7 to 1.9 |
| Q6_K, 5 | 2.5 to 2.8 | 1.5 to 1.7 |
| Q6_K, 8 | 3.5 to 3.7 | 2.3 to 2.6 |
| Q6_K, 9 (16-wide) | 5.5 to 5.8 | 3.2 to 3.9 |

The same twins serve the decode microbatch and the MoE expert matmuls, which gain the same way.

## Checks

- The PTX was regenerated with the pinned toolchain (`tools/ptx-toolchain/make-ptx.sh`, in a linux-64 container),
  which first reproduced main's `src/kernels_ptx.h` and `src/kernels_ptx80.h` byte for byte. Against main's PTX the
  only functions that differ beyond label and local-depot numbering are the Q4_K, Q5_K and Q6_K twins, and the nine
  exact-width kernels are new. `src/kernels_ptx80.h` is unchanged.
- `make test-cuda-batch-identity`, `test-cuda-spec-identity` and `test-cuda-moe-twin` on the RTX 3070, with
  llama-quantize available so the Q5_K_S and IQ4_XS twins run too. `test-cuda-spec-identity` now also quantizes
  its 256-wide model to Q5_K_S where llama-quantize is available.
- The microbenchmark checked every column of every twin against the batch-1 kernel bit for bit, on outputs cleared
  before each launch.
