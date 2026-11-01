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

## On a second GPU architecture (2026-11-01)

The twins were tuned on the RTX 3070 (sm_86). To check that they are not slower elsewhere, both trees ran on an
RTX PRO 6000 Max-Q (compute capability 12.0) in a MIG 1g.24gb partition that no other process used during the run.
The trees were pre = f61f6961 (main just before this change) and main = f4a1c212, both built with clang 22.1.8,
`-t 8` on eight host CPUs. The host was shared with other work, so host-bound time carries that load. The models
were the same as above (sha256-checked against the Hub), with three prompts, 200 tokens at most, temperature 0, and
one run per cell.

A k 4 round is one draft of four tokens plus one five-row verify forward; this run did not split the two, so the
round time below is both. The draft is also a Q4_K_M model and uses the same twins.

| Prompt | Rounds, tok/round | pre: ms a round, tok/s | main: ms a round, tok/s | Round time |
|---|---|---|---|---|
| 0 | 45, 4.44 | 59.7, 74.45 | 44.5, 99.84 | -25.4% |
| 1 | 34, 2.97 | 59.5, 49.95 | 44.4, 66.86 | -25.3% |
| 2 | 26, 3.38 | 59.3, 57.07 | 44.4, 76.19 | -25.1% |

Plain decoding was 75.38, 74.91 and 75.76 tok/s on pre against 75.47, 75.73 and 74.79 on main (13.2 to 13.4 ms a
token on both). Prompt 2 is 1.3% slower on main, inside the spread of the other two prompts, which moved the other
way by up to 1.1%. Every k 4 output equals its plain output on both trees, and plain output is byte-identical
between the trees.

The decode microbatch: a server at `--parallel 4` answered four prompts at once, and each reply matched the same
prompt answered alone (4 of 4 on both trees). The four replies are 437 tokens in all (counted with the Qwen2.5
tokenizer).

| | pre | main |
|---|---|---|
| batch wall time | 4.34 s | 3.44 s |
| aggregate decode | 100.7 tok/s | 127.0 tok/s (+26%) |
| the four alone, summed | 5.98 s | 5.99 s |

What got slower: prompt processing on these 15 to 20 token prompts, in three of the six cells. On k 4 prompt 0 it
went from 434.3 to 367.3 tok/s (-15%), on plain prompt 2 from 325.8 to 305.9 (-6%), and on k 4 prompt 2 from 326.9
to 312.2 (-4.5%). The other three cells got faster: plain prompt 0 +4%, plain prompt 1 +10%, and k 4 prompt 1 the
same. Each of these prompts takes 40 to 55 ms, one run per cell on shared host CPUs, and a prompt over eight tokens runs
the tiled GEMM, not a twin (outside a verify round the twins serve only 2 to 8 columns). We read this as noise, but
these runs cannot show it.

`make test-cuda-spec-identity` and `make test-cuda-batch-identity` exit 0 on this GPU at main.
