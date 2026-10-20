# Kernel routes and their numerical obligations

Each backend picks a kernel at one local decision point, keyed by facts it
already has: weight type and layout, column count, shape, scale, device
capability and an environment switch. This page lists those points, what each
route promises, and the test that holds the promise. The comment at each
decision point in the source carries the measurement history; this page is
the map.

A route has one of three obligations:

- **Bit-exact**: byte-equal to a named reference route. A gate compares bytes.
- **Deterministic within the route**: the same bytes on every run of the same
  shape on the same build, with no claim against another route.
- **Tolerance**: agreement with an independent reference under a stated bar.
  The bar used here is teacher-forced: every configuration gets the same
  tokens, and the gate counts top-1 flips (0 of 64 positions to promote) and
  the largest logit difference as a share of the logit range.

CPU against GPU on the scalar route is **token-identical, not
logit-bit-identical**: the CPU sums a row with SIMD lanes and the GPU with a
32-lane reduction tree, and those round differently. `test-gpu-identity`
states that bound. A route promoted to a default keeps an opt-out switch that
restores the reference route; every CPU-against-GPU byte comparison in the
suite sets those switches (`RUNNER_CUDA_TC=0`, `RUNNER_CUDA_ROUTE=reference`,
`RUNNER_METAL_MM=0`, `RUNNER_METAL_ATTN_COOP=0`, `RUNNER_METAL_MOE_MM=0`,
`RUNNER_MOE_EAGER=1`).

Several tolerance gates take a real model as an argument (`./test-tc-tol
MODEL.gguf`). They run on a promotion, not on every change; the rows they
produced are recorded beside the code that consults them.

## CPU: `mv_rows` in `src/model.c`

| Route | Taken when | Obligation | Gate |
| --- | --- | --- | --- |
| `vec_dot`, the native quantized dot | one column, f32 activations | the CPU reference | `test-quants-simd` against scalar code |
| `vec_dot_multi` | 2 to 7 columns (`MV_SMALL_BATCH` is 8) | bit-exact to the per-column `vec_dot` loop | `test-batch` (logits), `test-dot-route` (dispatch and a same-process cost ratio) |
| dequantized f32 tile, `vec_dot_f32_tile` | 8 or more columns | bit-exact per output to the f32 dot of the dequantized row, whatever the blocking (`fmaf` tails) | `test-quants-simd`, `test-batch` |
| fused int8 dot, `vec_dot_i8` | `RUNNER_CPU_I8=1`, fewer than 8 columns, at least 32 output rows, a supported type (`i8_dot_ok`) | tolerance; off by default because no format cleared 0/64 with a gain worth taking | `./test-i8-tol MODEL.gguf` |
| canonical-order kernels | a `make T3=1` build | bit-exact to an independent C definition of the same reduction tree, on every target | `test-canon-kernels`, `test-pmath` |

The thread count never changes which rows a worker sums or their order:
`test-attn-split` holds attention logits byte for byte at 1, 2, 3, 4 and 7
threads. Scoring defaults to the scalar teacher-forced path;
`RUNNER_SCORE_CHUNKED=1` takes the batch route and its contract.

The f32 tile is exact per output, but it is not the one-column `vec_dot`'s
arithmetic, so a prompt read in batches of 8 or more moves every logprob
after it: 4.2e-3 at most over 165 positions of MiniCPM5-1B Q4_K_M
(`RUNNER_SCORE_CHUNKED=1` against the solo default), and 3.1e-3 on the
tokens sampled after a 60-token prompt. Batches of 2 to 7 stay on
`vec_dot_multi` and match solo exactly, at about a quarter of the tile's
prefill speed (37 against 129 to 143 tok/s on that model, 422 tokens, 4
threads). Training rollouts therefore read the prompt with `-b 1`
(`scripts/train-grpo-lite.py`, gated by `tests/test_rollout_identity.py`).

## CUDA: `enc_mv` in `src/cuda.c`

The branches are tried in this order; the first that applies runs.

| Route | Taken when | Obligation | Gate |
| --- | --- | --- | --- |
| int8-activation `dp4a` GEMV | `RUNNER_CUDA_Q8X=1`, 1 to 8 columns, unscaled, `n_in` a multiple of 256, a kernel for the type (IQ3_S, IQ3_XXS, IQ4_XS, Q4_K, Q8_0) | tolerance against CPU `--score`; experimental and off | measured by hand (0 to 3 argmax flips in 480 positions) |
| small batch | 2 to `GPU_SMALL_BATCH` columns, unscaled, `RUNNER_CUDA_SMALL_BATCH` not `0` | bit-exact per column to the batch-1 kernel: a width-classed GEMV twin (`f_gemvb`), or the batch-1 kernel once per column | `test-batch` on a CUDA build |
| pipelined tensor-core GEMM (TC2) | the tensor-core GEMM's conditions, `n_in` a multiple of 64, at most the window's columns, and a kernel from `tc2_fn()`: by default the compute_80 module's (`k_gemm2_80_*`, one m16n8k16 per 16-deep step) on compute capability 8.0+ and none below; `RUNNER_CUDA_TC2=1` (or `tc2` in `--gpu-route`) also the compute_75 kernels where that module did not load (below 8.0, `RUNNER_CUDA_SM80=0`, or a route set with `tc2` and without `sm80`, which keeps it unloaded); `RUNNER_CUDA_TC2=0` or a route set without `tc2` none | compute_80 kernels: byte-identical to the tensor-core GEMM (the same fp32 sums; chunked `--score` on sm_86 and sm_120 with launch counts); compute_75 kernels: the tensor-core GEMM's tolerance class (fp32 rounding between the two m16n8k8 halves of a step) | `make test-cuda-tc2-bytes` in the CUDA gate (chunked `--score` against `RUNNER_CUDA_TC2=0`, byte for byte, with the launch counts); `./test-tc-tol MODEL.gguf` (prints the TC2 and compute_80 launch counts) |
| tensor-core GEMM | more than one column, unscaled, `tc_on()` (the `RUNNER_CUDA_TC` override, else `tc_promoted()` per type and architecture), kernel present | tolerance per promoted (type, architecture) row; token columns are staged scaled to the fp16 range | `./test-tc-tol MODEL.gguf`; `make test-tc-overflow` on the CUDA gate |
| tiled GEMM (`f_gemm`, Q8_0 and Q4_K) | more than one column, unscaled, kernel present | bit-exact to the batched `_b` kernel: same lane ownership, same inner order, only the source of `x` moves to shared memory | `test-gpu-identity` (prefill against the CPU) |
| BF16 split-K | one column, BF16, at most 1,024 rows of at least 2,048 | deterministic within the route: a fixed tree per split and an index-order sum, no atomics | `make test-cuda-qwen4exp`: launch count, the CPU bound, two device runs byte for byte |
| coalesced GEMV (`f_gemv`) | one column, kernel present | the decode route; token-identical to the CPU scalar route | the CUDA gate's `cpu-cuda` step, `test-gpu-identity` |
| generic `f_mv` and `f_mvb` | otherwise | `f_mvb` is a bit-exact twin of `f_mv` only for F32, F16 and BF16 (`mvb_is_mv_twin`) | `test-batch` |

Around the matvec:

- **Prefill and decode attention (experimental).** `RUNNER_CUDA_ATTN_TILE=1`
  takes prefill attention through `k_attn_tile32`/`16` (query strips over
  shared-memory K and V tiles; each score is `k_attn`'s, the softmax and the
  V sum reassociate), and `RUNNER_CUDA_ATTN_DEC=1` single-token attention
  through `k_attn_dec_gqa` and `k_attn_merge32` (a KV row read once per
  head group, 32 fixed splits). Obligation: tolerance, token-identical on
  the long-context checks. Gate: `test-gpu-identity` (with `GPU_ID_KV`).
- **Byte movers, on by default.** The one-copy KV gather (decode with full
  offload: every layer's new K and V rows gathered on the device, one copy
  back, `RUNNER_CUDA_KV_GATHER=0` restores the per-layer copies) and the
  whole-tile recurrent prefill (Gated DeltaNet and Mamba-2 tiles in one
  launch per kernel, `RUNNER_CUDA_RECUR_SEQ=0` restores the per-token
  launches) move the same bytes to the same places or repeat the per-token
  arithmetic in order. Obligation: byte-identical. Gates: `GPU_ID_RECUR=1
  ./test-gpu-identity` (launch count required), the CUDA gate's identity,
  injected-failure and recovery steps (which now run through the gather).
- **Slot microbatching.** `batch_mv_twin_ok()` admits a type to the
  shared-weight decode microbatch only when its batched kernels are twins
  of the batch-1 kernel. A refusal decodes the sequences one at a time; it
  never runs a non-twin kernel. Obligation: bit-exact to sequential decode
  (`docs/cuda-microbatch-identity-2026-08-18.md`).
- **MoE routing.** The default fused device routing selects the same experts
  as the eager host routing (`RUNNER_MOE_EAGER=1`) and keeps routing weights
  within about 2 ulp: device `expf` is not the host libm. Gate:
  `./test-moe-tol MODEL.gguf`.
- **Graph replay.** One-token decode with full offload replays a captured
  graph of the same launches. `RUNNER_CUDA_GRAPH_OFF=1` runs them eagerly, to
  separate a kernel from the graph path. See [buffer lifetimes](buffer-lifetimes.md)
  for when capture is refused.
- **Training primitives.** `gpu_mvcanon` (the forward matvec in the canonical
  order, F32, F16 and Q8_0) and the transposed matvec behind
  `RUNNER_TRAIN_GPU=1` are bit-exact to the CPU trainer's chain. Gates:
  `test-mvcanon`, `test-mvt`.
- **KV cache formats.** q8 KV is tolerance-gated (`./test-kv-tol`); fp4 KV is
  held to the host codec (`test-kv-fp4`, `make test-cuda-kv-fp4`).

## Metal: `enc_mv_n`, then `enc_mv_cols`, in `src/metal.m`

| Route | Taken when | Obligation | Gate |
| --- | --- | --- | --- |
| identity matvec, multi-column | 2 to 16 columns | bit-exact to that many single-column dispatches | `test-batch-identity` |
| Metal 4 tensor GEMM | more than one column, `RUNNER_METAL_TENSOR=1`, an M5 or newer GPU on macOS 26.2 or newer, the per-type self-test passed at load, `n_in` a multiple of 256 | tolerance; opt-in, a failing type falls back alone | the load-time self-test, then `./test-tc-tol` |
| simdgroup GEMM (`k_mm_*`) | more than one column, `RUNNER_METAL_MM` not `0` (promoted), `n_in` a whole number of k-steps and of superblocks | tolerance; activations staged scaled by the column maximum | `./test-tc-tol MODEL.gguf`, `make test-tc-overflow` on Darwin |
| identity matvec (`k_mv`) | otherwise | token-identical to the CPU scalar route | `test-gpu-identity`, `make test-metal-kquant` and the other `test-metal-*` text gates |
| fast decode matvec (`k_mvf_*`) | one column, `RUNNER_METAL_MV=1`, kernel present | tolerance; off by default (neutral when measured), never on prefill | `./test-mv-tol MODEL.gguf` |

Around the matvec:

- **Decode attention.** The cooperative KV read is the default and is
  tolerance-gated (`./test-attn-tol MODEL.gguf`); `RUNNER_METAL_ATTN_COOP=0`
  pins the byte-identical kernel. The grouped-query chunk kernel is
  bit-exact to the cooperative chunk kernel (`RUNNER_METAL_ATTN_GQA=0`), and
  the prefill tile kernel is bit-exact to the one-column kernel
  (`RUNNER_METAL_ATTN_TILE=0`).
- **Decode fusion.** `RUNNER_METAL_FUSE` (default on) folds rope and KV store,
  residual and norm, the MoE gate and up projections, and the attention
  front into single dispatches, each bit-exact to the unfused chain. Gate:
  `make test-metal-fuse`.
- **MoE prefill.** Grouped MMA (`RUNNER_METAL_MOE_MM`, default on) answers to
  the fidelity bar (margin-qualified top-1 at least 97% and mean KLD at most
  0.05) through `test-moe-mm-ab` in `make test`; `=0` restores the
  slot-major matvec that byte gates pin. Expert-major kernels
  (`RUNNER_METAL_MOE_EM=1`) are bit-exact to slot-major (`make
  test-metal-moe-em`).
- **Weight wraps.** A file above the per-buffer limit is wrapped in several
  buffers cut on tensor boundaries, bit-exact to one wrap (`make
  test-metal-multibuf`).
- **KV cache formats.** `make test-metal-kv-q8` and `make test-metal-kv-fp4`.

## Adding a kernel

Put a new kernel at the decision point for its backend, keyed by facts that
point already reads, with no new conditional in model logic and no indirect
call in an inner loop. Give it one of the three obligations and a gate
before it can be selected, keep it off by default until the gate and a
measured gain both pass, and give it a switch that restores the route it
replaces. Training recomputes through the inference primitives, so a new
route must not change what the backward pass differentiates. `RUNNER_DEBUG_ROUTE=1`
and the `route` object of `--bench-json` report the route a forward took
([performance](performance.md)).
