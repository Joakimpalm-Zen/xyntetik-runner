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
`RUNNER_METAL_MM=0`, `RUNNER_METAL_ATTN_COOP=0`, `RUNNER_METAL_MV=0`,
`RUNNER_METAL_MOE_MM=0`, `RUNNER_MOE_EAGER=1`).

Several tolerance gates take a real model as an argument (`./test-tc-tol
MODEL.gguf`). They run on a promotion, not on every change; the rows they
produced are recorded beside the code that consults them.

## CPU: `mv_rows` in `src/model.c`

| Route | Taken when | Obligation | Gate |
| --- | --- | --- | --- |
| `vec_dot`, the native quantized dot | one column, f32 activations | the CPU reference | `test-quants-simd` against scalar code |
| `vec_dot_multi` | 2 to 7 columns (`MV_SMALL_BATCH` is 8) | bit-exact to the per-column `vec_dot` loop | `test-batch` (logits), `test-dot-route` (dispatch and a same-process cost ratio) |
| dequantized f32 tile, `vec_dot_f32_tile` | 8 or more columns | bit-exact per output to the f32 dot of the dequantized row, whatever the blocking (`fmaf` tails) | `test-quants-simd`, `test-batch` |
| int8 prefill tiles, `i8_tile_dot` | `RUNNER_CPU_I8_PREFILL=1`, 8 or more columns, a type with a pack (`i8_tile_ok`: Q8_0, Q4_0, Q4_K, Q6_K); NEON with dot-product, the scalar twin elsewhere | tolerance; off by default: mean KL 1.22x the f32 route's against float32 on Qwen3-0.6B Q8_0 (p 5e-86), prefill +68% / +49% on an Apple M1 (Q8_0 / Q4_K_M) | `test_quants_simd` (the kernel equals its scalar twin bit for bit, both the same-quantization dot), `tests/test_cpu_i8_prefill.py` |
| fused int8 dot, `vec_dot_i8` | `RUNNER_CPU_I8=1`, fewer than 8 columns, at least 32 output rows, a supported type (`i8_dot_ok`) | tolerance; off by default because no format cleared 0/64 with a gain worth taking | `./test-i8-tol MODEL.gguf` |
| canonical-order kernels | a `make T3=1` build | bit-exact to an independent C definition of the same reduction tree, on every target | `test-canon-kernels`, `test-pmath` |

The thread count never changes which rows a worker sums or their order:
`test-attn-split` holds attention logits byte for byte at 1, 2, 3, 4 and 7
threads. Scoring defaults to the scalar teacher-forced path;
`RUNNER_SCORE_CHUNKED=1` takes the batch route and its contract.

The f32 tile is exact per output, but it is not the one-column `vec_dot`'s
arithmetic, so a prompt read in batches of 8 or more moves every logprob
after it: 4.2e-3 at most over 165 positions of MiniCPM5-1B Q4_K_M
(`RUNNER_SCORE_CHUNKED=1` against the solo default), and 1.45e-2 on the
tokens of 384 rollouts sampled after prompts of 49 to 51 tokens
([measured](train-serve-mismatch-2026-11.md)). Batches of 2 to 7 stay on
`vec_dot_multi` and match solo exactly, at about a quarter of the tile's
prefill speed (37 against 129 to 143 tok/s on that model, 422 tokens, 4
threads). Training rollouts therefore read the prompt with `-b 1`
(`scripts/train-grpo-lite.py`, gated by `tests/test_rollout_identity.py`).
A server's prefix cache meets the same contract: a request that continues a
cached prefix reads only its new tokens, so its logprobs can differ from the
same request on a fresh slot (measured with the 1.5.0 and 1.4.2 Linux builds:
every repeat, rewind and one-token extension of a cached SmolLM2-135M Q8_0
prompt moved the answer, by at most 4e-5, and a BF16 file's never did;
`"cache_prompt": false` reproduces the fresh slot, and so does a restart).

## CUDA: `enc_mv` in `src/cuda.c`

The branches are tried in this order; the first that applies runs.

| Route | Taken when | Obligation | Gate |
| --- | --- | --- | --- |
| int8-activation `dp4a` GEMV | `RUNNER_CUDA_Q8X=1`, 1 to 8 columns, unscaled, `n_in` a multiple of 256, a kernel for the type (IQ3_S, IQ3_XXS, IQ4_XS, Q4_K, Q8_0) | tolerance against CPU `--score`; experimental and off | measured by hand (0 to 3 argmax flips in 480 positions) |
| small batch | 2 to `GPU_SMALL_BATCH` columns, unscaled, `RUNNER_CUDA_SMALL_BATCH` not `0` | bit-exact per column to the batch-1 kernel: a width-classed GEMV twin (`f_gemvb`), or the batch-1 kernel once per column | `test-batch` on a CUDA build (`make test-gates-batch`: the Q8_0, BF16, Q4_K, Q5_K and Q6_K fixtures) |
| int8 tensor-core GEMM (`f_gemm_i8`) | `RUNNER_CUDA_I8=1`, more than one column, unscaled, `tc_on()` for the type, a kernel in the compute_80 module (Q8_0, Q4_K, Q6_K), `n_in` a multiple of 64, at most the window's columns | tolerance; off by default: `test-tc-tol` fails on Qwen3-0.6B Q8_0 (free-run divergence at a near-tie), 0/64 flips on four files, GEMM 1.66x (Q8_0) and 1.18x (Q4_K_M) faster than TC2 on an RTX 3070 | `make test-cuda-i8` (`test-tc-tol` on Q8_0, Q4_K and Q6_K fixtures) |
| pipelined tensor-core GEMM (TC2) | the tensor-core GEMM's conditions, `n_in` a multiple of 64, at most the window's columns, and a kernel from `tc2_fn()`: by default the compute_80 module's (`k_gemm2_80_*`, one m16n8k16 per 16-deep step) on compute capability 8.0+ and none below; `RUNNER_CUDA_TC2=1` (or `tc2` in `--gpu-route`) also the compute_75 kernels where that module did not load (below 8.0, `RUNNER_CUDA_SM80=0`, or a route set with `tc2` and without `sm80`, which keeps it unloaded); `RUNNER_CUDA_TC2=0` or a route set without `tc2` none; a GEMM over more than 64 columns takes the compute_80 module's 128-column twin (`k_gemm2_80w_*`) unless `RUNNER_CUDA_TC2_WIDE=0`; a GEMM whose input columns the previous TC2 GEMM staged, unwritten since (K and V after Q, up after gate, a grouped MoE up after its gate), reuses that staging unless `RUNNER_CUDA_XH_REUSE=0` | compute_80 kernels: byte-identical to the tensor-core GEMM (the same fp32 sums; chunked `--score` on sm_86 and sm_120 with launch counts); compute_75 kernels: the tensor-core GEMM's tolerance class (fp32 rounding between the two m16n8k8 halves of a step); the 128-column twin: byte-identical to the 64-column tile (`GPU_ID_TC2_WIDE=1 ./test-gpu-identity`, prompt longer than 64 columns); the staging reuse: byte-identical to each GEMM staging its own columns (the same staged values) | `make test-cuda-tc2-bytes` in the CUDA gate (chunked `--score` against `RUNNER_CUDA_TC2=0`, byte for byte, with the launch counts, and on 8.0+ against `RUNNER_CUDA_XH_REUSE=0` with the reuse count); `./test-tc-tol MODEL.gguf` (prints the TC2 and compute_80 launch counts) |
| tensor-core GEMM | more than one column, unscaled (or NVFP4 with its companion scale and no bias, the scale applied to the output after), `tc_on()` (the `RUNNER_CUDA_TC` override, else `RUNNER_CUDA_TC_NVFP4=1` forcing NVFP4, else `tc_promoted()` per type and architecture; NVFP4 promoted on #508's float32-reference row, `RUNNER_CUDA_TC_NVFP4=0` holding it off), kernel present | tolerance per promoted (type, architecture) row; token columns are staged scaled to the fp16 range | `./test-tc-tol MODEL.gguf`; `make test-tc-overflow` on the CUDA gate |
| prefill through the batch-1 kernel's twins (every type with a 16-wide twin but F32, F16 and BF16: Q4_0, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, MXFP4) | more than `GPU_SMALL_BATCH` columns, unscaled, no int8-activation route (`mv_twin_route`), a 16-wide twin (`k_gemvb_<type>_x16`) loaded, and not a type whose `f_mvb` already is its batch-1 kernel's twin (`mvb_is_mv_twin`): `prefill_takes_twins`, by capability since R6.57 | bit-exact per column to the batch-1 kernel: 16-column passes through the width-classed twins, a lone last column through the batch-1 kernel. It replaces `k_mvb_q4_0` (R6.35), the K-quants' `k_gemm_*` (R6.41, Q3_K R6.54), `k_gemm_q8_0` (R6.50) and the codebook i-quants' and MXFP4's `k_mv_*_b` (R6.57), which sum in another order | `./test-prefill-bits` on thirteen fixtures in `test-gates-core`: the route table (every type with a 16-wide twin prefills through it) and layer 0's KV rows byte-identical to a batch-1 prefill at batch 9, 17, 24 and 64 (before: through `k_gemm_*` Q4_K, Q5_K and Q6_K differed in 17 to 24 of 16,384 K values, Q3_K in 8 K and 1 V, Q8_0 in 4 K and 1 V of 4,096; through `k_mv_*_b` the codebook i-quants in 23 to 36 K, MXFP4 in 27 to 28 K at rows 2,048 long); `./test-kv-tol test-q40.gguf` (through `k_mvb_q4_0` its q8 arm read 10.7x the reassociation floor on an RTX 5070) |
| tiled GEMM (`f_gemm`) | more than one column, unscaled, kernel present, and the row above not taken: every `f_gemm` type is listed there, so only a type whose 16-wide twin did not load | bit-exact to the batched `_b` kernel: same lane ownership, same inner order, only the source of `x` moves to shared memory; not to the batch-1 GEMV, which sums in another order (R6.39) | `test-gpu-identity` (prefill against the CPU) |
| BF16 split-K | one column, BF16, at most 1,024 rows of at least 2,048 | deterministic within the route: a fixed tree per split and an index-order sum, no atomics | `make test-cuda-qwen4exp`: launch count, the CPU bound, two device runs byte for byte |
| coalesced GEMV (`f_gemv`) | one column, kernel present | the decode route; token-identical to the CPU scalar route | the CUDA gate's `cpu-cuda` step, `test-gpu-identity` |
| generic `f_mv` and `f_mvb` | otherwise | `f_mvb` is a bit-exact twin of `f_mv` only for F32, F16 and BF16 (`mvb_is_mv_twin`) | `test-batch` |

Around the matvec:

- **Prefill and decode attention (on by default).** Prefill attention takes
  `k_attn_tile32`/`16` (query strips over shared-memory K and V tiles; each
  score is `k_attn`'s, the softmax and the V sum reassociate) unless
  `RUNNER_CUDA_ATTN_TILE=0`, and single-token attention `k_attn_dec_gqa` and
  `k_attn_merge32` (a KV row read once per head group, whole 32-row tiles
  per split, one pass over each split with a running maximum since R3.29,
  each V pair read once for all of a thread's heads since R3.29.1;
  the same bytes run to run for a geometry and position,
  `tests/test_attn_dec_repeat.py`) unless `RUNNER_CUDA_ATTN_DEC=0`; both recorded as routes
  (`attn-tile`, `attn-dec`). Obligation: tolerance, and no measurable
  distance from the replaced kernels against the float32 reference at long
  context (2,000 positions at 4,000-6,000 tokens; Qwen3-4B, Mistral-7B, and
  gpt-oss-20b, whose attention has sinks and an alternating window). Gate:
  `test-gpu-identity` (with `GPU_ID_KV`). The tensor-core prefill attention
  (`k_attn_tc`, f16 caches, route `attn-tc`) is on by default for the llama,
  qwen2, qwen3, qwen3moe, granite and phi3 families without a sliding
  window, whose long-context rows (Llama-3.2-3B, Mistral-7B, Qwen2.5-3B,
  Qwen3-4B, Qwen3-30B-A3B, granite-4.2-8b, Phi-3.5-mini, Phi-4-mini) read no
  further from the float32 reference than the tiled kernel's; Phi-3.5 and
  Phi-4 publish a window longer than their context, which is no window.
  gpt-oss's row read further and the windowed files have none, so they keep
  the tile (`RUNNER_CUDA_ATTN_TC=1` / `0` forces it on / off). With Q and the
  softmax weights as single fp16 operands it read 0.19% further from the
  reference on Mistral-7B; it carries each in two halves (about 22 bits of
  the operand), which read the same as `k_attn` there. The tiled prefill
  attention also takes head widths 257-512 (`k_attn_tile8`, 8 query rows of
  8-key tiles: gemma-4's global layers), whose long-context rows on
  gemma-4-26B-A4B and the dense gemma-4-E2B read no further from the
  reference than `k_attn`'s (`RUNNER_CUDA_ATTN_TILE512=0` keeps `k_attn`).
  Head widths 256 to 512 take the tensor cores too where `k_attn_tcw` is
  on (R3.6.3.1, route detail `attn_wide=on`): `k_attn_tc`'s operands (Q and
  P in two halves, fp16 K and V, f32 accumulation) for 16 query rows a
  block, four warps splitting head_dim, the partial scores summed in warp
  order, one pass with a running maximum. On by default for gemma-4's
  E-series (gemma4 with per-layer embeddings: Nemotron 3.5 Bolt E4B,
  gemma-4-E2B and E4B), owner decision 2026-11-03, with its float32-reference
  row on Bolt E4B (2,000 positions: mean KL 0.9965x the tile's, Wilcoxon p
  0.188, McNemar 2/0); and for qwen35 at head width 256 (Qwen3.5, 3.6 and
  3.8), R3.6.3.2, on Qwen3.8-27B's row (1.82x prefill at 16K; 2,000
  positions: mean KL 0.9998x the tile's, Wilcoxon p 0.82, McNemar 0/0);
  and for deepseek2's absorbed MLA row through its twin `k_attn_tcw_mla`
  (R3.32): one 576-wide [k_pe 64 | latent 512] row that the 16 query heads
  share, S over the whole row, V the staged row's latent slice (the one
  table, so no V tile is staged), the 16 heads as a block's 16 rows (the
  packing of llama.cpp's mma_f16 <576, 512, 16>). On DeepSeek-V2-Lite-Chat
  Q6_K, prefill 1.05x the naive form's at 4K and 1.11x at 8K (1.76x and
  2.46x the previous path, `k_attn` one block per head and column); 1,999
  positions against the publisher's float32 model: mean KL 0.986x the
  previous path's, Wilcoxon p 0.79, McNemar 3/2. `RUNNER_CUDA_ATTN_TC_WIDE=0`
  keeps the tile there, `=1` forces it on any model it admits. fp16 caches only, and never a layer
  with sinks (gpt-oss, flint keep the tile). One global layer at about 31K
  on an RTX 3070: 53-57 ms against `k_attn_tile8`'s 352 ms; Bolt E4B prefill
  2.0x at 16K and 2.6x at 32K. Gate: `make test-cuda-attn-wide` (with the
  `--mla-wide` fixture at 512 tokens), `tests/test_attn_wide_repeat.py`.
  One opt-in stays off: `RUNNER_CUDA_ATTN_TC_FULL=1` puts the tensor-core
  attention on the layers without a window in a family the default keeps on
  the tile (gpt-oss's full-attention layers), whose gpt-oss-20b row read
  further from the reference than the tile's. 8,191-token prefill on the
  datacenter MIG slice: gemma-4-12B +36.5% and gemma-4-E4B +50.6% with the
  512-wide tile, Phi-3.5-mini +124.6%, Phi-4-mini +213.6% and Qwen3-30B-A3B
  +206% (one round) with the tensor-core attention; on an RTX 3070,
  gemma-4-E4B +41.8% and Phi-3.5-mini +135.6%. At about 3,500 tokens on the
  MIG slice: +21% to +121%, and +1.5% to +31% at about 700 (4 rounds).
- **Byte movers, on by default.** The one-copy KV gather (decode with full
  offload: every layer's new K and V rows gathered on the device, one copy
  back; since 2026-10-31 also a forward of up to 16 rows, such as a verify
  round, and each member of a decode microbatch;
  `RUNNER_CUDA_KV_GATHER=0` restores the per-layer copies) and the
  whole-tile recurrent prefill (Gated DeltaNet and Mamba-2 tiles in one
  launch per kernel; since 2026-10-21 a qwen35 decode step and each
  checkpointed verify row run the conv over one token and the per-token step
  with each head's columns over several blocks, `k_q35_delta1` and its norm,
  `RUNNER_CUDA_RECUR_SEQ=0` restores the per-token launches) move the same
  bytes to the same places or repeat the per-token arithmetic in order. Obligation: byte-identical. Gates: `GPU_ID_RECUR=1
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
- **MoE families on the tensor cores (moe-tc) and grouped (moe-grouped).**
  qwen3moe and gpt-oss run their dense projections through the tensor-core
  GEMM (and TC2 on 8.0+), and qwen3moe's prefill groups a tile's columns by
  expert and multiplies each expert's on the tensor cores
  (`gpu_moe_ffn_grouped`) when its experts are Q4_K or Q6_K, the types
  measured (Q8_0, Q5_K and Q4_0 experts take it on request). Both are
  routes: `moe-tc` recorded for those families, `moe-grouped` wherever the
  grouped prefill ran (a pin or `RUNNER_MOE_GROUPED=1` admits other MoE
  models), pinned by `--verify`, off under `reference`; `RUNNER_CUDA_TC` and `RUNNER_MOE_GROUPED` still set them when
  no route is pinned. Obligation:
  tolerance, no further from the publisher's float32 model than the route
  they replace (2,000 positions, mean KL within 1.01x, paired Wilcoxon and
  McNemar not worse at p < 0.05; the rows are in `tc_promoted`'s notes),
  scored chunked (`RUNNER_SCORE_CHUNKED=1`, with `RUNNER_SPEC_BATCH=64` as
  the chunk width) so the grouped path's batched expert GEMMs are what the
  rows measure; `RUNNER_SPEC_BATCH` alone scores solo and runs none of them.
  Gates: `test-tc-tol`, `test-moe-tol`.
- **What a record says about the tensor cores (R3.39.2).**
  - **The route detail's `tc=` token** names the route: `taken` where the default takes the tensor cores for one of
    the model's weight matrices, `off` where it does not, and `forced` under `RUNNER_CUDA_TC` (`moe-tc` for the MoE
    families).
  - **`profile.route.tc_types`** lists the weight types whose products the route takes on them. It is the policy fixed
    by the binary, the model and the switches, not a dispatch count, so a prefix-cache hit does not change it.
  - **`--verify` holds a replay to both.** A record written before R3.39.2 said `tc=promoted` for the mode, whatever
    ran, and has no `tc_types`. For such a record the token is read as not recorded, and the verdict says so; every
    other token still has to match.
- **MoE prefill by expert.** A prompt tile of more than 8 columns that
  averages at least four tokens an expert groups its (token, expert) pairs
  by expert (`k_moe_place`) and runs each expert's columns through the
  width-classed twins of the kernel the fused path uses for that weight
  (an expert slice: `k_moe_mv_<type>`'s body; gemma-4's dense shared MLP:
  the batch-1 dispatch), then `k_moe_sum_tile` adds each token's outputs in
  router order. On by default for gemma4 and gpt-oss, where it measured
  faster; `RUNNER_MOE_TWIN=1|0` overrides. Obligation: byte-identical to
  the per-token fused path. Gate: `GPU_ID_MOE_TWIN=1 ./test-gpu-identity`
  (launch count required), `make test-cuda-moe-twin`.
- **Graph replay.** One-token decode with full offload replays a captured
  graph of the same launches, qwen35's Gated DeltaNet steps included since
  2026-10-21 (`RUNNER_CUDA_GRAPH_RECUR=0` keeps that family eager).
  `RUNNER_CUDA_GRAPH_OFF=1` runs them eagerly, to separate a kernel from the
  graph path. See [buffer lifetimes](buffer-lifetimes.md)
  for when capture is refused.
- **Training primitives.** `gpu_mvcanon` (the forward matvec in the canonical
  order, F32, F16, Q8_0, Q4_0, Q4_K, Q5_K and Q6_K) is bit-exact to the host.
  The training context's `k_tmvt_*` (the transposed product, every weight
  type) is the CPU trainer's chain bit for bit, and its `k_tmm_*` (the
  forward product behind `--train-gpu`) is batch-invariant: a column's value
  does not depend on the columns beside it. Gates: `test-mvcanon`,
  `test-mvt`, `test-train-gpu`, the device-mode `test-lora-grad` runs.
- **KV cache formats.** q8 KV is tolerance-gated (`./test-kv-tol`); fp4 KV is
  held to the host codec (`test-kv-fp4`, `make test-cuda-kv-fp4`).

## Metal: `enc_mv_n`, then `enc_mv_cols`, in `src/metal.m`

| Route | Taken when | Obligation | Gate |
| --- | --- | --- | --- |
| identity matvec, multi-column | 2 to 16 columns | bit-exact to that many single-column dispatches | `test-batch-identity` |
| decode microbatch on the GEMM tiles (`enc_mb_mv`) | `RUNNER_METAL_BATCH_TILES=1` (opt-in), 6 to 8 microbatch columns, the tensor or simdgroup GEMM takes the weight (else the identity matvec) | tolerance; a slot's logits are not its solo decode's; D1 rows on K2-Horizon-1B: mean KL 1.006x (Q8_0) and 0.9996x (Q4_K_M) of the identity route's, not distinguishable; aggregate decode +20% / +40% at 8 slots on an Apple M1, slower below 6; named `micro=tiles` in the route detail | `tests/test_metal_batch_tiles.py` (0 flips against solo, the tiles ran, the 6-column threshold); D1 rows from `RUNNER_SCORE_MICRO=8` score files |
| Metal 4 tensor GEMM | more than one column, `RUNNER_METAL_TENSOR=1`, an M5 or newer GPU on macOS 26.2 or newer, the per-type self-test passed at load, `n_in` a multiple of 256 | tolerance; opt-in, a failing type falls back alone | the load-time self-test, then `./test-tc-tol` |
| simdgroup GEMM (`k_mm_*`) | more than one column, `RUNNER_METAL_MM` not `0` (promoted), `n_in` a whole number of k-steps and of superblocks | tolerance; activations staged scaled by the column maximum | `./test-tc-tol MODEL.gguf`, `make test-tc-overflow` on Darwin |
| identity matvec (`k_mv`) | otherwise | token-identical to the CPU scalar route | `test-gpu-identity`, `make test-metal-kquant` and the other `test-metal-*` text gates |
| fast decode matvec (`k_mvf_*`: Q4_0, Q8_0; Q4_K runs `k_mvg_q4_K` since R3.28.2, its microbatch columns walking the one-column arithmetic, `k_mvf_q4_K` under `RUNNER_METAL_MVG=0`) | decode columns (one token, or the microbatch's slots), kernel present; Q4_K by default on dense models since 2026-10-23 for inputs at least 2048 wide (R3.28.1: the fast kernel was 15% slower at 1536; narrower inputs keep identity; route detail `mv=mvg-q4_K>=2048+q6_K>=1536`; MoE models keep identity: their D1 rows do not exist), every type under `RUNNER_METAL_MV=1`, none under `=0` | tolerance; Q4_K promoted on D1 rows (mean KL 0.9993x / 0.9998x, decode +11-13% on the Llama 1B / 3B files), Q4_0 / Q8_0 neutral when measured and opt-in; never on prefill; each column is the one-column kernel's bits, so microbatch and verify rows equal solo decode | `./test-mv-tol MODEL.gguf`; `test-batch-identity` and `test-spec-identity` on `test-q4k.gguf` in `make test` |
| narrow-slice Q6_K matvec (`k_mvg_q6_K`, R3.28.3) | decode columns of a dense model's Q6_K tensor whose input is at least 1536 wide (every one under `RUNNER_METAL_MV=1`), not under `RUNNER_METAL_MVG=0` or `RUNNER_METAL_MV=0`; route detail `mv=mvg-q4_K>=2048+q6_K>=1536` | tolerance (a different summation order from `k_mv_q6_K`); each microbatch column runs the one-column arithmetic | `./test-mv-tol MODEL.gguf`; `test-batch-identity` and `test-spec-identity` on `test-q6k.gguf` in `make test`; its float32-reference row in R3.28.3 |
| two-row identity matvec (`k_mv2_q6_K`) | one column, a Q6_K tensor, not the fast route, `RUNNER_METAL_MV2` not `0`, the load-time self-test passed | bit-exact to `k_mv_q6_K` (the self-test runs both on pseudo-random blocks and refuses the kernel on a device where any bit differs) | the load-time self-test; the identity and `test-metal-*` gates through the default route |

Around the matvec:

- **Decode attention.** The cooperative KV read is the default and is
  tolerance-gated (`./test-attn-tol MODEL.gguf`); `RUNNER_METAL_ATTN_COOP=0`
  pins the byte-identical kernel. The grouped-query chunk kernel is
  bit-exact to the cooperative chunk kernel (`RUNNER_METAL_ATTN_GQA=0`), and
  the prefill tile kernel is bit-exact to the one-column kernel
  (`RUNNER_METAL_ATTN_TILE=0`). The tile takes heads up to 512 wide on
  gemma-4's E-series (R3.6.3.2, route detail `attn_wide=on`), the family
  CUDA's wide kernel defaults to; `RUNNER_METAL_ATTN_TILE_WIDE=0` keeps the
  one-column kernel for heads wider than 256, `=1` takes the tile on any
  model. Its obligation is unchanged (bit-exact), gated by
  `tests/test_metal_attn_tile.py` at 512-wide heads.
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

A tolerance route becomes a default only when it is no further from the
publisher's float32 model than the route it replaces: 2,000 positions, mean
KL within 1.01x, paired Wilcoxon and McNemar on the margin-qualified chosen
token not worse at p < 0.05, on rows of the families it reaches. The rule
holds as written for mixture-of-experts families too, where a handful of
positions at which the expert routing flips can decide the mean either way;
a MoE route that cannot pass stays opt-in.

A change that only reorders float32 sums (the same terms at the same
precision, such as the CPU K-quant dots keeping one accumulator per row in
1.5.0) is judged across files, not per file: it passes when no file moves
mean KL by more than 0.1%, none changes a margin-qualified chosen token,
and the paired verdicts have no consistent direction across the files
measured. At 2,000 positions the Wilcoxon test resolves such a reordering's
file-specific drift, which is not a fidelity change; for that change it read
closer on one file, further on one and level on two
([golden pass on 1.5.0](golden-pass-2026-10-22.md)). Those readings were solo
scores, which run only the single-column dots; on the batched route, where
AVX2's multi-column form runs, it reads level on Qwen3-4B Q4_K_M (mean KL
equal to six digits, Wilcoxon p 0.65;
[the chunked row](compat-reports/cpu-kquant-one-accumulator-chunked-2026-10-31/README.md)).
