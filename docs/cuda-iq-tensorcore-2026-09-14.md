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

## Steps 1 and 2, the nine kernels

`k_gemm_iq*_tc` for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
IQ4_XS and IQ4_NL: the `TC_GEMM_BLK` shape (the Q8_0/Q4_0 macro
generalised over the block's element count, their PTX unchanged) with the
format's block (50, 56, 66, 74, 82, 98, 110, 136 bytes per 256 weights;
IQ4_NL's 18 bytes per 32 in the Q4_0 stager's two-block shape). The
64-row x 128-K fp16 weight tile is staged two 64-element segments per row
per K-step by the `*_stage64` functions in `src/iq_decode.h`; the same
source, compiled for the host, is held by `tests/test_iq_decode.c` to
`dequant_row()` bit for bit on 500 random blocks and the scale corners per
format (IQ1_M's planted nibble by nibble into its scattered half) and to
the format's definition on a hand-built block per format, one per row of
the plan's hazard table. Each landed red first: `IQ_TC_TYPES` in
`tests/test_iquants.py` claimed the format and the forced-TC leg failed
until the backend's kernel table carried it (#95, #97, #98).

### The precision contract, and the model that broke the first draft

fp16 operands, fp32 accumulation, the class the four promoted formats live
in. The first draft staged activations as the promoted kernels do,
`__float2half(x)`. The plan's warning ("FP16 staging can overflow even
when FP32 computation is finite") came true on the first architecture row
beyond granite: Phi-4-mini requantized to IQ3_S has |x| = 1.6e5 at its
first position in layer 30 (the FFN activation, the `ffn_down` input; the
CPU path's `RUNNER_DEBUG_ACT` measured it), past fp16's 65504, so the
staged operand was Inf and every logit of the forced-TC arm came out NaN.
The non-finite check the review asked for (Step 0) is what caught it; the
run was deterministic and memcheck, racecheck, synccheck and initcheck
were clean, which is how a range problem was told apart from a kernel
defect. The scalar path and the promoted formats on the Q4_K_M sibling
stay under the limit.

The nine kernels therefore take the `XSCALED` form of the macro: each
token column is staged as `x * (2^14 / max|x|)` and the product is scaled
back in the epilogue; `k_colabsmax` computes the per-column maxima into a
64-float buffer right before each tile's launch and the kernel receives
them as a trailing parameter (the NVFP4 companion's shape, so no shared
argument layout moves). Cost: one 64-block reduction per tile, about 2%
of prefill throughput on the RTX 3070. The promoted Q4_K/Q6_K/Q8_0/Q4_0
kernels kept their unscaled form in the promotion PR; the owner ordered
the same treatment for them the same day (Step 6 below).

### Gate rows (forced on against forced off, same binary and split)

Every row is `tests/test_tc_tol.c`: 64 teacher-forced positions, then
free-running greedy at batch 64 / ctx 4096. Every free-running arm below
is token-identical; every flip listed is a near-tie inside the gate's
margin.

| model | box | forced-on dispatches | of range | flips |
|---|---|---|---|---|
| llama-quantize fixtures, all nine formats (256-wide) | RTX 3070 and Blackwell slice | each claimed type (IQ3_XXS=16 IQ3_S=4 IQ2_S=8; IQ3_S=28; IQ2_XXS=24; IQ2_XS=24; IQ2_XS=20 IQ3_S=8; IQ1_S=20; IQ1_M=20; IQ4_XS=28; IQ4_NL=28) | 3e-5 to 7e-5 | 0 |
| granite-4.2-3b IQ3_S (241 IQ3_S, 40 Q4_K), all layers on the device | Blackwell slice / RTX 3070 | Q4_K=80 IQ3_S=480 | 9e-5 / 8e-5 | 0 / 0 |
| granite-4.2-3b IQ3_XXS recipe (120 IQ3_XXS, 80 IQ2_S, 41 IQ3_S) | Blackwell / RTX 3070 | Q4_K=80 IQ3_XXS=240 IQ3_S=80 IQ2_S=160 | 6e-5 / 6e-5 | 0 / 0 |
| granite-4.2-3b IQ2_S recipe (195 IQ2_XS, 46 IQ3_S) | Blackwell | Q4_K=80 IQ2_XS=390 IQ3_S=90 | 1.1e-4 | 1 |
| granite-4.2-3b IQ2_XS | Blackwell | Q4_K=80 IQ2_XS=470 | 1.2e-4 | 0 |
| granite-4.2-3b IQ2_XXS | Blackwell / RTX 3070 | Q4_K=80 IQ2_XXS=470 | 1.3e-4 / 1.3e-4 | 0 / 0 |
| granite-4.2-3b IQ1_S (195 IQ1_S, 40 IQ2_XXS) | Blackwell | Q4_K=80 IQ2_XXS=80 IQ1_S=390 | 9.0e-4 | 2 |
| granite-4.2-3b IQ1_M | Blackwell / RTX 3070 | Q4_K=80 IQ2_XXS=80 IQ1_M=390 | 3.4e-4 / 3.5e-4 | 0 / 1 |
| granite-4.2-3b IQ4_XS (236 IQ4_XS, 45 Q5_K) | Blackwell | IQ4_XS=470 | 4e-5 | 0 |
| granite-4.2-3b IQ4_NL | Blackwell | IQ4_NL=470 | 4e-5 | 0 |
| Phi-4-mini IQ3_S (the overflow model) | Blackwell / RTX 3070 | IQ3_S=448 | 2.4e-4 / 2.5e-4 | 2 / 3 |
| Llama-3.2-3B IQ3_S | Blackwell | IQ3_S=392 | 9e-5 | 0 |
| Qwen3-4B IQ3_S | Blackwell | Q4_K=72 IQ3_S=432 | 2.4e-4 | 0 |
| SmolLM2-1.7B IQ3_S | Blackwell | IQ3_S=336 | 3e-5 | 0 |
| Mistral-7B IQ3_S | Blackwell | Q4_K=64 IQ3_S=384 | 4e-5 | 0 |
| gemma-3-4b IQ3_S | Blackwell | IQ3_S=476 | 6e-5 | 0 |
| gemma-4-E4B IQ3_S | Blackwell | Q4_K=48 IQ3_S=636 | 9e-5 | 0 |
| Llama-3.2-3B IQ2_XXS | Blackwell | IQ2_XXS=330 | 4e-5 | 0 |
| Qwen3-4B IQ2_XXS | Blackwell | Q4_K=72 IQ2_XXS=424 | 1.0e-4 | 1 |
| SmolLM2-1.7B IQ2_XXS | Blackwell | IQ2_XXS=282 | 1.0e-4 | 0 |
| Qwen3.8-27B GSQ-RCO (qwen35, the motivating file), 40 of 64 layers on the slice | Blackwell | resident IQ tensors | greedy at 64 tokens: 9 of 9 prompts identical | |

The IQ1 rows are the family's largest deviations, an order of magnitude
under the bound: the 1-bit formats' logits are the most sensitive to the
fp16 rounding of `scale * (grid + delta)`. The other architectures'
models were requantized from their Q4_K_M files (`--allow-requantize`,
imatrix from the runner's docs for IQ2_XXS), which says nothing about
their quality and everything the gate needs about the two paths.

Sanitizers: `compute-sanitizer` 13.3 on the RTX 3070, the whole
`test-tc-tol` run of each of the nine fixtures under memcheck, racecheck,
synccheck and initcheck: 0 errors, 0 hazards, with the claimed types
dispatching in each run.

### Prefill throughput (RTX 3070, 5 warmed runs, median, spread under 1%, `-n 1`)

| model, all layers on the device | 118 tokens | 475 tokens | 1892 tokens |
|---|---|---|---|
| granite-4.2-3b IQ3_S | 24.5 to 485.7 tok/s (20x) | 26.1 to 451.5 (17x) | 25.8 to 371.0 (14x) |
| granite-4.2-3b IQ3_XXS recipe | 24.4 to 452.9 (19x) | 26.1 to 428.7 (16x) | 25.8 to 354.9 (14x) |
| granite-4.2-3b IQ2_XXS | 25.1 to 320.3 (13x) | 26.9 to 318.7 (12x) | 26.6 to 273.2 (10x) |
| granite-4.2-3b IQ1_M | 24.6 to 329.9 (13x) | 26.3 to 330.4 (13x) | 26.0 to 281.4 (11x) |
| Phi-4-mini IQ3_S | 23.1 to 389.2 (17x) | 24.7 to 369.1 (15x) | 24.9 to 321.0 (13x) |

The scalar column is the generic warp-per-row matvec applied to 64
columns, which is why the gap is an order of magnitude larger than the
promoted formats' rows: the codebook formats had no batched GEMM at all.
The Blackwell slice was shared with another job's loads throughout this
program (one bench run was refused VRAM outright, spreads reached 50%), so
its throughput rows are not quoted; its gate rows are unaffected by
contention.

### Promotion

Promoted 2026-09-14: the nine formats join Q4_K/Q6_K/Q8_0/Q4_0 in
`tc_promoted()` for the same architecture list (llama, phi3, gemma4,
qwen3, qwen35, mistral, gemma3, smollm, granite), each architecture with
at least one row above and granite with all nine. The plan's gate (at
least 10% median improvement over five warmed runs above noise, no
regression over 5% on the workloads assigned to the path) is met by an
order of magnitude on both device families; single-token decode is not on
this path and is unchanged. `RUNNER_CUDA_TC=0` pins the scalar path as
before, and every CPU-versus-CUDA identity gate sets it.

Not claimed: a llama.cpp column on the tensor-core path (the CPU decoders
are unchanged and keep their anchor); Metal kernels; anything about the
IQ formats' own quality (the requantized models exist to compare two
paths of the same file).

Evidence files: `cuda-iq-tensorcore-evidence/`: the gate logs of every
row above (`tc-tol-*`), the fixture racecheck logs, the RTX 3070 benches
(`pp-bench-*`), the GSQ-RCO greedy comparison.

## Step 6, the promoted Q4_K/Q6_K/Q8_0/Q4_0 kernels scaled the same way

`k_gemm_q4_K_tc` and `k_gemm_q6_K_tc` (the hand-written aliasing kernels)
gain the trailing `xsc` parameter, the scaled staging and the epilogue
multiply; `k_gemm_q8_0_tc` and `k_gemm_q4_0_tc` take the macro's XSCALED
form. Every tensor-core GEMM dispatch now goes through the scaled launch
(the unscaled `launch_tiled` path serves only the fp32 tiled GEMM). PTX
from CUDA 13.3 on the Windows box: exactly the four kernels changed.

The regression instrument is a fixture, `make-test-model.py --wide --quant
q8_0 --act-fp16-overflow`: gate weights 4e4x and up weights 4e2x drive the
FFN activation (the `ffn_down` input) to 1.2e7, finite in fp32 and far
past fp16. `make test-tc-overflow` (in `make test`; it skipped on macOS
until 2026-09-17, when the Metal tiled path gained the same per-column
scaling and the gate started running there too) runs the forced-TC gate on it and
requires `Q8_0` dispatches and a pass: Q8_0=28 dispatches, 6e-5 of range, 0
flips, free-running identical on both CUDA boxes. The same fixture on the
previous binary returns NaN logits (the row below).

### Re-gate of the four formats on the scaled kernels

Same protocol as above; every free-running arm token-identical.

| model (all layers on the device) | box | forced-on dispatches | of range | flips |
|---|---|---|---|---|
| Llama-3.2-3B Q4_K_M | Blackwell slice | Q4_K=336 Q6_K=56 | 3e-5 | 0 |
| Qwen3-4B Q4_K_M | Blackwell | Q4_K=432 Q6_K=72 | 4e-5 | 0 |
| SmolLM2-1.7B Q4_K_M | Blackwell | Q4_K=288 Q6_K=48 | 5e-5 | 1 (near-tie) |
| Mistral-7B Q4_K_M | Blackwell | Q4_K=384 Q6_K=64 | 2e-5 | 0 |
| Phi-4-mini Q4_K_M | Blackwell | Q4_K=224 Q6_K=32 | 2e-5 | 0 |
| granite-4.2-3b Q4_K_M | Blackwell | Q4_K=480 Q6_K=80 | 5e-5 | 0 |
| gemma-3-4b Q4_K_M | Blackwell / RTX 3070 | Q4_K=408 Q6_K=68 / Q4_K=410 Q6_K=66 | 6e-5 / 3e-5 | 0 / 0 |
| gemma-4-E4B Q4_K_M | Blackwell / RTX 3070 | Q4_K=452 Q6_K=64 | 5e-5 / 4e-5 | 0 / 0 |
| Qwen3.8-4B Q4_K_M (qwen35) | Blackwell | Q4_K=432 Q6_K=64 | 1e-5 | 0 |
| granite-4.2-8b Q4_K_M | RTX 3070 | Q4_K=480 Q6_K=80 | 5e-5 | 0 |
| Llama-3.1-8B Q4_K_M | RTX 3070 | Q4_K=384 Q6_K=64 | 3e-5 | 0 |
| SmolLM2-135M Q4_K_M | RTX 3070 | Q8_0=28 Q4_K=32 Q6_K=28 | 5e-5 | 0 |
| Phi-4-mini Q8_0 | Blackwell | Q8_0=448 | 2e-5 | 0 |
| granite-4.2-8b Q8_0 | Blackwell | Q8_0=560 | 5e-5 | 0 |
| SmolLM2-135M Q8_0 | Blackwell / RTX 3070 | Q8_0=420 | 6e-5 / 6e-5 | 0 / 0 |
| Qwen3-8B Q8_0 | Blackwell | Q8_0=504 | 3e-5 | 0 |
| qwen3-0.6b Q8_0 | RTX 3070 | Q8_0=392 | 5e-5 | 0 |
| granite-4.1-3b Q8_0 | RTX 3070 | Q8_0=560 | 4e-5 | 0 |
| Phi-4-mini Q4_0 | Blackwell | Q4_0=448 | 3e-5 | 0 |
| Hermes-4-14B Q4_0 (qwen3) | Blackwell | Q4_0=550 | 4e-5 | 0 |
| SmolLM2-135M Q4_0 | RTX 3070 | Q4_0=420 | 5e-5 | 0 |
| granite-4.1-8b Q4_0 | RTX 3070 | Q4_0=560 | 4e-5 | 0 |

The rows sit where the promotion table's historical rows sat (3e-5 to
8e-5 of range, 0 flips): the scaling moves the operands' rounding, not
their class. Sanitizers on the RTX 3070: memcheck, racecheck, synccheck
and initcheck on the overflow fixture's whole gate run; memcheck,
racecheck and synccheck on one forced-TC prefill each of SmolLM2-135M
Q4_K_M (Q4_K and Q6_K kernels), Q4_0 and Q8_0: clean. The
previous binary on the overflow fixture with the tensor cores forced:
`<unk><unk><unk>` (NaN logits, token 0); with them off, and the new binary
either way: the same finite text.

### Cost (RTX 3070, 5 warmed runs, median, forced on, scaled against the previous binary)

| model, all layers on the device | 118 tokens | 475 tokens | 1892 tokens |
|---|---|---|---|
| granite-4.1-3b Q8_0 | 534.7 to 531.8 tok/s (-0.5%) | 509.6 to 503.8 (-1.1%) | 405.4 to 400.9 (-1.1%) |
| granite-4.2-8b Q4_K_M | 258.7 to 257.0 (-0.7%) | 253.4 to 251.8 (-0.6%) | 215.6 to 213.7 (-0.9%) |
| granite-4.1-8b Q4_0 | 181.8 to 178.8 (-1.7%) | 177.8 to 176.5 (-0.7%) | 161.2 to 160.4 (-0.5%) |

Under 2% everywhere, the column-max reduction per tile; the plan's 5%
regression bound holds. Every CUDA tensor-core GEMM now stages its
operands inside fp16's range. Metal's tiled GEMM (`k_mm_*`, `tg_x` is
half) stayed unscaled and carried the exposure on Apple silicon until
2026-09-17: `k_colabsmax` in `kernels.metal` and the scaled `MM_BODY`
(the same 2^14 / max|x| arrangement, scaled back in the epilogue) closed
it, and `make test-tc-overflow` runs on Darwin since, where the old
kernels produced 16,576 non-finite logits on the fixture and the scaled
ones pass. The default Metal MoE tiled GEMM (`MOE_MM_BODY`) stages in
float and was never exposed; its opt-in half-staged twin (`MOE_MMH_BODY`,
`RUNNER_METAL_MOE_MM=half`) was, and is scaled the same way since the same
day (gate: `make test-metal-moe-mm` on the `--act-fp16-overflow` moe1
fixture with Q8_0 experts, where the unscaled twin read `<unk>` for every
token). The Metal 4 tensor path (`kernels_tensor.metal`, opt-in) stages
its own operands.
