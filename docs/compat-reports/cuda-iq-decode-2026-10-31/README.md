# Codebook i-quant decode on CUDA (R4.12.29), 2026-10-31

The plan item: granite-4.2-3b at IQ3_S decoded at 12% of llama.cpp on the RTX 3070 (the 1.0.0 measure,
2026-10-03), "not diagnosed". By main 7b5c948d that row had moved to 58% (the lane-level i-quant GEMVs of
2026-10-08 and since), and the slow rows were IQ2_XXS (28%) and IQ1_M (37%). The gate: decode byte-identical to
the current path, the row re-measured and published either way.

## What was slow

Nsight Systems with graph-node tracing put the batch-1 codebook GEMVs at 72 to 86% of decode GPU time
(`k_mv_iq2_xxs` 80 us a launch against 38 for `k_mv_iq3_s` on comparable shapes). Nsight Compute on one
`k_mv_iq2_xxs` launch: 38 GB/s of DRAM, the warps stalled on short scoreboard, and the hottest SASS were
`I2F.U64`: `iq_w8` converted each grid byte to float while it was still a 64-bit integer, and a 64-bit integer
conversion runs on a slow pipe. The IQ3 and IQ1 kernels convert 8-bit values (`I2F.U8`, `I2F.S8`) on the
quarter-rate pipe.

## The change (`src/iq_decode.h`, `src/kernels.cu`)

- `iq_u8f`: a byte as a float by placing it in the mantissa of 2^23 and subtracting 2^23. Exact for every byte,
  so every weight is the same float as before; two full-rate instructions instead of an I2F. `iq_w8` (IQ2_XXS,
  IQ2_XS, IQ2_S) and `iq_w4` (IQ3_XXS, IQ3_S) use it on the device; the host keeps the plain cast, and
  `tests/test_iq_decode.c` holds the two to the same values.
- `iq1_w` keeps its I2F: moving IQ1's conversion onto the FP32 pipe measured slower (IQ1_M 69.5 to 64.5 tok/s),
  because the IQ1 GEMV already keeps that pipe busy. Its remaining gap is structural (one lane a 256-element
  block, so a 2,560-wide row keeps 10 of 32 lanes busy); closing it changes the reduction order, which this
  item's gate does not allow.
- The batch-1 lane GEMVs unroll their block loop by four, so a warp has several blocks' dependent load chains
  in flight; each lane's accumulation order is unchanged.

The same helpers feed the width-classed twins, the MoE GEMVs, the tensor-core stagers (compute_75 and the
compute_80 TC2 GEMMs) and the training kernels, so prefill and the verify and microbatch paths gain too, with
unchanged bits. The PTX was regenerated with the pinned toolchain; against main, only the codebook kernels
differ beyond label and local-depot numbering.

## Re-measure (the Windows CUDA host, RTX 3070; method in `env.txt`, raw rows in `rows.jsonl`)

| file | prompt | prefill tok/s: main / this change / llama.cpp | decode tok/s: main / this change / llama.cpp | decode of llama.cpp: main / this change |
|---|---:|---|---|---|
| granite-4.2-3b IQ1_M | 512 | 2468 / 2513 / 4553 | 64.8 / 68.6 / 176.6 | 37% / 39% |
| granite-4.2-3b IQ2_XXS | 512 | 2138 / 2597 / 5608 | 48.4 / 104.4 / 173.5 | 28% / 60% |
| granite-4.2-3b IQ3_S | 512 | 1889 / 2011 / 5494 | 90.3 / 96.2 / 154.7 | 58% / 62% |
| granite-4.2-3b IQ3_XXS | 512 | 2198 / 2230 / 5473 | 89.4 / 101.7 / 160.7 | 56% / 63% |
| granite-4.2-3b IQ1_M | 2048 | 2101 / 2135 / 4459 | 61.2 / 64.6 / 165.3 | 37% / 39% |
| granite-4.2-3b IQ2_XXS | 2048 | 1859 / 2192 / 5418 | 46.4 / 95.7 / 162.4 | 29% / 59% |
| granite-4.2-3b IQ3_S | 2048 | 1666 / 1757 / 5324 | 83.4 / 88.4 / 145.9 | 57% / 61% |
| granite-4.2-3b IQ3_XXS | 2048 | 1902 / 1930 / 5315 | 82.6 / 93.4 / 151.9 | 54% / 61% |

Decode output is byte-identical to main on all four files (128 greedy tokens each).

## Checks

- `make test-iq-decode` (host): sign expansions, grid signs, IQ1 values, field reads and every half; the
  staging of the nine i-quant types, 506 blocks each, bitwise.
- `make test-cuda-iquants` with `test-tc-tol` built: 37 passed (the GPU against the CPU on every i-quant
  fixture, decode and tensor-core prefill).
- `make test-cuda-batch-identity` and `test-cuda-spec-identity` with llama-quantize: every fixture ok,
  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS among them.
