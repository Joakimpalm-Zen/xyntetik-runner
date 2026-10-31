# NVFP4 tensor-core prefill on CUDA (R4.9.5), 2026-10-31

Measured on the Windows CUDA host (RTX 3070, 8 GB). Model: qwen3.5-4b NVFP4 (single-level, llama.cpp's
`block_nvfp4`; NVFP4 79% of the weights, Q6_K 21%), every layer on the card, a 1,834-token prompt.

## The defect

NVFP4 had no batched GEMM on CUDA: no tensor-core kernel, no TC2 kernel and no tiled GEMM, so a prompt tile ran the
generic batched kernel a column tile at a time. `RUNNER_CUDA_PROFILE=1`: 99.7% of prefill in matvec, 20.5 s a
512-column tile. Prefill 22.4 tok/s against llama.cpp b11748's 3,669.

## The change

- `nvfp4_stage64` (`src/kq_stage.h`): one 64-element block to 64 values, built from their bits so they equal
  `ue4m3_to_fp32()` and `kvalues_mxfp4[]` exactly (entry 8 of the E2M1 table is +0, not -0).
  `make test-kq-stage` holds it to `dequant_row()` bit for bit on 2,000 random blocks and every UE4M3 scale byte.
- `tc_stage_nvfp4` and the kernels over it: `k_gemm_nvfp4_tc` (the TC route), `k_gemm2_nvfp4` (compute_75 TC2),
  `k_gemm2_80_nvfp4` and `k_gemm2_80w_nvfp4` (compute_80 TC2). Ampere has no FP4 tensor cores: this is the
  dequantizing route, fp16 operands and fp32 accumulation as for the other types.
- A two-level file's companion tensor scale (`X.scale`, bound by `gguf_bind_scale_companions`): the GEMM runs on the
  blocks and `enc_scale` multiplies its output, where the tensor has no bias to order the scale against.
- **Opt-in**: `RUNNER_CUDA_TC_NVFP4=1` turns the route on for NVFP4 tensors alone (the route line reads
  `tc=promoted+nvfp4`); `RUNNER_CUDA_TC=1` forces every type's tensor-core route, NVFP4 among them. Promotion waits
  for its float32-reference row (owner rule D1).

## Numbers

| route | prefill tok/s (1,834 tokens) | decode tok/s | output |
|---|---|---|---|
| default (scalar batched) | 22.4 | 58.9 | |
| `RUNNER_CUDA_TC_NVFP4=1` | 1,170 | 58.8 | identical (32 greedy tokens) |
| llama.cpp b11748 | 3,669 | 107.1 | |

## Checks

- `make test-kq-stage`: 7 formats, every element bit-equal to `dequant_row()`.
- `make test-cuda-tc-kq` now includes a 256-wide NVFP4 fixture with its companion tensor scale
  (`make-test-model.py --wide --quant nvfp4`): `test-tc-tol` ok on the TC2 kernels (compute_80), the compute_75 TC2
  kernels and the TC route, free-running token-identical.
- The PTX was regenerated with the pinned toolchain; against main the four NVFP4 kernels are new and nothing else
  differs beyond label and local-depot numbering.
