# CUDA codebook quantization kernels

CUDA has decode/prefill operations for the codebook i-quants admitted by the
loader. Mixed-type files do not fall back merely because they include IQ1,
IQ2, or IQ3. Architecture/layout, placement/budget, and actual tensor types
still determine admission; the backend must report its real fallback cause.

Device decoders use the frozen format grids and CPU definitions.
`tests/test_cuda_iq_grids.py` checks tables/sign expansion, and
`tests/test_iq_decode.c` executes host/device-shared decode primitives against
independent boundary values and the production reader. `tests/test_iquants.py`
uses 256-wide blocks and asserts actual stored types so an automatic
requantizer substitution cannot masquerade as i-quant coverage.

`RUNNER_REQUIRE_IQ_GATES` turns missing prerequisites into failures on required
runs; `RUNNER_IQ_FIXTURES` and `make test-cuda-iquants` select the intended
fixture/device checks. Failure injection retains the unavailable-kernel
refusal even when the normal type table has a kernel.

## Scoped device evidence

Measured 2026-09-14 on the Qwen3.8-27B GSQ-RCO IQ3_S artifact: nine prompts,
128 generated tokens, scalar pin, 40 of 64 layers on the datacenter GPU:
**9/9 exact**, with the split recorded in
[the CPU/CUDA report](cuda-iquants-evidence/cpu_cuda-qwen38-27b-gsq-rco-iq3s-blackwell.json).
On the RTX 3070, 28 of 64 layers and 32 generated tokens also read 9/9 exact;
[that report](cuda-iquants-evidence/cpu_cuda-qwen38-27b-gsq-rco-iq3s-rtx3070-32tok.json)
is a shorter smoke, not the full-length contract.

Neither row is a full-offload or new-build certification. Scalar identity,
tensor-core tolerances, publisher reference, and useful task behavior are
separate checks. Metal has its own codebook kernels and backend/layout
refusals. See [model evidence](model-evidence.md), the
[tensor-core contract](cuda-iq-tensorcore-2026-09-14.md), and the manual.
