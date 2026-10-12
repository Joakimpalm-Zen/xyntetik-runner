# CUDA batch/scalar numerical contracts

Batch and scalar paths can prepare activations and reduce sums differently.
Exact equivalence belongs to a named model/type/shape/build/route; a batch
speed improvement does not establish it automatically.

Keep current scalar/eager identity pins separate from tensor-core,
fused-router, attention, quantized-KV, and other tolerance-gated routes.
Activation representation, layout/stride, padding, aliasing, tail handling,
and scratch lifetime are inputs to kernel selection. Unsupported layouts
must take the existing fallback or refusal rather than an unimplemented
batch operation.

`tests/test_batch.c`, `tests/test_tc_tol.c`, `tests/test_gpu_identity.c`,
attention/MoE tolerance gates, and current kernel-verification tools cover
these paths. Include odd verification widths and boundaries around tiles;
prove the intended device kernel ran rather than comparing two fallbacks.
Preserve stream/event ordering, graph-capture addresses/shapes, and shared
weight ownership. See [performance routes](performance.md) and
[compatibility verification](compatibility-program.md).
