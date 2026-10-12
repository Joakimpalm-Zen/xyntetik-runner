# Metal decode matvec numerical contract

A faster decode matvec must retain the numerical obligation of the route
it replaces. CPU/device parity and a tolerance-gated matrix route are
different contracts; a reduction-tree change cannot borrow the latter's
allowance while advertising the former's identity.

`tests/test_mv_tol.c` checks the device operation against its reference,
including quantized weights and real model shapes. Whole-model identity,
attention, quantization, and batch gates cover how those results are used.
Preserve quantizer extremes, zero blocks, scales, SIMD/tile tails,
strides/layout, rounding, and accumulation order beside randomized cases.

Do not present a kernel speed ratio as request speedup. Measure the full
request with the actual route, model bytes, cache state, and configuration.
Backend admission, fallback, and scratch/buffer ownership remain explicit.
See [performance routes](performance.md) and the [compatibility method](compatibility-program.md).

This path remains the source/test reference for the matvec contract.
