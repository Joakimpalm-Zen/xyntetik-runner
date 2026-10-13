# CUDA i-quant matrix precision and verification

The nine codebook tensor-core kernels stage fp16 operands and accumulate
in fp32. Activation columns are range-scaled before staging and the product
is rescaled in the epilogue, preventing finite fp32 activations from
becoming fp16 infinity. Keep maxima, scale companions, dimensions/strides,
padding, and trailing launch arguments consistent with the kernel contract.

`src/iq_decode.h` is compiled by the host compiler and nvcc. Its stagers
are checked bit for bit against the decoder, with quantizer extremes,
zero blocks, scale corners, tails, and independent hand-built blocks.
The frozen codebook tables are not regenerated from the implementation
being tested. Generated PTX/source/embedding checks remain mandatory.

## Actual route engagement

`test-tc-tol --types` reports the backend's kernel table. Forced-on checks
must show dispatches for each claimed type; forced-off checks must show
none. Missing device/model/shape support is not a tolerance pass. Exact
position counts, finite logits, the existing decisive/margin bar, and
rejected non-finite negative controls remain part of the gate.

`RUNNER_CUDA_TC=0` pins the scalar path. Forced and default matrix selection
have distinct purposes: a kernel existing does not automatically promote
it for every architecture/type/layout. The current backend selection table
and applicable shape/fallback guards define what runs.

## Retained evidence

The 94 recorded device cases (original commands, build/device/model
identities, per-type engagement, overflow checks, numerical comparisons and
performance configurations from the September measurement campaign) are
archived byte for byte in Shade under
`research/runner-engine/archive/runner-docs-2026-10-13c/`. They are evidence for those configurations,
not a new measurement of current hardware or a universal identity claim.
The wide fixtures and real-model numerical/performance cases guard different
properties; keep both where the route is affected.

Use [compatibility checks](compatibility-program.md),
[performance routes](performance.md), and the
[current backend table](../MANUAL.md#placement-and-memory) for operation.
