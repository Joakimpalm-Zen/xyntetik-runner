# GPU matrix numerical contract

Tensor-core/matrix execution is admitted per backend, tensor type/layout,
shape, and architecture. Faster routes that reassociate or use reduced
precision carry established tolerances against an independent reference;
they do not inherit scalar byte-identity claims.

`tests/test_tc_tol.c` exercises the actual matrix route and compares its
logits with the pinned scalar operation. Missing hardware, unsupported
weights/shapes, or a route that never engages are scoped skips, not device
passes. Preserve the existing tolerance and decisive/top-margin checks,
with deterministic quantizer/scale/zero/tile/tail boundaries beside random
inputs. Do not widen the bar to admit a new implementation.

Exact CPU/GPU comparisons pin the certified scalar/eager paths. Current
backend/source capability decisions determine promotion and fallback; an
old model name alone is not an execution policy. Record compiler flags,
model/build hashes, route, shape, and device when measuring a candidate.
See [compatibility checks](../compatibility-program.md) and the
[backend table](../../MANUAL.md#placement-and-memory).
