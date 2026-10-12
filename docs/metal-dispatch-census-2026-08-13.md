# Metal dispatch and batch shapes

Prompt tokens, an execution tile, speculative verification rows, and
concurrent serving sequences are distinct shapes. A batch entry point must
preserve row/sequence/position association, logits ownership, layer layout,
and cache updates; an unsupported shape falls back or refuses explicitly.

`gpu_forward_native_batch` uses admitted matrix and attention operations.
Single-token decode, batch prefill, attached adapters, sparse experts, and
recurrent/unsupported layouts do not automatically share a dispatch route.
Inspect the actual selected path before attributing a performance change.

Existing phase profiling reports dispatches, matrix/attention work, and
submission/waits under its documented instrumentation conditions. GPU
event timing and wall time remain separate; profiling can perturb command
submission and synchronization. Use ordinary runs for request-latency
claims. Preserve command/buffer ordering and cleanup on partial failure.

Batch/scalar, decode-only scratch initialization, split/shared ownership,
admission/fallback, and numerical gates guard these interfaces. See
[performance routes](performance.md) and [CI verification](ci.md).
