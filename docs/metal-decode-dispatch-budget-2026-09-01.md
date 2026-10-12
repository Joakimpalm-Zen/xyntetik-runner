# Metal decode profiling and resource lifetimes

Profile the route that actually ran, including matrix/attention operations,
host expert work, command encoding/submission, and waits. Separate GPU
event time from CPU wall time and account for overlap. A requested fused
route is not proof that every model property was admitted to it.

Fusion and attached-adapter routes retain their own admission and fallback
rules. Weight buffers wrap mapped storage without copying; activation, KV,
and scratch storage have different owners. Keep captured addresses,
capacities, aliasing, submission completion, and cancellation/unload order
explicit. A buffer cannot be reused or freed while a command still owns it.

Measure idle CPU and wired/retained memory separately from decode speed.
The profiler must not add ordinary-request synchronization or allocations
when disabled. Existing ownership/failure, decode-only, batch, adapter,
and numerical gates protect these boundaries. See [performance routes](performance.md)
and [the idle measurement](idle-coexistence-120b-m5max-2026-09-01.md).
