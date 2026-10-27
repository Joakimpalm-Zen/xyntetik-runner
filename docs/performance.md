# Performance and execution routes

Use the [benchmark report](benchmarks.md) for measured speed comparisons.
Its model bytes, build, hardware, route, and date define each result; a
number from that report is not a throughput guarantee for another setup. What the 2026-10 performance work tried and did not ship, with the
number that stopped each idea, is in
[negative results of the 2026-10 work](negative-results-2026-10.md).

## Choosing and measuring a route

- `--info fit` estimates placement before load; `--info caps` describes this build
  and hardware, and `--info doctor` reports the loaded model. A requested GPU
  is not proof that every layer or expert executed there.
- NVIDIA support starts at **NVIDIA Turing / compute capability 7.5 or newer**.
  The embedded PTX target and the manual are checked against the source
  constants by `scripts/check-generated.py`.
- `--cpu-moe` keeps selected expert work on the host. Record the actual
  layer/expert split, CPU threads, context, KV type, and shared slot count.
- CUDA and Metal matrix/tensor-core paths have per-type/shape admission and
  numerical tolerances. CPU/GPU bit identity uses the certified scalar
  route, with `RUNNER_CUDA_TC=0`, `RUNNER_MOE_EAGER=1`, and the applicable
  `RUNNER_METAL_MM=0` identity pin. Do not infer one route's guarantee from
  a different route passing. [Kernel routes](kernel-routes.md) maps every
  decision point to its route, obligation and gate.
- CPU thread pools are shared by serving slots and park between work.
  `RUNNER_TPOOL_SPIN` bounds the brief spin before parking. Additional
  slots share immutable weights but still own request/KV/runtime state.
- Default scoring follows the scalar teacher-forced path.
  `RUNNER_SCORE_CHUNKED=1` opts into the batch route and its numerical
  contract. Compare output as well as time when changing execution shape.

The manual documents current environment switches and supported backend
formats. [Compatibility checks](compatibility-program.md) describe how to
pin and verify a route. Batch/scalar, MoE routing, attention, SIMD boundary,
and allocation-failure tests protect the implementation.

## Profiling

Use the existing CPU/Metal/CUDA phase profilers and the request's
`runner_telemetry`. CPU wall time and GPU event time are different
measurements. CUDA graph replay, profiling synchronization, and overlapping
stages can change how a number is obtained; do not sum overlapping events
as serial work or use an instrumented run as ordinary request latency.
Record compiler/flags, executable/model hashes, actual backend/device,
placement, batch width, and tuning variables beside every comparison.

Compare baseline and candidate on the same machine with the same prompt,
model, context, threads, seed, and route. Separate cold model/page caches,
warm loads, prefix hits, and uncached prefill. Alternate repeated runs and
investigate reproducible differences beyond the measured baseline spread.

## Drafting and batching

`--draft`, `--draft mtp`, `--draft mtp:FILE`, and `--draft lookup` preserve target
verification and accepted-token accounting. Their supported architecture,
backend, and cache-state combinations differ; the manual and capability
response describe the actual admission. Draft count, verification width,
prompt tile width, and concurrent request count are separate quantities.
Acceptance rate is not speedup: draft, verify, and replay/fold must cost
less than producing the same committed output without drafting.

[`context-drafts.md`](context-drafts.md) explains lookup drafts. Existing
MTP/draft-consumption, speculative-telemetry, batch-identity, rewind, and
rejected-round checks guard these paths. Recurrent checkpoints and KV
rollback must produce the same next target step after a rejected round.

## Idle cost and ownership

Measure idle CPU and retained/wired/device memory separately from active
throughput. `scripts/idle_coexistence.py` measures loaded idle,
post-inference idle, and post-unload, with request latency beside them.
On macOS it reports system wired-memory deltas on a quiet machine.
Mapped, reclaimable weight pages are not the same as `--mlock` or an owned
GPU allocation. Metal wraps mapped weights without copying them.

Explicit `/unload` clears shared prefixes/contexts and releases resident
resources at a safe boundary; a model swap preserves shared prefixes.
TTL, `keep_alive`, and cooperative VRAM yield retain their documented
support/refusals. Check reload, target/draft cleanup, adapter/template
identity, and VRAM-lease release when changing resource ownership.
[Buffer lifetimes](buffer-lifetimes.md) lists each buffer's owner, its
transfers and completion signal, and the CUDA graph constraints.
See the [idle measurement](idle-coexistence-120b-m5max-2026-09-01.md) and
[residency reference](../MANUAL.md#serving-and-apis).
