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

## Typed decision cost

`/v1/decide` and `--decide FILE` sample no tokens, but they still run the
model to score the supplied options. The
[response contract](../MANUAL.md#typed-decisions-v1decide) describes the
returned probabilities. The execution contract in
[`src/decide.c`](../src/decide.c) and [`src/decide.h`](../src/decide.h)
deliberately fixes how the state and the option conditionals are fed:

- R10.3.3 (`5aed98eb`, 2026-10-02) seats the shared state alone from an empty
  KV. `canon_seat` keeps only the canonical state rows that match a
  question's prompt, then the question feeds its own remainder. The state
  is prefilled once per request while those rows remain reusable. A lost
  canonical prefix can require another fresh state feed; recurrent models,
  whose fold cannot be sliced, score each question cold.
- `score_mode` re-feeds the token before a scored conditional alone and
  advances through the option with one-token forwards. Sorted option
  traversal shares already computed conditionals for common token prefixes;
  it does not batch different option conditionals together.

CPU prefill is not batch-invariant on every host. Reusing whatever rows a
slot happened to retain used to change the remaining prefill's batch width
and the last digits of a score. The canonical state feed and solo
conditional steps make a score a function of the request on a fixed
model/backend/configuration, independent of the slot's previous requests
and question/option order. This is not a claim of bit identity between
different backends.

The historical TD-000 measurements recorded the cost as request-time totals:

| Workload | Recorded request time | Comparison |
|---|---|---|
| Slice 1 | 1.65 h on Runner 1.1.4 | 0.98 h on the 2026-09-22 build |
| Slice 2b | 4.0 h on the 4B | 3.3 h for Kvist-14B |

These workload totals are not current per-request latency guarantees. The
slice 2b comparison also changes the model, so it does not isolate the
overhead of this execution contract.

Questions sharing a state belong in one request's `questions` array.
[`scripts/decide-calibrate.py`](../scripts/decide-calibrate.py) groups equal
state strings by default (`--batch-state`), amortizing the state prefill
across questions. It leaves the solo conditional forwards and the
per-request restart. State rows are not reused across requests merely
because their state digest matches.

The retained policy is to keep this contract and document its cost.
Batching option conditionals waits for a backend whose batched step is
gated equal to the solo step for this scorer. Existing decoding gates on
CUDA or Metal do not establish that equality for decide; there is no such
evidence for R13.10. Sharing a state within a request is already supported
and does not require that future conditional-batching change.

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
