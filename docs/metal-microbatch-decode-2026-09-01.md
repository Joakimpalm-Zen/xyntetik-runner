# Metal microbatch decode — M5 Max, 2026-09-01

`--parallel` slots on Metal used to decode strictly one sequence at a time
(the FUTURE Phase 6 item). Now `gpu_batch_*` is implemented for Metal: N
slots share one weight sweep per decode step. The projections run as
multi-column identity matvecs — `enc_mv_cols`, the exact `k_mv` dot the
solo path runs per column, never the mm/tensor branches — so the win is
weight rows served from cache for columns 2..N while the arithmetic stays
the solo arithmetic. Rope, KV store and attention remain one dispatch per
column (each column has its own position and its own slot's KV buffers).
Unified memory removes the upload/copyback half CUDA needed.

**Contract: BIT-IDENTICAL to sequential decode** — the CUDA microbatch's
twin discipline, held by `test-batch-identity` in `make test`: three
sequences, different prompts and positions, decoded through both paths in
one process; every logit of every step byte-equal, with an engagement check
so a declined batch cannot pass vacuously, mutation-proven (an off-by-one
store position fails it).

Measured, Qwen3-8B Q4_K_M, 64 steps per arm, aggregate decode:

| slots | sequential | microbatch | speedup |
|---|---|---|---|
| 4 | 77.9 tok/s | **112.9 tok/s** | **1.45x** |
| 8 | 67.6 tok/s | **99.8 tok/s** | **1.47x** |

The same shape CUDA measured (1.62x): most of the shared-sweep win, capped
by the per-column attention/rope/store dispatches that stay sequential.
Default ON (`RUNNER_METAL_BATCH=0` disables). Scope mirrors CUDA's
admissions: dense gated transformers, full offload, uniform engine config
across slots; MoE, recurrent, output-gate and PLE families decline to the
sequential path, which remains correct for everything.

Found on the way and worth its own sentence: the lead slot's scratch is
sized by the largest batch any forward has used, and a slot that had only
ever decoded had one-column buffers — column 1's writes landed in whatever
allocation the heap placed next (x's column 1 aliased xb). The microbatch
now grows scratch on entry; the identity gate is what caught it.
