# Metal 4 tensor-op GEMM: first M5 measurement, and what it does to the premise

Measured 2026-08-31 on an Apple M5 Max (18 cores, 128 GB unified, macOS 26.5
build 25F71) — the first M5-class host this project has had. This is the
"short, pre-defined measurement" that
`docs/specs/2026-08-12-metal4-tensor-ops-scope.md` was written to enable, and
it resolves item 2 of that spec's "What cannot be validated without M5
hardware" list (the real prefill number on this codebase). It does **not**
resolve items 1, 3, 4, 5 or 6, which need the kernel itself.

Binary: `runner` 0.4.4 from `main` at `d14a51c`, Metal shader sha256
`a0a923b9...70cd0`.

## Hardware preconditions: both now satisfied

| precondition | required | this host |
|---|---|---|
| M5-class silicon (Neural Accelerator) | M5+ engages it at all | Apple M5 Max |
| macOS for MPP `matmul2d` | 26.0 for the API | 26.5 |
| functional bf16 in MPP | **26.2+** (26.0 bug, `ollama` #13460/#15862) | 26.5 — clears it |

The spec's blocking hardware condition is gone. Nothing below required the
tensor path to exist; it is all baseline.

## The measurement: the runner's current `k_mm_*` prefill on M5

`gemma-3-4b-it-Q4_K_M.gguf` (2.32 GB, dense, sha256 verified `882e8d2d...`) —
Q4_K chosen because the spec names it as the dense-arch promoted default on
both CUDA TC and Metal MM, and 4B-class to match the trigger's model size.
`--temp 0`, `-n 1`, prefill tok/s as reported by the runner:

| prompt tokens | prefill tok/s |
|---:|---:|
| 129 | 488.81 |
| 513 | 981.81 |
| 1025 | **1062.81** |
| 2049 | 1059.64 |

Prefill saturates just above **1,060 tok/s**. For contrast, the same host on
`gpt-oss-20b-MXFP4` (20B MoE) saturates at 216 tok/s prefill and 115 tok/s
decode; `gemma-3-4b` decode is 109.76 tok/s.

## What this does to the spec's premise

The spec's trigger was an external evaluation on an M5 Max: llama.cpp prefill
**~7,500 tok/s** on a 4.65B-class model against the runner's **~575 tok/s** in
the same comparison — a ~13x gap, large enough that "tune the existing kernel
harder" was judged not credible.

The runner half of that comparison does not reproduce here. On M5 Max with a
4B-class dense Q4_K model, the current `simdgroup_float8x8` kernel measures
**1,060 tok/s — 1.84x the 575 the spec assumed.**

That does not overturn the spec's recommendation, and it must not be read as
one. What it changes:

- **The gap is ~7.1x, not ~13x**, if the 7,500 figure holds on this host.
- **The ceiling arithmetic should be recomputed from 1,060, not 575.** The
  spec's honest-prediction band (1.2-4x prefill) was reasoned against the lower
  baseline; against the measured one, the same absolute target implies a
  smaller multiple.
- **The "not credible by tuning alone" argument weakens by the same factor**
  and should be restated against the measured baseline before it is relied on.

## What this measurement cannot settle, and why no go/no-go is recorded here

A verdict is deliberately **not** written into this document. Two links of the
chain are missing, and per AGENTS.md a gate without an external anchor is
evidence of self-consistency and nothing more:

1. **The 7,500 tok/s denominator was never measured on this box.** It is
   another project's number on different weights, quant unstated. The honest
   comparison is llama.cpp built and run here on this same file, and that has
   not been done. Comparing a measured numerator to an imported denominator is
   exactly the error the spec warns against in its own trigger section.
2. **`RUNNER_METAL_TENSOR` does not exist.** The spec's prescribed first action
   is a single `matmul2d` admission-test spike against one quant type with the
   flag wired to `gpu_tc_force()`-style plumbing and `test_tc_tol.c` extended
   to drive it. No tensor-path number can be produced until that lands, so the
   "is it worth it" question remains open by construction.

## Recommendation

**Proceed to the spike, with the baseline corrected.** The hardware gate the
spec was waiting on is satisfied, so its own next action is now unblocked. Two
amendments before it starts:

- Re-derive the promotion bar from a **1,060 tok/s** baseline.
- Measure llama.cpp on this host and this file first. It is cheap next to
  writing a tensor kernel, and it is the only way the 7,500 figure becomes an
  anchor rather than a quotation.

The LM Studio failure mode the spec cites (a shipped Metal 4 self-test that
failed on real M5 Macs and silently fell back, costing 2-3x prefill with only
a log line) remains the governing design constraint for the admission test:
fail loud, at load, defaulting to the known-good path.

## Same-host llama.cpp anchor (completed later 2026-08-31)

The missing denominator is now measured on this host and exact GGUF. Official
llama.cpp commit `010be9683afabe14ce299197b38c329f94bae568` was built Release,
native arm64, Metal enabled, with AppleClang 21.0.0. Five repetitions through
`llama-bench`, full GPU offload, batch 2048 / ubatch 512:

| prompt tokens | tensor API tok/s | tensor disabled tok/s | tensor speedup |
|---:|---:|---:|---:|
| 128 | 3,066.21 | 1,784.18 | 1.72x |
| 512 | 5,797.53 | 2,028.69 | 2.86x |
| 1,024 | 5,660.47 | 2,003.44 | 2.83x |
| 2,048 | 5,419.74 | 1,937.17 | 2.80x |

The enabled run's device probe reports `has tensor = true`; the disabled arm
uses llama.cpp's explicit `GGML_METAL_TENSOR_DISABLE=1` control and reports
`has tensor = false`. This is a behavioral A/B of the same binary, model, and
host—not a comparison across projects or artifacts.

The tensor path is a real ~2.8x prefill lever at the saturated prompt sizes.
Runner's measured ~1,060 tok/s is therefore about 5.1x behind the tensor-enabled
reference, not the provisional 7.1x inferred from the unmeasured 7,500 figure,
and not the original ~13x trigger. The admission spike remains justified: the
reference proves both that MPP works on this machine and that disabling it
removes most of the reference's advantage over its own simdgroup path.
