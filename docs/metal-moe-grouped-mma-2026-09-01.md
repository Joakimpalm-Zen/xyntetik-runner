# Grouped-MMA MoE prefill: the win, and the instrument that judges it — M5 Max, 2026-09-01

Round 2 of the MoE prefill grouping arc
(`docs/negative-result-metal-moe-expert-major.md` is round 1, which killed
the cache- and register-reuse shapes). This round builds the shape that
carries llama.cpp's advantage — per-expert simdgroup-MMA matmul tiles — and
it wins big. It is NOT promoted, for a reason worth stating precisely.

## Design

`RUNNER_METAL_MOE_MM=1`. A one-threadgroup kernel (`k_moe_group`) counting-
sorts the batch's (token, expert) slots by expert on-GPU; the expert matmuls
(`k_moe_mm_q8_0`, `k_moe_mm_mxfp4` so far) then run the dense `k_mm` tile
structure — same MM_TM/MM_TN/MM_TK — but with FLOAT-staged operands (see
below for why that is nearly free on this hardware), the x tile *gathered*
through the group's slot map and the outputs scattered back. No physical
repack of x exists anywhere. Column tiles beyond an expert's count exit on
two int reads; types without a `k_moe_mm` kernel stay on the matvec path
per-tensor; decode (slots below the engagement threshold) is untouched.
Column order within a group comes from atomic appends and is deliberately
allowed to vary: every MMA output column is an independent dot in fixed
k-order, so a slot's value does not depend on where it lands in the tile,
and outputs stay bit-stable across runs.

## Measured: the largest Metal prefill win this project has recorded

Interleaved warm A/B, 714-token prompt, batch 512, three reps per arm:

| model | matvec path | grouped-MMA | delta |
|---|---|---|---|
| Qwen3-30B-A3B Q8_0 | 168–172 tok/s | **225–232 tok/s** | **+33%** |
| gpt-oss-120b MXFP4 | 73–78 tok/s (warm) | **91–96 tok/s** | **+24%** |

Decode unchanged. Both clear round 1's recorded floor (+10%) severalfold.

## The dense identity bound fails, and the harnesses say why

`test-gpu-identity` (CPU vs GPU, the bound calibrated on dense models):
~0.0044 of the mean logit range on BOTH models, against the 0.002 limit —
and, decisively, **staging precision barely moves it**. The same session
rebuilt the kernels with FLOAT-staged operands (an Apple-specific economy:
fp16 simdgroup matmul runs only ~1.1x fp32 on M-series, per the community
metal-benchmarks — the half staging was an NVIDIA-shaped assumption), which
removes operand rounding entirely, leaving only k-tile summation order. The
aggregate barely changed. So the excess was never the staging: it is the
routing.

Two purpose-built harnesses now hold the account:

**Routing side — `scripts/moe-mm-flips.py`** (two traced prefills over the
same prompt, matvec vs grouped-MMA, RUNNER_MOE_TRACE carrying per-record
selected experts + full router logits; unit-gated on hand-computed margins).
Qwen3-30B, 34,320 routing records:

| | f32-staged | half-staged |
|---|---|---|
| top-k set flips | 1,566 (4.56%) | 1,606 (4.68%) |
| router logit mean \|Δ\| | 0.0062 | 0.0073 |
| router logit max \|Δ\| | 0.90 | 4.40 |
| flip margin, median | 0.0091 | 0.0090 |
| flip margin, max | 0.26 | 0.45 |

The flip rate is staging-invariant: flips initiate where the boundary margin
(median 0.057, p10 0.0085) sits below the accumulated reassociation
perturbation (~0.006), then CASCADE — a flipped token's stream diverges, so
deep-layer records compare two different computations, which is where the
wide-margin tail lives. Conclusion: **no kernel precision fix can pass the
dense bound for top-k MoE under any reassociating prefill.** The bound is
the wrong instrument for sparse routing — the same reason the 235B fails it
at 0.0033 with scalar kernels and no grouped path at all.

**Logit side — `test-moe-mm-ab`** (GPU-matvec vs GPU-grouped-MMA in one
process, teacher-forced, scored on the project's own dual-column fidelity
bar: margin-qualified top-1 >= 97% AND mean KLD <= 0.05, the
`kld-compare-raw.py` v3 definitions every published artifact answers to):

| model / arm | top-1 agree | margin-qualified | mean KLD | worst position |
|---|---|---|---|---|
| 30B, f32-staged | 24/24 | 100% | **0.00001** | 0.00007 |
| 30B, half-staged | 24/24 | 100% | 0.00015 | 0.00331 |
| 120B, f32-staged | 24/24 | 100% | **0.00987** | 0.06236 |
| 120B, half-staged | 24/24 | 100% | 0.01506 | 0.11019 |

**Every arm passes the house bar, most by orders of magnitude, with not one
top-1 flip in 96 scored positions.** The route-flip literature explains why
the two instruments disagree: flip damage is measured to be nearly
symmetric (harmful and beneficial flips cancel to ~0.004 nats) and
undetectable from router statistics, so flips inflate a logit-delta metric
while leaving the distribution — the thing the fidelity bar measures — at
the model's own noise floor.

## Where the f32 staging leaves performance

Warm interleaved, same protocol as above: 30B 170 → **225 tok/s (+31%)**,
120B ~72 → **87 tok/s (+21%)** — two to three points under the half-staged
figures, buying the 1.5-15x cleaner numerics above. On this backend that is
the right trade, and f32 staging is the default arm of RUNNER_METAL_MOE_MM
(`=half` keeps the comparison measurable).

## Standing, and the promotion question as it now stands

Opt-in, default off, with the complete three-instrument account on file:

1. performance: +31% / +21% prefill, decode untouched;
2. the dense identity bound: FAIL at ~0.0044, shown flip-dominated and
   staging-invariant — an instrument mismatch, not a kernel defect
   (mutation gates, fixture-scale byte identity through every tile
   boundary, and the f32-staging null all bracket the kernel itself);
3. the house fidelity bar, held mv-vs-mm: PASS on every arm, 100%
   margin-qualified top-1, KLD 5x-5000x inside the bar.

What promotion needed was one owner decision, not more engineering:
ratify that sparse-routing models hold reassociating prefill to the house
fidelity bar (`test-moe-mm-ab`, in `make test` at fixture scale and runnable
against any real model) rather than to the dense logit bound — the
routing-aware floor the 235B scalar FAIL already argued for, with its
measuring instrument built, unit-anchored, and applied.

**RATIFIED — owner, 2026-09-01, same day.** Grouped-MMA MoE prefill is the
DEFAULT (`RUNNER_METAL_MOE_MM=0` restores the matvec path and is pinned by
every byte-identity gate; `=half` keeps the comparison arm), and `k_moe_mm`
extends to the rest of the roster — q2_K, q3_K, q4_0, q4_K, q6_K, f16, f32
join q8_0 and mxfp4, each chunk its dense twin verbatim (q5_K has no dense
mm chunk and stays on the matvec path per-tensor). The 235B measurement
under the new default is recorded below.

**Qwen3-235B-A22B Q2_K (merged), first run on the extended roster:**
prefill 23.4 → **32.3 tok/s (+38%)** — the largest relative win of the
three models — decode untouched, and the house bar held with the cleanest
margin yet: 24/24 top-1 agreement, 100% margin-qualified, mean KLD
**0.00026** (190x inside the bar), worst position 0.00498. Against the
same-host llama.cpp `010be968` denominator the warm 120B non-tensor
prefill gap narrows to roughly 8.7x (752.65 vs ~87) from the 9.8x this
arc opened at — the remaining distance is the next arc's question.
