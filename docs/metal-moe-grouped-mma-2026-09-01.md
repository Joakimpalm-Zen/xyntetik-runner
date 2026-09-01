# Grouped-MMA MoE prefill: the win, and what blocks its promotion — M5 Max, 2026-09-01

Round 2 of the MoE prefill grouping arc
(`docs/negative-result-metal-moe-expert-major.md` is round 1, which killed
the cache- and register-reuse shapes). This round builds the shape that
carries llama.cpp's advantage — per-expert simdgroup-MMA matmul tiles — and
it wins big. It is NOT promoted, for a reason worth stating precisely.

## Design

`RUNNER_METAL_MOE_MM=1`. A one-threadgroup kernel (`k_moe_group`) counting-
sorts the batch's (token, expert) slots by expert on-GPU; the expert matmuls
(`k_moe_mm_q8_0`, `k_moe_mm_mxfp4` so far) then run the dense `k_mm` tile
structure — same MM_TM/MM_TN/MM_TK, same half-staged operands, same
mixed-precision `simdgroup_multiply_accumulate` — with the x tile *gathered*
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

## Why it is not promoted: the identity bound, and what the excess is

`test-gpu-identity` (CPU vs GPU, whole model, 24 teacher-forced positions):

| model | default path | grouped-MMA | limit |
|---|---|---|---|
| Qwen3-30B-A3B | 0.00109 | **0.00443** (worst 3.36) | 0.002 |
| gpt-oss-120b | 0.000215 | **0.00442** (worst 2.07) | 0.002 |

Both FAIL the dense-calibrated bound. What the excess is NOT, established by
isolation: not the down projection's post-activation dynamic range (mv-only
down still measures 0.00427); not multi-k-tile, multi-row-tile, or
multi-col-tile arithmetic (a widened fixture — 64-wide embed, 160-wide FF,
700-token prompt — stays byte-identical to the matvec path through every
tile boundary); not a scale or routing-map error (the mutation gates catch
both). What remains is the compound of two known mechanisms: legitimate
half-staging rounding — the same class dense `RUNNER_METAL_MM` carries
inside the bound — **amplified by discrete top-k routing**: the router runs
f32 on both sides, but its inputs carry the mm rounding of earlier layers,
and a flipped near-tie expert selection swaps a whole FFN. This is the exact
mechanism the 235B census located for the *scalar* cross-backend gap
(0.00332 with no mm anywhere), and the worst-vs-mean shape here (3.36 vs
0.167) is what a few discrete flips over mostly-tiny deltas look like.

Plausible is not proven. The instrument that would prove it — a
teacher-forced GPU-matvec vs GPU-MMA comparison with per-position deltas —
**does not exist yet**: `--score` under `RUNNER_SCORE_CHUNKED` was measured
here to never engage the grouped path (its chunks stay below the
slots >= n_expert threshold), so scoring compares mv against mv and reads
zero. Building that harness is the first step of any promotion pass.

## Standing

Opt-in, default off — the same shelf `RUNNER_METAL_TENSOR` sits on, for the
complementary reason (that one passed numerics and failed the perf bar;
this one passes perf severalfold and has an open numerics question).
Promotion needs, in order: the mv-vs-mm teacher-forced harness; a measured
routing-flip account (how many selections move, at what margins); and a
policy decision on whether sparse-routing models hold MMA prefill to the
dense bound or to a routing-aware floor — the same question the 235B's
scalar identity FAIL already put on the table. Gates in `make test`:
`test-metal-moe-mm` (engagement + fixture-scale byte agreement with the
matvec path + the identity bound on the mxfp4 fixture), mutation-proven
against a wrong scale. Extending `k_moe_mm` to q2_K/q3_K (the 235B) is
mechanical once promotion is settled.
