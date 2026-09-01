# Where the 120B's decode milliseconds go — M5 Max, 2026-09-01

gpt-oss-120b decodes at ~75 tok/s against llama.cpp's 102.8 (0.73x) — the
one place llama wins decode here. This session put numbers on every part of
the budget; the mechanism is neither weights nor encode.

Measured, per decode token (RUNNER_METAL_TIMING / RUNNER_METAL_STATS):

- **13.2 ms total**, of which CPU encode is 0.12 ms and GPU execution
  12.9-13.1 ms — the gap lives on the GPU timeline;
- **686 dispatches per token** (181 mv, 216 moe, 73 rmsnorm, 72 rope,
  72 elem, 36 store, 36 attn — ~19 per layer x 36 layers);
- weight-traffic floor: ~1.8 GB of active expert weights + ~0.6 GB
  attention + head per token ≈ **~5 ms** at this machine's bandwidth;
- pure dispatch-chain cost, measured directly with a hazard-chained no-op
  probe at the same count: 686 serialized dispatches = **2.3-3.3 ms**
  (3.4-4.8 us each) before any kernel does real work.

So the budget reads: ~5 ms unavoidable weights + ~3 ms dispatch chain +
~5 ms of small-kernel execution that is latency, not throughput (a
2880-element rmsnorm or rope dispatch cannot hide its own launch depth).
llama.cpp's 9.8 ms/token fits the same floor with a shallower chain.

**The route, quantified:** fusion. Folding the per-layer chain
(rope+store; norm into the following matvec's prologue; the MoE
route→gate→act→up→down slot chain into one or two kernels; add+norm) from
~19 dispatches per layer toward ~6 would recover an estimated **3-6 ms →
~90-130 tok/s**, llama-parity and past it. This supersedes the smaller
"fuse gate+up in grouped-MMA prefill" item (A4): the same fusion program
covers it, and decode is where the milliseconds are. Prior context: the
Codex pass parked "pure launch reduction" on the M1 — at the M1's budgets
that was right; at 686 dispatches inside 13 ms on an M5 it is the lever.

Nothing here is a defect: every dispatch computes what it should (the
identity gates say so). It is architecture headroom, now measured well
enough to be scheduled instead of guessed at.

## Phase 1, executed same day (RUNNER_METAL_FUSE, default on)

Three budget-line fusions, all in the byte-identity class, all landed:
rope(q)+rope(k)+f16-store (−2/layer), residual-add+rmsnorm at both per-layer
seams (−2/layer, the post-FFN add deferred into the next layer's attention
norm), and MoE gate+up+activation (−2/layer, same dot bodies, same bias
points, the actmul code verbatim including its mandated tanh clamp).

Measured on the 120B: dispatches ~686 → ~480 per token (−30%), decode
64.4 → 65.8 tok/s (**+2.1%**), byte-identical on the 120B, the 30B, and a
six-architecture fixture roster (`test-metal-fuse`, in `make test`,
engagement-checked; a 1% store perturbation is caught by the roster —
fixture-scale f16 swallows anything under its ULP, which is itself a
calibration note for future mutations). q8 caches, NoPE layers, sandwich
norms, muP scales and the gemma dual branch all keep the unfused path by
guard, and `RUNNER_METAL_FUSE=0` restores it everywhere.

The honest ledger: the cheap third of the chain is gone and bought 2%, not
the 3–6 ms projected for the full program — the remaining milliseconds sit
inside the mv/moe dispatches themselves, which only the persistent
layer-walk kernel (the summit item, the one llama.cpp's op-graph design
cannot follow) can fold. That is now the measured next step, not this one
re-run harder.

## Phase 2 — the attention-front megakernel, same day

The persistent layer walk, cut where Metal's execution model allows it. A
literal single-dispatch layer needs cross-threadgroup synchronization Metal
does not guarantee (no forward-progress contract; a stall is a GPU-watchdog
event, which fails the fail-closed bar), so the walk fuses everything whose
dependencies fit inside one threadgroup barrier scope: **residual-stream
norm + Q/K/V matvecs + rope + f16 KV store, one dispatch**
(`k_attn_front_q8_0`), whole heads per threadgroup, the normed row staged
once in threadgroup memory and read by all three projections from SRAM.
Byte identity throughout: the norm replays k_rmsnorm's 256-thread tree, the
dots replay k_mv_q8_0's packed body, rope/store replay k_rope_store.

Measured on the 120B: decode 64.5 → **67.6 tok/s (+4.8%)**, byte-identical;
census ~370 dispatches/token. Program cumulative: **686 → ~370 dispatches
(−46%), 64.4 → 67.6 tok/s (+5.0%)**, all of it byte-identity class.

Admission is static and honest: uniform Q8_0 attention (both flagship MoE
models), f16 KV, roped, no qk/v norms, no output gate, n_embd ≤ 7936 (the
32 KB threadgroup budget — the 70B's 8192-wide rows keep the split path).
The gate grew a logprob-level leg after a roped-K sign flip survived the
token-level compare at fixture scale: `--score` teacher-forced logprobs,
fused vs unfused, byte-equal — that leg catches it.

Remaining on the 15 ms → 5 ms road: the o/down/router/attention dispatches
and the ramp cost inside the heavy matvecs themselves; extending the front
kernel to Q4_K/Q4_0 attention and qk-norm models widens its coverage.

## Phase 3 — widening the front, same day: two wins, two honest refusals

The front kernel grew from one shape to a roster — Q4_0, Q4_K, Q6_K and the
Q4_K/Q6_K mix (the Q4_K_M attention layout) alongside Q8_0, each dot body a
character-for-character replay of its k_mv_* twin behind one activation
macro — and learned qwen-style qk-norm, replayed in-dispatch at exactly the
64-thread shape enc_qknorm_n gives k_qknorm. Measured:

- **Qwen3-30B-A3B-Q8_0 (qk-norm MoE): 72.8 → 77.1 tok/s (+6.0%),
  teacher-forced logprobs byte-identical.**
- 120B regression: byte-identical, 72.5 → 76.6 (+5.6% warm).

And what measurement kept OUT:

- **Dense models are not admitted.** Qwen3-8B-Q4_K_M ran the front
  byte-identically — 8.4% SLOWER (84.3 → 77.2). The front recovers
  dispatch-chain latency, which is where MoE decode's milliseconds sit; a
  fast dense model is matvec-bound, and one-threadgroup-per-head (48 TGs)
  starves a GPU the split mv grid fills. Admission is therefore MoE-only
  (n_expert > 0), a measured criterion, not a type list.
  RUNNER_METAL_FRONT=all lifts it for fixture-scale gates.
- **The wide (no-staging) form is removed.** Built for the 70B's 8192-wide
  rows: recompute the normed element inline as x[i]*r*nw[i] instead of
  staging it. The expression is bit-exact; the COMPILER is not — it
  contracts the inline renorm into the dot's fma chains, and the fused
  product never rounds where the unfused path's store rounds it. Fixture
  force-runs passed (small exact values round the same fused or not); the
  real 70B's teacher-forced logprobs failed. And the 70B is
  bandwidth-bound at 12 tok/s — dispatch latency isn't its problem, so
  even a fixed wide form has nothing to win. Both facts are the negative.

Gate mechanics that earned their keep: the qk-norm fixtures pin every new
kernel byte-identically (q8_0, q4_0, q4_k, q6_k, forced via
RUNNER_METAL_FRONT=all for the dense ones, with a default-admission MoE leg
against vacuity), and the teacher-forced-logprob leg — added this morning
after the rope-sign escape — caught BOTH planted mutations (qk-norm weight
dropped; Q4_K dmin sign flipped) while every 24-token compare waved them
through. Token-level fixture compares are officially not a gate for
arithmetic inside the attention front; logprob bytes are.

## Phase 4 — the layer tail, same day: one fold in, one fold out

**In: F5, the post-FFN residual add folded into the expert sum.** k_moe_sum
already visits every element of the summed expert output; it now writes the
residual stream directly — `x[i] + s`, character for character what k_add
computed from the stored sum — on exactly the layers whose add F2b was not
already deferring (the two are decided together, before the FFN encodes, and
are mutually exclusive by construction). One dispatch per MoE layer gone
(−36/token on the 120B), byte-identical on the full fixture roster and on
both flagships' teacher-forced logprobs; decode speed within noise on top of
the megakernel (30B 75.5, 120B 75.3 tok/s in the same session-state run
where unfused read 70.9/71.5). A dropped-residual mutation is caught by the
plain moe4-q8 byte leg instantly.

**Out: F4, the router matvec + route fused into one dispatch.** Built,
byte-identical — and 30B decode COLLAPSED 77 → 40 tok/s. One threadgroup
means one GPU core, and the router is a real matvec: ~1 MB of router weights
per layer that the split mv grid streams across the whole chip were being
pulled through a single core's load path. Same lesson as the dense-model
front admission, at matvec scale, and it is now a rule: **never trade a
parallel weight sweep for a dispatch.** The route kernel itself (a few
hundred scalar ops) stays the one honest tiny dispatch in the tail.

Program state after phases 1–4: **686 → ~330 dispatches per token (−52%)
on the 120B, all byte-identity class**, decode +5–6% on both MoE flagships,
and the negatives ledger (expert-major ×2, wide front, F4 router, dense
admission) is as load-bearing as the wins — each one is a measured reason
the shipped shape is what it is.

## Phase 5 — the down+sum fold, and the bug class it flushed out

**F6 (down-projection + weighted sum + residual add, one dispatch) is a
negative.** Byte-identical once its accumulate was pinned, and a CONSISTENT
−0.5…−0.9% on both MoE flagships across six paired reps: collapsing the
down matvec's (n_embd × slots) grid to n_embd rows narrows the heaviest
weight sweep of the layer 4×, and that costs more than the one dispatch it
saves. The mild corollary of the F4 rule — a fold that narrows the sweep is
still a trade. Removed; F5 (which touches no sweep) stays.

**What the hunt earned — a new bug class, twice confirmed and once latent:**
per-inlining-site fp contraction. A `static inline` dot body shared verbatim
between kernels does NOT guarantee identical arithmetic: each inlining site
contracts multiply-add chains independently. F6's register-operand
accumulate needed an explicit `fma` to match k_moe_sum's load-operand form;
and the same disease was found LATENT in the morning's q4_0 attention-front
kernel — an ULP off at teacher-forced-logprob level, invisible to every
24-token compare, missed because only q4_k had a score leg. Fixed by
composing float4s so the front's expression is verbatim k_mv_q4_0's (the
codegen then matches; two hand-fma guesses did not).

**The rule that falls out:** every fused kernel type joins its roster with
a teacher-forced-logprob gate, not by analogy to a sibling type. The
Makefile now score-gates every attention-front type (q8_0, q4_0, q4_k,
q6_k; the q4_K+Q6_K mix verified live on the 8B), and RUNNER_METAL_FRONT=0
exists as a permanent isolation lever. Also re-learned at CPU speed: a
Metal-source error fails the LIBRARY, not the build — the day's fixture
runs fell back to CPU silently until stderr was read. Check "Metal backend"
in stderr before believing any Metal measurement.

## Phase 6 — the load-widening probe: negative, and what it proves

The last scalar-load dot bodies in the file (moe_dot_q8_0, moe_dot_mxfp4 —
carrying the MoE flagships' ~2 GB/token) were load-widened to packed
vector reads, accumulation order untouched, all byte gates green including
CPU identity. Measured: 30B dead even (75.41 → 75.46), 120B consistently
SLOWER (75.2 → 74.3). Reverted. The compiler was already coalescing these
loads; source-level load shape is not where the 120B's milliseconds are.

What this pins down for the bandwidth track: the token achieves ~200 GB/s
against a ~5 ms weight floor that assumed ~480. llama.cpp's 9.7 ms is only
~250 GB/s — BOTH engines sit far from peak at batch-1, so the remaining
gap is latency structure (dispatch chain, inter-kernel serialization,
in-flight depth), not per-kernel load emission. The honest instruments for
the next attack are a per-kernel isolation bench (achieved GB/s per shape)
and the concurrent-encoder track — not more source-level dot rewrites.

## Phase 7 — spec decode measured, and the tensor roster extended

**Metal-4 tensor GEMM: Q8_0 and Q4_0 join Q4_K** (macro-factored dequant
blocks, per-type self-tests with unit-weight rows, harness checks all
three pipelines). Measured at 2K-token prefill: 30B-Q8_0 264.9 simdgroup
vs 261.2 tensor; 70B-Q4_0 parity within noise; temp-0 tokens identical
both arms. The ≥1.2× promotion bar stays unmet on M5, so the whole path
stays opt-in (RUNNER_METAL_TENSOR) — the tracer now covers the flagship
types for the day an M6 or a better tile geometry changes the answer.
(Build note: backslash-continued lines make a // comment swallow a whole
macro body — the tensor library compiled to nothing until the comments
moved out.)

**Speculative decode on Metal: works, and loses on this hardware's pairs —
root-caused.** The 70B + 1B-draft pair, byte-gated and healthy at 62–70%
acceptance (3.5–4.9 tok/round), decodes SLOWER than plain: 12.0 plain vs
9.3 best spec. The verify is one properly batched n=k+1 command buffer —
and it costs a full plain step PER COLUMN (75 → 116 → 613 ms at n=1/2/8),
because at batch-1 the 70B step is ~90% bandwidth-bound with the q4_0
dequant ALU hidden under it; add columns and the per-column dequant ALU
comes out from under the bandwidth floor as the new critical path. Weight
amortization is real; ALU amortization does not exist on the identity-mv
route, and RUNNER_METAL_COL_TILE (1/2/8) measurably does not matter.

Two structural conclusions. (1) A winning verify needs a dequant-once
route — a narrow GEMM tile (4–8 columns) that dequantizes each weight
once into threadgroup memory — but that route is not byte-identical to
the mv chain, and spec verify is currently GATED byte-identical against
plain decode; admitting it is a fidelity-bar decision, not a kernel
patch. (2) MoE targets are structurally weak for spec verify: each
column routes to different experts, so even weight traffic fails to
amortize. Both recorded here so the next session starts from the
measurement, not the hope.
