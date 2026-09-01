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
