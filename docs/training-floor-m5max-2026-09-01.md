# The local training floor, re-measured — M5 Max, 2026-09-01

`docs/adaptation-engine.md` records the honest limit: the CPU-hosted
training loop caps M1-class machines at ~2B bases (T0 memory audit). This
session measures where the floor sits on a 128 GB M5 Max (12 worker
threads, `--gpu off`, ctx 128, the deterministic tool-use corpus from
`scripts/make-tooluse-data.py`, 158 examples).

| base | s/step | peak RSS | notes |
|---|---|---|---|
| Qwen3-8B Q4_K_M (5.03 GB, sha256 `d98cdcbd…c5745785` [local digest of the official Qwen file]) | **31–34** | 5.36 GB | loss 0.2716 → 0.0016 over 4 steps |
| Llama-3.3-70B Q4_0 (40.1 GB) | **457** | 41.7 GB | one full step, completes cleanly |

**Determinism extends to 8B.** Two independent runs (same data, same seed
1234, same config) produced **byte-identical adapter files** — sha256
`cec2ce6d…95cef37f` both times. The byte-identity contract that `make test`
gates at fixture scale and the published artifact demonstrated at 4B holds
unchanged at twice that size.

**The floor question, answered.** The suite plan hypothesized the local
adaptation floor "rises to ~8-14B" on a 128 GB M5-class machine. Measured:
8B is comfortable (half a minute per step; a 100-step run is under an
hour), and 70B is *feasible* rather than practical (~7.6 min per step —
overnight for a small run, but it completes, where an 8 GB machine cannot
load it at all). The practical ceiling for iterative work on this box is
the 8–14B band, as hypothesized.

**Against the many-core reference.** `docs/adaptation-engine.md` records 4B
at ~20 s/step on a 96-thread host. 8B here at ~32 s/step with 12 threads is
roughly 1.25× faster per parameter than that reference — indicative only
(different model family and quant type), but it answers the plan's
"bandwidth-bound backward may rival the EPYC" question in the affirmative:
the unified-memory M5 Max does not need the core count to keep up.
