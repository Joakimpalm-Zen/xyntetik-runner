# Activation trace across runtimes: Runner against transformers on Qwen3-0.6B (2026-11-01)

The R4.11 trace (`RUNNER_ACT_TRACE`) and its transformers writer (`scripts/act-trace-hf.py`) were compared on a real
model, to check that the two write the same points and to measure how far apart two correct runtimes sit. That
measured floor is what `--rtol` has to clear in a cross-runtime comparison.

## Setup

- **Reference:** `Qwen/Qwen3-0.6B` at `c1899de289a0`, transformers 5.16.1, torch 2.14.0 on the CPU, float32 weights
  and arithmetic, 16 threads.
- **Runner:** commit `4a8a0c54` on x86-64 with AVX-512, CPU only (`--gpu off`, 16 threads), on
  `Qwen3-0.6B-bf16.gguf` (the same weights in BF16, which float32 holds exactly).
- **Prompt:** "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is", 19
  tokens with the same ids on both sides. Positions 0, 5, 12 and 18 were traced with values. Both runtimes continue
  with " Rome".

## Result

`scripts/act-trace-diff.py run.jsonl ref.jsonl --rtol 1e-3`: 229 common points, every one matches. The 3 points
only on the reference side are its logits at positions 0, 5 and 12; Runner computes logits only for a forward's
last row.

| Point | Records | Largest relative L2 distance |
|---|---|---|
| `embd` | 4 | 1.1e-9 |
| `mixer` | 112 | 6.2e-4 |
| `resid` | 112 | 6.8e-4 |
| `logits` (position 18) | 1 | 3.7e-4 |

At position 0, the distance is 1e-4 to 2e-4 in the first two layers. It then settles near 4e-6, because the first
token's attention is a single row. At positions 5 to 18 it stays between 1e-4 and 7e-4 through all 28 layers, with
no growth with depth. That is the difference between two float32 forwards with different summation orders. An
`--rtol` of 1e-3 clears it on this model. The gate fixture's dropped attention edit reads 0.04 to 0.06 at its
mixer point, about two orders of magnitude above this floor (on the toy model, not this one). A larger model or a quantized file has a higher floor: measure it the same
way before choosing `--rtol`.

## Files

- `act-trace-diff.json`: `act-trace-diff.py --json`, the distance at every common point.
