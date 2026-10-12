# Measured MoE sensitivity floors

Measured 2026-09-01 on the named M5 Max configurations below. These are
scoped comparisons, not a live artifact-status table. Current admission
and numerical routes are described in [model evidence](model-evidence.md)
and the [manual](../MANUAL.md#support-matrix).

## What this licenses, and what it does not

**gpt-oss:** the measured self-sensitivity is shown by
measurement. gpt-oss disagrees with ITSELF on 2/16 (20b) to 5/16 (120b) of
prompts within 16 greedy tokens under a bounded KV-format change. The 1/6
cross-engine greedy identity row is therefore consistent with the model's
own numerical sensitivity, and the README's position — hold numerically
sensitive models to a measured self-sensitivity floor rather than
cross-engine token identity — now has its measured floor for this family.

**235B:** the model sits in the same chaotic class (11/16; the five
diverging prompts average under four shared tokens). This CONTEXTUALIZES the 0.00332-of-range
CPU/Metal logit delta — a chaotic model amplifies any reordering — but it
does not RE-GRADE it: the floor here is measured in sampled-token logprob
deltas and the identity gate in mean logit range fractions, different units
over different quantities. The identity row stays FAIL against its stated
bound; what this measurement adds is that a bound tightened for placid
dense models cannot be assumed transferable to this artifact, and any
future re-derivation of the 235B bound must start from this floor.

Raw JSON (all three `xyntetik.runner.sensitivity-floor.v1` records) is
reproducible with:

```sh
python3 scripts/sensitivity_floor.py --mode runner --tokens 16 \
  --model models/<artifact>.gguf
```
