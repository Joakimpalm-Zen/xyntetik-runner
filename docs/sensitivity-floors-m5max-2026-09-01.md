# Self-sensitivity floors: 235B and gpt-oss — M5 Max, 2026-09-01

Two standing gaps said "measure the floor before reading the gap":
`docs/qwen3-235b-metal-m5max-2026-09-01.md` declined to claim a
self-sensitivity floor for the 235B's cross-backend identity FAIL, and
`docs/cert-matrix-status.md` records that the gpt-oss self-floor "is
plausibly below 6/6 and has never been measured." Both are measured here.

Method: `scripts/sensitivity_floor.py --mode runner` — the same build, the
same weights, an f16 KV cache vs a q8_0 KV cache, 16 greedy prompts × 16
tokens, CPU. This is the smallest bounded numeric perturbation available
inside one engine: if a model diverges from itself under it, no cross-engine
or cross-backend comparison can be expected to do better.

| model | self-identical | mean tokens before divergence | mean \|Δlogprob\| | max |
|---|---|---|---|---|
| Qwen3-235B-A22B Q2_K (merged) | **11/16** | 12.19 | 0.0159 | 0.270 |
| gpt-oss-20b MXFP4 | **14/16** | 15.50 | 0.0208 | 0.252 |
| gpt-oss-120b MXFP4 | **11/16** | 12.69 | 0.0218 | 0.303 |

For scale, the worked example in the script's header (b10076, on
llama.cpp's own cold-vs-warm perturbation) put a placid dense 7B at 16/16
and gemma-4-26B-A4B at 12/16: all three models here sit in the measured
chaotic class, not the placid one.

## What this licenses, and what it does not

**gpt-oss:** the suspicion in `cert-matrix-status.md` is confirmed by
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
