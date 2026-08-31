# Full Metal router-logit trace

`RUNNER_MOE_TRACE=PATH` now captures the complete pre-softmax expert vector on
Metal, not only the selected top-k. Each JSONL row contains `pos`, `layer`,
`experts`, `gates`, `norms` (empty on Metal), and `logits`.

The implementation cannot read `g->moe_logits` only after inference: that
scratch is overwritten by the next MoE layer. When tracing is enabled, a small
compute copy snapshots each layer into `[layer][token][expert]` device memory.
The host reads the accumulated buffer once, after the normal command-buffer
wait. There is no per-layer synchronization. With tracing unset, the snapshot
buffer is not allocated and the copy dispatch is not encoded.

`tests/test_metal_moe_batch.py` is the lifetime gate: it requires full vectors
for every token/layer pair and independently asserts that layer 0 did not turn
into the last layer's reused vector. The embedded shader gate also requires the
copy kernel and pipeline.

This format is intentionally raw. Selection/gating transformations can be
replayed offline, while top-k-only output cannot reconstruct the discarded
ordering or margins.
