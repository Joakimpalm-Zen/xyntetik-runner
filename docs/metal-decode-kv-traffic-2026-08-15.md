# Attention and KV numerical checks

Attention depends on the admitted cache format/layout, layer window, head
geometry, position mask, and execution shape. Full, sliding/shared, ring,
derived-K, recurrent, and quantized caches do not have interchangeable
addressing or rewind/prefix contracts.

`tests/test_attn_tol.c` checks the implemented attention path against its
reference. Cache-format, ring/tied-V, attention-split, rewind, prefix,
and batch checks cover the surrounding state transitions. Include exact
boundary positions, tails, zero/large values, mask edges, and independent
expected values; random agreement alone is not an equivalence proof.

Preserve sum order and established precision/tolerances. A cache or shader
optimization must retain current admission/refusals, state ownership,
partial-failure cleanup, and deterministic next-step behavior. Record
actual memory traffic and route beside a performance comparison rather
than inferring a bandwidth limit from tensor size alone.
