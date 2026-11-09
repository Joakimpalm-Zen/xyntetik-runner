# Readout cancellation

The issue still reproduced on current main at `f075a535`, after the October
refactors: an abandoned decide, rerank, or embeddings request held the only
slot and a following short completion timed out. The HTTP regression uses a
generated dense fixture, CPU execution, batch size one, and a live long request
as the duration control. The follow-up must complete in less than half that
control time and match a cold short completion's literal output.

All three routes failed before the fix. A separate decide case with a long
option also timed out: its scoring uses one-token `engine_feed` calls, which
never cross the feed function's between-chunk cancellation poll. This case
prevents a fix that only attaches the existing engine callback from looking
complete. No production delay hook or external model is used.

The generation handler installed a socket stop predicate; the three readout
handlers did not. Decide and rerank now attach a scoped predicate, poll it
between option-scoring forwards, and clear it before their handler returns.
Embeddings polls between inputs and passes an optional stop callback into the
model's chunked embedding loop. Partial output is discarded, embedding prefix
state is invalidated, and all replies happen after the device turn is released.
The existing non-consuming peer probe still distinguishes an orderly close
from a quiet live socket or pipelined bytes. Cancellation waits for a complete
forward; it does not interrupt a kernel or tokenization.

The calibration instrument's former implicit 120-second HTTP timeout is now
`--timeout SECONDS`, finite and positive. The same value reaches retries and
single-question fallback and a non-default value is recorded in provenance.
The CLI test failed on the missing flag before implementation; its boundary
test covers refused batches and fallback against a real local HTTP fixture.

README impact: its brief serving account remains accurate. The manual records
readout cancellation and the instrument flag. The site's serving account and
model cards contain neither detailed cancellation nor instrument timeout
claims, so they need no corresponding change.
