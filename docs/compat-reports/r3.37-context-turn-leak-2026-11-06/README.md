# R3.37: a released context left the device turn held for good

**Result: fixed. One branch of a completion request returned without giving the device turn back, so every later
completion on the server hung, with `/health` still reading `ok`.** After the fix the same request gets the same 409
and every slot keeps serving.

**Limits, first:**
- **This test needs a CUDA device.** The device turn exists only on a GPU-backed server (`device_turn` in
  `scheduler.c`), so on a CPU build this path holds nothing and the leak cannot be seen. No GPU-free form of it exists,
  and it is not in `make test`. `repro.py` is the record, run by hand on an RTX 3070.
- **The hang was watched for 2 minutes, not to the end of time.** The code has no path that releases the turn, which
  is why it reads as permanent.
- **The repro holds the turn with a `--draft lookup` generation.** Any request that holds the turn should open the same
  window, but only this case is measured.

## The defect

`run_completion` takes the turn at `sched_prefill_begin()` and gives it back at `sched_prefill_end()`, about 80 lines
later. Between them, the `context_unavailable` branch returned without giving it back, and without clearing the
prefill yield hook, after writing its 409 to the client under the turn. `completion_cleanup` touches neither.

**To reach it:** the request passes the prefix check, queues for the turn behind a running generation, and its context
is deleted while it waits. The prefix reuse then falls short of the context and the branch is taken. Any client allowed
to use named contexts can do this. `cache_prompt: false` does not reach it: validation answers 400 before the turn.

## The fix

The branch now does what the end of the span does, in that order: `sched_prefill_end()`, then
`engine_set_prefill_yield(e, NULL, NULL)`, and only then frees, cleans up and writes.

**Nothing else stays held.** That `return` is the only exit in the span. The stop hook goes in `completion_cleanup`, as
on every other exit. The yield hook mattered too: left set, a later `engine_feed` on the slot, in a session or a
context pin, would have given the turn away in the middle of that span.

## The reads

`repro.py`, same model (Qwen3-1.7B Q8_0), same requests, `--parallel 3`. Each round asserts the answer was 409
`context_unavailable`, so a round that misses the branch fails instead of passing.

| | Rounds | Served after the 409 |
|---|---:|---|
| main e2579150 (`before.log`) | 1 | **0 of 4**, each 30 s timeouts; `/health` reads `ok` with 3 requests active |
| this change (`after.log`) | 3 | **12 of 12**: short requests in 0.07-0.10 s, a long prompt in 0.49-0.51 s |

The long prompt yields the turn between prefill chunks, so it also shows the turn stayed balanced.

**The gates:** `tests/test_named_contexts.py`, `tests/test_kv_snapshots.py`, `tests/test_server_sessions.py` and
`tests/test_request_disconnect.py` pass, 38 of 38, and the turnstile test `test-sched-turn` passes. Full `make test` was
not run.

## Files

- `repro.py`: exits 0 when every completion is served.
- `before.log`, `after.log`: the two runs.
