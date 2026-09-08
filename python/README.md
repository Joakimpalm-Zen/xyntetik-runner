# Xyntetik Runner Python Client

The supported Python endpoint, process-launch, and startup-ownership boundary for Runner consumers.

`StartupLease` atomically arbitrates one parent-owned Runner launch. It only
tracks the owning parent and never kills an unrelated child process. PID reuse,
dead owners, and unreaped zombie owners are treated as stale claims.

`ManagedRunner.start()` owns the child for the whole call. A False return means
nothing is left running: a runner that never answered before the deadline is
terminated rather than handed to a caller that has no reference to it. A
healthy endpoint counts as ready only when its reported PID matches the spawned
child, so an existing Runner on the port cannot be mistaken for that child.

`RunnerEndpoint.stream_chat()` treats a malformed `data:` frame as a protocol
error rather than skipping it, so a corrupt stream cannot be certified complete
by a later `finish_reason`; non-data SSE lines (comments, `event:`, `id:`,
`retry:`) are ignored as the spec requires. `stall_seconds` is a watchdog over
the time between stream events, and the raised `RunnerStallError` reports the
measured silence. Both errors carry the text received so far in `.partial`.
Passing `cancel_event` also interrupts a silent blocking SSE read promptly;
`RunnerCancelledError.partial` preserves any text received before cancellation.

Transport-level breakage is translated too: a peer holding the port that is not
speaking HTTP, or a body cut short mid-stream, raises `RunnerProtocolError`
rather than a raw `http.client` exception — so `RunnerEndpoint.healthy()` reports
False for a squatting service instead of raising through `ManagedRunner.start()`.

## Shadow mode: `xyntetik_runner.shadow`

The instrument half of shadow mode (plan epic R14), stdlib only. It does not
route requests, train, or sandbox.

- `evidence`: one `EpisodeEvidence` record per observed episode with a
  `Disposition` from a closed list, an `Identity` of the whole measured stack
  and a `VerifierOutcome`. A record cannot claim `verified_local_attempt`
  without a passing verifier outcome and cannot carry frontier content;
  both are refused in `__post_init__`. `summarize` and `render` print counts
  first and both denominators, and no percentage before thirty independent
  eligible episodes.
- `baseline`: `Baseline.capture` hashes a tree; `changes` and `patch_sha256`
  give a patch its identity against that baseline.
- `verifier`: `ProtectedTests.freeze` / `load` keep the deciding tests outside
  the workspace under a hashed manifest; `calibrate` refuses a protected set
  that passes on the untouched baseline; `verify` runs the frozen tests over a
  scratch copy of the workspace with the verifier's own pytest configuration
  and returns a `VerifierOutcome`. A no-op, a changed protected test file, a
  changed `conftest.py` or pytest configuration file, a skipped or missing
  expected test cannot pass; a timeout is `passed=None`.

The negative controls that prove the verifier can fail live in
`python/tests/test_shadow_verifier.py` against the frozen repair task in
`python/tests/fixtures/repair_task_v1`.
