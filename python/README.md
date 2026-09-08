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

- `importer`: `scan_all` reads the Codex (`~/.codex/sessions`) and Claude Code
  (`~/.claude/projects`) session files and yields one `Episode` per task
  boundary: source, session, turn, working directory, start and end, the
  user's request. Never an assistant message, tool argument or tool result.
- `tasks`: `pair` finds the commit ranges an episode's window covers in any
  repository at or under its directory; `choose` gives each fix commit to one
  prompt; `build_task` freezes the post-state test files, requires them to
  pass on the post-state and fail on the pre-state, and writes a `RepairTask`
  that records which tests fail at base. The source diff is never given to
  the attempt.
- `importer` also carries `heads_start` / `heads_end` (every repository HEAD
  the capture hook saw at both ends) and `context` (earlier requests of the
  session, the user's words only).
- `attempt`: `Workspace` confines paths to a scratch worktree and offers
  `list_files`, `read_file`, `write_file`, `run_tests`; `attempt` drives a
  model through `finish` under a `Budget` and keeps the transcript;
  `probe_speed` measures decode speed for the fit-first floor that
  `replay --min-tps` applies before any attempt.
- `cli`: `python -m xyntetik_runner.shadow import --out DIR --python PY` scans
  and admits; `replay --out DIR --endpoint URL` attempts each admitted task
  against a runner and verifies it; `report --out DIR [--by-reason] [--tasks]`
  prints counts and both denominators. Pass `--python` the interpreter that
  can run the repositories' tests.

The pipeline is exercised without a model in `python/tests/test_shadow_pipeline.py`
on synthetic traces and a synthetic repository; a scripted chat function
proves the plumbing and the confinement, the pilot measures the model.
