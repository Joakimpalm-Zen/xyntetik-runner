# Nested snapshot test completion

The repository-coverage test called synchronous `snapshot_all` with the
production hook's shared three-second budget. That budget includes discovery
and Git subprocesses. If the nested repository's `git status` returns after
its remaining budget, `snapshot` marks it truncated before recording the first
changed file. Both repositories can still be present, with the nested file map
empty. The test did not check truncation.

Before changing the test, 50 executions with eight concurrent writers repeatedly
writing and fsyncing one MiB each all passed. This disk-contention experiment
alone did not reproduce the intermittent failure.

A controlled process-boundary stall did reproduce the missing `pkg/a.py`
assertion before the fix: after the real nested `git status` completed,
`subprocess.run` was held by repeated one-MiB writes and fsyncs for 3.1 seconds.
The original test returned two repositories, an empty nested file map, and
`truncated=true`, with a `snapshot deadline` skipped reason. The overall
snapshot took 3.198 seconds. This establishes a sufficient failure mechanism;
it does not prove that the historical failure had the same cause.

The test now passes `deadline_s=float("inf")` so its synchronous call finishes
without the hook budget, asserts completeness, and checks the captured blob
against the literal fixture bytes. The same controlled stall passes after the
change. The existing spent-deadline test still exercises production truncation.
Disk experiments ran under a resource guard with pressure below 2.

README impact: none. Only the test's capture budget and assertions change;
production behavior, the manual, site, and model cards are unchanged.
