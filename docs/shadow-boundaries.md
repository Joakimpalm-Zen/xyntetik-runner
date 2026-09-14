# Shadow execution and optional learning

Shadow collects local evidence, reconstructs eligible workspaces, runs bounded
attempts, verifies results and offers patches. It does not require Shade or a
training run. The standard-library Python client ships beside Runner.

## Learning command migration

`shadow adapt` and `shadow optimize` now return exit code 2 with migration
instructions. They do not import Shade or start a subprocess. Exit code 2 is
the client's usage-error code, the same one an unknown subcommand gets from
argparse; it is not the verification-replay convention, where 2-5 means the
recorded command could not run. Install the optional Shade tools to use
`python -m xyntetik_shade.shadow adapt` or `optimize`.
Shade produces candidates and evaluation records; selecting a candidate does
not activate it in Shadow. Existing configured adapters and Runner's `--lora`
interface continue to work. Runner's native training commands remain available.

The evidence format and task artifacts are the boundary. Shade consumes Runner's
versioned records and execution APIs. Runner never imports Shade. New learning
objectives and eligibility rules do not change the recorder's meaning.

## Capture contents and limits

Capture v2 retains raw hook payloads, including agent text or tool data present
in those payloads. It stores large payloads and workspace contents in a local
deduplicated blob store. Git heads are storage references, not proof that a
commit passed tests. Reconstructing a workspace requires its stored contents
and the referenced Git objects; exporting a record alone is insufficient.

The capture file is `~/.xyntetik/shadow/capture2.jsonl`; blobs live alongside
it. The v1 capture remains readable. No records or blobs are uploaded by these
commands. Installation makes the Shadow directory owner-only where the platform
supports POSIX permissions. Keep local captured content out of source-control
commits.

Bounds are 256 KiB per file, 4 MiB and 200 files per repository snapshot, with a
shared three-second snapshot deadline. Reports retain truncated records and
state which bounds were reached. Inherited task associations are heuristics;
uncertain background-job ownership is recorded explicitly.

`shadow install` installs prompt, stop and post-edit hooks for Claude Code and
Codex. For Codex it also installs a `PostToolUse` hook that records recognized
test-runner commands from direct shell and unified `exec` tools as verification
events; Codex reviews new or changed hook
commands before trusting them. The same event remains available directly as
`shadow capture --event verify`, accepting the tool's command, response and exit
code. The captured snapshot is taken after the tool returns; it does not prove
the tree stayed unchanged during execution.

`shadow capture --report --json` checks completeness, stored-content recovery,
manifest associations and attribution. It does not execute captured commands.
Explicit verification reproduction must resolve the recorded manifest and
working directory and refuses incomplete reconstruction. Reproducing an outcome
does not prove patch causality or reproduce dependencies and external services.

Learning eligibility belongs to Shade and is separate from these observations.
Neither successful capture nor a passing fixture establishes useful training
yield or reduced frontier usage.
