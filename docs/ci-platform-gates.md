# Platform core gates before merge (R6.29)

A test registered in `make test-gates-core` was not exercised by the PR's
Windows smoke/conformance job. The full `windows-suite` job allowed only
schedule, dispatch, or an engine push to main; `macos-suite` allowed only
schedule or dispatch. Their job-level conditions excluded every PR. The
required `pr-gate` also omitted both platform suites, so it could pass before
the Windows core gate ever ran.

This let a Windows-only finite-difference failure in `test-full-grad` first
appear after merge. The test repair and its numerical evidence are separate
from the CI policy change.

## Decision and duration

Run the existing **Windows gates-core** part on relevant PRs automatically.
Do not require a manual dispatch as a normal merge step: it is easy to miss,
requires a separate check for a main run in flight, and starts the complete
platform suites even when one Windows core result is the missing evidence.

One verification run of the numerical repair
([run 42610898224](https://github.com/Joakimpalm-Zen/xyntetik-runner/actions/runs/42610898224))
read 5m52s for Windows gates-core and 16m09s for macOS gates-core. These are
job durations from that run, not a promised latency. The Windows job runs
beside the existing PR jobs; adding only that part limits the impact on the
roughly eleven-minute CI target. Keep macOS's existing schedule/dispatch
policy: its core part was not cheap in this observation, and the hosted
Apple Silicon capacity/queue problem is already recorded in ci.yml.

## Selection and failure behavior

`scripts/ci-gates-core.py` classifies the complete PR diff. The workflow uses
NUL-separated paths and disables rename detection so moving/deleting a core
input cannot hide its old path. Selection is conservative:

- all `src/` files and `scripts/` gate/fixture helpers;
- C/Objective-C/Metal/header/resource inputs under `tests/`, including newly
  added gates without a second manually maintained gate roster;
- `tests/fixtures/`, `Makefile`, CI workflows and policy-contract tests;
- `MANUAL.md`, agent-rule files and `docs/model-tiers.json`, which core gates
  consume as data rather than treating as prose.

Prose under `docs/`, the README/changelog, site pages, Python client files
and Python-only tests do not select this extra job by themselves. Some helper
or source changes will conservatively select core gates even if only another
test part directly uses the file. That avoids incomplete dependency lists;
it does not expand the PR to the batch and Python suite parts.

On PR events the selected `windows-suite` matrix contains only `gates-core`.
On push/schedule/dispatch its previous three-part coverage is preserved.
`pr-gate` depends on `windows-suite`, so a selected failure or cancellation
blocks merge. An intentional skip for a prose-only diff remains acceptable.
No workflow dispatch is needed for this policy or its probe PRs.

## Verification

Local policy tests cover the actual regression path, source/helper changes,
new C gates, mixed diffs and the prose/client fast path. The job wiring test
checks the selected Windows result belongs to the required gate.

Two probe PRs targeting the implementation branch distinguish a C-gate-only
diff from a prose-only diff against the same workflow. Their check results
and classifier logs are recorded in the implementation PR body. The probes
are evidence-only and must not be merged into main.

README impact: none. This changes internal pre-merge validation, not the
product's documented runtime behavior, site account or model cards.
