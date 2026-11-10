# Changelog entries waiting for the next release

Each pull request adds its CHANGELOG entry here as a file of its own, instead of editing
CHANGELOG.md. No two pull requests touch the same file, so entries never conflict. The release
folds the files into CHANGELOG.md's `## Unreleased` section and deletes them
(`python3 scripts/changelog-entries.py fold --write`, step 2 of
[the release checklist](../docs/release-checklist.md)).

- **Name:** `changes/<plan ID>.md`, for example `changes/R6.21.md`. With no plan ID, use a short
  slug such as `changes/fix-typo-in-manual.md`. Use letters, digits, `.`, `_` and `-`.
- **Text:** the entry exactly as CHANGELOG.md will show it. That is one top-level list item
  starting with `- `, with any detail nested under it, and no headings. Links are relative to the
  repository root, as from CHANGELOG.md. The fold puts the newest entry first.
- **One per pull request.** CI checks this in the commit-hygiene job
  (`scripts/changelog-entries.py check`). A pull request with nothing to record says why: add a
  line `Changelog: none, <why>` to its description or to one of its commit messages. After
  editing the description, re-run the job.
- **A merge train** is one pull request that is `main` plus the merged heads of several reviewed
  pull requests, merged in order with nothing of its own. Its description has a line
  `Changelog: train`, and each member keeps its own entry: the check reads every merge on the
  train's first-parent line as one member and holds it to the rule above on its own. A member's
  reason for no entry is read from its own commit messages, the merge commit's included, never
  from the train's description. A train with any commit that is not a merge fails.
- **Before the switch:** a branch cut before this folder existed may keep its entry in
  CHANGELOG.md. The check accepts it with a warning. Moving the entry here is how to settle its
  next CHANGELOG conflict.

This file is not an entry, and the fold leaves it in place.
