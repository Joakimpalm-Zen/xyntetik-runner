# 1.22.0 draft gate record

**Not release-ready.** This is preparation from main `83da2dc3`, with
candidate version pins at `0ea15c6e`. Nothing has been tagged, pushed or
published by this preparation. No release-specific device gate is claimed.

## Version and changelog

1.22.0 is a compatible minor release: new model families, execution paths
and additive protocol support, with the 1.x interface promises unchanged.
1.21.0 remains the last published release. README explicitly labels this
candidate unpublished; its final published banner and the release date
remain for the actual cut.

`python3 scripts/changelog-entries.py fold --write` assembled 45 entries:
41 pending entries from the cut plus R4.48.24 and release Claims 1, 2 and 3
result entries. The Claim 1 correction preserves the pre-registered result first
and labels erratum 3 as decided after the verdicts. Only the
release heading and fresh empty Unreleased heading were added outside the
fold. The folded body was compared with all 45 corrected source texts in the tool's
order. `changes/README.md` remains; no pending entry is left behind.

## Dry-run status

`scripts/check-release.py --help` exposes no dry-validation/skip-artifact
option. `make -n release-check` was run: it prints the runner build and the
actual gate invocation but executes neither. A zero dry-run exit is not a
release-check pass. No fake binary, compatibility report or device row was
created to make the real gate green.

The real target requires a version-matching binary, BUILD-INFO metadata
naming the tag and pin commit, synchronized package/docs pins, folded
CHANGELOG, site/Hub-link parity, clean public-text scans, a dated 1.22.0
compat report with executed checks, current device evidence, and a passing
full CUDA gate for this version with its committed agreeing summary.
At prep, no `1.22.0-*.json` compat report or 1.22.0 full-CUDA record exists.

## Before-cut CI snapshot

At this preparation query, [scheduled-CI-red issue #1034](https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/1034)
was open. Its scheduled run at `84daf177` had cancelled macOS jobs and a
failed aggregate gate; no Runner code failure is inferred from those
cancellations. The checklist still requires the issue to be settled by a
green workflow. At the cut SHA, [run 45652059814](https://github.com/Joakimpalm-Zen/xyntetik-runner/actions/runs/45652059814)
was green and [run 45653009538](https://github.com/Joakimpalm-Zen/xyntetik-runner/actions/runs/45653009538)
was in progress. Recheck both status and open scheduled-red issues before
cutting; this snapshot is not an assertion about their later state. The
unpushed candidate has no candidate CI run yet.

## Remaining artifact and Windows checks

- Build the Linux x86_64, Windows x86_64 and macOS arm64 release artifacts
  under the workflow's recipes. x86 uses x86-64-v3, macOS apple-m1, not a
  native x86 build. Preserve the pin commit and actual binary hashes.
- Run the Windows full CUDA gate and smoke/activation probe at that pin,
  with `tools/release-make` first on PATH. Commit the gate summary, class
  ledger row and version-specific reports; preserve any real failure.
- Check Windows portable math/T3, the venv health/timeout cases and the
  recurrent-context lane's skip checker. Old fixture passes on 1.21.0 are
  not fresh 1.22.0 artifact gates. The real Inkling files do not fit the
  available 12 GB Windows device; its fixture gate is separate from the
  Linux real-file marks.
- Package BUILD-INFO, source commit, checksums, licenses and notices. The
  workflow's glibc floor/dot-product/version checks must pass on the produced
  assets. Download-and-checksum verification needs the actual assets.
- A full CUDA gate for 1.22.0, with agreeing committed summary, is still
  required even though the feature-level 284-check Inkling runs passed.

## Remaining Mac checks

One heavy job at a time, `-j2`, pytest at most four workers. Before builds,
check disk and stop below 4 GiB free. Before a full suite, read memory pressure
and swap and announce its start/end. Follow the checklist's 15-second guard:
stop on swap growth above 1.5 GB or pressure warning/critical (2/4), recording
start and peak rise, not an absolute swap ceiling.

1. `make -j2 STRICT=1 test PYTEST_WORKERS=4 PYTHON=<venv>/bin/python` on the
   candidate. Record compile toolchain, warnings, actual counts/skips and
   the memory guard. This full run has not been started by draft prep.
2. Conformance with the optional Python and AI SDK clients available;
   report actual skips rather than implying the clients were exercised.
3. Verify the pinned Gemma-4 E4B QAT Q4_0 file (5,154,941,280 bytes, SHA256
   `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`).
   The shared checkout has that pathname, but this prep has not rehashed it.
   Hard-link it into a real `models/` directory in the gate worktree, not a
   symlink that leaks a resolved home path into evidence. Run the CPU compat
   matrix and commit its 1.22.0 report with executed checks.
4. Run the actual `make release-check` and read its exit status. Demonstrate
   the stale-version scan fails on the old pin and clears on the new one;
   repeat the real gate at the final release head.
5. Metal serve soak: 100 cycles, 20 cancels, 120 seconds concurrent work,
   reporting the final line, cancellation counts and every finding.

The main ledger at the cut still names owned Windows and Mac rows from
1.6.0 and the borrowed Linux CUDA row from 1.21.0. Those dated records must
not be relabeled as this candidate's tests. The device ledger's backstop and
the current-version full-CUDA requirement are separate checks.

## Remaining result fields and final decision

- Claim 2: sixteen-thread half reported provisionally; one-thread cells
  are scheduled after the release, not counted as completed.
- Claim 3: final pre-registered outcome TIE; additional Unsloth pairs do
  not change it. The secondary unit-test diagnostic was not computed.
- **PLACEHOLDER-OLMO2-CPU-SPEED:** final CPU timings and limits.
- Fresh candidate CI, no open scheduled-red issue, final device reports,
  release-check and artifact verification.
- The candidate date is provisionally 2026-11-25; confirm the actual cut
  date and replace the unpublished README banner before release approval.
- Merge/tag/publish, container visibility, site rebuild, asset verification
  and the Homebrew update wait for the release decision. This document is
  not authorization to perform those steps.
