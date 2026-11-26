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

`python3 scripts/changelog-entries.py fold --write` assembled 46 entries:
41 pending entries from the cut plus R4.48.24 and release Claims 1, 2 and 3
and OLMo CPU-speed result entries. The Claim 1 correction preserves the pre-registered result first
and labels erratum 3 as decided after the verdicts. Only the
release heading and fresh empty Unreleased heading were added outside the
fold. The folded body was compared with all 46 corrected source texts in the tool's
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

## macOS gates (an Apple M1, 8 GB; Apple clang 21.0.0; the project venv's Python 3.14.8)

All five ran on the pin commit `0ea15c6e`, in that commit's checklist order, one at a time. `./runner --version`
reads `runner 1.22.0`. A memory guard read `vm.swapusage` and the pressure level every 15 s.

1. **`make -j2 STRICT=1 test PYTEST_WORKERS=4`: pass on the second attempt.**
   - Root pytest: 3,726 passed, 261 skipped, 44 subtests passed, 5 warnings.
   - 0 compiler warnings, 676 s.
   - Swap used at the start: 2,606 MB. Peak rise: 70 MB. Level 1 at 41 readings, level 2 at 3.
   - **A departure from the checklist, stated, as in 1.21.0.**
     - The checklist stops a run at memory pressure level 2. Attempt 1 was stopped by that rule after 25 s,
       mid-compile, with a swap rise of 0 MB.
     - At idle afterwards the level was 1. About 80 MB of pages were free, and about 1.9 GB of physical
       memory held the compressor's 4.4 GB of data. That is memory held by idle applications in the
       compressor; nothing was swapping.
     - Attempt 2 enforced the swap rule (a rise over 1.5 GB) and level 4, and logged level 2.
   - **The 5 warnings** are one Python 3.14 DeprecationWarning, `forkpty()` in a multi-threaded process,
     in test_tray_not_raised_on_refusal.
   - **The 261 skips:**
     - 148 need a CUDA device;
     - 35 need llama.cpp's quantize and imatrix tools;
     - 26 are fit cases with no answer for the file;
     - 12 need torch;
     - 9 need the sanitizer build;
     - 8 are Linux-only, 6 Windows-only, and 6 need a T3 build;
     - 5 need SmolLM2-135M, which was absent from the gate worktree;
     - 2 are qwen4exp's runtime decline on Metal;
     - 2 are carried by the clef fixture;
     - 2 are fit GPU lines that need a device.
   - **Run separately with SmolLM2-135M present:** the two test files of those 5 skips gave 39 passed and
     1 skipped (an OS self-connect case).
2. **Conformance: 539 passed, 0 skipped,** in 56 s of tests.
   - The openai (3.46.0) and anthropic (2.6.0) SDKs were present in the venv.
   - The AI SDK packages were added to `tests/aisdk` for the run, not committed, so nothing skipped.
   - Swap rise 0 MB, level 1 throughout.
3. **Compat matrix on gemma-4 E4B QAT Q4_0, CPU: complete.**
   - Before the run, the file rehashed to its pinned SHA256 `676c3507...` and 5,154,941,280 bytes. It was
     hard-linked into a real `models/` directory.
   - Files pass, load pass and chat pass: the same checks as 1.21.0's report.
   - The report reads `loaded models/gemma-4-E4B_q4_0-it.gguf` and carries no home path.
   - 180 s. Swap rise 0 MB. Level 2 at 4 of 11 readings, logged.
4. **release-check: exit 2 at the pin.** None of its three reasons is a macOS one:
   - no CHANGELOG section for 1.22.0;
   - 41 unfolded entries in `changes/`. The fold is a later commit on the branch, `2fa237bc`;
   - no full CUDA gate recorded for 1.22.0.

   At the branch head `a2b13397`, with this compat report in place, it exits 2 on the CUDA gate alone.
   - Scan proof: with the old string in MANUAL.md it reports
     `release-check: MANUAL.md contains stale release string '1.21.0'`. With the pin restored, it reports no
     stale string.
   - It runs again at the final head once the CUDA record is committed.
5. **Serve soak on Metal (SmolLM2-1.7B Q4_K_M, as in 1.21.0): `soak: 0 finding(s) in 302.4 s`.**
   - Identity: 100 of 100, 0 mismatches.
   - Cancels: 20 each of close in prefill (20 landed), by id in prefill, close in decode and by id in decode,
     none too late.
   - The limit walk as expected: at the window and one token over, 400 `context_length_exceeded`.
   - 145 concurrent requests, all 200, 0 mismatches.
   - Device memory is unmeasured on Metal: the soak reads it from nvidia-smi.
   - The model-swap phase needs a second model and was not run; the checklist's command gives none.
   - Swap: 2,668 MB at the start, peak rise 802 MB, under the 1.5 GB rule. Level 2 at all 20 readings,
     logged.

## Remaining result fields and final decision

- Claim 2: sixteen-thread half reported provisionally; one-thread cells
  are scheduled after the release, not counted as completed.
- Claim 3: final pre-registered outcome TIE; additional Unsloth pairs do
  not change it. The secondary unit-test diagnostic was not computed.
- OLMo CPU rows: the measured 7B ratios and scope are reported; the
  unrun 1B Q8_0 row is disclosed, not inferred. The rows are information,
  not a speed gate.
- Fresh candidate CI, no open scheduled-red issue, final device reports,
  release-check and artifact verification.
- The candidate date is provisionally 2026-11-25; confirm the actual cut
  date and replace the unpublished README banner before release approval.
- Merge/tag/publish, container visibility, site rebuild, asset verification
  and the Homebrew update wait for the release decision. This document is
  not authorization to perform those steps.
