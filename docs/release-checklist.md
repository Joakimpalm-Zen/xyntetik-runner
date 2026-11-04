# Release checklist

How a Runner release is cut, gated and published, written so a session with no
memory of earlier cycles can run it. The worked example throughout is 1.13.0
(2026-11-04, PR #624, tag on d0b0672f).

A release has two halves. The **worker half** prepares the release branch and runs
the macOS gates. The **CUDA host** runs the CUDA gate on the pin commit while the
macOS gates run. The **coordinator half** merges, tags and publishes. One person
may do all three, but the order holds.

## 0. Before the cut

- Every PR meant for the release is merged into main. Note the cut SHA: the merge
  commit of the last one (1.13.0: 3b9c0726, the #623 merge).
- Main's CI runs for the last merges may still be queued. Start anyway; a red run
  for a real cause stops the release.
- The macOS compat gate needs its model file present before you start:
  `models/gemma-4-E4B_q4_0-it.gguf` (5,154,941,280 bytes, sha256
  `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`), from
  `https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf`.
  Check it at prep time, not at gate time.
- Read the whole `## Unreleased` section of CHANGELOG.md. Merges can keep both sides
  of a rewritten entry, or leave a stub heading. Fix those in the release branch.

## 1. Branch and pins (worker)

```sh
git fetch origin
git switch -c release/X.Y.Z <cut SHA>
```

Seven pins move to the new version. `scripts/check-release.py` checks all of them.

| File | Line pattern |
|---|---|
| `src/runner.h` | `#define RUNNER_VERSION "X.Y.Z"` |
| `python/pyproject.toml` | `version = "X.Y.Z"` |
| `README.md` | `> **Published release: X.Y.Z.** ...` (rewrite the paragraph for this release, limits first) |
| `README.md` | ``> **Version `X.Y.Z`.** The command line, ...`` |
| `README.md` | `./runner --version   # -> runner X.Y.Z` |
| `MANUAL.md` | ``> **Version `X.Y.Z`.** The command line, ...`` (the banner near the top) |
| `MANUAL.md` | `./runner --version   # -> runner X.Y.Z` and the docker tag example ``e.g. `:vX.Y.Z` `` |
| `docs/moe-support.md` | `Runner: X.Y.Z (core support plus the follow-ups at the end of this doc)` |

The MANUAL banner became the seventh pin in 1.13.0. The scan read a version only
after the word Runner, so that banner said 1.6.0 through six releases until a fourth
regex alternative (``Version `X.Y.Z` ``) was added. The site has no pin:
`site/build.py` takes the version and date from the CHANGELOG's first released
heading.

Older versions must not appear in the scanned files (README, MANUAL, the release
workflow, SECURITY.md, CONTRIBUTING.md, docs/moe-support.md) in the forms
`vX.Y.Z`, `Runner X.Y.Z` or ``Version `X.Y.Z` ``. Write history as "Runner's 0.2.0
release" or "its 0.11.0 release".

Commit the pins alone, push, and send the pin commit's SHA (read it back with
`gh api repos/<owner>/<repo>/branches/release%2FX.Y.Z -q .commit.sha`) to the CUDA
host straight away, so the CUDA gate runs in parallel with the macOS gates
(1.13.0: 2d34685b).

## 2. CHANGELOG (worker)

- Retitle `## Unreleased` to `## vX.Y.Z - YYYY-MM-DD`, with a fresh empty
  `## Unreleased` above it. Every entry merged before the cut goes under the release
  heading.
- Apply the record fixes found in step 0. Keep history strings in the CHANGELOG
  itself: it is not scanned for stale versions.
- Run `python3 scripts/check-content-hygiene.py --staged` before each commit. Commit
  the CHANGELOG and docs fixes as their own commit and push. A docs commit does not
  change the CUDA gate's binary.

## 3. macOS gates (worker), in this order, one at a time

Run them on Apple Silicon, in the release worktree, with a Python that has pytest (a
project venv, not the system python3).

1. **make test**, under a swap guard:

   ```sh
   make test PYTHON=<venv>/bin/python
   ```

   Every 15 s, read `sysctl -n vm.swapusage` and
   `sysctl -n kern.memorystatus_vm_pressure_level`. Stop the run if swap used rises
   more than 1.5 GB over its value at the start of the run, or if the pressure
   level is 2 (warning) or 4 (critical). Log the start value and the peak rise in
   the PR's gate record. Do not use an absolute swap figure: macOS keeps the swap
   file allocated after a spike, so an absolute stop trips on memory that is not
   under pressure. In 1.13.0 the first run was stopped at an absolute 1,900 MB
   (2,245 MB, with 73% of memory free). The rerun under the rise rule started at
   2,141 MB and rose 55 MB: 2,227 passed, 115 skipped, exit 0.
2. **conformance**, with the optional client SDKs installed for the run so nothing
   skips:

   ```sh
   PYTHON=<python with openai and anthropic installed> scripts/conformance.sh
   ```

   Install the AI SDK packages in `tests/aisdk/` (`npm install --ignore-scripts`).
   `tests/aisdk/node_modules` is not ignored by git: stage files by explicit path and
   never commit it.
   Without the SDKs the suite skips 17 cases. 1.13.0: 526 passed, 0 skipped. State
   in the notes that the SDKs were added for the run.
3. **compat matrix** on the gemma-4 E4B QAT file, on the CPU:

   ```sh
   <venv>/bin/python scripts/compat_matrix.py --runner ./runner --models-root . \
     --gpu off --execute-checks --verify-files --model gemma-4-e4b-it-qat-q4_0 \
     --out docs/compat-reports/X.Y.Z-YYYY-MM-DD-macos-gemma-4-e4b-qat.json
   ```

   `models/` must be a real directory in the worktree. Put the file there as a hard
   link (`ln <main checkout>/models/<file> models/<file>`), never as a symlink: the
   runner prints the resolved path of a symlinked file, and the report's
   `stderr_tail` then carries a home-directory path. The content hygiene check did
   not catch that in 1.13.0. Before committing, check that the report reads
   `loaded models/<file>`. The script refuses to overwrite an existing report
   unless you pass `--force`; use that only over your own uncommitted output.
   Commit the report.
4. **release-check**, which must exit 0:

   ```sh
   make release-check PYTHON=<venv>/bin/python; echo rc=$?
   ```

   It is silent on success, so read the exit code, not the output. Before the
   compat report exists it exits 2 naming the missing report; that is expected.
   When a pin or the scan changes, prove the scan: run it once with the old string
   in place (exit 2, naming the stale string) and once with the pin (no stale
   string). Put both lines in the PR's gate record. Run it again at the branch's
   final head.

## 4. CUDA gate (CUDA host), on the pin commit

Build the binary at the pin commit with the pinned PTX, then:

```sh
python scripts/release-cuda-gate.py --binary ./runner.exe \
  --model qwen3-0.6b-q8_0.gguf --version X.Y.Z [--card rtx3070]
```

It runs the smoke (`--expect-version`) and the activation probe and writes
`docs/compat-reports/cuda-smoke-X.Y.Z-DATE-CARD.json` and
`cuda-act-probe-X.Y.Z-DATE-CARD.json`, with LF line endings and the home directory
written as `~/`. Exit 0 only when both pass. 1.13.0: smoke pass, 13 checks, 0
warnings; probe 16/16 finite and within float rounding.

Two reads on the same binary:

- **NVFP4 prefill**: `qwen3.5-4b-nvfp4.gguf`, a 1,971-token prompt,
  `-n 32 --temp 0 -c 4096`, every layer on the card. Run it by default and with
  `RUNNER_CUDA_TC_NVFP4=0`; the default must be far faster (the tensor-core path is
  engaged). Read it with the card alone; a shared card is not a speed read. 1.13.0:
  1,326.99 against 21.71 tok/s (1.12.0: 1,200.84 and 22.68).
- **MoE replay**: `make test-cuda-moe-replay`, exit 0, every case within the gate.
  1.13.0: 24 cases, 4,696 s. It is long. If it is not done by the merge, the
  release ships on the CUDA gate and the record follows on main (1.13.0: #626).

Commit the reports to the release branch, or to main if they arrive after the cut.

## 5. The PR (worker)

- Open the PR from `release/X.Y.Z` to main as non-draft. The body lists the
  commits and the gates, limits first.
- First comment: the release notes, limits first. The first sentence says what the
  release is for and what it is not. Scan the text before posting: no em dashes, no
  machine or host names, no host paths, and the chip as "Apple M1", never the bare
  model number.
- Second comment: the gate record (swap figures, the scan proof lines, anything
  rerun and why).
- When a late record arrives, update the notes comment's gates section and rerun
  release-check at the final head.

## 6. Coordinator half

1. Merge the PR once its CI is green and main is idle and green. Do not dispatch
   ci.yml while a main run is in flight.
2. Watch main's CI for the merge commit until it is green.
3. Tag the merge commit with an annotated tag and push it:
   `git tag -a vX.Y.Z -m "Runner X.Y.Z" <merge SHA> && git push origin vX.Y.Z`.
   `.github/workflows/release.yml` builds the platform binaries, checks that each
   binary's `--version` matches the tag, smoke-tests them, writes SHA256SUMS, drafts
   the release from the CHANGELOG, and pushes the container image.
4. Replace the draft body with the notes comment, and drop the PR footer:
   `gh release edit vX.Y.Z --notes-file <notes.md> --draft=false --latest`.
5. Verify:
   - the macOS asset: download it, check its sha256 against SHA256SUMS, and run
     `--version` (prints `runner X.Y.Z`);
   - the container: `ghcr.io/joakimpalm-zen/xyntetik-runner` tags `vX.Y.Z` and
     `latest` share one digest (check anonymously: token from
     `https://ghcr.io/token?scope=repository:joakimpalm-zen/xyntetik-runner:pull`,
     then `/v2/joakimpalm-zen/xyntetik-runner/tags/list`);
   - the site: the pages workflow rebuilds on a CHANGELOG change to main, and
     xyntetik.com reads "vX.Y.Z, released YYYY-MM-DD".
6. Delete the merged release branch.

1.13.0's coordinator half: tag on d0b0672f, release run green, 9 assets, notes from
the PR comment, marked latest; container `v1.13.0` and `latest` on one digest; site
"v1.13.0, released 2026-11-04".
