# R3.33: why a first request decodes slower under `RUNNER_MOE_HOT`

**Result: no defect. The gap is demand paging of a 12.9 GB mapped file on a host with about 11 GB of RAM to spare. The
lead that opened this item, the hot bank's unreleased host copies, is not the cause.**

**Limits, first:**
- **Scope:** Mellum2.1-12B-A2.5B-Thinking Q8_0, `hot-all.txt` (768 slots), the RTX 3070, a 16 GB host, one prompt, one
  seed.
- **The two shares below are read in one cache state.** They are not a general split.
- **Why the first request read lower on 2026-11-04 is an inference** (see "Page-cache state at launch").

## The reads

First request and warm requests in one server (`ab.sh`, `ab.log`), decode tok/s, medians of three interleaved rounds.
`release` is a scratch build that releases the hot bank's host copies after upload. `primed` reads the first request's
missed experts (752 experts, 4.95 GB, about 5 s) after load and before the request.

| | First request | Warm |
|---|---:|---:|
| shipped | 32.93 | 40.9 |
| release | 33.54 | 40.8 |
| shipped, primed | 35.96 | 40.7 |
| release, primed | 37.71 | 41.0 |

- **Hard reads:** priming lifts the shipped binary from 32.93 to 35.96. The experts a first request misses are not all
  in the page cache.
- **Soft first-touch faults:** the rest of the way to warm, about 8%. A file that needed no disk at all shows the same
  gap: step 3's `m2` read 52 cold and 57 warm with almost no page reads.
- **Release:** about 2% here, so it is not the cause.

## Against the fresh-process bench

Three fresh-process rounds first suggested 1.25x for release (`fresh3.log`: 32.09, 33.02, 32.93 against 26.32, 21.42,
30.72). Six ABBA rounds on the same seed bring that to **1.063x** (`abba.log`: medians 34.47 against 32.42). Decode
page reads barely move between the two builds.

The shipped binary's worst runs are low outliers: 23.66 and 21.42, where release's worst is 32.09. That is a handful
of runs and no mechanism is established, so this record does not claim it.

## Page-cache state at launch

The first request read 26.2-29.7 tok/s when runs alternated between different hot files and placements (step 3's
`matrix.log`), and 32.7-33.0 when the same hot file ran back to back (`ab.log`). The likely reason is what the runs
before it left in the page cache: the bytes a run touches, about 11.7 GB with `hot-all`, do not fit what the host can
hold, so what survives depends on the order of runs. **This is an inference.** The two figures come from different
sessions and the state was not set directly. It would also account for step 2's fresh-process numbers moving between
rounds.

## What the first run of this read got wrong

A first server A/B ran every configuration on the scratch build, and priming never ran, because a patch to the script
failed in a background log nobody read. Its result was thrown out. `serve_read.py` (step 3's report) now takes its
binary from the environment with no default and refuses a run whose load-time RAM line does not match it: 5.8 to 6.0
GB on the shipped binary, 10.9 to 11.0 on release. That check was tested both ways.

## Files

- `ab.sh`, `ab.log`: the table above.
- `abba.log`, `fresh3.log`: the fresh-process rounds. `one.sh` is one run.
- `serve_read.py`, `pagein.py` and `adm_list.py` are in `docs/compat-reports/r3.30-lru-cache-2026-11-06/`, on that
  item's branch.
