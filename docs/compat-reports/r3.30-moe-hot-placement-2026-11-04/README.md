# R3.30 step 2: hot experts on the device for `--cpu-moe` host layers (`RUNNER_MOE_HOT`)

**Result: 1.52x warm decode over `--cpu-moe auto`, above the 1.3x kill.** The file is Mellum2.1-12B-A2.5B-Thinking Q8_0,
on the RTX 3070 (the Windows CUDA host). Warm decode in one long-lived process, on a prompt the routing trace never
saw, reads **40.69 against 26.75 tok/s**, medians of three interleaved rounds.

**Corrected 2026-11-06.** This record first read 1.96x (27.82 against 14.19 tok/s). That is a fresh-process reading:
each run was its own process, so it measured a first request, which pays for paging the mapped file. The first
request reads 26 to 33 tok/s against the warm 40.69. The original table and raw records are kept below, labelled.

**Limits, first:**
- **Scope:** opt-in, one file, one card.
- **The hot file comes from a trace.** It is the 768 most frequent (layer, expert) pairs over step 1's six traced
  prompts (`hot-all.txt`, from `hotset.py`), so it fits the text the trace saw. The bench prompt is not one of
  those six.
- **v1 covers the plain top-k SiLU expert FFN only.** The runner refuses a shared expert, expert biases or scales,
  gemma-4's dual branch, hyper-connections, a malformed file, or the knob without `--cpu-moe`.
- **The first request is slower than the warm figure,** and by an amount that depends on the page cache at launch
  (see "Against the projection").

## The warm read (2026-11-06)

**Run:** runner main eb145de0, the same file and `hot-all.txt`, one long-lived server per run, three interleaved
rounds (`warm/matrix.log`, its `all` and `auto` rows). Identical requests at a fixed seed, on step 1's bench prompt,
128 decode tokens. The request's routing matches a `--bench-json` run at that seed record for record.

| | First request | Warm requests, by round | Warm median |
|---|---:|---|---:|
| `hot` | 26.23, 29.68, 29.35 | 40.38, 40.86, 40.69 | **40.69** |
| `auto` | 16.89, 18.29, 18.29 | 26.71, 26.75, 26.86 | **26.75** |

- **The ratio:** 1.52x on the medians.
- **The first request:** 26 to 33 tok/s across sessions. The 33 comes from `hot` run back to back, the 26 from runs
  alternating with other placements (`docs/compat-reports/r3.33-first-request-paging-2026-11-06/`).

## The fresh-process reads (2026-11-04)

These are the original measurements, kept as run. They are not the placement's steady speed.

**Run:** runner at this PR's head on main 90cddfe2, the Q8_0 (`JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF@4159d684`,
sha256 `69e0ac02...`), `bench2.sh`.
- **The prompt:** step 1's bench prompt, 128 decode tokens, `--bench-json`.
- **The two placements:**
  - `auto`: `--cpu-moe auto`, 12 of 28 expert banks on the device.
  - `hot`: `--cpu-moe 99` (every bank on the host) with `RUNNER_MOE_HOT=hot-all.txt`, which holds 768 slots
    (5.07 GB) on the device.
- **Rounds:** each run is its own foreground job. The order alternates by round.

| Round | `auto` tok/s | `hot` tok/s | Hit share (`hot`) |
|---|---:|---:|---:|
| 1 (auto first) | 12.69 | 26.72 | 0.735 |
| 2 (hot first) | 14.19 | 30.31 | 0.719 |
| 3 (auto first) | 15.15 | 27.82 | 0.716 |
| **median** | **14.19** | **27.82** | |

- **The ratio:** 1.96x on the medians. The smallest ratio across rounds is 1.76x (round 1's hot against round 3's
  auto).
- **Sampling:** `--bench-json` samples at the file's defaults, so each round's text and routing differ. The hit share
  counts every routing of the run, prefill included.
- **Step 1's simulator** read the hot set leave-one-out at 0.785 on the traced prompts. Here the prompt was never
  traced, and the share is 0.716 to 0.735.

## Where the time goes (`RUNNER_MOE_HOT_PROF=1`)

Per mixed-layer call in round 2 (`bench/r2-hot.err`; 129 calls a layer, the prompt's tile and 128 decode steps):

| | per call, ms (range over 28 layers) |
|---|---|
| route: router, sync, read-back, remap | 0.159-0.213 |
| host: the host experts, while the device runs the bank | 0.597-2.685 |
| wait: the sync after both | 0.002-0.071 |
| fill: the host terms' upload and the sum | 0.025-0.069 |

- **The host experts are the cost that is left.** Over the run they take 4.78 s of 5.60 s in the mixed layers.
- **Per layer,** the host time follows the layer's misses: blk.0 (hit 0.785) spends 2.685 ms a call, blk.21 (hit
  0.846) 0.597 ms.
- **The device side barely waits:** the bank's kernels finish inside the host's share.

## Against the projection

Step 1 projected 1.66x. It charged each routed host expert the whole-bank slope (0.370 ms over 16, 18 and 20 host
banks) and left the rest of the token unchanged.

**The open question is answered: warm, the measurement does not beat the model.**
- **Step 1's 1.66x assumed a served share of 0.785.** This prompt's decode share is 0.738, where the same model gives
  1.53x. The warm read is 1.52x.
- **Both placements sit on one line.** Warm, `auto` takes 37.39 ms a token with 128 routed host experts and `hot`
  24.58 ms with about 59, which is about 0.19 ms a host expert. Step 3's static files give 0.218 ms from four shares
  (`docs/compat-reports/r3.30-lru-cache-2026-11-06/`).
- **The 1.96x came from the arms paying different cold costs.** From the fresh-process reading to warm, `auto` gains
  1.89x (14.19 to 26.75) and `hot` 1.46x (27.82 to 40.69). Those readings are two days apart.
- **Why they paid different amounts is an inference,** from page-cache state at launch, and was not set or measured
  directly. The first request's own cost is measured in the R3.33 report.

The note on grouped misses, the CPU router and differing text, which this section first listed as possible causes,
is not needed to reach 1.52x. None of those was measured separately.

## Identity (`tests/test_moe_hot.py`, the RTX 3070)

The fixture is `make-test-moe.py`'s moe4 (top-2 of 4 experts, a real router, two layers), at F32 and at Q8_0, scored
chunked so every row goes through the FFN.

1. **Every expert listed:** the score body is `--cpu-moe 0`'s bytes on both fixtures. The route record reads
   `experts_on_host` 2/2 and `moe_hot` 8.
2. **The host terms are the CPU's by construction.** The host half calls the CPU path's own `moe_grouped_expert`,
   and the term is the routing weight times its down output. There is no separate byte test.
3. **Mixed, 2 of each layer's 4 experts listed** (hit share 0.492):
   - The run is the same bytes twice.
   - Perplexity against all-host: F32 259.74357 against 259.743569 (4e-9 relative), Q8_0 259.821603 against
     259.82158 (9e-8). The bound is 1e-3.
   - A mixed layer's sum is the device's sum of both sides' terms, so it is neither placement's bytes, and this
     tolerance is its gate.

## Files

- `warm/matrix.log`: the warm read's server log, every row of step 3's matrix (`all` is `hot`).
- `bench/`: the six fresh-process runs' `--bench-json` records and stderr; the hot runs carry the profile. Host paths
  read `~/`.
- `bench2.sh`: one run a call.
- `hotset.py`: the hot file from step 1's traces (`docs/compat-reports/r3.30-moe-expert-cache-design-2026-11-04/step1/run/`,
  PR #619).
- `hot-all.txt`: the file used.
