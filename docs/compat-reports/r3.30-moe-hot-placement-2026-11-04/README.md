# R3.30 step 2: hot experts on the device for `--cpu-moe` host layers (`RUNNER_MOE_HOT`)

**Result: 1.96x decode over `--cpu-moe auto`, above the 1.3x kill.** The file is Mellum2.1-12B-A2.5B-Thinking Q8_0, on
the RTX 3070 (the Windows CUDA host). Decode on a prompt the routing trace never saw reads **27.82 against 14.19
tok/s**, medians of three interleaved rounds.

**Limits, first:**
- **Scope:** opt-in, one file, one card.
- **The hot file comes from a trace.** It is the 768 most frequent (layer, expert) pairs over step 1's six traced
  prompts (`hot-all.txt`, from `hotset.py`), so it fits the text the trace saw. The bench prompt is not one of
  those six.
- **v1 covers the plain top-k SiLU expert FFN only.** The runner refuses a shared expert, expert biases or scales,
  gemma-4's dual branch, hyper-connections, a malformed file, or the knob without `--cpu-moe`.
- **The measurement beats step 1's projection of 1.66x, and why is not established** (see "Against the projection").

## The reads

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
banks) and left the rest of the token unchanged. The measured 1.96x is above that, and this record does not explain
the difference.

Three things differ from the model, and none is measured separately:
- The host's misses run grouped (`moe_grouped_expert`) and overlap the device's work in the same layer.
- `auto`'s host layers also pay the norm and the router on the CPU.
- The two runs' texts and routings are not the same.

The record claims the measured number only.

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

- `bench/`: the six runs' `--bench-json` records and stderr; the hot runs carry the profile. Host paths read `~/`.
- `bench2.sh`: one run a call.
- `hotset.py`: the hot file from step 1's traces (`docs/compat-reports/r3.30-moe-expert-cache-design-2026-11-04/step1/run/`,
  PR #619).
- `hot-all.txt`: the file used.
