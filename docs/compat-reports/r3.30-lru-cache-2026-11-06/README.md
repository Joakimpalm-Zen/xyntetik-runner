# R3.30 step 3: the dynamic expert cache, killed at the charged read

**Result: killed. The capped LRU projects 1.06x mean over static `RUNNER_MOE_HOT` across six out-of-sample prompts,
and 1.08x at the lowest charge this method can read, under the 1.15x kill. No kernel code was written.**

**Limits, first:**
- **The verdict depends on the charge.** The mean is 1.22x at zero charge and passes 1.15x at or below 0.141 ms an
  admission. The lowest charge read here is 0.299 ms, but that is the floor of this method, not a physical floor: it
  comes from a second CUDA context, and no in-process number exists.
- **Scope:** one file (Mellum2.1-12B-A2.5B-Thinking Q8_0, sha256 `69e0ac02...`), one card (RTX 3070), one host (16 GB,
  PCIe 3.0 x16).
- **Only the bench prompt is measured end to end.** The six prompts are projected from a fit to its measurements.
- **The cache itself was emulated, not built** (see "How the read was taken"). The emulation errs both ways.

## The kill, as set before measuring

At least 1.15x decode over static `RUNNER_MOE_HOT` at equal slots, on the same file and card, on an out-of-sample
prompt, in interleaved rounds, with no identity miss. Earliest read: the policy simulation at equal bytes with the
admission cap charged, before any kernel code.

## The baseline moved first

Step 2's 27.82 tok/s is a fresh-process number. In one long-lived server, identical requests read:

| | first request | warm requests (median of 3 rounds) |
|---|---:|---:|
| static hot, 768 slots | 26.2-29.7 | **40.69** |
| `--cpu-moe auto` | 16.9-18.3 | 26.75 |

The kill was read against the warm 40.69. Against 27.82 the cache would have looked far better than it is.

## Results

Each cell is decode tok/s, the median of three interleaved rounds (`matrix.log`). All static files hold 768 slots.
The decode served share is measured on the request's own routing, which matches bench seed 1 on 4,172 of 4,172
records.

| Static file | Served share | No uploads | Under the side load | Side load | Over warm static |
|---|---:|---:|---:|---|---:|
| `hot-all` (the baseline) | 0.738 | 40.69 | | | 1.000x |
| `m1` | 0.824 | 49.42 | 45.40 | 180 experts/s (R=4) | 1.116x |
| `m2` | 0.879 | 57.03 | 48.98 | 380 experts/s (R=8) | **1.204x** |
| `m3` | 0.916 | 62.71 | 51.71 | 550 experts/s (R=16) | 1.271x |

On this text the simulated LRU serves 0.827-0.842 at R=4, 0.872-0.886 at R=8 and 0.888-0.901 at R=16 (the ranges are
prefill admissions off and on). `m2` is therefore the like-for-like read for R=8. `m1` sits just under R=4, and `m3`
sits above R=16, so `m3`'s row flatters R=16.

**The fit (`project.py`):**
- `T(h) = 60.53 - 48.80 h` ms a token, which is 0.218 ms a routed expert.
- One admission costs 0.353 ms of critical path, the slope through the origin of the side load's added time.

**The projection at R=8, static set from the other five prompts (`project.txt`):**

| Prompt | Static share | LRU share | Uploads a token | Over static |
|---|---:|---:|---:|---:|
| copy | 0.664 | 0.858 | 7.6 | 1.32x |
| prose | 0.738 | 0.875 | 7.5 | 1.20x |
| code | 0.837 | 0.892 | 7.0 | 1.01x |
| long | 0.824 | 0.868 | 7.8 | 0.97x |
| tool | 0.803 | 0.845 | 7.6 | 0.97x |
| edit | 0.843 | 0.858 | 7.9 | 0.90x |
| **mean of six** | | | | **1.06x** |

- Two of six clear 1.15x and three lose outright.
- R=4 reads a mean of 1.04x. Pinning 512 or 640 of the hot set and running the LRU on the rest reads 0.98-1.00x.
- The bench prompt, in none of the six traces, projects 1.20-1.23x over three seeds, which agrees with its measured
  1.204x.
- **The reading:** the cache pays only where the hot file fits the prompt badly. Where it fits, uploads cost more
  than the extra hits return.

## How far the verdict rests on the charge

The first read gave 1.06x without saying how much that number depended on the charge. The rule below was set after
that was found and before the lower bound was measured: the kill stands if a pinned-buffer side load charges above
0.141 ms an admission, the break-even for a 1.15x mean, and step 3 stays open at or below it.

| Charge, ms an admission | Mean of six | Clear 1.15x | Lose | code | copy | edit | long | prose | tool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.217x | 3 of 6 | 0 | 1.16 | 1.51 | 1.04 | 1.12 | 1.37 | 1.11 |
| 0.141 (break-even) | 1.150x | 2 of 6 | 1 | 1.09 | 1.43 | 0.98 | 1.05 | 1.30 | 1.05 |
| **0.299 (pinned, the floor of this method)** | **1.083x** | 2 of 6 | 3 | 1.03 | 1.34 | 0.92 | 0.99 | 1.22 | 0.99 |
| 0.352 (pageable) | 1.062x | 2 of 6 | 3 | 1.01 | 1.32 | 0.90 | 0.97 | 1.20 | 0.97 |

**The pinned read (`pinned.sh`, `pinned.log`):** the R=8 row, three interleaved rounds on the shipped binary, one DMA
of one expert from a pinned buffer at 380 a second. The charge reads 0.272, 0.299 and 0.310 ms by round, each above
0.141. The pageable control in the same session reads 0.352 ms, against the fit's 0.353. On the bench prompt, the
pinned load reads 50.45 tok/s, 1.24x over the warm 40.69.

Dropping the CPU copy and the file read took only 15% off the charge, so much of what remains may be the second
context or the bus. That is not separated here.

## What charging changed

- **Queue time costs nothing at R=8.** An expert's next routing comes at least a token later, so its upload has
  landed. The `late` counter does work: forced to a 40 ms upload it reads 112 a token.
- **R=16 outruns the queue on the serial pageable path.** At 2.2 ms an upload and a 20 ms token the served share
  falls to about 0.75 (`sim-c.txt`). A real cache would drop admissions instead, so this is its worst case.
- **The upload is dearer than the design assumed.**

| One expert, 6,580,224 bytes | ms |
|---|---:|
| Pinned, one copy | 0.53 |
| Pageable from the mapped file, one copy | 0.76 |
| Staged: memcpy to pinned, then DMA | 1.06 |
| Replayed as three pageable slices, under decode | 1.15-1.91 |
| Replayed as three pageable slices, alone | 2.1-2.4 |

The design's 0.55 ms estimate holds only for one pinned copy. Its 0.22 ms for computing one expert from DDR4 holds:
the fit reads 0.218 ms.

## How the read was taken

- **`sim_charged.py`** adds the upload queue and the victim slot's loss to step 1's policy, against the static hot
  set. As a control, its free-admission column with step 1's warm start reproduces step 1's 0.790, 0.862 and 0.883.
- **Static files stand in for a cache's hit share** at zero upload cost. `m1` to `m3` are `hot-all` with its least
  used entries swapped toward the prompt's own most used (`blend.py`), so they fit this prompt by construction. They
  are an instrument, not a proposal.
- **The side load** replays the LRU's own admission stream for this text (`adm_list.py`) from the file's real tensor
  offsets, at the real rate, from a second process (`upload_bench.py`).
- **Each server** (`serve_read.py`, `matrix.sh`) takes one cold request, two warm, then two warm under the side load,
  so a row's clean and loaded numbers come from one process.

**Where the emulation errs:**
- A second CUDA context shares the card by time-slicing, which likely overstates the charge.
- It omits slot-map updates and the fence an eviction needs against a kernel still reading the slot, which
  understates it.
- Neither effect is measured, and the verdict is sensitive to them (see "How far the verdict rests on the charge").
- A first run of the pinned row used a scratch binary and was stopped and rerun on the shipped one. `serve_read.py`
  now names its binary and refuses a run whose load-time RAM line does not match it.

## Against step 2's 1.96x

Warm hot over warm `auto` reads **1.52x** (40.69 against 26.75). Step 1's model at the measured share of 0.738 gives
1.526x. Both placements fall on one line, about 0.19 ms a host expert.

## Files

- `sim_charged.py`, `project.py`: the simulation and the projection. `sim-a.txt` (0.55 ms upload), `sim-c.txt`
  (2.2 ms), `sim-sv.txt` (the request's own trace), `project.txt`.
- `serve_read.py`, `matrix.sh`, `matrix.log`, `pinned.sh`, `pinned.log`: the measured reads. Host paths are replaced
  by `R330_DIR`, `R330_MODEL`, `R330_BASE`; the scripts are otherwise as run, and `pinned.sh` and `serve_read.py` are
  the guarded versions.
- `upload_bench.py`, `upload-bench.txt`: the upload timings and the side load.
- `adm_list.py`, `blend.py`, `pagein.py`, `hot-m1.txt` to `hot-m3.txt`: the instruments.
- `trace-s1.jsonl.gz`, `trace-s2.jsonl.gz`, `trace-s3.jsonl.gz`, `trace-svq.jsonl.gz`: the bench prompt's traces.
