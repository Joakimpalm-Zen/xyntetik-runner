# R3.35: the hot bank gives back its experts' host copies

**Result: a `RUNNER_MOE_HOT` server holds 5.05 GB less host RAM. After warm decode, available memory on a 16 GB host
reads 5.85 GB against 0.78 GB before.** The identity gates pass, 13 of 13 in `tests/test_moe_hot.py`, the item's kill.

**Limits, first:**
- **This claims RAM, not speed.** Warm decode is unchanged. Fresh-process runs read 1.063x in R3.33, but that is one
  prompt and one seed and is not claimed here. The shipped build's low outliers are not claimed either: they have no
  mechanism.
- **Scope:** Mellum2.1-12B-A2.5B-Thinking Q8_0, `hot-all.txt` (768 slots, 5.07 GB), the RTX 3070, one 16 GB Windows
  host. The Linux path calls the same `plat_release_pages` and has not been measured.
- **The fixtures cannot read resident memory.** Their slices are smaller than a page, so the test asserts the count
  of experts given back, and the bytes are this report's measurement. The test says so where it lives.

## The change

`moe_hot_upload` copies a listed expert's slices into the bank, and those pages stayed in the working set for the
process's life, because `cuda_release_host_weights` skips a layer whose experts stay on the host. It now releases the
listed experts' slices in that layer and keeps the rest. The host path never reads a listed expert, since
`model_moe_hot_terms` runs only the misses, so no value changes. A page something does read later faults back in with
the file's bytes, as for every other released tensor.

Load prints `gpu-split: RUNNER_MOE_HOT gave back the host copies of 768 experts`.

## The reads

`ram_read.py`, one server a run, three interleaved rounds (`ram.log`). GB, the same in every round to within 0.13.

| | Shipped | This change |
|---|---:|---:|
| Working set after load | 5.28 | **0.23** |
| Available after load | 5.84-5.97 | **10.99-11.01** |
| Working set after four requests | 10.43-10.44 | **5.39** |
| Available after four requests | 0.77-0.79 | **5.84-5.86** |
| Peak working set | 10.43-10.44 | 6.14 |

- **The working set falls by 5.05 GB.** The anchor is the file: 768 experts of 6,580,224 bytes are 5.05 GB. The
  5.07 GB the load line prints also counts the routers, which were already released.
- **The gap holds after decode,** so misses do not pull the pages back.
- **Load still peaks at 6.14 GB.** The upload touches the pages before they are released.

**Warm decode, fourth request:** shipped 42.53, 41.86, 42.09; this change 42.36, 42.08, 41.76 tok/s.

**Greedy text on the real file:** the same sha256 (`422485c1b666db04`) in all six runs.

## The gates

- `tests/test_moe_hot.py`: 13 of 13. The new test failed on main first, with the 11 others passing.
- `tests/test_cuda_buffer_lifetime.py`, `tests/test_cuda_upload_report.py`, `tests/test_gpu_split_order.py`,
  `tests/test_score_route.py`: 17 of 17.
- `scripts/help-parity.py`: ok. No flag changed.

## Files

- `ram_read.py`, `ram.log`: the reads. The first-request A/B is in
  `docs/compat-reports/r3.33-first-request-paging-2026-11-06/`, on that item's branch.
