# Big-file limits exercised for real — M5 Max, 2026-09-01

The multi-buffer wrap, `--mlock`, and the residency warning were all written
against machines whose models fit comfortably inside every limit, and the
tree said so in three places ("every model this is developed on fits a
single MTLBuffer"). This session exercised each limit with real checkpoints
on Apple M5 Max, 128 GB unified RAM, macOS 26.5.

## The real buffer ceiling, measured

`MTLDevice.maxBufferLength` on the M5 Max is **86,586,540,032 bytes
(80.6 GiB)**; `recommendedMaxWorkingSetSize` is 115,448,725,504 (107.5 GiB).
The largest single file on the shelf — the merged 235B at 85,691,001,248
bytes (79.8 GiB) — fits ONE buffer with ~0.8 GiB to spare. So no on-disk
artifact crosses the real multi-buffer ceiling naturally, and none of the
big-model runs to date has taken the split path. A genuine crossing needs a
single GGUF above 80.6 GiB on this device class.

## Forced multi-GB wrap on a real model

Since the natural crossing is out of reach, the honest exercise is a forced
one at real scale: `make test-metal-bigmodel-multibuf BIGMODEL=<path>` (new,
opt-in by path like `test-metal-bigmodel`) caps `RUNNER_METAL_MAX_BUF` at
16 GiB and demands ≥2 buffers, no copy fallback, and byte-identical output
against the natural single-buffer run. Measured:
Llama-3.3-70B-Instruct-Q4_0 (40.1 GB) wraps into **3 buffers,
byte-identical** over 32 greedy tokens. This is the first time the split
path has carried multi-GB buffers with cuts on real tensor boundaries; the
fixture-scale `test-metal-multibuf` (up to 5 buffers on a 145 MB model)
remains the fast everyday gate.

## `--mlock` at 40 and 85.7 GB

Never before exercised above fixture scale (`src/compat.h` records the
16 GB-era caution). Both claims succeed and both release fully:

| model | mlock result | wired peak (vm_stat) | after exit |
|---|---|---|---|
| 70B Q4_0, 40.1 GB | `mlock: 40.1 GB of weights wired into RAM` | 45.3 GB (baseline 3.6) | 2.9 GB |
| 235B Q2_K, 85.7 GB | `mlock: 85.7 GB of weights wired into RAM` | 89.8 GB (baseline 2.9) | 2.9 GB |

macOS wired 85.7 GB of 128 GB for an unprivileged process without any
sysctl change. Notably, `plat_ram_available_bytes()` reported only 43.9 GB
"available" at that moment — the lock succeeded anyway, because available
is an instantaneous figure that excludes reclaimable page cache.

## The bug the 85.7 GB run found

With the lock reported successful, the loader still printed the sparse-MoE
residency warning — "expect … disk reads whenever routing reaches a cold
expert" — which is impossible for wired pages. `model_residency_warning()`
used `locked` only to trim the `--mlock` hint off the message, not to
suppress the prediction. Fixed: a locked model gets no eviction warning at
all, and `tests/test_paging_warn.c` now pins that (the old test pinned the
wrong behavior and was corrected red-to-green).

`test-file-identity` needed nothing: it already manufactures its own 5 GB
sparse file inside `make test`, so the >2 GB `stat()` cliff gate was never
vacuous on this host.
