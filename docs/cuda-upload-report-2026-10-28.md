# A silent weight upload read as a hang (2026-10-28)

**Report:** `runner -m <4B bf16> --lora <rank-8 adapter> -p <110 tokens> --score` on the lab's
24 GB MIG slice printed its `gpu-split` lines and then nothing for nine minutes, process alive,
and was killed. The same command with the GPU off finished in about 90 s; the same adapter served
over the Q4_K_M base on the GPU answered 29 completions in 21 s. Read as "`--score` with `--lora`
hangs on the CUDA path".

**Cause:** the upload of the weights, not the adapter and not the scorer. The CUDA backend copies
the offloaded layers to the device straight from the model file's mapping in one `cuMemcpyHtoD`,
so the copy runs at the speed the storage faults the pages in. The box keeps its model files on
an array whose reads are capped (about 90 MB/s and 60 IOPS at best, shared with other tenants),
and the bf16 file had never been read on this host: the stderr of the killed run ends before the
`gpu: CUDA backend on ...` line that follows the upload, and before the `lora:` line that follows
the backend line. The adapter code had not run. The Q4_K_M runs were fast because the CPU runs
minutes earlier had left that file in the page cache; the later CPU run of the bf16 file was fast
because the killed upload had already pulled part of it in and the CPU path reads at the
bandwidth cap.

**Measured on the box, CPU only (no slice):**

| Probe | Result |
|---|---|
| `mincore` on the bf16 and Q4_K_M files after the day's runs | 8.05 of 8.05 GB and 2.50 of 2.50 GB resident |
| a never-read 8.05 GB F16 file | 0.27 GB resident |
| cold sequential faulting through the mapping, 2 MB pieces | 7.6 MB/s |
| the same with `readahead(2)` ahead of 64 MB pieces | 7.7 MB/s (the array, not the fault pattern, is the limit) |
| `dd bs=64M` on the same cold file | 7.7 MB/s |

At 7.7 MB/s an 8.05 GB file is 17 minutes; the killed run had waited nine.

**Fix (this change):** every copy of weight bytes to the device goes through one chunked helper
(64 MB pieces). After two seconds of copying it says, every five seconds, how much is on the device,
the rate from the file and the time left, then one closing line with the whole upload's rate. A
warm load prints nothing new. Policy in `src/upload_report.h`; `tests/test_upload_report.c` drives
it with a synthetic clock; `tests/test_cuda_upload_report.py` holds every weight copy in `cuda.c`
to the helper (it found the tensor-run binding path on its first run).

**Demonstration on the slice with the fixed binary (2026-10-28, 16:15 to 16:19 UTC):**

The cold F16 base with the adapter, `--score`, cut at 150 s:

```
gpu-split: budget=25.13GB fixed=0.83GB G=36/36 full=1 used=8.70GB
gpu-split: prompt window 512 columns (batch 512, 15.65 GB spare, 181252 bytes a column with TC2's activations)
gpu: weights 0.07 of 8.05 GB on the device after 8 s (8 MB/s from the file; about 976 s more)
gpu: weights 0.13 of 8.05 GB on the device after 17 s (8 MB/s from the file; about 1026 s more)
...
gpu: weights 0.87 of 8.05 GB on the device after 113 s (8 MB/s from the file; about 929 s more)
```

The reported command, bf16 base with the adapter, `--score` on the GPU, file warm:

```
gpu: CUDA backend on <the slice> (8.0 GB weights in VRAM)
lora: <adapter> - 252 adapted projections, alpha 16, scale x1
gpu: adapter on device for 36 blocks (66.1 MB VRAM)
score: route {"backend":"cuda", ... "gpu_layers":36, "layers":36, ... "paths":{"solo":{"forwards":128,"rows_max":1}} ...}
```

12.8 s wall; a valid `score.v1` object, 129 tokens, 128 scored. `--score` with `--lora` on CUDA
works; it was waiting for the disk.

**What this does not do:** make the disk faster. On that array a cold 8 GB file is a quarter of
an hour whichever path reads it; the box's practice of prefetching with `dd bs=64M` before a run
stands, and the new lines say when it was skipped.
