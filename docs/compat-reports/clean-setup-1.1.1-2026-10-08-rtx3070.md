# Clean-setup run: Runner 1.1.1 candidate on Windows, RTX 3070

2026-10-08, ZEN-GAMING (Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode. The
candidate binary built from the `release-1.1.1` branch at its final commit
before the tag (the device head, CUDA speculative verify, the small-batch GEMV
routing and the lane-level i-quant kernels all in), the steps of
[docs/workflow-opencode.md](../workflow-opencode.md) as written on that
branch, the example files fetched from it, and an empty model cache
(`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3 really downloaded;
85 s to the server). Result: **pass**. Home paths are shown as `~/`.

## Driver output

```text
=== candidate binary (release-1.1.1 build, pre-tag)
runner 1.1.1
=== 2 fit on a ranged header read
  split         f16: 35 of 40 layers on the GPU, 1.22 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 0.47 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 0.00 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.00 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 8.06 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 85 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 385 s
sentinel lines: 1
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
