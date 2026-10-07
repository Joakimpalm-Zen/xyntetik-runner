# Clean-setup run: Runner 1.1.0 candidate on Windows, RTX 3070

2026-10-07, ZEN-GAMING (Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode. The
candidate binary built from the `release-1.1.0` branch before the tag (the
1.0.1 lesson), the steps of [docs/workflow-opencode.md](../workflow-opencode.md)
as written on that branch, the example files fetched from it, and an empty
model cache (`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3 really
downloaded; 319 s to the server, the box's line). Result: **pass**. Home paths
are shown as `~/`.

## Driver output

```text
=== candidate binary (release-1.1.0 build, pre-tag)
runner 1.1.0
=== 2 fit on a ranged header read
  split         f16: 35 of 40 layers on the GPU, 1.22 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 0.47 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 0.00 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.00 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 6.81 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 319 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 379 s
sentinel lines: 1
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
