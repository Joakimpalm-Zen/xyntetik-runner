# Clean-setup run: Runner 1.5.0 candidate on Windows, RTX 3070

2026-10-21 evening (CEST), for the 2026-10-22 release: the Windows CUDA host
(Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode. This run used:

- the candidate binary built from the `release-1.5.0` branch at its release
  commit (8d680cf7), before the tag;
- the steps of [docs/workflow-opencode.md](../workflow-opencode.md) as written
  on that branch, with the example files fetched from it;
- an empty model cache (`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3
  really downloaded; 71 s to the server).

Result: **pass**. The agent's answer carries the sentinel word (one line). The
fit split is better than 1.4.2's on the same host (36 of 40 layers in f16
where 1.4.2 placed 35, and every layer with `--kv q8` where it placed 39),
because the token embedding now stays in RAM; the server placed all 40
layers with `--kv q8`. The RAM to spare is the host's free memory at the time.

## Driver output

```text
=== candidate binary (release-1.5.0 candidate, pre-tag)
runner 1.5.0
=== 2 fit on a ranged header read
  split         f16: 36 of 40 layers on the GPU, 3.50 GiB in RAM  | --kv q8: 40 of 40 layers on the GPU, 1.55 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 1.23 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.92 GiB in RAM
  verdict       FITS — 36 of 40 layers on the GPU, 5.84 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 71 s; G=40/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 104 s
sentinel lines: 1
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
