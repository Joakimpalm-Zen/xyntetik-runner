# Clean-setup run: Runner 1.4.2 candidate on Windows, RTX 3070

2026-10-17, the Windows CUDA host (Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode.
This run used:

- the candidate binary built from the `release-1.4.2` branch at its release
  commit (3fed02a6), before the tag;
- the steps of [docs/workflow-opencode.md](../workflow-opencode.md) as written
  on that branch, with the example files fetched from it;
- an empty model cache (`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3
  really downloaded; 426 s to the server, against 70 s on 1.4.1, the download
  being the difference).

Result: **pass**. The agent's answer is the sentinel word, and two lines carry
it. One is the `cat` of `NOTES.md` that the agent's tool printed, and the other
is the answer. The fit split is the same as 1.4.1's, and the RAM to spare is
the host's free memory at the time. Home paths are shown as `~/`.

## Driver output

```text
=== candidate binary (release-1.4.2 candidate, pre-tag)
runner 1.4.2
=== 2 fit on a ranged header read
  split         f16: 35 of 40 layers on the GPU, 3.41 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 1.76 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 1.02 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.71 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 5.84 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 426 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 395 s
sentinel lines: 2
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
