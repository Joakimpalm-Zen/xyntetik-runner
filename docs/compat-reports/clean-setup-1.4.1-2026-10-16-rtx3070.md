# Clean-setup run: Runner 1.4.1 candidate on Windows, RTX 3070

2026-10-16, the Windows CUDA host (Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode. The
candidate binary built from the `release-1.4.1` branch at its release commit
(d6df86c9), before the tag, the steps of
[docs/workflow-opencode.md](../workflow-opencode.md) as written on that
branch, the example files fetched from it, and an empty model cache
(`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3 really downloaded;
70 s to the server). Result: **pass**: the agent's answer is the sentinel
word. One sentinel line, the answer: the model first guessed
`/workspace/NOTES.md`, the `external_directory` rule refused it, and it then
found `NOTES.md` with its glob tool and read it with its read tool, whose
output the transcript does not echo. The split is the same as 1.4.0's; the
RAM to spare is the host's free memory at the time. Home paths are shown as
`~/`.

## Driver output

```text
=== candidate binary (release-1.4.1 candidate, pre-tag)
runner 1.4.1
=== 2 fit on a ranged header read
  split         f16: 35 of 40 layers on the GPU, 3.41 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 1.76 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 1.02 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.71 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 5.84 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 70 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 360 s
sentinel lines: 1
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
