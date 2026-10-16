# Clean-setup run: Runner 1.4.0 on Windows, RTX 3070

2026-10-16, the Windows CUDA host (Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode. The
binary built from the `v1.4.0` tag (62f8c400, after the release), the steps
of [docs/workflow-opencode.md](../workflow-opencode.md) as written at that
tag, the example files fetched from it, and an empty model cache
(`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3 really downloaded;
71 s to the server). Result: **pass**: the agent's answer is the sentinel
word. Two sentinel lines: the `cat NOTES.md` tool output and the answer. As
in 1.3.0, the model first guessed `/workspace/NOTES.md`, the
`external_directory` rule refused it, and it then listed the directory and
read `NOTES.md`. The split is the same as 1.3.0's; the larger RAM to spare
(5.77 GiB against 4.56 GiB) is the host's free memory at the time, since the
figure is available RAM minus the 3.41 GiB the split keeps in RAM. Home paths
are shown as `~/`.

## Driver output

```text
=== candidate binary (v1.4.0 tag build)
runner 1.4.0
=== 2 fit on a ranged header read
  split         f16: 35 of 40 layers on the GPU, 3.41 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 1.76 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 1.02 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.71 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 5.77 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 71 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 366 s
sentinel lines: 2
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
