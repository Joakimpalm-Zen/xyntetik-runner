# Clean-setup run: Runner 1.1.4 candidate on Windows, RTX 3070

2026-10-08, ZEN-GAMING (Windows 11, RTX 3070 8 GB, i7-7700K), OpenCode. The
candidate binary built from the `release-1.1.4` branch at its final commit
before the tag, the steps of
[docs/workflow-opencode.md](../workflow-opencode.md) as written on that
branch, the example files fetched from it, and an empty model cache
(`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3 really downloaded;
106 s to the server). Result: **pass**: the agent's answer is the sentinel
word. One sentinel line where 1.1.2 and 1.1.3 printed two: on this run the
model first asked for `/workspace/NOTES.md`, which OpenCode's directory rule
refused, then found the file with a glob and read it, so the read's echo did
not repeat the word. OpenCode sends `tool_choice: auto`, which the 1.1.4
change does not touch. Home paths are shown as `~/`.

## Driver output

```text
=== candidate binary (release-1.1.4 build, pre-tag)
runner 1.1.4
=== 2 fit on a ranged header read
  split         f16: 35 of 40 layers on the GPU, 1.22 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 0.47 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 0.00 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.00 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 6.00 GiB of RAM to spare at ctx 16384
fit rc=0
=== 3 serve: -hf into an empty cache, --kv q8
server up after 106 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 377 s
sentinel lines: 1
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```
