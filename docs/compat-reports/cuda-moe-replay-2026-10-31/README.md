# MoE routing trace and replay on CUDA, a real model (R3.21.2), 2026-10-31

The check the CPU's replay was held to (#499), on the CUDA routers: a run's own
trace replayed must override nothing and change nothing, and a control trace
must be followed and change the text.

- Model: `gpt-oss-20b-MXFP4.gguf` from ggml-org/gpt-oss-20b-GGUF at revision
  15dcfc7e7c608c6161507211ed43c5b590801b4b, sha256
  27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901 (24 MoE
  layers, 32 experts, 4 routed a token). Deleted after the run.
- Host: the Windows CUDA host (RTX 3070, 8 GB, driver 596.36, 16 GB RAM). The
  file does not fit the card: 15 of 24 layers ran on CUDA (6.8 GB) and 9 on
  the CPU, so the trace and the replay cross both routers in one run.
- Runner: this PR's branch (CUDA trace and replay, k_moe_route_replay in the
  pinned-toolchain PTX).
- Instrument: `real-replay.py` (here): greedy generation of 64 tokens from a
  fixed prompt, `-c 1024`, GPU auto; plain, traced (`RUNNER_MOE_TRACE`),
  replayed (`RUNNER_MOE_REPLAY` on that trace), and a rotated-layer control
  (each layer takes the next layer's recorded experts at the same position).

| run | forced | differed | uncovered | text |
|---|---|---|---|---|
| traced | | | | same as plain |
| self-replay | 7,776 (1,944 records x 4) | 0 | 0 | same as plain |
| rotated-layer control | 7,776 | 7,530 | 0 | changed |

The trace holds 1,944 records: positions 0 to 80 across all 24 layers, the 15
device layers' routings written by the CUDA routers and the 9 host layers' by
the CPU's.
