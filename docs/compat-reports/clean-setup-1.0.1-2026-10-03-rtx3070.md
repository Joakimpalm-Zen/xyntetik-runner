# Clean-setup run: Runner 1.0.1 on Windows, RTX 3070

2026-10-03, ZEN-GAMING (Windows 11, RTX 3070 8 GB, driver 596.36,
i7-7700K), OpenCode 1.18.31. The released `runner-windows-x86_64.exe` from
the v1.0.1 GitHub release, the steps of
[docs/workflow-opencode.md](../workflow-opencode.md) exactly as written at
the v1.0.1 tag, the example files fetched from that tag, and an empty model
cache (`RUNNER_HF_CACHE` pointed at a fresh directory, so step 3 really
downloaded). Result: **pass**. Home paths are shown as `~/`.

The same run on 1.0.0 failed at step 3 (`--fit` said PAGES on this machine)
and at step 5 (the model's first path guess ended `opencode run`); both are
fixed in 1.0.1 (CHANGELOG).

## Driver output

```text
=== 1 download + checksum
runner-windows-x86_64.exe: OK
checksum rc=0
runner 1.0.1
=== 2 fit on a ranged header read, nothing running
fit: head.gguf
  model         granite, 40 layers
  weights       4.98 GiB
  kv cache      2.50 GiB at ctx 16384, f16   |  1.33 GiB with --kv q8   |  1.02 GiB with --kv k8v4   |  0.70 GiB with --kv fp4
  available RAM 9.63 GiB right now
  gpu           NVIDIA GeForce RTX 3070, offload budget 6.95 GiB right now (0.85 GiB of it for embeddings, scratch and headroom)
  split         f16: 35 of 40 layers on the GPU, 1.22 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 0.47 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 0.00 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.00 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 8.41 GiB of RAM to spare at ctx 16384
  note          KV is an upper bound: models with per-layer KV geometry (shared KV, MLA) use less
fit rc=0
=== 3 serve: -hf into an empty cache
server up after 87 s; G=39/40
16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311
=== 4-5 the example
opencode rc=0 in 358 s
sentinel lines: 2
✗ Read /work/NOTES.md failed
Error: The user has specified a rule which prevents you from using this specific tool call. Here are some of the relevant rules
=== 6 shutdown
server stopped
=== doctor
doctor rc=0
CLEANDONE
```

## OpenCode transcript

```text
> build · granite-4.2-8b-Q4_K_M.gguf

✗ Read /work/NOTES.md failed
Error: The user has specified a rule which prevents you from using this specific tool call. Here are some of the relevant rules [{"permission":"*","action":"allow","pattern":"*"},{"permission":"external_directory","pattern":"*","action":"ask"},{"permission":"external_directory","pattern":"~/.local\\share\\opencode\\tool-output\\*","action":"allow"},{"permission":"external_directory","pattern":"C:\\msys64\\tmp\\opencode\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\shadow\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\grill-me\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\xlsx\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\skill-creator\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\pptx\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\morning\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\pdf\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\import-memory\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\docx\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\google-workspace\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\docs\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\deep-research\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\computer-use\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\chrome-browser\\*","action":"allow"},{"permission":"external_directory","pattern":"~/.claude\\skills\\synced\\e9714a09-e99c-4ff1-b888-926ab7348037_2960f107-0e79-46d0-b9d1-26f7175d7c62\\built-in-browser\\*","action":"allow"},{"permission":"external_directory","action":"deny","pattern":"*"},{"permission":"external_directory","pattern":"~/.local\\share\\opencode\\tool-output\\*","action":"allow"}]

$ ls -la
totalt 6
drwxr-xr-x 1 zen Ingen   0  3 okt 16.22 .
drwxr-xr-x 1 zen Ingen   0  3 okt 16.22 ..
-rw-r--r-- 1 zen Ingen 134  3 okt 16.22 NOTES.md
-rw-r--r-- 1 zen Ingen 529  3 okt 16.22 opencode.json

$ cat NOTES.md
# Release notes draft

The request timeout moved from 30 to 60 seconds in this release.
The sentinel for this fixture is ORANGE-7319.

ORANGE-7319
```

## Server log

```text
hf: fetching ibm-granite/granite-4.2-8b-GGUF/granite-4.2-8b-Q4_K_M.gguf
hf: ibm-granite/granite-4.2-8b-GGUF:Q4_K_M -> ~/clean101/hf\ibm-granite--granite-4.2-8b-GGUF\granite-4.2-8b-Q4_K_M.gguf
gpu: OS video memory budget 7.60 GB for this process, 0.14 GB in use; offload budget 7.46 GB, headroom 0.54 GB
gpu: output weights do not fit alongside the layer split — running the last layer on the CPU so the boundary is consistent
gpu-split: budget=7.46GB fixed=0.92GB G=39/40 full=0 used=7.12GB
gpu: the file's tensor order spreads the 39 offloaded layers over 5.22 GB of the file against 4.88 GB of weights; uploading per tensor
gpu: CUDA backend on NVIDIA GeForce RTX 3070 (39/40 layers, 4.6 GB in VRAM; CPU runs the rest)
gpu: VRAM 0.32 GB free of 8.59 GB after init (kv 1.39 GB + scratch 0.15 GB this instance)
note: the KV cache for ctx 16384 is 1.39 GB on the device and 1 of 40 layers ran out of room because of it — a smaller -c or --kv k8v4 (about a quarter less) moves layers back
loaded ~/clean101/hf\ibm-granite--granite-4.2-8b-GGUF\granite-4.2-8b-Q4_K_M.gguf | granite | 40 layers | ctx 16384 | 4 threads (divided across slots) | 1.78s
sampling: granite42 (temp 1.00, top_p 0.95, top_k 0, min_p 0.00, repeat_penalty 1.00)
sampling: from the file (general.sampling.*): temp 1.00, top_p 0.95
server listening on http://127.0.0.1:8080 — 1 slot x 4 threads
  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions
  POST /v1/embeddings | POST /v1/rerank | POST /v1/messages | POST /v1/messages/count_tokens
  GET /v1/models | GET /v1/capabilities | GET /health | GET /metrics
  GET /v1/runner/prefix-cache | POST /v1/runner/prefix-cache/clear | POST /unload
  GET /v1/runner/provenance | POST /v1/runner/contexts | GET /v1/runner/contexts | DELETE /v1/runner/contexts/{id} | POST /v1/runner/contexts/{id}/snapshot
  GET /v1/responses/{id} | GET /v1/responses/{id}/input_items | DELETE /v1/responses/{id}
[slot 0] chatcmpl-0: start, 583 prompt (0 cached, none)
[slot 0] chatcmpl-0: 583 prompt (0 cached) + 1024 gen tok (21.3 tok/s) [183954 page-ins — weights not resident]
[slot 0] chatcmpl-1: start, 11151 prompt (5 cached, kv)
[slot 0] chatcmpl-1: 11151 prompt (5 cached) + 65 gen tok (11.1 tok/s)
[slot 0] chatcmpl-2: start, 12501 prompt (11181 cached, kv)
[slot 0] chatcmpl-2: 12501 prompt (11181 cached) + 50 gen tok (10.4 tok/s)
[slot 0] chatcmpl-3: start, 12675 prompt (12520 cached, kv)
[slot 0] chatcmpl-3: 12675 prompt (12520 cached) + 42 gen tok (10.4 tok/s)
[slot 0] chatcmpl-4: start, 12768 prompt (12685 cached, kv)
[slot 0] chatcmpl-4: 12768 prompt (12685 cached) + 49 gen tok (10.3 tok/s)
```

## `--doctor` (after shutdown, same file, `-c 16384 --kv q8`)

```json
{
  "schema": "xyntetik.runner.doctor.v1",
  "runner": {
    "version": "1.0.1"
  },
  "model": {
    "file": "granite-4.2-8b-Q4_K_M.gguf",
    "bytes": 5347917952,
    "architecture": "granite",
    "name": "Granite 4.2 8b",
    "layers": 40,
    "trained_context": 131072,
    "adapter": false
  },
  "placement": {
    "requested": "auto",
    "backend": "cuda",
    "gpu_layers": 39,
    "layers": 40,
    "device": "NVIDIA GeForce RTX 3070",
    "vram_total_bytes": 8589279232,
    "vram_free_bytes": 521854976
  },
  "context": {
    "tokens": 16384,
    "kv": "q8"
  },
  "template": {
    "name": "granite42",
    "recognised": true,
    "from_file": true,
    "tool_family": "qwen3_xml",
    "native_tool_protocol": true
  },
  "sampling": "granite42 (temp 1.00, top_p 0.95, top_k 0, min_p 0.00, repeat_penalty 1.00)",
  "memory": {
    "ram_total_bytes": 17109143552,
    "ram_available_bytes": 5104861184
  },
  "timings": {
    "load_s": 1.685,
    "threads": 4,
    "prompt_tokens": 28,
    "prompt_s": 0.346,
    "prompt_tok_s": 80.88,
    "generated_tokens": 1,
    "gen_s": 0.048,
    "gen_tok_s": 20.76
  },
  "probe": {
    "ran": true,
    "thinking": "off",
    "stopped_by_itself": true,
    "answer_found": true,
    "empty": false,
    "template_markup_in_reply": false
  },
  "findings": [
    {
      "severity": "degraded",
      "code": "partial_offload",
      "detail": "only some layers are on the GPU, so every token crosses between device and host",
      "next_step": "use a smaller quantization or a shorter -c so the whole model fits, or accept the split"
    }
  ],
  "verdict": "degraded",
  "shareable": "no prompt or reply text; file name and machine sizes only"
}
```
