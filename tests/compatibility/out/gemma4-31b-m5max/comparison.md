# Runner vs llama.cpp comparison

## Provenance

- Schema: `xyntetik.runner.llamacpp-comparison.v1`
- Generated UTC: `2026-09-01T06:17:52Z`
- Status: `complete`
- Model path: `models/gemma-4-31B-it-Q4_0.gguf`
- Model SHA256: `031dc1c5fa9c5a0abbf3c39c5173fb2af65f5ac2dc2a090268561d3c72dcd834`
- Model bytes: `17992313088`
- Runner: `runner 0.4.4`
- Runner commit: `e8c9fbb1a601e5633972233bb19f2f65dd2dc254`
- llama.cpp: `version: 0.3.0-dev (build 1, commit 010be96)`
- llama.cpp commit: `010be968`

## Settings

- Context: `2048`
- Maximum generated tokens: `64`
- Quantization: `None`
- Temperature: `0`
- Top-p: `1`
- Sampling: `greedy`
- Prompt: `The capital of France is`

The TTFT request is a separate warmed streaming request after model load. The auxiliary top-k comparison sends the same chat payload to both runtimes; it is distinct from the raw-completion throughput request.

## Hardware and driver

```json
{
  "cpu_brand": "Apple M5 Max",
  "machine": "arm64",
  "nvidia_smi": null,
  "processor": "arm",
  "python": "3.12.14",
  "system": "macOS-26.5-arm64-arm-64bit"
}
```

## Commands

Runner: `runner -m models/gemma-4-31B-it-Q4_0.gguf --serve --no-tray --port 52233 -c 2048 --gpu auto -n 64`

llama.cpp: `llama-server -m models/gemma-4-31B-it-Q4_0.gguf --host 127.0.0.1 --port 52234 -c 2048 -ngl -1`

## Results

Derived columns are measured identically for every engine from the streaming response (see `derived_metrics`); self-reported columns come from each engine's own `timings` block, which not every engine emits and which engines do not all define the same way.

| Runtime | Derived prefill tok/s | Derived decode tok/s | Self-reported prompt tok/s | Self-reported decode tok/s | TTFT s | Tokens | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runner | 4.16 | 23.65 | 4.00 | 23.66 | 1.4437 | 64 | 2.76 |
| llamacpp | 39.36 | 26.31 | 51.90 | 26.05 | 0.1524 | 64 | 2.52 |

## VRAM

Load deltas are `nvidia-smi` used-memory changes from immediately before process start to server readiness; they are not peak VRAM.

```json
{
  "llamacpp_after_start": null,
  "llamacpp_load_delta_mib": null,
  "runner_after_start": null,
  "runner_load_delta_mib": null
}
```

## Correctness comparison

Text comparison: `pass`

Top-logprob comparison: `captured`
Maximum absolute common-token logprob delta: `None`
Correctness gate: `pass`

## Generated output

Runner:

```text
 Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is
```

llama.cpp:

```text
 Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is Paris.

The capital of France is
```

## Raw artifacts

The complete buffered responses, benchmark JSON, top-k values, exact requests, and VRAM snapshots are in `comparison.json`. Server output is in `runner.log` and `llamacpp.log` for real runs.

Real Qwen3/MoE GPU results are pending unless this report status is `complete`.
