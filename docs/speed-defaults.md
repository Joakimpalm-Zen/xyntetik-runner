# Speed switches and their defaults

Runner turns a speed route on by default wherever it keeps the quality claim for that
model, backend or quant type. You should not need a flag to get the fast path. This page
lists every speed switch, its class, and what it does by default today.

**Classes.**
- **Identity:** the same tokens as the default route (byte-identical kernels, or
  speculative decoding, whose verify walk emits plain decoding's tokens). It is on by
  default wherever it measured faster on that workload.
- **Tolerance:** a different reduction or precision. It is on by default for a family,
  backend or type only with a row against the publisher's float32 reference within the
  house bar, and no worse than the current default on the paired test.
- **Never:** it changes sampling, or it is fitted to one workload.

A switch below is an environment variable unless it starts with `--`. The route line
(`RUNNER_DEBUG_ROUTE=1`) and the receipts name every route a run took, so a default is
never silent.

## Speculative decoding (identity class)

| Source | Default | Scope and evidence |
|---|---|---|
| `--draft lookup` | **On for a server request that carries `draft_hints`.** Off for every other request and for the CLI. | CUDA with every layer on the device, at width 7: 2.86x plain on an in-context edit (Qwen2.5-Coder-7B-Instruct Q4_K_M, RTX 3070). There is no prose row, so a request without hints decodes plain. Before this default, a request with hints was refused unless the server ran with `--draft lookup`. [Report](compat-reports/r3.3.7-draft-width-defaults-2026-11-03/README.md). |
| `--draft mtp` | **On for a one-shot CLI run whose output is one JSON document (`--json`, `--json-schema`) on Qwen3.5-4B Q4_K_M with its NextN block.** Off for every other model, mode and backend. | CUDA with every layer on the device, at width 2: 1.13x plain on a tool call and on code. On prose the head reads 0.88x, so free text decodes plain. If binding the head would cost the run its full offload, Runner reloads without the head and decodes plain. [Report](compat-reports/r3.3.7-draft-width-defaults-2026-11-03/README.md). |
| `--draft PATH` | Off. A draft model is the caller's choice. | |
| `--draft off` | Turns the defaults above off for the run or the server. | |

Not on by default, and why:
- **The NextN head per request on the server.** While the head is on, it is fed every
  token the target reads, prompt included. If it were fed always, every prose request
  would pay for it. If it were fed only when a request turned it on, a reused prefix
  would have no head state under it.
- **Metal and CPU.** On Metal a verify row costs a full decode step, so no drafted arm
  beats plain there. The CPU has no per-workload rows.

The startup line names a default and its scope, and the route line marks it as
`draft=mtp:default` or `draft=lookup:default`. A named source shows as `draft=lookup`.

## Already on by default (promoted with measured rows)

CUDA: the tensor-core GEMM per promoted combination (`RUNNER_CUDA_TC`), NVFP4
tensor-core prefill on the promoted architectures (`RUNNER_CUDA_TC_NVFP4`), the attention
routes `RUNNER_CUDA_ATTN_TC`, `RUNNER_CUDA_ATTN_TILE512` and their neighbours, graphs and
the small-batch route. Metal: the fused, matrix and matvec kernels and the cooperative
attention routes. CPU: batched and split attention, fused matvec. `RUNNER_MOE_GROUPED`
per family. `--kv-ring` and `--moe-prefetch` choose for themselves (`auto`).

## Off, with a failing row

These stay off unless a narrower scope gets its own passing row.

| Switch | Why it is off |
|---|---|
| `RUNNER_CUDA_I8` | Q8_0 reads 1.51x the default route's mean KL. |
| `RUNNER_CPU_I8_PREFILL` | 1.22x the default route's mean KL. |
| `RUNNER_CPU_I8K`, `RUNNER_CPU_I8` (decode) | Fail the bar. |
| `RUNNER_TIEDV` | Saves memory, not byte-identical. |

## Off, measured no gain

| Switch | Measurement |
|---|---|
| `RUNNER_METAL_TENSOR` | Parity. |
| `RUNNER_METAL_MOE_EM` | 3 to 4% slower. |
| `RUNNER_PREFETCH` on macOS | 60% slower cold load. |
| `RUNNER_PREFETCH` on Linux | No gain on a host capped at 60 read IOPS: time with it on over off 1.229 on the CPU, 1.006 and 0.9995 on CUDA. |
| `RUNNER_PREFETCH` on Windows | Slower on every row with CUDA: time with it on over off 1.159 cold, 1.122 and 1.118 warm on two files. On a CUDA load the hint runs after the upload has read the weights. |

## Candidates, not yet decided

| Switch | What it needs |
|---|---|
| `RUNNER_METAL_BATCH_TILES` | Its float32 rows pass (+40% at 8 slots); receipts must carry the route before it turns on. |
| `RUNNER_CUDA_Q8X` | Rows of its own. |
| `RUNNER_GRAMMAR_FF` | Its class established (speculation under a grammar). |

## Never on by default

`RUNNER_MOE_HOT` is fitted to one trace. `--top-k` on a preset with top-k 0 changes
sampling.
