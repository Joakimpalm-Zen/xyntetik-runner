# R3.6.3.1 on gemma-4-E2B (2026-11-03)

`k_attn_tcw`'s default family (gemma4 with per-layer embeddings) covers gemma-4-E2B, which had no row when the
default was set. This is its row, on the Tier 2 file `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`
(Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF at `37c8fa43`, sha256 `16ffaa2ea568e7a0...`, 2,631,015,904 bytes),
runner main 36b17876, the Windows CUDA host (RTX 3070 8 GB, driver 596.36).

The kill number, set before the first measurement: E2B leaves the default family if `test-gpu-identity` with
the kernel on reads over its 1e-3 limit, if the CUDA cell's reply differs from its pin, or if 16K prefill
with the kernel reads under 1.0x of the tile. None did.

## Route

With `RUNNER_CUDA_ATTN_TC_WIDE` unset, the route records `attn_wide=on` (35 of 35 layers on the device);
under `=0` it does not.

## Gates

- `test-gpu-identity`: 2.84e-5 of the mean logit range with the kernel (`=1`), 1.39e-5 with the tile (`=0`);
  limit 1e-3.
- The Tier 2 row's CUDA cell on the new default: the reply byte-identical to its pin (output sha256
  `4132fa520bc517ed`), recorded in `docs/model-tiers.json` at 36b17876.
- The 32 greedy tokens after each prompt are identical with and without the kernel at 4K and 16K.

## Speed

`-n 32 --temp 0`, the tile (`=0`) and the kernel (`=1`) alternated, two rounds, warm cache, prompt tok/s:

| Prompt | Tile | `k_attn_tcw` | Gain |
|---|---|---|---|
| 3,901 tokens (`-c 4800`) | 1008.4 / 1020.8 | 1455.3 / 1521.2 | 1.47x |
| 16,251 tokens (`-c 17600`) | 528.0 / 521.4 | 1230.0 / 1227.5 | 2.34x |

Decode is unchanged (90.5 to 92.2 tok/s at 4K, 45.5 to 45.8 at 16K, either way).

## Not read

No float32-reference row on this file: the default's D1 row is Bolt E4B's (in `d1-result.md`). This mix file
already fails the fidelity bar against its BF16 parent by its own row's claim.
