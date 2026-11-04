# #574 evidence row: `RUNNER_CUDA_ATTN_TC_WIDE` on Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head (qwen4exp, lab, 2026-11-04)

**Verdict: no widening to qwen4exp.** The 4K prefill reads **0.992x** with `k_attn_tcw` on, with 26 of 48 layers on
the CPU. 16K and the D1 read were not run. The default stays as it is: gemma-4's E-series (R3.6.3.1, #574), plus
qwen35 at head width 256 if R3.6.3.2 (#607) lands.

**The identity gate passes with pinned routing (4.42e-6 of the logit range); free routing tips 3 of 45,600 expert
slots.**

- **Free routing** fails with `k_attn_tcw` on (`=1`): 2.41e-3 of the range against the routed limit of 2e-3, where
  `=0` reads 5.7e-6.
- **R1.25's bisect (2026-11-04) shows that is not the kernel.**
  - Every one of the 15 device attention calls at `=1` matches a float64 reference over its own inputs to at most
    1.73e-6, as the `=0` tile does.
  - With the routing pinned to the CPU's expert choices, `=1` passes at 4.42e-6 of the range (`=0`: 5.66e-6; limit
    1e-3).
  - The free run's miss is 3 of 45,600 expert slots tipping at near-ties.
  - The raw attention dumps stay with the lab.

## The run

- **Build:** runner main 7567c94 on the CUDA MIG slice (1g.24gb, sm_120).
- **File:** the row's four parts, `Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF` at c104e657 (`trunk-with-head/`, 86.4
  GB), streamed into memory with each part's sha256 checked against the Hub's LFS record (`FILES.sha256`).
- **Placement:** CUDA with 22 of 48 layers on the device and no expert banks on the host. The route record shows
  `attn_wide=on` at `=1` (`bench-4k-probe.json`).
- **The file's attention:** 24 query heads on 2 KV heads (12 per group), key and value width 256, full attention one
  layer in four, an attention indexer (4 heads, key 128, top-k 2048), and hyper-connections.

## gpu-identity (`tests/test_gpu_identity.c`)

The run was `CUDA_VISIBLE_DEVICES=<the slice> RUNNER_CUDA_ATTN_TC_WIDE=1|0 ./test-gpu-identity <part 1>`: 96 tokens,
24 teacher-forced positions, prefill batch 32, 5,959,680 logits.

| Setting | Free routing: mean \|Δlogit\| / range (limit 2e-3) | Worst | Pinned routing (limit 1e-3) | Verdict |
|---|---|---|---|---|
| `=1` | 0.0729 / 30.2 = **2.41e-3** | 2.34 | not reached | **FAIL** |
| `=1`, `CUDA_LAUNCH_BLOCKING=1` | 0.0729 / 30.2 = 2.41e-3 (bit-identical) | 2.34 | not reached | FAIL |
| `=0` | 0.000173 / 30.2 = 5.74e-6 | 0.0044 | 5.66e-6 (8 of 45,600 expert slots pinned off the device's choice) | ok |
| `=1`, pinned arm run after the free arm (R1.25, a lab-only test switch) | (the free arm as above) | | **4.42e-6** (3 of 45,600 slots pinned off the device's choice), worst 0.00306 | ok |

Files: `gpu-identity-tcw1.log`, `gpu-identity-tcw1-blocking.log`, `gpu-identity-tcw0.log`,
`gpu-identity-tcw1-pinned.log`.

- **The blocking run reads the same numbers bit for bit,** so the failure is deterministic. A missing stream
  ordering on the 22/48 split path is ruled out.
- **The kernel itself runs correctly elsewhere.** Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered (qwen35: 24 heads on 4 KV heads,
  6 per group, key 256, full offload) passes `=1` at 6.79e-6 with `k_attn_tcw` dispatched (the R3.6.3.2 report,
  #607).
- **What separates the two files:** the group size (12 against 6), the attention indexer, hyper-connections and the
  partial split. A code read rules out the group size alone, since the `=0` tile route has the same head map at 12.
  On the 24:2 toy (R1.25, #608), run on the Windows CUDA host, the kernel matches its reference at a group of 12.

## The tier smoke against the pin

The file matches the row's pin with `=1` and with `=0` (8.3-8.4 s, fact found; `tier-smoke-tcw*.json`). The pin is 48
greedy tokens at ctx 512, too short to show the drift the identity gate measures.

## Prefill A/B (4K only)

The method is the one the R3.6.3.2 report used: `--bench-json -f prompt-4k.txt -c 4608 -n 1 -b 512 -t 40`, one
discarded warm-up, five rounds interleaved with the order alternating, 4,005 tokens (`bench-4k.jsonl`,
`bench-4k-*.err`).

| `=0` prompt tok/s | `=1` prompt tok/s | Ratio of medians |
|---|---|---|
| 33.6, 33.5, 33.5, 34.2, 33.9 (median 33.6) | 33.4, 31.9, 33.3, 34.4, 31.7 (median 33.3) | 0.992x |

With 26 of 48 layers on the CPU, the prefill is CPU-bound and the wide kernel has nothing to win.

**16K: not run.** The identity gate failed first, which keeps the default where it is, so the D1 read (gated on 1.2x
at 16K) does not run either. `bench-16k-warmup.err` is the warm-up's log from before the stop; `prompt-16k.txt` is
kept for a rerun.

## Not measured

- 16K and the D1 read.
- The layer where `=1` first departs from `=0` under free routing.
- A full-offload placement (the file is larger than the slice).

## Files

- **The lab's package:** 29 files and a `SHA256SUMS` that hashes to `547141b4186419340974f57afaafc01ab3d7c9f8...`.
  Every digest was re-checked on the Windows CUDA host after extraction.
- **Committed here:** 27 of the 29, unchanged. Two are left out:
  - the lab's `RESULT.md` (sha256 `6093f3bd5c7b48a3...`), which this README carries with session names replaced by
    roles;
  - the run's timeline `fn.log` (`03f11dee229511d7...`), whose numbers are all in the files above.
- **`SHA256SUMS`** covers every file here except itself and this README.
