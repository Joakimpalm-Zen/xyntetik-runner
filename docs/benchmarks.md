# GPU benchmarks — Runner vs llama.cpp (CUDA)

First published 2026-07-29, runner `94ce01f`; MoE rows updated the same day
on runner `26cf7e6` after the device-routing work landed. One machine, one
method, both engines on the same files — and the losing rows published
alongside the winning ones.

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q, **MIG 1g.24gb slice** (not the full card) |
| CPU | AMD Ryzen Threadripper (128 threads available; runner used its defaults) |
| Runner | `94ce01f` (2026-07-29), built `-O3 -march=native`, CUDA via `sm_75` PTX, JIT to `compute 12.0` |
| llama.cpp | build `ea12b27` (CUDA), same GGUF files |
| Method | 512-token prefill / 128-token greedy decode (`--temp 0`), full GPU offload (verified `full=1` per run), median of 3 runs; llama.cpp via `llama-bench -p 512 -n 128 -ngl 99 -r 2` |

**MIG caveat:** a 1g.24gb slice has roughly ~200 GB/s of memory bandwidth and a
fraction of the card's SMs. Absolute tok/s will differ on other hardware; the
*ratios* between the two engines are the meaningful result, and even those shift
with the compute/bandwidth balance (the same kernels measured different ratios
on an RTX 3070).

## Results — 2026-09-02, three hosts, runner `6d705e9`

Same method as the 2026-08-13 table on the MIG (`--bench-json -n 128 -b 64`,
mean of 2; `llama-bench -p 512 -n 128 -ngl 99 -r 2`; full offload verified
`full=1` on every row), plus two consumer hosts. Absolute numbers on the MIG
box are lower than in August on BOTH sides (llama.cpp's Llama-3.2-3B decode
reads 101.8 against 169.0 then) because the box carried a load average of 18
from other work during the run and the driver moved to 610.43; the ratios are
the result, as the setup section says.

### MIG 1g.24gb, llama.cpp b10353 CUDA (same build as August)

| model | quant | decode tok/s: runner / llama.cpp | prefill tok/s: runner / llama.cpp |
|---|---|---|---|
| Llama-3.2-3B | Q4_K_M | 87.9 / 101.8 (**86%**) | 525.7 / 6387.6 (8.2%) |
| Phi-4-mini | Q8_0 | 53.8 / 57.7 (**93%**) | 355.6 / 5960.6 (6.0%) |
| granite-3.3-8b | Q4_K_M | 39.7 / 45.4 (**87%**) | 220.5 / 2288.9 (9.6%) |
| granite-4.1-8b | Q4_0 | 41.7 / 46.5 (**90%**) | 151.1 / 2359.2 (6.4%) |
| Phi-3.5-mini | Q4_K_M | 72.0 / 83.2 (**87%**) | 284.9 / 5120.3 (5.6%) |
| gemma-4-12B | Q4_K_M | 23.2 / 28.8 (**81%**) | 123.7 / 1439.2 (8.6%) |
| Qwen3-30B-A3B (MoE) | Q4_K_M | 62.2 / 86.6 (**72%**) | 128.5 / 2326.3 (5.5%) |
| gemma-4-26B-A4B (MoE) | Q4_0 | 23.3 / 56.2 (41%) | 34.3 / 2429.7 (1.4%) |
| Qwen2.5-32B | Q3_K_S | 1.8 / 14.3 (13%) | 15.3 / 469.0 (3.3%) |

Dense decode moved from 77-87% to 81-93% and the MoE rows from 67% and 22%
to 72% and 41% (the KV-ring accounting fix and the August decode work);
prefill is unchanged at 5-10% and stays the column that decides the next
piece of work. The Q3_K row is still the naive token-identical kernel.

### RTX 3070 8 GB (Windows 11, driver 596.36), llama.cpp b10754 CUDA 13.3 prebuilt

| model | quant | decode tok/s: runner / llama.cpp | prefill tok/s: runner / llama.cpp |
|---|---|---|---|
| Qwen2.5-7B | Q4_K_M | 65.6 / 81.6 (**80%**) | 112.3 / 2224.5 (5.0%) |
| granite-4.1-3b | Q8_0 | 78.6 / 94.3 (**83%**) | 493.5 / 5614.1 (8.8%) |
| Qwen2.5-3B | Q4_K_M | 113.3 / 161.2 (**70%**) | 234.0 / 6069.9 (3.9%) |

A desktop card with a display attached, a newer llama.cpp than the MIG
table's, and the same shape of result: decode within 20-30%, prefill an
order of magnitude behind.

### Apple M1 8 GB, Metal, llama.cpp from Homebrew (ggml 0.22.0), 2026-09-02

| model | quant | decode tok/s: runner / llama.cpp | prefill tok/s: runner / llama.cpp |
|---|---|---|---|
| granite-4.1-3b | Q8_0 | 11.6 / 14.4 (**80%**) | 47.8 / 266.5 (17.9%) |
| gemma-3-4b | Q4_K_M | 8.4 / 20.3 (41%) | 62.9 / 250.3 (25.1%) |

Measured with 1.3 GB of RAM available: the runner logged its own warning
("weights are 3.6 GB but only 1.3 GB of RAM is available, expect the model to
be evicted") and paged, while llama.cpp wires the model into memory. That is
the coexistence trade the README describes, read from the losing side: on a
pressured machine the resident engine decodes faster and the evictable one
gives the memory back. The gemma row is that effect, not a kernel result;
re-measure on a quiet M1 before reading it as one.

## Results — 2026-08-13, both sides re-measured on this box

The 2026-07-29 table below is kept as history. It was taken before a CUDA
prefill correctness bug was found (`TC_GEMM_32B` published uninitialised shared
memory for token columns 16..63 — its Q8_0 prefill rows were therefore
measuring a kernel that was not computing what it reported), and before Q4_0,
the granite arch and the Q4_0 decode GEMV landed. This table replaces it: both
engines re-run on the same slice, same files, same day, on the fixed kernel.

| model | quant | decode tok/s: runner / llama.cpp | prefill tok/s: runner / llama.cpp |
|---|---|---|---|
| Llama-3.2-3B | Q4_K_M | 130.3 / 169.0 (**77%**) | 735.1 / 8440.6 (8.7%) |
| Phi-4-mini | Q8_0 | 80.2 / 92.2 (**87%**) | 509.8 / 8397.0 (6.1%) |
| granite-3.3-8b | Q4_K_M | 61.0 / 73.9 (**83%**) | 326.4 / 3335.8 (9.8%) |
| granite-4.1-8b | Q4_0 | 64.4 / 75.9 (**85%**) | 230.4 / 3710.6 (6.2%) |

Runner via `--bench-json -n 128 -b 64` (mean of 2, spread < 2.2%); llama.cpp
b10353 built with `-DGGML_CUDA=ON` from the same source tree on this box, via
`llama-bench -p 512 -n 128 -ngl 99 -r 2`.

**The reference reproduces the published one.** llama.cpp measures 8440.6 /
169.0 on Llama-3.2-3B against the 8373.6 / 169.0 recorded on 2026-07-29 with a
different build — so the denominators in the old table were sound, and the
movement in the ratios is runner's.

**Decode: 77-87%, up from 73-79%.** The granite-4.1-8b Q4_0 row (85%) is new
coverage rather than tuning: Q4_0 had no coalesced decode GEMV until
2026-08-13 and ran at 11.9 tok/s, which was slower than the same model on the
CPU.

**Prefill: 6.1-9.8%, up from 4.3-5.6%,** and still the honest weak column.
The gain came from admitting types and architectures to the tensor-core path
(Q6_K 2026-08-08, Q4_0 + granite 2026-08-13), not from a faster kernel — the
tile is still 64 columns wide where llama.cpp's stack is deeper. See
[performance.md](performance.md) for what was measured and rejected.

## Results — default configuration, 2026-07-29

"Default" means what each engine does out of the box on these files. For Runner
that includes the tensor-core prefill GEMM on the seven gated dense (Q4_K, arch)
combos (promoted 2026-07-29 behind a measured tolerance gate; see
[the TC spec](specs/2026-07-22-tensor-core-gemm-scope.md)) and the scalar path
everywhere else.

| model | quant | decode tok/s: runner / llama.cpp | prefill tok/s: runner / llama.cpp |
|---|---|---|---|
| Llama-3.2-3B | Q4_K_M | 130.7 / 169.0 (**77%**) | 438.1 / 8373.6 (5.2%) |
| Phi-3.5-mini | Q4_K_M | 112.4 / 142.8 (**79%**) | 302.3 / 6965.6 (4.3%) |
| gemma-4-12B | Q4_K_M | 35.9 / 48.9 (**73%**) | 131.8 / 2349.0 (5.6%) |
| Qwen3-30B-A3B (MoE) | Q4_K_M | 102.2 / 151.7 (**67%**) | 194.0 / 3233.5 (6.0%) |
| gemma-4-26B-A4B (MoE) | Q4_0 | 24.7 / 114.2 (22%) | 23.6 / 3694.2 (0.6%) |
| Qwen2.5-32B | Q3_K_S | 3.0 / 25.1 (12%) | 1.4 / 794.4 (0.2%) |

## Reading the numbers honestly

**Dense decode is the story: 73–79% of llama.cpp.** Single-stream decode is
memory-bandwidth-bound on this slice — both engines read the same quantized
weights per token — so parity is the physical target, not victory. Runner's
decode GEMVs (aligned 8-byte quant loads, `float4` activation loads, factored
per-group affine) close most of the remaining gap while keeping the engine
dependency-free.

**Prefill is llama.cpp's win, and we publish it as such.** llama.cpp's prefill
throughput comes from a mature tensor-core GEMM stack across every quant.
Runner's TC path covered Q4_K and Q8_0 when this table was taken, and lifted
promoted dense models from ~3% to ~4–6% of llama.cpp. It now also covers Q6_K
(2026-08-08) and Q4_0 (2026-08-13, with the granite arch), and the 32-byte-block
kernel it shares was carrying a 48-of-64-columns bug until 2026-08-13 — see the
re-measured table above. Further coverage is tracked work, and the gap is
reported, not hidden.

**Known-slow rows are kept in the table.** Q3_K decode (12%) uses a
token-identical but naive kernel — its rewrite is a tracked item, including the
measured root cause (accumulator spill to local memory at the widened tile).
MoE decode reached 67% via device-side routing and fused indirect expert
matvecs (2026-07-29); gemma's dual-branch MoE (22%) and MoE prefill remain
the tracked remainders. The certified byte-identity property for MoE is
defined over the eager routing path (`RUNNER_MOE_EAGER=1`, pinned in the
certification harnesses); the fused default is verified
selection-identical with routing weights within ~2 ulp.

**Correctness gates every speed number.** The scalar path is certified
token-identical CPU vs GPU (and against a pinned llama.cpp revision where
recorded — see [the compatibility program](compatibility-program.md)). The
tensor-core path is fp16-tile arithmetic and is instead held to a teacher-forced
tolerance gate (`make test-tc-tol`): every promoted row measured 0/64 top-1
flips and ≤0.012% mean logit deviation; in free-running checks to date its
greedy output has matched the scalar path exactly.

## Trajectory (same box, same method, same llama.cpp build)

| model | decode, 2026-07-25 | decode, 2026-07-29 | prefill, 07-25 | prefill, 07-29 |
|---|---|---|---|---|
| Llama-3.2-3B | 52% | **77%** | 3.1% | **5.2%** |
| Phi-3.5-mini | 50% | **79%** | 2.9% | **4.3%** |
| gemma-4-12B | 48% | **73%** | 3.1% | **5.6%** |
| Qwen3-30B-A3B | 36% | **67%** | 2.4% | **6.0%** |
| gemma-4-26B-A4B | 21% | 22% | 0.8% | 0.6% |
| Qwen2.5-32B | 12% | 12% | 0.3% | 0.2% |

Six days of kernel work (decode GEMV bandwidth pass, MVB-16 tiles, MMQ-style
TC prefill GEMM + its tolerance gate, MoE device-side routing + fused
indirect expert matvecs) — plus one silent MoE GPU→CPU fallback found by
this benchmark's own control run, fixed and now guarded by a test.

## Reproducing

```sh
# runner rows (median of 3; verify the gpu-split line reports full=1)
./runner -m model.gguf -f prompt-512tok.txt -n 128 --temp 0 -s 1 \
         --ignore-eos --gpu-layers 99

# llama.cpp rows
llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -r 2

# pin runner's scalar path (byte-identical CPU==GPU) if comparing outputs
RUNNER_CUDA_TC=0 ./runner ...
```
