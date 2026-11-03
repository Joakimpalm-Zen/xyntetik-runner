# R3.29: CUDA decode attention in one pass (2026-11-03)

**Unflattering first.**
- Decode at depth is still short of llama.cpp b11748 on the same card: 0.86x at 16K on Llama-3.2-3B, and only
  0.66x on Qwen2.5-3B, whose eight query heads per KV head make the kernel bound by shared-memory work, not by
  DRAM.
- A split count set by geometry (more slices for a file with fewer KV heads) was tried and reverted. On
  Qwen2.5-3B it read +10% at 16K and -3.5% at 512, and on Qwen2.5-7B nothing.
- The Qwen3-4B 32K row is unstable: at that context its fp16 cache leaves 3 of 36 layers on the CPU, and the
  one-pass arm read 32.5 and 42.0 tok/s in its two rounds. It is shown, not read.
- At 512 tokens the change is worth 0 to 2%.
- The read against the publisher's float32 model (D1) is owed: the lab runs it (below).

## What was slow

Before any code, Llama-3.2-3B Q4_K_M at 17.4K on the RTX 3070 (main 3d541f61, `RUNNER_CUDA_PROFILE=1`):
attention took 10.8 of 18.6 ms a decoded token (58%). The decode path there is already `k_attn_dec_gqa`, one
block per KV head and slice serving the head group, at 32 slices. `ATTN_SPLITS 8` belongs to the fallback
kernel only. Nsight Compute on one layer: 380 us, 107 MB read from DRAM, 282 GB/s (63% of peak), 53% warps
active. The layer's fp16 cache is 72 MB. The kernel read K twice: a pass over every K tile for the maximum
alone, then K again with V for the probabilities and the weighted sum. The floor is about 160 us a layer
(72 MB at 448 GB/s).

## The change

- **One pass.** A tile's K and V rows are staged together. Each head keeps a running maximum; the tile's
  `expf(s - max)` are taken against it, and the sum and the value accumulators are rescaled by
  `expf(old - new)` where it rises. The partials (un-normalised V, max, sum) and `k_attn_merge32` are
  unchanged.
- **Staging.** For fp16 caches with head_dim a multiple of 64 and 16-byte-aligned rows, the tile moves as
  16-byte chunks. The next tile is loaded into registers while this one computes, and chunk `c` of row `k` is
  stored at `c ^ (k & 7)` so the stores, the per-key score reads and the value reads cover every bank. The
  score is `kv_dot`'s fp16 arithmetic in its order, read through the swizzle. Other caches (q8, fp4, other head
  widths) keep word staging, in the same one pass.
- **`__launch_bounds__(256, 3)`.** 80 registers, three blocks an SM. Uncapped it took 125 registers (two
  blocks); capped at four blocks it spilled and read 286 us.

One layer at 17.4K after the change: 227 us, 71.4 MB read (the cache once), 74% of DRAM peak.

## Speed

The Windows CUDA host: RTX 3070 8 GB, driver 596.36, i7-7700K. Runner main 853e478c against the branch at
ff613375, and llama.cpp b11748 (3bb63cb, the win-cuda-13.4 release). Same session, two rounds with the arm
order reversed in the second, warm cache, the 2026-10-21 campaign's driver: runner
`--bench-json -n 128 -c L+384 --gpu-layers 99 -t 4`, llama.cpp `llama-bench -p 0 -n 128 -d L -ngl 99 -r 2`.
Decode tok/s, median of the rounds. Every round within 1% of its median except the Qwen3-4B 32K rows.

| File | L | main | one pass | gain | llama.cpp | one pass / llama.cpp |
|---|---|---|---|---|---|---|
| Llama-3.2-3B Q4_K_M | 512 | 136.0 | 136.2 | 1.00x | 155.5 | 0.88 |
| | 4K | 102.7 | 112.6 | 1.10x | 134.9 | 0.83 |
| | 8K | 83.2 | 98.8 | 1.19x | 117.5 | 0.84 |
| | 16K | 59.2 | 79.6 | 1.34x | 92.7 | 0.86 |
| | 32K | 38.3 | 57.3 | 1.50x | 65.1 | 0.88 |
| Qwen2.5-3B Q4_K_M | 512 | 136.4 | 139.0 | 1.02x | 160.3 | 0.87 |
| | 4K | 104.6 | 118.8 | 1.14x | 151.3 | 0.78 |
| | 8K | 84.8 | 105.0 | 1.24x | 143.0 | 0.73 |
| | 16K | 62.0 | 85.3 | 1.38x | 129.3 | 0.66 |
| | 32K | 40.2 | 62.4 | 1.55x | 108.3 | 0.58 |
| Qwen3-4B Q4_K_M | 512 | 104.3 | 104.9 | 1.01x | 120.9 | 0.87 |
| | 4K | 77.7 | 85.6 | 1.10x | 104.0 | 0.82 |
| | 8K | 61.8 | 74.7 | 1.21x | 90.7 | 0.82 |
| | 16K | 43.0 | 59.6 | 1.38x | 71.6 | 0.83 |
| | 32K (unstable) | 27.5 | 32.5 / 42.0 | | 50.3 | |
| Qwen2.5-7B-Instruct Q4_K_M | 512 | 72.0 | 72.7 | 1.01x | 82.5 | 0.88 |
| | 4K | 61.8 | 66.5 | 1.08x | 79.3 | 0.84 |
| | 8K | 54.4 | 61.9 | 1.14x | 75.8 | 0.82 |
| | 16K | 44.1 | 54.6 | 1.24x | 69.8 | 0.78 |
| | 32K | 32.2 | 44.3 | 1.37x | 60.2 | 0.74 |

Prefill is unchanged (0.97x to 1.03x; the kernel is decode-only). The coordinator's kill number for the
remaining work, set before the A/B, was 1.15x over main at 16K on two of the four files; all four read 1.24x
or more. `ab-rows.jsonl` holds every run (host paths and command lines left out).

## Gates

- `tests/test_attn_dec_repeat.py` (new): a text scored twice at two depths and two head widths (128, the
  16-byte path; 16, the word path), the bytes compared. Passes.
- `make test-cuda-batch-identity`: every fixture, with the grouped decode on and off, byte-identical.
  `test-batch-identity` on the four real files: byte-identical, microbatch engaged.
- `test-gpu-identity` on the four real files: 1.5e-5 to 3.1e-5 of the logit range (limit 1e-3).
- `tests/test_kv_ring.py` and `tests/test_bench_json.py`: pass (the latter holds `cuda_attn_dec_gqa` counted).
- Greedy text after 17.4K tokens on Llama-3.2-3B: identical to main's.

Determinism is per geometry and position: the slice bounds, the tile order, the warp butterflies and the
rescale sequence are fixed. The order differs from the two-pass kernel's, so a receipt from an earlier build
replays at tier T2 (token replay across builds), as any kernel change between builds does.

## Against the publisher's float32 model (D1): owed

The score files are made: `runner --score 20 -f tests/fixtures/gold-corpus-2k.txt -c 8192`, scored solo so
every position takes the decode attention, on `qwen3-0.6b-q8_0.gguf` (sha256 `9465e63a...`), main 853e478c
against ff613375 (8,007 positions each). The read over positions 4,000 to 6,000 against `Qwen/Qwen3-0.6B` at
`c1899de2` in float32 needs more memory than this 16 GB host has. It was stopped for low memory after its
forward, so it runs on the lab host. The decision rule is the one fixed for attn-dec on 2026-10-20: the paired
Wilcoxon on per-position KL not worse at p < 0.05, mean KL at most 1.01x, and McNemar on margin-qualified top-1
not worse. A miss keeps the one-pass kernel behind a flag, not a second rewrite.

## Not in this change

- The 8-heads-per-KV-head case (Qwen2.5-3B: 39% of DRAM peak even at 256 blocks). Its score phase reads each K
  row from shared memory once per head; reading a chunk once for all heads of the group is the next lever.
- The split count by geometry (trial kept outside the tree; see above).
- The D1 row (lab).
