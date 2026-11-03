# R3.29.1: CUDA decode attention for eight query heads per KV head (2026-11-03)

**Unflattering first.**
- **One file gains much.** Qwen2.5-3B (16 query heads on 2 KV heads) reads 1.18x at 16K. Qwen2.5-7B (7 per
  KV head) reads 1.08x, and Qwen3-4B (4) and Llama-3.2-3B (3) read 1.03x to 1.04x.
- **Qwen2.5-3B is still the furthest from llama.cpp b11748.** At 16K it moves from 0.66x to 0.78x of llama.cpp,
  where the other three files read 0.85x to 0.88x.
- **Only 512 and 16K were read.** R3.29's 4K, 8K and 32K rows were not repeated.

## What was slow

After R3.29, Nsight Compute on one Qwen2.5-3B layer at about 16K put `k_attn_dec_gqa` at 157 us, bound by
shared-memory wavefronts, not by DRAM. With eight heads in a block, the value sum read each V pair once per
head a thread owned, and each probability as its own load.

## The change

`k_attn_dec_gqa` (`src/kernels.cu`), on the 16-byte path:
- When the block size is a whole number of value rows (`AD_THREADS % P2 == 0`), every column a thread owns has
  the same V pair. The pair is read once per key for all of the thread's heads, and the probabilities are
  read four keys at a time as a float4. Each output adds the same terms in the same order as before.
- Q rows sit at a stride of `hd + 4` floats (was `hd + 1`), so the score reads a chunk's eight Q values as
  two float4. The shared-memory size in `src/cuda.c` follows.

Other geometries and the word path keep the previous loop. The scores are the same bytes as main's.

## Kill read (set before the first measurement)

One layer on Qwen2.5-3B at about 16K had to beat the merged kernel by 1.2x, or stop. Nsight Compute: **157 us
to 97 us (1.62x)**; shared-memory wavefronts 3.06M to 1.96M.

## Speed

The Windows CUDA host: RTX 3070 8 GB, driver 596.36. Runner main 340fffa0 against the branch, and llama.cpp
b11748. Same driver as R3.29: runner `--bench-json -n 128 -c L+384 --gpu-layers 99 -t 4`, llama.cpp
`llama-bench -p 0 -n 128 -d L -ngl 99 -r 2`, two rounds with the arm order reversed, warm cache. Decode tok/s,
median of the rounds, every round within 1% of its median.

| File | Heads per KV head | L | main | R3.29.1 | gain | llama.cpp | R3.29.1 / llama.cpp |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B Q4_K_M | 8 | 512 | 139.1 | 140.8 | 1.01x | 160.2 | 0.88 |
| | | 16K | 85.5 | 100.8 | 1.18x | 129.3 | 0.78 |
| Qwen2.5-7B-Instruct Q4_K_M | 7 | 512 | 72.8 | 73.2 | 1.00x | 82.5 | 0.89 |
| | | 16K | 54.7 | 59.2 | 1.08x | 69.8 | 0.85 |
| Qwen3-4B Q4_K_M | 4 | 512 | 104.9 | 105.4 | 1.01x | 120.9 | 0.87 |
| | | 16K | 59.7 | 61.9 | 1.04x | 71.6 | 0.86 |
| Llama-3.2-3B Q4_K_M | 3 | 512 | 136.4 | 138.1 | 1.01x | 156.8 | 0.88 |
| | | 16K | 79.5 | 81.6 | 1.03x | 92.6 | 0.88 |

A second read through the runner's own `-n 64 --temp 0` decode, alternated twice, agrees: 1.19x, 1.08x,
1.04x and 1.03x at 16K. Prefill is unchanged (1.00x to 1.02x). `ab-rows.jsonl` holds every run (host paths
and command lines left out).

## Gates

R3.29's gates, on the four files:
- `tests/test_attn_dec_repeat.py` and `tests/test_bench_json.py`: 15 passed.
- `make test-cuda-batch-identity`: every fixture byte-identical.
- `test-batch-identity`: byte-identical, microbatch engaged, on all four.
- `test-gpu-identity`: 1.53e-5 (Llama-3.2-3B), 2.14e-5 (Qwen2.5-3B), 2.23e-5 (Qwen3-4B) and 3.07e-5
  (Qwen2.5-7B) of the mean logit range, limit 1e-3.
- Solo `--score 20 -f tests/fixtures/gold-corpus-2k.txt -c 8448`: body bytes identical to main
  (`scripts/score_record.py`) on all four files.

These ran on the branch before main took #574. After moving the change onto a997905a (pinned PTX
regenerated), the repeat tests (6 passed, with `tests/test_attn_wide_repeat.py`), batch-identity and
gpu-identity on Qwen2.5-3B, and the score bytes on Qwen2.5-3B and Qwen2.5-7B were read again: unchanged.

## Against the float32 reference (D1)

No row: the scores are the same bytes as main's, so R3.29's D1 row stands.
