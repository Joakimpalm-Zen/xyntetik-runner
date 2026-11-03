# R3.6.3.1: CUDA prefill attention on the tensor cores for 256- to 512-wide heads (2026-11-03)

**Unflattering first.**
- **The 4K row is the noisy one.** On Google's gemma-4-E4B QAT file, the second interleaved round read +2%
  (757.9 against 740.5 tok/s) where the first read +35%. Bolt E4B reads +30% and +40% at 4K.
- **It is a long-context gain.** Decode is unchanged, and at 512 tokens the gain is 8% to 25%.
- **The kernel is far from the tensor cores' peak.** At head_dim 512 a block needs 76 KB of shared memory, so
  one block fits an SM (two at 256). On a 256-wide sliding layer, Nsight Compute reads 167 registers, 16%
  achieved occupancy, and 29% each of SM throughput and memory throughput.
- **No sinks.** A layer with attention sinks keeps the tile (gpt-oss and flint carry sinks), even under
  `RUNNER_CUDA_ATTN_TC_WIDE=1`.
- **Only the E-series is promoted.** The other gemma-4 files (12B, 26B-A4B, 31B), whose global layers are
  512 wide too, keep `k_attn_tile8` by default: no row was read for them.

## What was slow

gemma-4's E-series has 512-wide global layers (7 of 42 on E4B; 8 query heads on 2 KV heads) and 256-wide
sliding layers with a 512-token window. `k_attn_tc` takes head widths up to 128, so prefill attention ran on
the tile: `k_attn_tile8` for the global layers and `k_attn_tile16` for the sliding ones, both fp32 FMA.

On the RTX 3070, gemma-4-E4B QAT Q4_0 prefill fell from 738 tok/s at 3.9K tokens to 289 at 32.5K.
`RUNNER_CUDA_PROFILE` put attention at 79% of the 32K prefill's GPU time. Nsight Compute on one 512-token tile
at about 31K:
- the 7 global layers took `k_attn_tile8` at 352 to 355 ms each, 2.48 s a tile (about 0.7 TFLOPS);
- the 35 sliding layers took `k_attn_tile16` at 2.37 ms each.

## The kernel

`k_attn_tcw` uses `k_attn_tc`'s operands:
- Q scaled into fp16 by an exact power of two per row;
- Q and P each in two halves;
- the fp16 cache rows as they are;
- `mma.sync` m16n8k8 with f32 accumulation.

`k_attn_tc` keeps a warp's whole O row in registers, 256 registers a thread at head_dim 512. Instead, a block
of `k_attn_tcw` owns 16 query rows (the heads of one KV group times a strip of tokens), and its four warps
split head_dim in quarters. For each 32-key tile:
1. Each warp takes S = Q K^T over its quarter.
2. The four partial S tiles are summed in shared memory in warp order, so every warp holds the same S.
3. A running maximum per row, the tile's `expf(s - max)`, and the row sum and the warp's quarter of O
   rescaled where the maximum rises.
4. O += P V for the quarter, V through `ldmatrix.trans`.

For a geometry and position the order is fixed, so a run is the same bytes twice. Admission: fp16 K and V
caches, head_dim 256, 384 or 512, no sinks, the GQA ratio dividing.

## Speed

One global layer at about 31K, Nsight Compute: **53 to 57 ms against 352 to 355 (6.4x)**. One sliding layer:
313 us against 2.37 ms (7.6x).

End to end, `-n 32 --temp 0`, the tile (`RUNNER_CUDA_ATTN_TC_WIDE=0`) and the kernel alternated, two rounds,
warm cache, prompt tok/s:

| File | 512 | 4K | 16K | 32K |
|---|---|---|---|---|
| Nemotron 3.5 Bolt E4B Q4_K_M, tile | 881 / 878 | 728 / 758 | 446 / 450 | 293 / 291 |
| Bolt, `k_attn_tcw` | 1045 / 947 | 1017 / 982 | 891 / 920 | 767 / 765 |
| gemma-4-E4B QAT Q4_0, tile | 794 / 811 | 756 / 740 | 446 / 443 | 292 / 291 |
| QAT, `k_attn_tcw` | 1009 / 923 | 1018 / 758 | 894 / 880 | 769 / 768 |

Decode tok/s is unchanged on both files, and the 32 greedy tokens are identical on both at every length.

gemma-4-E2B, in the same default family, has its own row in `e2b.md`: prefill 1.47x at 4K and 2.34x at 16K,
gpu-identity 2.84e-5, its Tier 2 CUDA cell byte-identical to its pin.

## Gates

- `make test-cuda-attn-wide` (new): the CPU identity at 1e-5 on 256- and 512-wide fixtures, the route required
  to say `attn_wide=on`. 4.3e-7 and 1.1e-6 of the logit range.
- `tests/test_attn_wide_repeat.py` (new): a fixture scored twice through the chunked prefill at both widths,
  the same bytes; the route names the kernel under `=1` and not under `=0`.
- `test-gpu-identity` on Bolt E4B (the default, the kernel on): 2.68e-5 of the range, limit 1e-3. The tile
  reads 2.56e-5.
- The Tier 1 smoke for Bolt's CUDA cell on the new default: the reply byte-identical to its pin (output sha256
  `0ae60d3e43338350`), recorded in `docs/model-tiers.json` at 1a2d85ee.
- The KV ring on Bolt, with the kernel. `RUNNER_KV_RING=1` (1,024 rows on 20 sliding layers) against `=0`,
  scored chunked over 8,196 positions: logprobs and top-k identical. `tests/test_kv_ring.py`: 12 passed.

## Against the publisher's float32 model (D1): passes

Read on the lab host against `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` at `49843eed` in float32, positions
4,000 to 5,999 of `tests/fixtures/gold-corpus-2k.txt`. Both sides were scored chunked on Bolt E4B (sha256
`c8af2b8a...`): main 340fffa0 (the tile) against 1a2d85ee (`k_attn_tcw`). The rule is the attention routes'
(2026-10-20):

| Rule | Tile | `k_attn_tcw` | Reading |
|---|---|---|---|
| Mean KL at most 1.01x | 0.14269 | 0.14220 | 0.9965x, pass |
| Paired Wilcoxon on per-position KL not worse at p < 0.05 | | | z 1.32, p 0.188, pass |
| McNemar on margin-qualified top-1 not worse | 1,607 of 1,718 | 1,605 of 1,718 | discordant 2 / 0, p 0.5, pass |

The scorer's verdict is "not distinguishable". The read needed `--force-bos`: the score files begin with the GGUF's
BOS (id 2), which the publisher's tokenizer does not add to a bare text (the documented Gemma 4 case). The
command, the environment and every hash are in `d1-result.md`; the scorer's output is `gold-main-vs-r3631.json`
(sha256 `750668a03af38296...`). The score files: `RUNNER_SCORE_CHUNKED=1 RUNNER_SPEC_BATCH=64 runner --score 20
-f tests/fixtures/gold-corpus-2k.txt -c 8448`, 8,196 positions each.

## Not in this change

- The other 256-wide Tier 1 files (Qwen3.8-27B x2, Qwen3.8-Flash-Next x2, Mellum): their identity and smoke
  rows after the merge, the ones under 8 GB on the RTX 3070 and the 13 GB ones on the lab slice. They are not
  in the default family, so the kernel reaches them only under `=1`.
- Occupancy: Q from registers instead of shared memory, two blocks an SM.
- The 4K prefill's other costs on E4B: elementwise work at 31% and host launch overhead at 20%.
