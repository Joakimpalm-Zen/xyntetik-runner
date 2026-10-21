# GPU benchmarks — Runner vs llama.cpp (CUDA)

First published 2026-07-29, runner `94ce01f`; MoE rows updated the same day
on runner `26cf7e6` after the device-routing work landed. One machine, one
method, both engines on the same files — and the losing rows published
alongside the winning ones.

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Max-Q, **MIG 1g.24gb slice** (not the full card) |
| CPU | AMD Ryzen Threadripper (128 threads available; runner used its defaults) |
| Runner | `94ce01f` (2026-07-29), built `-O3 -march=native`, CUDA via `sm_75` PTX, JIT to `compute 12.0` |
| llama.cpp | build `ea12b27` (CUDA), same GGUF files |
| Method | 512-token prefill / 128-token greedy decode (`--temp 0`), full GPU offload (verified `full=1` per run), median of 3 runs; llama.cpp via `llama-bench -p 512 -n 128 -ngl 99 -r 2` |

**MIG caveat:** a 1g.24gb slice has roughly ~200 GB/s of memory bandwidth and a
fraction of the card's SMs. Absolute tok/s will differ on other hardware; the
*ratios* between the two engines are the meaningful result, and even those shift
with the compute/bandwidth balance (the same kernels measured different ratios
on an RTX 3070).

## Results, 2026-10-21: after the 2026-10 performance work, three hosts

The same hosts, files and method as the baseline section below, measured on
the performance branch's final tree, with the baseline runner `990b8a27` and
llama.cpp b11748 run again beside it in the same session, rounds
interleaved: every row on the RTX 3070 and the Apple M1, and the 512- and
2,048-token rows on the datacenter MIG slice. The slice's 8K and 16K rows
join the evening's final runs to the same morning's baseline and llama.cpp
columns; the evening's same-session run put those columns within 0.6% of
their evening values in every cell (the faulted Qwen2.5-32B baseline decode
aside). A ratio here never sets two engines measured hours apart on a host
whose speed moved in between: on the Apple M1 a final build measured in the
evening read up to 9% below the morning's baseline, against which a
same-session A/B finds it equal.

**Read the limits first.** Prompt processing on CUDA is still where the
runner is behind. With every layer on the device it reaches 31 to 59% of
llama.cpp's prefill at 512 tokens and 16 to 33% at 16K on the dense K-quant,
Q4_0, Q8_0 and IQ3_S files, and much less on the gemma family, the
mixture-of-experts files and two others: gemma-4-26B-A4B 4 to 7%,
gemma-4-E4B 7 to 18%, gemma-4-12B 8% at 16K, Qwen3-30B-A3B 6.5% at 16K,
Phi-4-mini 7% at 16K, and Xyntetik-Kvist-14B 10 to 17% on the 8 GB card.
Decode on CUDA is 85 to 96% of llama.cpp at 512 tokens on the dense K-quant,
Q4_0 and Q8_0 files and 48 to 76% at 16K; the IQ3_S files decode at 45 to
61%, the gemma files at 40 to 76% and the mixture-of-experts files at 69 to
80% at 512. On the Apple M1, Metal runs at 16 to 38% of llama.cpp's prefill
and 30 to 42% of its decode (Q8_0 50 to 73%), as before this work, and the
CPU at 22 to 32% of llama.cpp's prefill and 36 to 52% of its decode on the
K-quant files (IQ3_S 60 to 69%, Q8_0 71 to 80%).

What changed: on CUDA, prefill runs 2.0 to 12.4 times the baseline's rate at
512 tokens and 1.45 to 13.7 times at 16K (gemma-4-E4B, whose sliding-window
attention keeps the tiled kernel, gains least), and decode 1.10 to 1.31
times at 512 and 1.4 to 3.0 times at 16K, where the baseline's attention
dominated; the gemma files, whose sliding windows keep most attention short,
decode 1.07 to 2.1 times faster. Xyntetik-Kvist-14B IQ4_NL decodes 3.4 to
3.8 times faster on the 8 GB card (its IQ4_NL kernel, and the VRAM the token
embedding no longer takes), Qwen2.5-32B Q3_K_S 14 to 21 times because the
baseline's Q3_K kernel faulted and it finished on the CPU (fixed), and the
two cells the baseline could not finish within the timeout (granite-4.2-8b
at 16K and Kvist-14B at 8K on the 3070) complete. On the Apple M1, Metal is
unchanged (0.97 to 1.02 times the baseline in every cell); the CPU decodes
1.10 to 1.17 times faster on the Q4_K_M files and 1.24 to 1.43 times on
IQ3_S, with prefill unchanged, and Q8_0, whose dot did not change, reads
0.95 to 1.05 times on two rounds (-0.5% in a 7-round A/B with an A/A arm).
Every default route behind these numbers either gives the baseline's bytes
or was held to the publisher's float32 model over 2,000 positions before it
became a default (the CHANGELOG names the rows); `--gpu-route reference`
runs the previous arithmetic.

Method: runner `--bench-json -n 128 -b 512 --gpu-layers 99` (`--gpu off`
for the Apple M1's CPU arm; 16 threads on the slice, 4 on the RTX 3070 host
and the Apple M1), default routes; llama.cpp `llama-bench -p L -n 0` for
prefill and `-p 0 -n 128 -d L` for decode at the same depth, `-r 2 -ngl 99`
(`-ngl 0` for the CPU arm), flash attention and the 512-token micro-batch at
their defaults. Two rounds, a warm-up discarded; cells are medians.
`--gpu-layers 99` forces every layer onto the card: on the RTX 3070 the
baseline could not hold granite-4.2-8b at 16K and Kvist-14B at 8K within the
600-second timeout, and llama.cpp's granite-4.2-8b 16K prefill falls to 158
tok/s from 2,418 at 8K (a memory-pressure row for it, not a ratio to read).
On the Apple M1 both engines page on the Llama-3.2-3B 8K CPU row. Raw rows
(host paths and device strings removed), with each host's method:
[2026-10-21-linux-cuda-mig/](benchmarks-raw/2026-10-21-linux-cuda-mig/),
[2026-10-21-windows-rtx3070/](benchmarks-raw/2026-10-21-windows-rtx3070/),
[2026-10-21-macos-m1/](benchmarks-raw/2026-10-21-macos-m1/).

### MIG 1g.24gb slice, CUDA

| model | prompt | prefill tok/s: runner / llama.cpp | of llama.cpp | vs baseline | decode tok/s: runner / llama.cpp | of llama.cpp | vs baseline |
|---|---:|---|---:|---:|---|---:|---:|
| Llama-3.2-3B Q4_K_M | 512 | 4648 / 8480 | 55% | 6.51x | 146 / 165 | 88% | 1.19x |
| Llama-3.2-3B Q4_K_M | 2048 | 3825 / 8489 | 45% | 7.39x | 133 / 154 | 86% | 1.48x |
| Llama-3.2-3B Q4_K_M | 8192 | 2247 / 7300 | 31% | 9.62x | 92.2 / 122 | 76% | 2.11x |
| Llama-3.2-3B Q4_K_M | 16384 | 1458 / 5523 | 26% | 11.40x | 65.7 / 94.0 | 70% | 2.55x |
| Qwen2.5-3B Q4_K_M | 512 | 4609 / 8016 | 57% | 6.63x | 143 / 168 | 85% | 1.21x |
| Qwen2.5-3B Q4_K_M | 2048 | 4046 / 8187 | 49% | 7.77x | 130 / 160 | 81% | 1.56x |
| Qwen2.5-3B Q4_K_M | 8192 | 2692 / 7539 | 36% | 10.25x | 92.2 / 149 | 62% | 2.37x |
| Qwen2.5-3B Q4_K_M | 16384 | 1819 / 6715 | 27% | 11.80x | 68.2 / 135 | 51% | 3.00x |
| Qwen2.5-7B Q4_K_M | 512 | 2459 / 4197 | 59% | 6.49x | 75.6 / 85.0 | 89% | 1.10x |
| Qwen2.5-7B Q4_K_M | 2048 | 2241 / 4205 | 53% | 7.26x | 71.7 / 82.3 | 87% | 1.26x |
| Qwen2.5-7B Q4_K_M | 8192 | 1619 / 3918 | 41% | 9.07x | 58.8 / 78.0 | 75% | 1.72x |
| Qwen2.5-7B Q4_K_M | 16384 | 1175 / 3540 | 33% | 10.87x | 48.5 / 71.5 | 68% | 2.21x |
| granite-4.1-3b Q8_0 | 512 | 3680 / 7665 | 48% | 6.04x | 91.6 / 96.4 | 95% | 1.17x |
| granite-4.1-3b Q8_0 | 2048 | 2974 / 7542 | 39% | 6.69x | 85.9 / 93.2 | 92% | 1.44x |
| granite-4.1-3b Q8_0 | 8192 | 1678 / 6304 | 27% | 7.89x | 66.4 / 83.5 | 79% | 2.13x |
| granite-4.1-3b Q8_0 | 16384 | 1035 / 5059 | 20% | 8.80x | 51.0 / 73.4 | 69% | 2.72x |
| granite-4.2-8b Q4_K_M | 512 | 1973 / 3354 | 59% | 6.72x | 64.4 / 70.8 | 91% | 1.13x |
| granite-4.2-8b Q4_K_M | 2048 | 1731 / 3333 | 52% | 7.58x | 60.0 / 67.4 | 89% | 1.33x |
| granite-4.2-8b Q4_K_M | 8192 | 1161 / 3057 | 38% | 9.98x | 46.2 / 57.7 | 80% | 1.84x |
| granite-4.2-8b Q4_K_M | 16384 | 780 / 2535 | 31% | 11.64x | 35.1 / 48.5 | 72% | 2.25x |
| granite-4.2-3b IQ3_S | 512 | 2195 / 7098 | 31% | 3.92x | 100 / 166 | 61% | 1.19x |
| granite-4.2-3b IQ3_S | 2048 | 1915 / 7256 | 26% | 4.60x | 93.6 / 156 | 60% | 1.48x |
| granite-4.2-3b IQ3_S | 8192 | 1264 / 6152 | 21% | 6.15x | 70.9 / 131 | 54% | 2.21x |
| granite-4.2-3b IQ3_S | 16384 | 866 / 4967 | 17% | 7.50x | 53.6 / 107 | 50% | 2.81x |
| Phi-4-mini Q4_0 | 512 | 3178 / 8572 | 37% | 12.38x | 132 / 148 | 89% | 1.31x |
| Phi-4-mini Q4_0 | 2048 | 1824 / 8588 | 21% | 8.22x | 120 / 138 | 86% | 1.60x |
| Phi-4-mini Q4_0 | 8192 | 665 / 7217 | 9.2% | 4.78x | 82.3 / 109 | 76% | 2.21x |
| Phi-4-mini Q4_0 | 16384 | 364 / 5242 | 7.0% | 4.08x | 58.0 / 83.4 | 70% | 2.61x |
| granite-4.1-8b Q4_0 | 512 | 2154 / 3770 | 57% | 9.49x | 70.2 / 74.9 | 94% | 1.14x |
| granite-4.1-8b Q4_0 | 2048 | 1854 / 3739 | 50% | 9.96x | 65.0 / 71.0 | 92% | 1.36x |
| granite-4.1-8b Q4_0 | 8192 | 1206 / 3398 | 35% | 11.59x | 49.1 / 60.4 | 81% | 1.90x |
| granite-4.1-8b Q4_0 | 16384 | 804 / 2772 | 29% | 12.81x | 36.8 / 50.3 | 73% | 2.31x |
| gemma-4-12B Q4_K_M | 512 | 825 / 2372 | 35% | 4.66x | 35.8 / 47.0 | 76% | 1.07x |
| gemma-4-12B Q4_K_M | 2048 | 539 / 2374 | 23% | 3.63x | 33.5 / 45.8 | 73% | 1.11x |
| gemma-4-12B Q4_K_M | 8192 | 288 / 2291 | 13% | 2.48x | 27.1 / 45.3 | 60% | 1.09x |
| gemma-4-12B Q4_K_M | 16384 | 172 / 2179 | 7.9% | 1.88x | 21.5 / 44.6 | 48% | 1.07x |
| Qwen3-30B-A3B Q4_K_M | 512 | 1030 / 3354 | 31% | 4.85x | 117 / 147 | 80% | 1.25x |
| Qwen3-30B-A3B Q4_K_M | 2048 | 736 / 3400 | 22% | 4.32x | 102 / 132 | 77% | 1.64x |
| Qwen3-30B-A3B Q4_K_M | 8192 | 313 / 3091 | 10% | 3.30x | 64.8 / 116 | 56% | 2.41x |
| Qwen3-30B-A3B Q4_K_M | 16384 | 176 / 2707 | 6.5% | 3.17x | 45.4 / 94.6 | 48% | 2.99x |
| gemma-4-26B-A4B Q4_0 | 512 | 271 / 4018 | 6.7% | 4.92x | 77.5 / 113 | 69% | 2.07x |
| gemma-4-26B-A4B Q4_0 | 2048 | 245 / 4103 | 6.0% | 4.61x | 71.2 / 108 | 66% | 2.02x |
| gemma-4-26B-A4B Q4_0 | 8192 | 194 / 3951 | 4.9% | 3.89x | 52.8 / 105 | 50% | 1.76x |
| gemma-4-26B-A4B Q4_0 | 16384 | 150 / 3610 | 4.2% | 3.24x | 40.1 / 100 | 40% | 1.58x |
| Qwen3.8-27B IQ3_S | 512 | 362 / 1117 | 32% | 9.03x | 16.2 / 30.3 | 54% | 1.17x |
| Qwen3.8-27B IQ3_S | 2048 | 336 / 1137 | 30% | 8.53x | 16.1 / 30.1 | 53% | 1.19x |
| Qwen3.8-27B IQ3_S | 8192 | 243 / 1117 | 22% | 6.83x | 15.3 / 29.2 | 52% | 1.27x |
| Qwen3.8-27B IQ3_S | 16384 | 179 / 1059 | 17% | 5.70x | 14.5 / 28.1 | 52% | 1.37x |
| Qwen2.5-32B Q3_K_S | 512 | 313 / 796 | 39% | 12.15x | 12.1 / 26.1 | 46% | 14.04x |
| Qwen2.5-32B Q3_K_S | 2048 | 299 / 788 | 38% | 12.20x | 11.8 / 25.2 | 47% | 17.11x |
| Qwen2.5-32B Q3_K_S | 8192 | 252 / 738 | 34% | 12.32x | 10.5 / 23.0 | 46% | 20.53x |

### RTX 3070 8 GB, CUDA

| model | prompt | prefill tok/s: runner / llama.cpp | of llama.cpp | vs baseline | decode tok/s: runner / llama.cpp | of llama.cpp | vs baseline |
|---|---:|---|---:|---:|---|---:|---:|
| Llama-3.2-3B Q4_K_M | 512 | 2743 / 6533 | 42% | 4.62x | 135 / 153 | 88% | 1.17x |
| Llama-3.2-3B Q4_K_M | 2048 | 2475 / 6446 | 38% | 5.62x | 121 / 144 | 84% | 1.42x |
| Llama-3.2-3B Q4_K_M | 8192 | 1661 / 5462 | 30% | 8.26x | 81.8 / 115 | 71% | 1.96x |
| Llama-3.2-3B Q4_K_M | 16384 | 1154 / 4474 | 26% | 12.16x | 58.1 / 90.3 | 64% | 2.36x |
| Qwen2.5-3B Q4_K_M | 512 | 2843 / 6064 | 47% | 4.58x | 134 / 157 | 86% | 1.20x |
| Qwen2.5-3B Q4_K_M | 2048 | 2655 / 6181 | 43% | 5.64x | 122 / 152 | 80% | 1.49x |
| Qwen2.5-3B Q4_K_M | 8192 | 1920 / 5559 | 35% | 8.43x | 83.5 / 140 | 59% | 2.13x |
| Qwen2.5-3B Q4_K_M | 16384 | 1377 / 4828 | 29% | 10.32x | 61.0 / 127 | 48% | 2.68x |
| Qwen2.5-7B Q4_K_M | 512 | 1539 / 3394 | 45% | 4.79x | 70.8 / 80.6 | 88% | 1.10x |
| Qwen2.5-7B Q4_K_M | 2048 | 1507 / 3386 | 45% | 5.63x | 67.0 / 79.4 | 84% | 1.23x |
| Qwen2.5-7B Q4_K_M | 8192 | 1161 / 3074 | 38% | 7.62x | 53.5 / 74.1 | 72% | 1.63x |
| Qwen2.5-7B Q4_K_M | 16384 | 882 / 2719 | 32% | 9.33x | 43.5 / 67.3 | 65% | 2.05x |
| granite-4.1-3b Q8_0 | 512 | 2371 / 5494 | 43% | 4.61x | 87.8 / 91.5 | 96% | 1.16x |
| granite-4.1-3b Q8_0 | 2048 | 2078 / 5378 | 39% | 5.48x | 81.9 / 88.8 | 92% | 1.40x |
| granite-4.1-3b Q8_0 | 8192 | 1276 / 4381 | 29% | 7.30x | 61.3 / 79.7 | 77% | 2.04x |
| granite-4.1-3b Q8_0 | 16384 | 834 / 3473 | 24% | 8.39x | 46.0 / 70.4 | 65% | 2.53x |
| granite-4.2-8b Q4_K_M | 512 | 1269 / 2686 | 47% | 4.83x | 60.0 / 67.3 | 89% | 1.12x |
| granite-4.2-8b Q4_K_M | 2048 | 1183 / 2684 | 44% | 5.79x | 55.5 / 63.3 | 88% | 1.30x |
| granite-4.2-8b Q4_K_M | 8192 | 846 / 2418 | 35% | 8.37x | 41.4 / 54.3 | 76% | 1.75x |
| granite-4.2-8b Q4_K_M | 16384 | 393 / 158 | 248% | baseline timeout | 24.6 / 36.6 | 67% | baseline timeout |
| granite-4.2-3b IQ3_S | 512 | 1804 / 5336 | 34% | 3.81x | 88.4 / 151 | 59% | 1.17x |
| granite-4.2-3b IQ3_S | 2048 | 1602 / 5225 | 31% | 4.53x | 82.0 / 143 | 57% | 1.40x |
| granite-4.2-3b IQ3_S | 8192 | 1073 / 4290 | 25% | 6.34x | 61.2 / 122 | 50% | 2.03x |
| granite-4.2-3b IQ3_S | 16384 | 744 / 3409 | 22% | 7.62x | 46.0 / 102 | 45% | 2.52x |
| Qwen3-4B Q4_K_M | 512 | 2296 / 4960 | 46% | 4.47x | 103 / 119 | 87% | 1.17x |
| Qwen3-4B Q4_K_M | 2048 | 2035 / 4896 | 42% | 5.99x | 91.6 / 111 | 83% | 1.43x |
| Qwen3-4B Q4_K_M | 8192 | 1265 / 4142 | 31% | 9.39x | 60.8 / 89.0 | 68% | 1.99x |
| Qwen3-4B Q4_K_M | 16384 | 836 / 3410 | 25% | 13.72x | 42.5 / 70.1 | 61% | 2.37x |
| gemma-4-E4B Q4_K_M | 512 | 748 / 4086 | 18% | 1.97x | 66.4 / 95.0 | 70% | 1.14x |
| gemma-4-E4B Q4_K_M | 2048 | 695 / 4257 | 16% | 2.07x | 60.6 / 93.4 | 65% | 1.13x |
| gemma-4-E4B Q4_K_M | 8192 | 398 / 4074 | 9.8% | 1.60x | 45.5 / 89.3 | 51% | 1.10x |
| gemma-4-E4B Q4_K_M | 16384 | 257 / 3784 | 6.8% | 1.45x | 34.0 / 84.8 | 40% | 1.07x |
| Xyntetik-Kvist-14B IQ4_NL | 512 | 315 / 1908 | 17% | 2.41x | 24.4 / 47.8 | 51% | 3.40x |
| Xyntetik-Kvist-14B IQ4_NL | 2048 | 253 / 1895 | 13% | 3.97x | 21.6 / 47.4 | 46% | 3.84x |
| Xyntetik-Kvist-14B IQ4_NL | 8192 | 186 / 1809 | 10% | baseline timeout | 19.4 / 46.7 | 41% | baseline timeout |
| Qwen3.5-4B Q4_K_M | 512 | 1558 / 4573 | 34% | 6.25x | 93.7 / 107 | 87% | 1.26x |
| Qwen3.5-4B Q4_K_M | 2048 | 1400 / 4632 | 30% | 5.90x | 90.9 / 106 | 86% | 1.31x |
| Qwen3.5-4B Q4_K_M | 8192 | 910 / 4334 | 21% | 4.80x | 81.1 / 100 | 81% | 1.48x |
| Qwen3.5-4B Q4_K_M | 16384 | 616 / 3958 | 16% | 4.47x | 72.0 / 94.2 | 76% | 1.69x |

### Apple M1 8 GB, Metal

| model | prompt | prefill tok/s: runner / llama.cpp | of llama.cpp | vs baseline | decode tok/s: runner / llama.cpp | of llama.cpp | vs baseline |
|---|---:|---|---:|---:|---|---:|---:|
| Llama-3.2-1B Q4_K_M | 512 | 308 / 866 | 36% | 1.00x | 27.7 / 65.5 | 42% | 1.01x |
| Llama-3.2-1B Q4_K_M | 2048 | 225 / 820 | 27% | 1.00x | 23.8 / 62.1 | 38% | 1.00x |
| Llama-3.2-1B Q4_K_M | 8192 | 101 / 629 | 16% | 1.00x | 15.5 / 52.2 | 30% | 1.00x |
| Qwen2.5-1.5B Q4_K_M | 512 | 238 / 629 | 38% | 1.02x | 21.7 / 52.3 | 41% | 1.00x |
| Qwen2.5-1.5B Q4_K_M | 2048 | 203 / 598 | 34% | 0.99x | 20.3 / 50.3 | 40% | 1.00x |
| Qwen2.5-1.5B Q4_K_M | 8192 | 108 / 475 | 23% | 1.00x | 13.6 / 42.5 | 32% | 1.00x |
| Qwen2.5-1.5B Q8_0 | 512 | 199 / 695 | 29% | 0.99x | 25.8 / 35.2 | 73% | 1.00x |
| Qwen2.5-1.5B Q8_0 | 2048 | 204 / 653 | 31% | 0.97x | 23.9 / 33.8 | 71% | 1.00x |
| Qwen2.5-1.5B Q8_0 | 8192 | 112 / 508 | 22% | 1.00x | 15.2 / 30.1 | 50% | 1.00x |
| Llama-3.2-1B IQ3_S | 512 | 310 / 833 | 37% | 0.99x | 30.0 / 70.9 | 42% | 1.00x |
| Llama-3.2-1B IQ3_S | 2048 | 225 / 783 | 29% | 1.00x | 25.6 / 67.2 | 38% | 1.00x |
| Llama-3.2-1B IQ3_S | 8192 | 101 / 600 | 17% | 1.00x | 16.1 / 52.9 | 31% | 1.00x |
| Llama-3.2-3B Q4_K_M | 512 | 103 / 301 | 34% | 0.97x | 11.2 / 27.1 | 41% | 1.00x |
| Llama-3.2-3B Q4_K_M | 2048 | 92.5 / 285 | 32% | 1.00x | 9.7 / 25.1 | 39% | 1.00x |
| Llama-3.2-3B Q4_K_M | 8192 | 51.4 / 228 | 23% | 1.01x | 6.5 / 18.4 | 35% | 1.00x |

### Apple M1 8 GB, CPU (4 threads)

| model | prompt | prefill tok/s: runner / llama.cpp | of llama.cpp | vs baseline | decode tok/s: runner / llama.cpp | of llama.cpp | vs baseline |
|---|---:|---|---:|---:|---|---:|---:|
| Llama-3.2-1B Q4_K_M | 512 | 91.6 / 322 | 28% | 0.97x | 23.2 / 60.2 | 38% | 1.10x |
| Llama-3.2-1B Q4_K_M | 2048 | 81.7 / 278 | 29% | 1.03x | 21.6 / 52.3 | 41% | 1.13x |
| Llama-3.2-1B Q4_K_M | 8192 | 54.1 / 193 | 28% | 1.03x | 15.0 / 28.9 | 52% | 1.13x |
| Qwen2.5-1.5B Q4_K_M | 512 | 69.5 / 228 | 30% | 0.97x | 17.5 / 48.9 | 36% | 1.10x |
| Qwen2.5-1.5B Q4_K_M | 2048 | 62.1 / 209 | 30% | 0.99x | 17.5 / 44.4 | 39% | 1.17x |
| Qwen2.5-1.5B Q4_K_M | 8192 | 46.8 / 148 | 32% | 1.01x | 14.2 / 27.7 | 51% | 1.13x |
| Qwen2.5-1.5B Q8_0 | 512 | 69.9 / 283 | 25% | 0.99x | 23.0 / 32.2 | 71% | 0.95x |
| Qwen2.5-1.5B Q8_0 | 2048 | 61.3 / 255 | 24% | 1.00x | 22.3 / 30.2 | 74% | 1.05x |
| Qwen2.5-1.5B Q8_0 | 8192 | 46.4 / 170 | 27% | 1.01x | 17.7 / 22.0 | 80% | 1.02x |
| Llama-3.2-1B IQ3_S | 512 | 88.0 / 383 | 23% | 0.99x | 18.4 / 30.7 | 60% | 1.43x |
| Llama-3.2-1B IQ3_S | 2048 | 79.9 / 334 | 24% | 1.01x | 17.5 / 27.7 | 63% | 1.35x |
| Llama-3.2-1B IQ3_S | 8192 | 52.8 / 207 | 26% | 1.01x | 13.6 / 19.6 | 69% | 1.24x |
| Llama-3.2-3B Q4_K_M | 512 | 33.0 / 111 | 30% | 1.00x | 9.2 / 22.7 | 41% | 1.13x |
| Llama-3.2-3B Q4_K_M | 2048 | 31.4 / 101 | 31% | 1.03x | 8.5 / 17.5 | 49% | 1.10x |
| Llama-3.2-3B Q4_K_M | 8192 | 15.7 / 70.7 | 22% | 0.98x | 4.6 / 5.0 | 92% | 1.00x |

## Results, 2026-10-20: three hosts, four prompt lengths, runner `990b8a27`

The baseline of the 2026-10 performance work, measured before any of it
landed: runner `990b8a27` (1.4.2 plus the changes merged after it) against
llama.cpp build `3bb63cb` (b11748), the same files on both engines, on the
datacenter MIG 1g.24gb slice described above, an RTX 3070 8 GB (Windows 11,
driver 596.36, i7-7700K) and an Apple M1 8 GB (Metal and the CPU).

**Read the limits first.** Prompt processing is where the runner is far
behind: on CUDA, with every layer on the device, it reaches 1.3 to 10.7% of
llama.cpp's prefill rate on every file and length, and the gap widens with
the prompt (Llama-3.2-3B Q4_K_M on the slice: 8.5% at 512 tokens, 2.3% at
16K). The weakest rows are a mixture of experts prompt (gemma-4-26B-A4B Q4_0
at 1.3 to 1.4%) and Phi-4-mini Q4_0 (1.7 to 3.0%). Decode on the dense
K-quant, Q8_0 and Q4_0 files is 52 to 82% of llama.cpp at 512 and 2K tokens
of context and falls to 17 to 32% at 16K, where attention over the context
dominates (gemma-4-12B, mostly sliding-window layers, keeps 45%); the IQ3_S
files decode at 46 to 51% and the two mixture of experts files at 33 to 64%
at 512. Two rows are defects, not ratios: Qwen2.5-32B Q3_K_S decodes at 0.5
to 0.8 tok/s on the slice because the Q3_K decode kernel faulted on the
device and the runner finished on the CPU (fixed after this baseline, see
the changelog), and granite-4.2-8b Q4_K_M at 16K on the RTX 3070 timed out
at 35 of 40 layers on the device (its f16 cache does not fit beside the
weights). The Flash-Next rows run the experts on the host CPU, where
llama.cpp's own rounds differ by up to 5x on a shared machine; read them as
a range, not a ratio.

Method: runner `--bench-json -n 128 -b 512 --gpu-layers 99` (`--gpu off` for
the Apple M1's CPU arm; 16 threads on the slice, 4 on the RTX 3070 host and
the Apple M1); llama.cpp `llama-bench -p L -n 0` for prefill and `-p 0 -n
128 -d L` for decode at the same depth, `-r 2 -ngl 99` (`-ngl 0` for the
Apple M1's CPU arm), flash attention and the 512-token micro-batch at their
defaults (flash attention on for CUDA). Two rounds, engine order
alternating, a warm-up discarded; cells are medians with the round range in
brackets where the rounds differ by more than 0.5%. The runner build at this
commit runs its device prompt window at 64 columns whatever `-b` asks,
against llama.cpp's 512-token micro-batch, so the prefill rows mix kernel
speed with batch width. On the Apple M1 both engines page on the
Llama-3.2-3B 8K CPU row (8 GB of memory, decode about 4 tok/s on each); that
row is a memory-pressure row. Raw rows (host paths and device strings
removed):
[2026-10-20-linux-cuda-mig/](benchmarks-raw/2026-10-20-linux-cuda-mig/),
[2026-10-20-windows-rtx3070/](benchmarks-raw/2026-10-20-windows-rtx3070/),
[2026-10-20-macos-m1/](benchmarks-raw/2026-10-20-macos-m1/).

### MIG 1g.24gb slice, CUDA

| model | prompt tokens | prefill tok/s: runner / llama.cpp | ratio | decode tok/s: runner / llama.cpp | ratio |
|---|---:|---|---:|---|---:|
| Llama-3.2-3B Q4_K_M | 512 | 715 / 8453 | 8.5% | 123 / 165 | 74% |
| Llama-3.2-3B Q4_K_M | 2048 | 518 / 8496 | 6.1% | 89.6 / 154 | 58% |
| Llama-3.2-3B Q4_K_M | 8192 | 234 / 7298 | 3.2% | 43.5 / 122 | 36% |
| Llama-3.2-3B Q4_K_M | 16384 | 128 / 5525 | 2.3% | 25.7 / 93.9 | 27% |
| Qwen2.5-3B Q4_K_M | 512 | 698 / 8007 | 8.7% | 119 / 168 | 71% |
| Qwen2.5-3B Q4_K_M | 2048 | 522 / 8184 | 6.4% | 83.7 / 160 | 52% |
| Qwen2.5-3B Q4_K_M | 8192 | 263 / 7528 | 3.5% | 39.0 / 150 | 26% |
| Qwen2.5-3B Q4_K_M | 16384 | 154 / 6711 | 2.3% | 22.7 / 135 | 17% |
| Qwen2.5-7B Q4_K_M | 512 | 380 / 4198 | 9.0% | 68.7 / 85.0 | 81% |
| Qwen2.5-7B Q4_K_M | 2048 | 310 / 4213 | 7.4% | 56.9 / 82.3 | 69% |
| Qwen2.5-7B Q4_K_M | 8192 | 179 / 3912 | 4.6% | 34.2 / 77.9 | 44% |
| Qwen2.5-7B Q4_K_M | 16384 | 108 / 3537 | 3.1% | 22.0 / 71.5 | 31% |
| granite-4.1-3b Q8_0 | 512 | 613 / 7628 [7606-7651] | 8.0% | 78.3 / 96.4 | 81% |
| granite-4.1-3b Q8_0 | 2048 | 447 / 7542 | 5.9% | 59.7 / 93.2 | 64% |
| granite-4.1-3b Q8_0 | 8192 | 213 / 6292 | 3.4% | 31.0 / 83.5 | 37% |
| granite-4.1-3b Q8_0 | 16384 | 118 / 5057 | 2.3% | 18.7 / 73.4 | 25% |
| granite-4.2-8b Q4_K_M | 512 | 295 / 3347 | 8.8% | 57.0 / 70.8 | 81% |
| granite-4.2-8b Q4_K_M | 2048 | 230 / 3330 | 6.9% | 45.2 / 67.4 | 67% |
| granite-4.2-8b Q4_K_M | 8192 | 116 / 3051 | 3.8% | 25.0 / 57.7 | 43% |
| granite-4.2-8b Q4_K_M | 16384 | 67.0 / 2532 | 2.6% | 15.6 / 48.4 | 32% |
| granite-4.2-3b IQ3_S | 512 | 561 [559-563] / 7093 [7074-7113] | 7.9% | 84.4 / 166 | 51% |
| granite-4.2-3b IQ3_S | 2048 | 418 / 7243 | 5.8% | 63.5 / 156 | 41% |
| granite-4.2-3b IQ3_S | 8192 | 206 / 6138 | 3.4% | 32.1 / 132 | 24% |
| granite-4.2-3b IQ3_S | 16384 | 116 / 4964 | 2.3% | 19.1 / 108 | 18% |
| Phi-4-mini Q4_0 | 512 | 258 / 8576 | 3.0% | 101 / 148 | 68% |
| Phi-4-mini Q4_0 | 2048 | 223 / 8589 | 2.6% | 75.2 / 138 | 54% |
| Phi-4-mini Q4_0 | 8192 | 140 / 7213 | 1.9% | 37.3 / 109 | 34% |
| Phi-4-mini Q4_0 | 16384 | 89.4 / 5243 | 1.7% | 22.2 / 83.3 | 27% |
| granite-4.1-8b Q4_0 | 512 | 228 / 3764 | 6.0% | 61.4 / 74.8 | 82% |
| granite-4.1-8b Q4_0 | 2048 | 186 / 3737 | 5.0% | 47.9 / 71.0 | 68% |
| granite-4.1-8b Q4_0 | 8192 | 104 / 3396 | 3.1% | 25.9 / 60.3 | 43% |
| granite-4.1-8b Q4_0 | 16384 | 62.8 / 2770 | 2.3% | 15.9 / 50.3 | 32% |
| gemma-4-12B Q4_K_M | 512 | 177 / 2378 | 7.4% | 33.4 / 47.0 | 71% |
| gemma-4-12B Q4_K_M | 2048 | 149 / 2373 | 6.3% | 30.3 / 45.8 | 66% |
| gemma-4-12B Q4_K_M | 8192 | 116 / 2290 | 5.1% | 24.8 / 45.2 | 55% |
| gemma-4-12B Q4_K_M | 16384 | 91.2 / 2178 | 4.2% | 20.0 / 44.6 | 45% |
| Qwen3-30B-A3B Q4_K_M (MoE) | 512 | 212 / 3350 | 6.3% | 93.5 [93.2-93.8] / 147 | 64% |
| Qwen3-30B-A3B Q4_K_M (MoE) | 2048 | 170 / 3397 | 5.0% | 61.8 / 131 | 47% |
| Qwen3-30B-A3B Q4_K_M (MoE) | 8192 | 95.0 / 3086 | 3.1% | 26.8 / 116 | 23% |
| Qwen3-30B-A3B Q4_K_M (MoE) | 16384 | 55.5 / 2701 | 2.1% | 15.2 / 88.5 [87.9-89.1] | 17% |
| gemma-4-26B-A4B Q4_0 (MoE) | 512 | 55.1 / 4020 | 1.4% | 37.5 / 113 | 33% |
| gemma-4-26B-A4B Q4_0 (MoE) | 2048 | 53.1 / 4105 | 1.3% | 35.2 [35.1-35.3] / 108 | 33% |
| gemma-4-26B-A4B Q4_0 (MoE) | 8192 | 49.9 / 3947 | 1.3% | 30.0 / 105 | 29% |
| gemma-4-26B-A4B Q4_0 (MoE) | 16384 | 46.4 / 3604 | 1.3% | 25.2 [25.0-25.4] / 100 | 25% |
| Qwen3.8-27B IQ3_S | 512 | 40.1 [40.0-40.2] / 1075 [1038-1113] | 3.7% | 13.6 [13.5-13.8] / 29.8 [29.4-30.3] | 46% |
| Qwen3.8-27B IQ3_S | 2048 | 39.2 / 1093 [1049-1137] | 3.6% | 13.3 [13.1-13.4] / 29.6 [29.2-30.1] | 45% |
| Qwen3.8-27B IQ3_S | 8192 | 35.6 / 1074 [1032-1116] | 3.3% | 12.1 / 28.8 [28.4-29.2] | 42% |
| Qwen3.8-27B IQ3_S | 16384 | 31.3 / 1021 [986-1056] | 3.1% | 10.6 [10.5-10.6] / 27.7 [27.4-28.1] | 38% |
| Qwen2.5-32B Q3_K_S | 512 | 25.7 / 796 | 3.2% | 0.8 [0.8-0.9] / 26.1 | 3% |
| Qwen2.5-32B Q3_K_S | 2048 | 24.5 / 789 | 3.1% | 0.8 [0.7-0.8] / 25.2 | 3% |
| Qwen2.5-32B Q3_K_S | 8192 | 20.5 / 738 | 2.8% | 0.5 [0.4-0.6] / 23.0 | 2% |
| Qwen3.8-Flash-Next IQ3_S, experts on the host | 512 | 43.2 [40.9-45.6] / 191 [85-298] | 22.6% | 17.6 [17.2-17.9] / 27.6 [25.7-29.4] | 64% |
| Qwen3.8-Flash-Next IQ3_S, experts on the host | 2048 | 41.5 [39.9-43.2] / 162 [75-248] | 25.7% | 17.1 [16.5-17.7] / 18.0 [6.0-30.0] | 95% |
| Qwen3.8-Flash-Next IQ3_S, experts on the host | 8192 | 25.0 [20.5-29.5] / 114 [73-155] | 22.0% | 15.4 [15.1-15.6] / 27.3 [25.4-29.2] | 56% |
| Qwen3.8-Flash-Next IQ3_S, experts on the host | 16384 | 31.1 [29.0-33.2] / 194 [81-308] | 16.0% | 13.1 [12.9-13.4] / 14.9 [5.5-24.3] | 88% |

### RTX 3070 8 GB, CUDA

| model | prompt tokens | prefill tok/s: runner / llama.cpp | ratio | decode tok/s: runner / llama.cpp | ratio |
|---|---:|---|---:|---|---:|
| Llama-3.2-3B Q4_K_M | 512 | 593 / 6534 [6454-6614] | 9.1% | 116 [115-116] / 154 | 75% |
| Llama-3.2-3B Q4_K_M | 2048 | 440 / 6452 [6424-6479] | 6.8% | 86.2 [86.0-86.5] / 144 | 60% |
| Llama-3.2-3B Q4_K_M | 8192 | 202 / 5473 | 3.7% | 41.8 / 115 | 36% |
| Llama-3.2-3B Q4_K_M | 16384 | 95.1 / 4489 | 2.1% | 24.8 / 90.7 | 27% |
| Qwen2.5-3B Q4_K_M | 512 | 614 / 6158 | 10.0% | 113 / 157 | 72% |
| Qwen2.5-3B Q4_K_M | 2048 | 473 / 6186 | 7.7% | 81.7 [81.4-82.1] / 153 | 54% |
| Qwen2.5-3B Q4_K_M | 8192 | 229 / 5580 | 4.1% | 39.3 / 140 | 28% |
| Qwen2.5-3B Q4_K_M | 16384 | 134 / 4848 | 2.8% | 22.9 / 127 | 18% |
| Qwen2.5-7B Q4_K_M | 512 | 323 [322-324] / 3014 [2668-3359] | 10.7% | 64.8 [64.6-64.9] / 79.7 [78.4-80.9] | 81% |
| Qwen2.5-7B Q4_K_M | 2048 | 268 [268-269] / 3368 [3346-3390] | 8.0% | 54.4 / 79.3 [78.9-79.6] | 69% |
| Qwen2.5-7B Q4_K_M | 8192 | 153 / 3083 | 5.0% | 32.9 / 73.5 [72.5-74.5] | 45% |
| Qwen2.5-7B Q4_K_M | 16384 | 95.1 / 2729 | 3.5% | 21.3 / 67.7 | 31% |
| granite-4.1-3b Q8_0 | 512 | 513 [511-514] / 5543 [5522-5563] | 9.2% | 75.7 / 91.9 | 82% |
| granite-4.1-3b Q8_0 | 2048 | 379 / 5371 | 7.1% | 58.5 / 88.9 | 66% |
| granite-4.1-3b Q8_0 | 8192 | 175 / 4385 | 4.0% | 30.2 / 79.9 | 38% |
| granite-4.1-3b Q8_0 | 16384 | 99.8 / 3479 | 2.9% | 18.3 / 70.4 | 26% |
| granite-4.2-8b Q4_K_M | 512 | 264 / 2716 [2710-2727] | 9.7% | 53.7 / 67.4 | 80% |
| granite-4.2-8b Q4_K_M | 2048 | 205 / 2694 | 7.6% | 43.0 / 63.4 [62.3-64.5] | 68% |
| granite-4.2-8b Q4_K_M | 8192 | 101 / 2411 | 4.2% | 23.7 / 54.9 [54.2-55.6] | 43% |
| granite-4.2-8b Q4_K_M | 16384 | timeout / 165 | - | timeout / 40.3 | - |
| granite-4.2-3b IQ3_S | 512 | 475 / 5365 | 8.9% | 75.4 / 150 | 50% |
| granite-4.2-3b IQ3_S | 2048 | 355 / 5229 | 6.8% | 58.1 / 143 | 41% |
| granite-4.2-3b IQ3_S | 8192 | 170 / 4290 | 4.0% | 30.3 / 121 | 25% |
| granite-4.2-3b IQ3_S | 16384 | 97.6 / 3415 | 2.9% | 18.3 / 102 | 18% |

### Apple M1 8 GB, Metal

| model | prompt tokens | prefill tok/s: runner / llama.cpp | ratio | decode tok/s: runner / llama.cpp | ratio |
|---|---:|---|---:|---|---:|
| Llama-3.2-1B Q4_K_M | 512 | 281 [276-286] / 855 | 32.9% | 27.5 / 64.6 | 43% |
| Llama-3.2-1B Q4_K_M | 2048 | 223 [221-224] / 799 [796-802] | 27.8% | 23.5 / 60.8 [60.1-61.4] | 39% |
| Llama-3.2-1B Q4_K_M | 8192 | 99.2 [98.9-99.6] / 617 | 16.1% | 15.3 / 49.3 [48.0-50.7] | 31% |
| Qwen2.5-1.5B Q4_K_M | 512 | 231 [220-241] / 620 [618-622] | 37.2% | 21.5 / 50.8 | 42% |
| Qwen2.5-1.5B Q4_K_M | 2048 | 198 [195-201] / 587 [584-589] | 33.8% | 20.1 / 47.7 [47.4-48.0] | 42% |
| Qwen2.5-1.5B Q4_K_M | 8192 | 105 / 462 | 22.6% | 13.4 / 40.1 [39.4-40.9] | 33% |
| Qwen2.5-1.5B Q8_0 | 512 | 215 / 676 | 31.8% | 25.1 [25.0-25.3] / 33.9 | 74% |
| Qwen2.5-1.5B Q8_0 | 2048 | 202 [200-203] / 634 [632-636] | 31.8% | 23.0 [22.8-23.3] / 32.8 | 70% |
| Qwen2.5-1.5B Q8_0 | 8192 | 107 [107-108] / 495 | 21.7% | 14.8 / 29.3 | 50% |
| Llama-3.2-1B IQ3_S | 512 | 296 [292-300] / 820 [815-825] | 36.1% | 29.7 / 68.8 | 43% |
| Llama-3.2-1B IQ3_S | 2048 | 220 [220-221] / 763 | 28.9% | 25.2 / 64.1 [63.3-64.9] | 39% |
| Llama-3.2-1B IQ3_S | 8192 | 98.1 / 593 | 16.5% | 15.9 / 52.4 | 30% |
| Llama-3.2-3B Q4_K_M | 512 | 103 [99-107] / 295 | 34.8% | 11.0 / 25.8 [25.8-25.9] | 42% |
| Llama-3.2-3B Q4_K_M | 2048 | 91.0 [89.0-93.1] / 276 [275-278] | 32.9% | 9.5 / 24.0 [23.8-24.3] | 40% |
| Llama-3.2-3B Q4_K_M | 8192 | 49.7 [49.4-50.0] / 222 | 22.3% | 6.3 / 17.6 | 36% |

### Apple M1 8 GB, CPU (4 threads)

| model | prompt tokens | prefill tok/s: runner / llama.cpp | ratio | decode tok/s: runner / llama.cpp | ratio |
|---|---:|---|---:|---|---:|
| Llama-3.2-1B Q4_K_M | 512 | 97.7 [96.0-99.5] / 320 | 30.6% | 19.8 [19.3-20.4] / 59.5 [59.3-59.7] | 33% |
| Llama-3.2-1B Q4_K_M | 2048 | 83.3 [82.3-84.2] / 275 [267-283] | 30.3% | 18.9 / 49.1 [48.1-50.2] | 38% |
| Llama-3.2-1B Q4_K_M | 8192 | 52.8 [51.7-53.9] / 190 [188-192] | 27.8% | 13.4 / 27.0 [25.4-28.6] | 50% |
| Qwen2.5-1.5B Q4_K_M | 512 | 73.4 [70.9-75.9] / 222 [214-230] | 33.1% | 15.1 [14.6-15.7] / 48.4 [47.8-49.0] | 31% |
| Qwen2.5-1.5B Q4_K_M | 2048 | 64.4 [63.4-65.4] / 198 [197-199] | 32.6% | 15.2 [15.2-15.3] / 42.1 [41.9-42.2] | 36% |
| Qwen2.5-1.5B Q4_K_M | 8192 | 47.8 [46.7-48.9] / 144 [143-144] | 33.2% | 12.4 [12.3-12.5] / 26.7 [25.8-27.5] | 47% |
| Qwen2.5-1.5B Q8_0 | 512 | 73.3 [72.7-74.0] / 289 [274-304] | 25.4% | 22.1 [21.2-23.1] / 30.1 [29.1-31.0] | 74% |
| Qwen2.5-1.5B Q8_0 | 2048 | 64.0 [63.6-64.4] / 249 [237-261] | 25.7% | 21.3 [21.2-21.5] / 27.4 [26.4-28.3] | 78% |
| Qwen2.5-1.5B Q8_0 | 8192 | 47.6 [46.3-48.9] / 166 [164-168] | 28.6% | 16.4 [15.9-17.0] / 19.6 [18.5-20.6] | 84% |
| Llama-3.2-1B IQ3_S | 512 | 92.7 [91.3-94.0] / 375 [374-376] | 24.7% | 13.3 [13.2-13.5] / 29.4 [28.2-30.6] | 45% |
| Llama-3.2-1B IQ3_S | 2048 | 80.9 [80.6-81.2] / 325 | 24.9% | 12.5 [12.2-12.7] / 27.8 [27.6-27.9] | 45% |
| Llama-3.2-1B IQ3_S | 8192 | 51.8 [51.6-52.0] / 205 [201-208] | 25.3% | 10.4 [10.1-10.6] / 18.6 [18.3-18.9] | 56% |
| Llama-3.2-3B Q4_K_M | 512 | 33.7 [33.1-34.3] / 106 [105-106] | 31.9% | 7.8 [7.6-8.0] / 21.0 [20.3-21.6] | 37% |
| Llama-3.2-3B Q4_K_M | 2048 | 31.3 [30.9-31.6] / 97.2 [96.3-98.1] | 32.2% | 7.4 [7.3-7.5] / 15.8 [15.2-16.3] | 47% |
| Llama-3.2-3B Q4_K_M | 8192 | 15.7 / 69.6 | 22.5% | 4.2 [3.9-4.5] / 4.3 [4.0-4.6] | 97% |

What the ratios leave out: the runner's default device routes are held to
its CPU path (decode token-identical in the CUDA gate, prefill within a
recorded tolerance), and that is part of what these numbers cost. The
2026-10-21 section above reports the same matrix on the same hosts after
the 2026-10 work.

## Results, 2026-10-03: RTX 3070, runner 1.0.0

The 1.0.0 release binary (`2cc9c4e`) on the same desktop card, driver and
llama.cpp build as the 2026-09-02 consumer table below: RTX 3070 8 GB,
Windows 11, driver 596.36, i7-7700K, llama.cpp b10754 (`43d87ff2d`) CUDA
prebuilt. Runner via `--bench-json -n 128 -b 64` (512-token prefill,
128-token decode, mean of 2 runs, spread at most 2.5% decode and 3.3%
prefill); llama.cpp via `llama-bench -p 512 -n 128 -ngl 99 -r 2`. Full
offload verified (`full=1`) on every runner row. Raw outputs:
[benchmarks-raw/2026-10-03-rtx3070-1.0.0/](benchmarks-raw/2026-10-03-rtx3070-1.0.0/).

| model | quant | decode tok/s: runner / llama.cpp | prefill tok/s: runner / llama.cpp |
|---|---|---|---|
| Qwen2.5-7B | Q4_K_M | 64.8 / 81.7 (**79%**) | 322.7 / 3445.8 (9.4%) |
| granite-4.1-3b | Q8_0 | 74.9 / 92.6 (**81%**) | 515.4 / 5539.5 (9.3%) |
| Qwen2.5-3B | Q4_K_M | 112.8 / 159.5 (**71%**) | 622.6 / 6229.3 (10.0%) |
| Llama-3.2-3B | Q4_K_M | 109.2 / 158.4 (**69%**) | 596.3 / 6691.2 (8.9%) |
| granite-4.2-8b | Q4_K_M | 52.0 / 68.4 (**76%**) | 265.5 / 2758.1 (9.6%) |
| granite-4.2-3b | IQ3_S | 19.0 / 154.1 (12%) | 469.6 / 5315.6 (8.8%) |

Decode on the dense K-quant and Q8_0 rows sits at 69 to 81% of llama.cpp,
the same band as a month ago. Prefill moved from 3.9-8.8% to 8.9-10.0%:
Runner's own Qwen2.5 prefill roughly tripled (Qwen2.5-7B 112.3 to 322.7
tok/s, Qwen2.5-3B 234.0 to 622.6) since `qwen2` took the tensor-core prefill
path, while llama.cpp's prefill on the same build and driver also reads
higher than on 2026-09-02 (Qwen2.5-7B 2224.5 to 3445.8), so the ratio, not
either absolute, is the result. Prefill is still an order of magnitude
behind and remains the column that decides the next piece of work.

The IQ3_S row is the weak one: granite-4.2-3b at IQ3_S decodes at 12% of
llama.cpp while the same family at Q4_K_M (granite-4.2-8b) decodes at 76%,
so the gap belongs to the codebook i-quant decode path, not the
architecture. Its prefill takes the tensor-core path and sits with the
other rows. Found by this table; the cause is not yet diagnosed and the
item is open in the plan.

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
