# Metal tuning evidence

The measurements behind the Metal backend's tuned constants and routes, moved
here from the comments in `src/metal.m`, which now state only the rule. Times
are from an Apple M1 with 8 GB unless a row says otherwise.

## Chunked decode attention

The split is chosen from the head count: chunking exists to fill the GPU, so
the target is a number of threadgroups (`n_head * n_chunks`).

- First sweep, 3,137 tokens of context, tok/s: single pass 7.32, chunk 256
  7.86, 384 7.94, 512 7.85, 1024 7.51. 384 was exactly span/8, which a target
  of about 64 threadgroups gave an 8-head model.
- Second sweep (2026-08-07), 3,259 tokens on gemma-3-4b: chunk 128 gives 8.130
  tok/s against 8.060 for span/8 (medians of 3 interleaved rounds, arms
  disjoint: [8.13, 8.14] against [8.06, 8.07]). The points both sweeps share
  agree (256 and 512 at 7.96 and 7.95 here against 7.86 and 7.85), so 128 is
  a new point on the curve. The target was raised to 256 threadgroups, which
  leaves many-head models where they were (a 32-head model already gets 32
  threadgroups from the single-pass kernel) and makes few-head models finer.
- The honest size of that tune: against the previous target, interleaved, 3
  rounds, GPU time 119.6 to 119.7 against 120.7 to 120.9 ms a token, disjoint,
  about 1.0%. End to end the medians differ by +0.87% but the arms overlap, so
  it is not separable from noise on this machine. It was kept because the
  direct measure is clean and nothing regressed (short-context decode 9.46
  tok/s, unmoved). A second A/B was abandoned, not reported: macOS began
  rendering an animated aerial wallpaper on the same GPU partway through and
  both arms fell about 7%. Check `WallpaperAerialsExtension` and
  `WindowServer` before trusting a GPU number on an idle Mac.
- For scale, the split itself is worth +9.4% over `RUNNER_METAL_ATTN_CHUNK=0`
  (7.975 against 7.290 tok/s at 3,259 tokens, arms disjoint).
- The floor of 128 positions a chunk: at 40 tokens of context an 8-head model
  would be cut into 8 chunks of 5, and short-context decode measured 9.28
  against 9.20 tok/s doing that. The floor used to be a veto (a target implying
  chunks below it disabled the split), which would have turned the path off at
  exactly the contexts it helps once the target was raised.
- Provenance: the 2026-08-07 change was pushed inside 5d7808b, whose message
  describes only a Makefile fix; its measurement record lived in the comment
  for that reason.

## Partial offload and residency

- The design behind partial offload argued that wiring weights into the Metal
  working set makes them non-evictable, so an oversubscribed model would stop
  paging. You cannot pin more bytes than the machine has. Measured twice on an
  8 GB machine with about 2.8 GB available: gemma-4-26B with 3.8 GB wrapped
  went from 1.5 to 1.7 down to 0.21 tok/s (8x slower), and e4b-q4km with 4.3
  GB wrapped from 5.6 to 0.15 tok/s (35x slower). Both passed the old
  "the tail fits" check; neither could hold its wrap resident. Partial offload
  helps only when the whole model fits in RAM but its weights plus KV and
  scratch exceed Metal's aggregate working-set budget.
- The budget covers KV and scratch, not weights alone: a 5.17 GB wrap was
  refused under a 5.73 GB limit.
- `maxBufferLength` is 4.29 GB against a 5.73 GB working set (0.75x); it bounds
  each wrap only, and the multi-buffer wrapper cuts a larger file at tensor
  boundaries.
- The gemma-4 E-series per-layer token table (2.31 GB of the 5.15 GB E4B QAT
  file) is gathered a row per token on the host; with 3.6 GB available, E4B
  was sent to the CPU at about 0.2 words per second while counting it,
  although its hot set is 2.8 GB.
- Layer order: Qwen2.5-1.5B-Instruct Q8_0 is stored in name order (blk.0,
  blk.1, blk.10 to blk.19, blk.2, ...) with `output_norm.weight` last, and both
  refused every split of it until the wrap reached to the farthest device
  layer.

## Attention front (MoE only)

The front recovers dispatch-chain latency, which is where MoE decode's time
sits: 120B +5.6%, 30B +6.0%, both byte-identical. A fast dense model is
matvec-bound, and the head-partitioned grid starves the GPU the split matvec
grid fills: Qwen3-8B measured -8.4%.

## Fast decode matvec (k_mvf_*)

Promoted for Q4_K on 2026-10-23 on its fidelity rows: mean KL 0.9993x and
0.9998x against the publisher's float32 model, decode +10.8% and +12.5% on the
Q4_K_M Llamas. Q4_0 and Q8_0 measured neutral and stay opt-in.

Q4_K decode time per input width, fast against identity, ms a token (three
rounds each): 1536 wide 11.11 / 9.62 (K2-Horizon-1B Q4_K_M: the fast kernel
15% slower), 2048 10.96 / 12.96 (Llama-3.2-1B) and 1.36 / 1.66 (K2-Horizon),
3072 36.48 / 40.21 (Llama-3.2-3B), 5120 1.88 / 2.45, 8192 1.61 / 2.98 and
5.38 / 7.91. Only 1536 lost, so inputs under 2048 keep the identity kernel
(widths between were not measured).

## Narrow-slice matvecs (k_mvg_*)

Q4_K decode time per input width, fast / identity / k_mvg, ms a token (three
rounds each): 1536 12.28 / 10.28 / 10.35 (a tie with identity, so the exact
kernel keeps it), 2048 11.41 / 12.67 / 9.31, 2560 47.88 / 45.88 / 34.56, 3072
38.05 / 42.14 / 27.68, 4096 5.95 / 8.21 / 5.74, 5120 1.20 / 2.63 / 0.91, 8192
4.78 / 7.73 / 4.45, 9728 7.01 / 10.38 / 6.67.

Q6_K decode time per input width, k_mv2_q6_K / k_mvg_q6_K, ms a step (three
rounds): 1536 (K2-Horizon attn_v and head) 4.77 / 2.05, 2048 0.41 / -0.02 (a
6.9 MB bucket), 2560 2.13 / 1.22, 3072 2.18 / 1.08, 5120 3.95 / 1.85, 8192
3.70 / 2.07 and 9.68 / 6.23, 9728 13.31 / 7.14; the LM heads 11.98 / 5.82
(Qwen3-4B), 8.61 / 3.71 and 10.70 / 5.83 (Llama-3.2-1B, -3B). Narrower inputs
were not measured and keep the exact kernel.

## Cooperative KV score read

Promoted on 2026-08-17 and opt-out since (`RUNNER_METAL_ATTN_COOP=0`): 0 of 64
teacher-forced flips on every local model that engages the route, both KV cache
formats, at +3.0 to 4.3% decode across 2.3k to 8.1k spans (the certificate is
the suite's m1-night-results-2026-08-17.md).

## Small batches: matvec or tiles

GPU ms a forward at n columns, multi-column matvec against GEMM tiles, on the
Metal test host: Qwen3.5-0.8B Q8_0 n=4 43 / 57, n=6 61 / 58, n=16 148 / 63;
granite-4.0-h-micro Q4_K_M n=3 251 / 265, n=4 335 / 267, n=16 1332 / 282 (the
matvec reuses nothing on Q4_K there). So the matvec keeps 2 and 3 columns and
the tiles take 4 and up. Until 2026-10-31 the matvec held 2 to 16 columns,
sized for the speculative window when verify rounds came through this path.

Microbatch decode, aggregate tok/s of N concurrent 96-token requests on
K2-Horizon-1B, matvec against tiles (2026-11-02): Q4_K_M N=2 34.4 / 19.5,
N=4 40.7 / 35.8, N=5 42.2 / 43.0, N=6 43.5 / 50.4, N=8 45.2 / 63.2; Q8_0
N=4 54.1 / 39.9, N=6 56.7 / 56.0, N=7 57.3 / 62.9, N=8 58.2 / 69.9. The tiles
start at 6 columns.

## Speculative verify rounds

A round runs as its solo steps because its rows must be a solo step's bits and
n > 1 takes other kernels (the fixtures differ from the first 2-row round).
Running the rows as one decode microbatch instead measured no faster
(2026-10-21: Qwen2.5-1.5B lookup 21.3 against 21.8 tok/s, Llama-3.2-3B with a
1B draft 6.4 against 6.7): the identity matvecs and unfused stages it runs per
column cost what the fused decode steps do, so a round still costs about a
decode step a row.

## Autorelease pools

Runner calls Metal from plain C threads, which have no autorelease pool, so
command buffers and encoders were never drained: every load left about 113 MB
in the GPU driver until the process exited, invisible to RSS and to
`currentAllocatedSize`, while `POST /unload` reported the model gone.
