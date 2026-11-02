# R3.24 (b): the Metal decode microbatch on the GEMM tiles (2026-11-02)

**Unflattering first: the gain is 1.2x to 1.4x at eight slots, not the 1.5x to 2x estimated, and below six slots the
tiles are slower.** With `RUNNER_METAL_BATCH_TILES=1`, eight concurrent requests on K2-Horizon-1B decode at 63.2 tok/s
in aggregate on the Q4_K_M (45.2 on the identity route) and 69.9 on the Q8_0 (58.2). At two slots the tiles take
19.5 and 21.8 tok/s, against 34.4 and 47.2. The route is opt-in and starts at six columns. It is not bit-identical
to solo decode: over 4,099 teacher-forced positions, each file's tiled scores pick a different top token than solo
decode at 6 positions. Against the publisher's float32 model, the two routes are not distinguishable on either file.

## What the route does

The decode microbatch (the `--parallel` slots that decode together) runs its weight products as multi-column identity
matvecs, each column its solo decode's bits. After R3.24 (a), each block is read and converted once for the column
tile, and what remains per column is the multiply-add on scalar lanes. Under `RUNNER_METAL_BATCH_TILES=1`, a
microbatch of 6 to 8 columns sends those products to the prefill GEMM tiles (`enc_mm_tiles`: the Metal 4 tensor GEMM
where it admitted the type, else the simdgroup tile), which accumulate in another order. Any weight the tiles do not
take stays on the identity matvec. The route detail names it (`micro=tiles`), and so do receipts and the score route.

## Throughput

Apple M1, 8 GB. Server `--parallel 8 -c 4096`, N concurrent `/v1/completions` requests of 96 tokens (`ignore_eos`),
aggregate tok/s by wall time. The two routes were measured alternately, one server each, two rounds at 2, 4 and 8 (the
rounds agree within 0.6 tok/s) and one at 5, 6 and 7. The tiles took every width here; the threshold came from these
rows. `tput.py` is the script and `throughput.jsonl` the rows.

| K2-Horizon-1B, tok/s | N=2 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|
| Q4_K_M identity | 34.4 | 40.7 | 42.2 | 43.5 | 45.0 | 45.2 |
| Q4_K_M tiles | 19.5 | 35.8 | 43.0 | 50.4 | 56.9 | 63.2 |
| Q8_0 identity | 47.2 | 54.1 | 55.5 | 56.7 | 57.3 | 58.2 |
| Q8_0 tiles | 21.8 | 39.9 | 47.7 | 56.0 | 62.9 | 69.9 |

The tiles cost about the same at any width up to their 32 columns, and the identity route grows with each column, so
they cross between 5 and 7 columns. The route starts at 6: +16% on the Q4_K_M there and level on the Q8_0 (56.0
against 56.7).

## Against the publisher's float32 model (D1)

`RUNNER_SCORE_MICRO=8` scores every position as one column of an eight-wide microbatch, beside seven more instances of
the file decoding the same text. On the default route it scores solo's bytes (checked over 199 positions of the Q8_0,
and in `tests/test_metal_batch_tiles.py`). Each file was scored solo (the default decode route) and as tiled
microbatch columns over the first 4,100 tokens of `tests/fixtures/gold-corpus-2k.txt` on an Apple M1 GPU, then read against
`IFM/K2-Horizon-0.9B` at `d50881f4f8dc` in float32 on a CPU (`scripts/gold-logits.py --solo-ok --force-bos`), 2,000
positions at stride 2 from position 8, every position matched by token id.

| File | Route | Mean KLD | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| Q8_0 | solo | 0.003604 | 0.063 | 97.7% | 100% |
| Q8_0 | tiles | 0.003625 (1.006x) | 0.083 | 97.7% | 100% |
| Q4_K_M | solo | 0.10496 | 2.013 | 84.9% | 95.98% |
| Q4_K_M | tiles | 0.10491 (0.9996x) | 2.011 | 84.8% | 95.84% |

- Q8_0: Wilcoxon p 0.86 on the per-position KLD, McNemar p 1.0 (no discordant qualified position), no position past
  1 nat. Not distinguishable.
- Q4_K_M: Wilcoxon p 0.97, McNemar p 0.5 (2 qualified positions only solo got right, none only the tiles), no position
  past 1 nat. Not distinguishable. The file itself fails the fidelity bar on either route (see
  `docs/compat-reports/k2-horizon-0.9b-reference-2026-10-30/`); that is its quantization, not the route.
- Against solo decode directly, 6 of 4,099 positions change their top token on each file, and the largest per-position
  logprob difference is 0.043 (Q8_0) and 0.034 (Q4_K_M).

`gold-logits-q8_0-solo-vs-tiles.json` and `gold-logits-q4_k_m-solo-vs-tiles.json` hold every row.

## Not measured

One model family on one Apple M1 GPU. Larger models, an M5's tensor GEMM, and mixes where the tiles take only some
weights were not measured. A narrower tile than 32 columns, which would cut the padding at 6 to 8 slots, is not part
of this change.
