# R3.3.7: draft width defaults per backend and source (2026-11-03)

**Unflattering first.**
- **The adaptive width was stopped by its kill read.** A width that followed each turn's acceptance read 0.972x
  of the best fixed width (geometric mean over three rows; it needed 1.05x). On the NextN tool row it read
  0.902x of that row's own best width. What ships is fixed defaults.
- **On prose, the NextN head loses at the new default too.** In SS-001 (runner 1.12.0, the same host),
  Qwen3.5-4B's head on prose read 0.88x plain at 2, and 0.68x at the old default of 4. The head is opt-in;
  leave it off for prose.
- **One host, three rows.** All rows are from an RTX 3070. The CPU and Metal defaults stay at 4, without a
  new row of their own.

## What changed

With no `--draft-k`, a round's width is `engine_draft_k()` (`src/engine.c`):

| Backend | `--draft lookup` | `--draft mtp` | `--draft PATH` |
|---|---|---|---|
| CUDA | 7 | 2 | 4 |
| CPU, Metal | 4 | 4 | 4 |

An explicit `--draft-k` wins, as before. The startup line of each source prints the width the run takes.
`tests/test_draft_k_defaults.py` pins the table: the CPU column on every host, the CUDA column where a CUDA
device is. The verify walk is untouched, so the output is plain decoding's at any width.

## Kill read (stated before the first measurement, agreed with the coordinator)

**Rows.** The SS-001 study's rows on CUDA:
- Qwen2.5-Coder-7B-Instruct Q4_K_M (sha256 `1664fccab734674a...`) with `--draft lookup` on the edit trace
  (`edit.qwen25.txt`, sha256 `b8a66bd2bdb52bc7...`);
- Qwen3.5-4B-MTP Q4_K_M (`3874209241c9a397...`) with `--draft mtp` on code (`code.qwen3.txt`,
  `dab49f4da4c9501c...`);
- the same model on a tool call under `--json` (`tool.qwen3.txt`, `cfa261e75c9c49fc...`; schema
  `01cf332e801e020b...`).

**Arms.**
- lookup: plain, K 1/2/4/7 and auto;
- NextN: plain, K 1/2/3/4 and auto (capped at 4 for the head).

**Pass.** "Best fixed K" is, per draft source, the one K with the highest pooled tok/s over that source's
rows. To pass, auto had to read at least 1.05x of it (geometric mean over the three rows), and every row's
auto had to read at least 0.98x of that row's own best K. Identity with plain was required on every run.

**Procedure.**
- One runner process per run, a warm-up round discarded, then three rounds with the arm order rotated.
- Command: `runner -m MODEL -f PROMPT -n 256 -c 4096 --temp 0 -s 1 --gpu 99 --bench-json [--json SCHEMA]
  --draft SOURCE --draft-k K|auto`, with `RUNNER_SPEC_STATS=1 RUNNER_SPEC_PROF=1 RUNNER_DEBUG_TOKENS=1
  RUNNER_GPU_FALLBACK=fail`.
- The runner was this branch's base (main 8b2729f8) plus `auto-rule.patch` (the rule as measured).

**Host.** The Windows CUDA host: RTX 3070 8 GB, driver 596.36, every layer on the device, GPU lock held.

**Result.** Identity held on 72 of 72 runs (every run's token ids equal its row's plain ids), with no void
runs. Tok/s is pooled over the three rounds; every arm's spread is under 1.5%.

| Row | plain | K1 | K2 | K3 | K4 | K7 | auto | widths auto took (rounds) |
|---|---|---|---|---|---|---|---|---|
| lookup, edit | 70.0 | 95.7 | 125.0 | | 169.8 | **200.0** | 200.3 | 6: 3, 7: 31 |
| NextN, code | 96.2 | 99.3 | **108.4** | 105.2 | 97.5 | | 104.0 | 0: 33, 2: 19, 3: 25, 4: 27 |
| NextN, tool | 97.1 | 102.4 | 109.9 | **116.3** | 114.1 | | 104.9 | 0: 11, 2: 8, 3: 17, 4: 10 |

Best fixed K per source, pooled:
- lookup: K7 (199.95 tok/s);
- NextN: K2 (108.83, against K3 108.47).

| Row | auto / best fixed K | auto / the row's own best |
|---|---|---|
| lookup, edit | 1.002 | 1.002 (K7) |
| NextN, code | 0.960 | 0.960 (K2) |
| NextN, tool | 0.955 | 0.902 (K3) |

Geometric mean: 0.972. **KILL.**

## Why auto lost

The rule estimated per-token acceptance from the turn's rounds (decay 0.8 a round, a prior of 0.9 for lookup
and 0.7 for the head). It then took the width with the most expected tokens per step, from the measured CUDA
verify costs (SS-001 R1) and a head draft cost of 0.23 of a step per token.

On lookup it found 7 and matched fixed 7, so it had nothing to win. On the head, a third of the code rounds
went to width 0. A width-0 round still pays the head's one-row verify forward, and the turn loses the
acceptance it would have measured. The other rounds swung between 2 and 4 instead of staying at 2 to 3. The
widths are logged per row in `kill-rows.jsonl` (`auto_widths`).

## The defaults

- **CUDA lookup: 7.** The best fixed width on the edit trace, 2.86x plain against 2.43x at 4. A verify row
  costs about 0.2 of a step up to eight rows (SS-001 R1), and the ninth steps off the exact route, so 8 is
  never the default.
- **CUDA NextN: 2.** The best pooled width over code and tool: 1.13x plain on both, where 4 read 1.01x on code.
  Tool alone does best at 3 (1.20x).
- **Everything else: 4.** No new rows. On the CPU a verify row costs 0.45 to 0.78 of a step (SS-000), and
  lookup there reads about 1.37x at 4 and 1.35x at 8. Metal pays K + 1 steps for K + 1 rows until its verify
  twins land (R3.3.9).

## Files

- `kill-rows.jsonl`: one line a run (cell, seq, arm, tok/s, spec counts, auto's widths, the ids' sha256;
  no host paths).
- `auto-rule.patch`: the adaptive rule exactly as measured, against main 8b2729f8. It is not merged.
