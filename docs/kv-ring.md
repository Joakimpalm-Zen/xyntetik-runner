# The sliding-window KV ring: what it holds, and when reuse can use it

A sliding-window layer reads back only `swa_window` positions: query position
`q` attends to `[max(0, q - W + 1), q]` (model.c's `t0` clamp). The ring gives
such a layer `R = W + n_batch` rows and stores position `p` at row `p % R`
(`model_kv_row_at`). Full-attention layers keep their `n_ctx` flat rows.

## What the ring holds

A batch fed at `pos` writes rows for `[pos, pos + nb)` with `nb <= n_batch`,
overwriting the slots of positions `[pos - R, pos + nb - R)`, all older than
`pos - W`, which nothing in that batch reads. That is the whole correctness
argument for the ring itself, held by `tests/test_kv_ring.py` against the flat
allocation.

What it still holds is not `[pos - R, pos)`. A slot that rewinds and stops
short of where it was has had its older slots taken by the writes past the
rewind point. So the instance keeps a low watermark, `kv_ring_lo`, the oldest
position every ring layer still holds: each write of `[pos, pos + n)` sets it
to `max(lo, pos + n - R)` (`model_kv_ring_note`, called by every forward that
writes KV), and a write at 0 starts it afresh.

## The one rule every reuse path asks

A state can resume at `keep` when every row position `keep` reads is still
held:

    max(0, keep - W + 1) >= kv_ring_lo

`keep = 0` reads nothing and always resumes. Straight after a prefill of `pos`
tokens the watermark is `max(0, pos - R)`, so a rewind of up to `n_batch + 1`
tokens resumes (with `R = W + n_batch`). `model_kv_ring_resumes(m, keep)` is
the only place that answers it.
Rows for positions in `[keep, pos)` are stale but are rewritten by the next
feed before any position reads them.

## The paths that used to refuse, and what they do now

- **Partial rewind** (`engine_rewind`): a slot keeps `keep` tokens when the rule
  holds, else recomputes from 0 (`prompt_reuse: ring_reset`). Speculative
  rollback and a re-rendered last turn are within `n_batch + 1` tokens and
  keep their KV.
- **Snapshots** (the prefix cache, named contexts, KV snapshots, session
  images; `pfx_save` / `pfx_load`): an entry over `n` tokens stores a flat
  layer's rows `[0, n)` as before and a ring layer's last `min(n, R)` positions
  in position order, so a snapshot's size follows the ring, not `n_ctx`, and
  records the watermark it was taken at. Installing a prefix of `best <= n`
  tokens writes each needed ring position back at `p % R` and sets the
  instance's watermark. A hit is served only when the rule holds against the
  entry's watermark; otherwise it is a miss. The server publishes an entry
  right after a prompt's prefill, while the ring holds the prompt's end. An
  entry is stored whole or not at all (a ring has no state at a shorter
  length to store instead). A file (a saved prefix cache, an exported named
  context, a session image) carries no watermark, so it takes only a ring
  with no hole below its span and reads back as the whole span.
- `/v1/decide`'s shared-prefix seat goes through `engine_rewind` and follows.
- Metal's multi-sequence decode microbatch still declines ring members and
  runs them one at a time.

## The cost, first

Under the ring a shared-prefix hit is lost when the cached prompt runs past
the shared part by more than `n_batch + 1` tokens. A long system prompt
followed by a long first user turn, then a second user turn, is a miss where
the flat cache hits. A short divergent tail (a re-rendered last turn, a short
user message after a shared system prompt), an exact repeat and a
conversation that extends the cached prompt all still hit.

## Prior art

llama.cpp's server keeps a reduced SWA cache by default and reuses prompts
through SWA-only context checkpoints taken when a prompt finishes processing
(PR 15293; `--ctx-checkpoints`, `--checkpoint-min-step`), with `--swa-full`
to branch from any past position at the full cache's cost. Runner's snapshot
of the ring at the end of a prompt is the same idea, held by the same rule.
