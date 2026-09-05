# Context-grounded drafts: `--draft-lookup`

Measured 2026-09-04 on an Apple M1 (8 physical cores, 8 GB, macOS 26.6.1),
CPU path, runner 0.4.7 plus this change, greedy, `-c 2048`, default threads
(4). Another build-and-test job shared the machine throughout, so the
throughput numbers carry that noise; each cell is the best of three
alternating plain/lookup runs, and the per-run spread is given below the
tables. The negative rows stay.

## What it is

The fourth draft source of the speculative walk, after the draft model
(`--draft`), the target's own MTP head (`--mtp`) and grammar fast-forward.
It needs no weights and no draft forward. Each round the engine takes the
last n tokens of the context (prompt plus everything generated so far, the
pending token included), n from 5 down to 3, longest match first, finds the
most recent earlier occurrence of that n-gram in the context, and proposes
the tokens that followed it, up to `--draft-k`. Past the context end the
proposal continues the match's own period (a run "t t t t" proposes
"t t t t"). No match proposes nothing, so that round is plain decoding plus
one integer search.

The target verifies the proposals in the same batched walk the other three
sources use and samples every token from its own logits, so output is
token-identical to plain decoding, greedy and seeded alike. Where it pays is
input-grounded output: repeating or quoting a document, summaries that reuse
the source's wording, small code edits, tool results echoed back into the
reply. Where it costs is a compute-bound decode with drafts that get
rejected, because each proposal is one more column of the verify batch.

Composition: one draft source per run. `--draft-lookup` beside `--draft` or
`--mtp` is refused at startup, not silently ignored. Grammar fast-forward
still preempts it under an active constraint, as it does the other sources.

## Gate 1, identity (anchor)

`tests/test_draft_lookup.py`, in `make test`, on the random-weight fixture:
greedy bytes equal to the undrafted run at `--draft-k` 1 and 4 on a
repeating prompt (the lookup drafted every round), a prompt with no repeats
(it drafted nothing) and a multi-turn chat that echoes a tool result;
seeded-sampling token ids (`--transcript`, seed 11, temperature 0.9,
40 tokens with `--ignore-eos`) equal on the repeating and tool-echo prompts.
On SmolLM2-135M-Instruct Q8_0 (skipped when the file is absent): a sentence
seen three times is continued byte-identically with 35 of 40 drafts accepted,
3.69 tokens per round, so the search is matching the right tokens and not
merely being rejected.

The search itself is the absolute anchor: `tests/test_lookup_draft.c` pins
`engine_lookup_draft` against eight hand-computed proposals (longest n wins,
most recent occurrence wins, period continuation, the sub-minimum 2-gram is
not a match, k caps, empty and one-token contexts).

Every measured row below also asserted `stdout` equal between the plain and
the lookup run of the same prompt, and the never-draft mutation runs (gate 3)
produced the same bytes as plain decoding.

## Gate 2, accounting

`runner_telemetry.speculation` on every buffered surface: `source`
(`model`, `mtp`, `lookup`, or `grammar` when only grammar fast-forward
drafted), `rounds`, `drafted`, `accepted`, and the lookup's share as
`lookup_drafted`/`lookup_accepted`. The transcript record carries the same
`speculation` object whenever a source was configured (absent for plain
decoding, so those records are byte-for-byte what they were).
`GET /v1/capabilities` reports `draft.source`. The three
`runner_speculation_*` counters on `/metrics` keep their names and keep
counting every source; the server test asserts that one lookup request
advances each of them by exactly the request's own figure.

## Gate 3, cost

The search, `engine_lookup_draft`, on a context with no repeat at all (every
n from 5 down to 3 scans the whole context and nothing matches, the worst
case) and on a context whose only match sits at the far start:

| context tokens | no match | match at start |
|---|---|---|
| 512 | 0.76 us/round | 0.19 us/round |
| 2048 | 2.82 us/round | 0.70 us/round |
| 8192 | 11.18 us/round | 2.85 us/round |
| 32768 | 46.72 us/round | 11.16 us/round |

Bound stated: under 50 us per round at 32k context, under 12 us at 8k,
against a decode step of 8 to 100 ms on this machine, so below 0.1% of a
token at any context the M1 can hold. `RUNNER_SPEC_PROF=1` reported 0 ms of
draft time on every measured run.

The round itself when nothing is drafted: a binary mutated to never draft
(`engine_lookup_draft` returning 0, confirmed by `spec: 127 rounds, 0
drafted`) against plain decoding on the story prompt, SmolLM2-135M, three
alternating pairs: plain 115.2 / 124.7 / 128.4 tok/s, mutated lookup
110.4 / 127.1 / 126.0 tok/s, identical bytes. A draftless round is a one-row
verify and costs nothing measurable.

## Gate 4, measured speedup

Three input-grounded prompts (repeat a paragraph verbatim; summarize it
reusing its wording; rewrite a Python function with one default changed) and
two unrelated prompts (a short story; explain photosynthesis), each in the
model's chat shape, 128 tokens requested, greedy, `--draft-k 4`.

### granite-4.1-3b Q8_0: not measurable on this machine today

The brief named this model. `runner --fit` on the day:

```
  weights       3.37 GiB
  available RAM 1.15 GiB right now
  verdict       PAGES — 2.39 GiB over even with the smallest cache. Every token would page from disk.
```

A probe run confirmed it (the repeat prompt: 50 of 50 drafts accepted,
4.57 tokens per round, and 0.31 tok/s with every token paging). A paging
decode measures the disk, not the walk, so the row is withheld rather than
reported as a speedup. The 8 GB M1 with a concurrent build job is the
reason; the number should be taken on a machine where the model is
resident.

### SmolLM2-135M-Instruct Q8_0 (145 MB, resident)

| prompt | kind | tokens | plain tok/s | lookup tok/s | speedup | rounds | drafted | accepted | acceptance | tok/round |
|---|---|---|---|---|---|---|---|---|---|---|
| repeat | input-grounded | 102 | 116.37 | 171.02 | 1.47x | 24 | 80 | 78 | 98% | 4.25 |
| summary | input-grounded | 75 | 112.57 | 114.56 | 1.02x | 66 | 32 | 9 | 28% | 1.14 |
| codeedit | input-grounded | 128 | 85.92 | 88.90 | 1.03x | 49 | 99 | 79 | 80% | 2.61 |
| story | unrelated | 128 | 86.15 | 87.49 | 1.02x | 127 | 4 | 0 | 0% | 1.01 |
| science | unrelated | 128 | 75.65 | 89.39 | 1.18x | 111 | 24 | 16 | 67% | 1.15 |

Per-run spread (plain / lookup, three pairs): repeat 104.7-116.4 /
164.9-171.0; summary 67.6-112.6 / 60.0-114.6; codeedit 71.2-85.9 /
75.9-88.9; story 75.4-86.2 / 80.7-87.5; science 64.0-75.7 / 69.4-89.4.
Only the repeat row is outside the noise; the others are parity within it.
The science row's 1.18x is noise (24 drafts in 128 tokens cannot buy 18%).

### TinyLlama-1.1B Q2_K (480 MB, resident)

| prompt | kind | tokens | plain tok/s | lookup tok/s | speedup | rounds | drafted | accepted | acceptance | tok/round |
|---|---|---|---|---|---|---|---|---|---|---|
| repeat | input-grounded | 122 | 11.46 | 12.27 | 1.07x | 41 | 88 | 81 | 92% | 2.98 |
| summary | input-grounded | 103 | 12.38 | 10.76 | 0.87x | 77 | 48 | 26 | 54% | 1.34 |
| codeedit | input-grounded | 89 | 16.87 | 15.40 | 0.91x | 71 | 36 | 18 | 50% | 1.25 |
| story | unrelated | 128 | 13.50 | 11.39 | 0.84x | 122 | 20 | 5 | 25% | 1.05 |
| science | unrelated | 128 | 12.32 | 11.51 | 0.93x | 119 | 20 | 8 | 40% | 1.08 |

Per-run spread: repeat 10.0-11.5 / 9.7-12.3; summary 10.5-12.4 /
10.3-10.8; codeedit 11.6-16.9 / 10.2-15.4; story 11.8-13.5 / 10.6-11.4;
science 12.0-12.3 / 9.5-11.5.

### Reading the tables

The literature's 2 to 4x is for a bandwidth-bound decode, where a verify
column is nearly free and the acceptance rate converts almost directly into
speed. Neither model here decodes that way on this box. The 135M model is
overhead-bound (a 145 MB weight pass at 100+ tok/s is far under the M1's
bandwidth), so a 5-row verify is not free, and the summary and code-edit
rows land at parity despite 80% acceptance on the edit. TinyLlama at Q2_K is
compute-bound in the dequant (11 to 17 tok/s for 1.1B parameters), which is
the regime `docs/performance.md` already records for `--mtp` on a 4-core
AVX2 desktop: every added column costs close to a full forward, and even
92% acceptance on the repeat row buys only 1.07x while every other row loses
7 to 16%. These are the negative results and they stay: on a compute-bound
CPU decode, prompt lookup at `--draft-k 4` is a loss unless the output is a
near-verbatim copy of the input.

What the gate does establish: the walk is exact, the accounting is exact, a
draftless round is free, and the search finds what it should (98% and 92%
acceptance on the repeat rows, 80% on the code edit). The speedup itself is
a property of the target's decode regime and remains to be measured on a
resident 3B to 8B Q8_0 target, which is where the M1's decode is
bandwidth-bound.

### The minimum n, measured

The first build tried n down to 2. On SmolLM2-135M the unrelated story
prompt drafted 24 tokens and had 1 accepted; the science prompt 48 with 20;
the summary 84 with 15. At a minimum of 3 the same runs drafted 4 (0
accepted), 24 (16) and 32 (9), while the repeat and code-edit rows kept
their drafts (80 and 99 drafted, 98% and 80% accepted). A 2-gram match is
mostly spurious and each spurious draft is a wasted verify column, so the
minimum is 3.

## Gate 5

`make test`, `scripts/conformance.sh` and `scripts/help-parity.py` green;
counts in the pull request.

## R3.7.2, designed and not built

A session suffix tree over earlier outputs, with the draft length set by the
match score, was to follow only if time remained after this item was fully
gated. It was not built; the design, for the next session:

- **Store.** One per-slot array of the last N tokens the slot generated
  across requests (a ring of, say, 8k tokens), plus a hash table from
  4-gram to its latest position in the ring. The n-gram search above is
  linear in the context and already cheap; a suffix tree buys nothing at
  these sizes, a 4-gram hash does. The current context stays the primary
  source; the session store is consulted on a miss.
- **Score.** The length of the match (n) and how far the earlier occurrence
  continued matching after the n-gram (extend the comparison forward past
  the suffix into the proposal region on the previous round's outcome).
  Draft length = min(`--draft-k`, matched-extension + 1), so a 3-gram match
  with no history of continuing proposes one token and a long verbatim run
  proposes the full window. On the tables above this is the missing piece:
  the compute-bound rows lose because every match drafts four columns
  regardless of how likely they are.
- **Gates.** The identity gate unchanged (the walk decides); an accounting
  split of `lookup_drafted` into context and session shares; a cost gate on
  the hash table per token; and the same five-prompt table with the
  adaptive length, where the summary and unrelated rows are the ones that
  must move toward parity.
