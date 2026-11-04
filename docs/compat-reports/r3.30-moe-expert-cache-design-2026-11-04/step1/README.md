# R3.30 step 1: the routing trace and the three policies

**Result: not a kill. A static hot set alone projects 1.66x decode over today's `--cpu-moe auto`, above the 1.3x
kill, so step 2 is a placement change with no cache.** At the bytes today's placement gives the card's expert banks
(12 of 28 layers, 768 expert slots of 1,792), the mean over six prompts reads:

| | (a) static layers, today | (b) hot set, leave one out | (b) in sample | (c) LRU R=4 | (c) LRU R=8 | (c) LRU R=16 |
|---|---:|---:|---:|---:|---:|---:|
| served from the card, decode routings | 0.429 | **0.785** | 0.875 | 0.790 | 0.862 | 0.883 |
| implied decode gain over (a) | 1.000x | **1.660x** | 1.994x | 1.676x | 1.936x | 2.031x |
| uploads a decode token | 0 | 0 | 0 | 4.0 | 7.1-7.9 | 10.1-12.8 |

- **The hot set needs no uploads,** and its leave-one-out frequencies are honest: the in-sample row is printed only
  to show the overfit.
- **The LRU's gain** assumes its uploads stay off the critical path. At about 0.55 ms an expert over PCIe 3.0 on this
  host, R=8's 7.6 uploads cost about 4 ms a token if they do not.
- **Caveat for step 2:** the gain is projected from whole-bank slopes. A per-expert placement splits every layer
  between the card and the host, so step 2's first read is measured tok/s against `--cpu-moe auto` on the same file
  and card.

**Kill numbers, stated before the first measurement and read in this order:**
1. **The ceiling, from the bench:** the gain if every routed expert were served from the card. Under 1.3x stops the
   item before any trace. It reads **2.76x**.
2. **Flat routing, from the first prompt's trace:** a layer's top-32 share at the uniform noise floor (about 0.56)
   stops the item. It reads **0.940** on the first prompt and 0.923-0.940 on all six.
3. **The policies:** the best honest policy under 1.3x over (a) stops the item. The hot set reads **1.660x**.

## The run (2026-11-04, the RTX 3070, runner main cc62365d)

- **File:** `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF@4159d684`, the Q8_0 (sha256 `69e0ac02...`, 12,925,417,760
  bytes, verified against the Hub's LFS digest).
- **Memory:** the host has 16 GB and the file is 12.9 GB. A first CPU trace, and then `bench.sh`'s loop, were stopped
  by the host for low memory. Everything in `run/` ran one config or one prompt a call as its own foreground job
  (`bench1.sh`, `trace.py` with `R330_PROMPTS`). The bench's 28 and 24 host banks were not run: their expert bytes
  alone (11.8 and 10.1 GB) do not fit the 10.1 GB of RAM available.
- **Placement:** `--cpu-moe auto` puts 12 of 28 layers' expert banks on the card and 16 on the host (5.9 GB in VRAM).

### The bench: today's speed and the host cost of one expert

`bench1.sh`, 128 decode tokens, three rounds with the order rotated (`run/bench/`):

| Host banks | Routed host experts a token | Decode tok/s (3 rounds) | ms a token, median |
|---|---:|---|---:|
| 16 (`auto`) | 128 | 13.37, 14.44, 13.45 | **74.33** |
| 18 | 144 | 13.01, 13.83, 13.76 | 72.67 |
| 20 | 160 | 11.25, 12.42, 11.84 | 84.49 |

- **Host cost of one routed expert:** 0.370 ms, the least-squares slope over all nine runs (intercept 23.8 ms).
- **The noise:** the round-to-round spread is about 8%, and 18 banks' median is under 16's. The slope is the one
  number used from this table besides `T_static` = 74.33 ms.
- **The ceiling:** 74.33 / (74.33 - 128 x 0.370) = 2.76x. 1.3x needs a served share of at least 0.635.

### The trace

`trace.py ... 256 -- --gpu 99 --cpu-moe auto`, greedy, through Mellum's chat template (`run/runs.jsonl`, the
servers' logs in `run/server/`):

| Prompt | Prompt tokens | Working set: 1 / 8 / 32 / 128 tokens | Reuse distance p50 / p90 | Top-32 / top-16 share |
|---|---:|---|---|---|
| code | 102 | 224 / 624 / 1,007 / 1,390 | 1 / 10 | 0.940 / 0.763 |
| copy | 527 | 224 / 671 / 1,077 / 1,418 | 1 / 11 | 0.923 / 0.713 |
| edit | 3,167 | 224 / 662 / 1,086 / 1,415 | 1 / 12 | 0.929 / 0.751 |
| long | 3,866 | 224 / 665 / 1,067 / 1,404 | 1 / 11 | 0.939 / 0.763 |
| prose | 63 | 224 / 662 / 1,053 / 1,418 | 1 / 11 | 0.936 / 0.742 |
| tool | 126 | 224 / 633 / 1,045 / 1,434 | 1 / 10 | 0.927 / 0.742 |

- **The skew is strong:** a layer's 32 most used experts take 92-94% of its decode routings, against about 0.56 on
  uniform routing of this length.
- **The working set is not small:** 128 tokens touch about 1,410 of the 1,792 experts. The gain comes from the skew
  of the frequencies, not from a small working set.
- **Per prompt, the hot set reads 0.664 (copy) to 0.843 (edit).** Every prompt clears the 0.635 a 1.3x needs.

### Files

- `run/sim-k12.txt`: `sim.py run 12 4,8,16 --model 74.33,0.3703`, the tables above.
- `run/trace-*.jsonl.gz`: the traces with each record's position, layer and experts. The gates and norms are dropped
  to keep them small (55 MB raw). Decompressed beside `run/runs.jsonl`, they give `sim.py` the same output byte for
  byte; `run/raw-traces.sha256` names the raw files.
- `run/bench/`: the nine bench runs' JSON and stderr.
- Host paths in the logs read `~/`.

## The tools

| File | What |
|---|---|
| `trace.py` | one server per prompt, each with its own trace, the turn through the model's chat template, greedy |
| `user-turns.json` | the six SS-000 frozen prompts' user turns (code, copy, edit, long, prose, tool), cut from the `*.qwen3.txt` renders |
| `sim.py` | the trace statistics, then the three policies at equal bytes and the implied gain |
| `bench1.sh` | one bench config a call, as `run/` used it |
| `bench.sh` | decode at `--cpu-moe auto` and at 28, 24, 20 and 16 host banks, three interleaved rounds: today's speed, the placement record, and the slope that gives the host cost of one routed expert |
| `synth.py` | the control below |

**What is scored:** decode routings only (positions at or after the prompt's length). The policies see the
prefill's routings, since those happen before decode, but are never scored on them.

**Trace statistics:**
- **Working set:** distinct experts touched in 1, 8, 32 and 128 decode tokens, of 1,792.
- **Reuse distance:** decode tokens between two routings of one expert; uniform routing has a mean of 8.
- **Per-layer skew:** the share of a layer's routings taken by its 32 and 16 most used experts.

**The policies, at the bytes today's placement gives whole banks (K banks, B = 64 K slots):**
- **(a) static layers:** the shallowest K banks, which is today's `--cpu-moe auto`.
- **(b) static hot set:** the B most frequent (layer, expert) pairs, with frequencies taken from the other five
  prompts (leave one out). The in-sample figure is printed beside it only to show the overfit.
- **(c) capped LRU:**
  - B slots, warm-started with (a)'s experts.
  - A miss runs on the host.
  - A missed expert is admitted on its second touch (a ghost list of B keys).
  - At most R admissions a token, evicting the least recently used.

**Implied gain over (a):** `T_static / (T_static - (h - h_a) * 224 * t_e)`, where:
- `T_static` is today's milliseconds a token;
- `t_e` is the host cost of one routed expert, from `bench.sh`'s slope;
- `h` is the policy's share of routed experts served from the bank.

## The control: uniform routing

`synth.py` writes six prompts of uniform random top-8-of-64 routing in Mellum's shape (100 prompt and 300 decode
tokens each, seed 1). `sim.py` on it at K = 14 (half the banks), R = 4 and 8, with a stand-in model of 40 ms a
token and 0.25 ms an expert (`synthetic-check.txt`):

| | (a) static | (b) hot set, leave one out | (b) in sample | (c) LRU R=4 | (c) LRU R=8 |
|---|---:|---:|---:|---:|---:|
| served from the bank, mean of six | 0.500 | 0.500 | 0.552 | 0.498 | 0.498 |
| implied gain over (a) | 1.000x | 1.001x | 1.078x | 0.998x | 0.997x |

- **Honest policies:** every one reads 0.50 and about 1.00x on a flat routing, which is what the arithmetic
  requires.
- **The in-sample row** is the overfit the leave-one-out guards against: 0.55 from noise alone.
- **The skew noise floor:** on 300 uniform tokens, a layer's top-32 share reads 0.56, not 0.50. The real trace's
  skew is read against that floor, not against 0.50.
