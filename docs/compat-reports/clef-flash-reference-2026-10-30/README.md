# Clef-Flash: both ggml-org files against Cloudflare's float32 reference (2026-10-30)

**Unflattering first: the Q4_K_M changes 3 of 29 decisions.** Against Cloudflare's own `joint_schema_model.py` run in
float32, `Clef-Flash-Q4_K_M.gguf` picks the reference's option on 26 of 29 questions, with a largest probability
difference of 0.199. All three flips sit where the reference itself is nearly undecided: its top two options are 0.020,
0.050 and 0.066 apart. `Clef-Flash-Q8_0.gguf` agrees on 29 of 29, with a largest probability difference of 0.012, and is
the file to use when a decision has to match the publisher's model.

## Against the reference

| File | Decisions agree | Max \|dprob\| | Mean \|dprob\| | Max \|dlogit\| |
|---|---|---|---|---|
| Q8_0 | **29 of 29** | 0.0119 | 0.0027 | 0.049 |
| Q4_K_M | **26 of 29** | 0.199 | 0.025 | 2.1 |

The three Q4_K_M flips (reference margin is the gap between the reference's top two probabilities):

| Record / question | Reference margin | Reference's choice, its probability | Q4_K_M's choice, its probability |
|---|---|---|---|
| `long-state` / `worst` | 0.020 | 0.466 | 0.516 |
| `choice-keys` / `route` | 0.050 | 0.272 | 0.238 |
| `instructions-types` / `empty` | 0.066 | 0.533 | 0.506 |

Both files read every record as the reference does: the input ids and the question and option spans are equal on all 15
records. `compare-q8_0.txt` and `compare-q4_k_m.txt` hold `scripts/clef-gold.py compare`'s own output, which lists every
question whose logits differ by more than its default tolerance of 0.001. That tolerance is the one for a float file of
the same weights; on a quantized file each question exceeds it, so those listings are the per-question differences, not
a verdict. `decisions-q8_0-q4_k_m.json` holds the decision, margin and differences of every question.

### Setup

- **Reference:** `Cloudflare/clef-flash` at revision `f0b7d3cd`, its `joint_schema_model.py` (sha256 `0e304cf7c650...`)
  loaded as shipped by `scripts/clef-gold.py reference`, float32 on the lab CPU lane (16 cores, no GPU), transformers
  5.16.1, torch 2.14.0, attention `sdpa`, the head in float32. Another tenant ran about 5 CPUs on the same lane.
- **Records:** the 15 records of `tests/fixtures/clef-records.jsonl`, 29 questions.
- **Runner:** `runner --systemone` with `runner_logits` on, `-t 16 --gpu off`, on the same lane at `686a31da`
  (`model/clef-flash`'s head).
- **Files:** from `ggml-org/Clef-Flash-GGUF` at revision `6e2a164d`: `Clef-Flash-Q8_0.gguf` (9,657,260,288 bytes, sha256
  `25f801e2...ff7d906e`) and `Clef-Flash-Q4_K_M.gguf` (6,732,781,824 bytes, sha256 `eef2968c...f9a9b54e`).
- **The Q8_0's CPU cell** on the lab class (`linux-x86_64-cuda`, the GPU hidden) is recorded in the row by tier-smoke at
  the same commit.

## llama.cpp column (2026-11-01)

**Unflattering first: on the Q4_K_M, llama.cpp's decision server is slightly the closer of the two on the questions it
answers.** On the 12 questions both engines answer, both pick the reference's option on 10. Mean |dprob| is 0.031 for
llama.cpp against Runner's 0.035, with Runner nearer the reference on 5 of the 12. On the Q8_0, both engines agree on all
12, and Runner is the closer: mean |dprob| 0.0023 against 0.0121, max 0.011 against 0.049, nearer on 9 of 12.

| File | Engine | Questions answered | Decisions agree (all answered) | Shared 12: agree | Shared 12: max \|dprob\| | Shared 12: mean \|dprob\| |
|---|---|---|---|---|---|---|
| Q8_0 | Runner | 29 of 29 | 29 | 12 | 0.011 | 0.0023 |
| Q8_0 | llama.cpp | 12 of 29 | 12 | 12 | 0.049 | 0.0121 |
| Q4_K_M | Runner | 29 of 29 | 26 | 10 | 0.199 | 0.035 |
| Q4_K_M | llama.cpp | 12 of 29 | 10 | 10 | 0.175 | 0.031 |

- **llama.cpp:** mainline `dde879bc` (`llama-server`, CPU build, flash attention off, `-c 17408 -b 8192 -ub 8192`),
  serving `/v1/systemone` from `tools/server/server-decision.cpp`, asked each record by
  `scripts/clef-gold.py compare --decisions`. The decision pass needs the whole input in one physical batch, so the
  default 512 refuses the 3,789-token `long-state` record. A larger `-b`/`-ub` is required.
- **Runner:** `runner --systemone` at `e0c6e2c9`, same lane, `--gpu off`. It reproduces the numbers above from the
  10-30 build exactly.
- **What the column does not check:** llama.cpp's server returns no input ids, spans or logits. Its column is
  decisions and probabilities only; the id, span and logit checks above are Runner's alone.
- **Shared flips:** both engines flip `choice-keys`/`route` and `long-state`/`worst` on the Q4_K_M, where the
  reference's top two are 0.050 and 0.020 apart. Runner's third flip, `instructions-types`/`empty` (margin 0.066), is
  on a record llama.cpp refuses.

Why the columns differ, one numbered sentence each:

1. llama.cpp's server refuses 17 of the 29 questions (6 of the 15 records) that the publisher's `joint_schema_model.py`
   answers. It requires `instructions` on every question and a non-empty state, instructions and option list, while
   the reference applies its defaults (a question without instructions, a null state); the refusals are request
   validation, not arithmetic.
2. llama.cpp's CPU dot products quantize the activations to 8 bits and Runner keeps them in float32. That is the Q8_0
   gap (0.0121 against 0.0023). At Q4_K_M the file's own rounding dominates, and the two engines are within each other's
   spread on 12 questions.

Files: `llamacpp-decisions-q8_0.json`, `llamacpp-decisions-q4_k_m.json` (per question, as `decisions-q8_0-q4_k_m.json`
is for Runner), and `llamacpp-compare-q8_0.txt`, `llamacpp-compare-q4_k_m.txt` (the comparison's own output, listing
the refused questions as unreadable).
