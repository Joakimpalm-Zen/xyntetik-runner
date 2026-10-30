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
