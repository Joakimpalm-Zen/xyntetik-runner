# MiMo-V2.6-Distill-Qwen-9B: both files against the float32 reference, and the quant bar (2026-10-29, reference 2026-10-30)

**Unflattering first: bartowski's Q4_K_M fails the fidelity bar.** Against XiaomiMiMo's BF16 checkpoint computed in
float32, `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf` reads a mean KLD of **0.0717** against a bar of 0.05, with a top-1 of
**91.7%**. Its margin-qualified top-1 is 98.2%, inside the bar of 97%, so the mean fails, not the qualified top-1. It
also fails the quant bar against its own Q8_0 (0.064). The Q8_0 of the same checkpoint reads **0.0075**, 99.0% and 100%
margin-qualified, so the gap is the Q4_K_M file's quantization, not the arithmetic: the Q8_0 is the fidelity file. No
llama.cpp column was run.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0075** | 0.0006 | 1.08 | 99.0% | 100% (220 positions) |
| Q4_K_M | 300 | **0.0717** (bar 0.05) | 0.022 | 5.91 | 91.7% | 98.2% (bar 97%) |

The paired test in `gold-logits-cpu-q8-q4.json` says the Q8_0 is the closer of the two: Wilcoxon on per-position KLD p
below 1e-49. On the 220 margin-qualified positions the Q4_K_M misses 4 that the Q8_0 keeps (McNemar exact p 0.125).

By depth (mean KLD, top-1 and margin-qualified top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,493 | 0.0037, 98%, 100% | 0.041, 91%, 98.7% |
| 1,508 to 2,993 | 0.016, 99%, 100% | 0.140, 91%, 97.1% |
| 3,008 to 4,493 | 0.0029, 100%, 100% | 0.035, 93%, 98.7% |

- **Q4_K_M worst position:** 2,243, KLD 5.91, the markdown table separator below. The reference's top-1 is a newline
  (margin 0.68 nats), as the Q8_0's is; the Q4_K_M's is `|`.
- **Q8_0 worst position:** 2,513, KLD 1.08, where the reference's top-1 is a space and the Q8_0's is `-`.

### Setup

- **Reference:** `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` at revision `1f09ddcc1a0e`, loaded by transformers 5.16.1 as
  `Qwen3_5ForCausalLM` with torch 2.14.0, float32 on the lab CPU lane (16 cores, no GPU), by `scripts/gold-logits.py` at
  `fcd461ef` on this branch (the scored rows only, `logits_to_keep`). Weights loaded in 253 seconds and one forward over
  4,509 tokens took 87. Another tenant ran about 5 CPUs on the same lane throughout; the reference is deterministic, so
  that costs wall time only.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. The reference and
  both Runner sides see the same token ids at all 300 (matched by id, no dropped mass); the tokenizer adds no BOS and
  neither does Runner for this family.
- **Runner sides:** the two score files of the quant bar below, made on the same lane at `f5b25f0a`, the pre-rebase form
  of `e7f03eaf` on this branch.

## Quant bar: Q4_K_M against Q8_0

| Comparison | Positions | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q4_K_M vs Q8_0, from the two score files | 300 | **0.064** (bar 0.05) | 0.022 | 4.69 | 92.0% | 97.7% (218 positions; bar 97%) |

By depth (100 positions each):

| Token positions | Mean KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|
| 8 to 1,500 | 0.066 | 91% | 98.7% (75) |
| 1,500 to 3,000 | 0.091 | 92% | 95.6% (68) |
| 3,000 to 4,493 | 0.034 | 93% | 98.7% (75) |

Two positions carry most of the mean; without them it is 0.038. Both are matched by token id, so neither is a
spelling artefact:

- **Position 2,243** closes a markdown table's separator row (`|---|---|---`). The Q4_K_M puts 61% on `|`, the token
  the text has. The Q8_0 puts 57% on a newline and less than 1% on `|`. KLD 4.69.
- **Position 128** ends a Swedish clause about multilingual models mixing grammar between languages ("...blandar ihop
  grammatik mellan språk, och"). Both files put their top probability on the English ` why`, the Q8_0 at 92% and the
  Q4_K_M at 25%; the text continues ` ge`. KLD 3.25.

The float32 reference answers both (below): at 2,243 its top-1 is the newline, as the Q8_0 has it, so the Q4_K_M's
`|` is the error there; at 128 it puts its top on ` why` too, with a margin of 3.9 nats.

## Method

- **Files:** `MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf` (sha256 `2fad0aa1...3b90ae`) and
  `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf` (sha256 `4bca6f18...048c81`), both from `bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF`
  at revision `4371da10c84f`.
- **Scoring:** both on the lab CPU lane (16 cores), Runner at `f5b25f0a`,
  `RUNNER_SCORE_CHUNKED=1 runner --score --score-top 20 -c 6144 --gpu off`. That is the batched-prefill route, whose
  measured envelope against the one-forward-per-position route is about 1e-6 in log probability
  (`tests/test_score.py`).
- **Corpus:** the first 5,030 tokens of `tests/fixtures/gold-corpus-2k.txt`, the head the reference protocol spans.
- **Score:** KLD of Q4_K_M from Q8_0 over the union of the two top-20 heads, each renormalised. The positions are
  8 + 15n for n below 300, the reference protocol's positions, without its re-tokenization filter, since no tokenizer
  was run here. The per-position rows are in `quant-bar-q4_k_m-vs-q8_0.json`.

## Files

- `gold-logits-cpu-q8-q4.json`: the reference run, both sides, per position (host paths replaced by file and
  checkpoint names).
- `quant-bar-q4_k_m-vs-q8_0.json`: the quant bar, per position, with its method.
