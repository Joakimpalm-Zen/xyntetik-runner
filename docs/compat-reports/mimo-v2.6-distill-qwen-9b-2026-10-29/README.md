# MiMo-V2.6-Distill-Qwen-9B: quant bar measured, float32 reference owed (2026-10-29)

**Unflattering first: bartowski's Q4_K_M fails the quant bar against its own Q8_0.** Over 300 positions it reads a mean
KLD of **0.064** against a bar of 0.05. Margin-qualified top-1 is 97.7%, inside its bar of 97%. **The float32 reference
against XiaomiMiMo's BF16 checkpoint is not measured.** Its run on another host's CPU stalled while loading the weights
and was ended, so no number in this record says how far either file is from the publisher's model.

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

Which file is right at these two positions is the float32 reference's question, so this record states the bar as
measured and makes no fidelity claim.

## Method

- **Files:** `MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf` (sha256 `2fad0aa1...3b90ae`) and
  `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf` (sha256 `4bca6f18...048c81`), both from `bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF`
  at revision `4371da10c84f`.
- **Scoring:** both on another host's CPU (16 cores), Runner at `f5b25f0a`,
  `RUNNER_SCORE_CHUNKED=1 runner --score --score-top 20 -c 6144 --gpu off`. That is the batched-prefill route, whose
  measured envelope against the one-forward-per-position route is about 1e-6 in log probability
  (`tests/test_score.py`).
- **Corpus:** the first 5,030 tokens of `tests/fixtures/gold-corpus-2k.txt`, the head the reference protocol spans.
- **Score:** KLD of Q4_K_M from Q8_0 over the union of the two top-20 heads, each renormalised. The positions are
  8 + 15n for n below 300, the reference protocol's positions, without its re-tokenization filter, since no tokenizer
  was run here. The per-position rows are in `quant-bar-q4_k_m-vs-q8_0.json`.

## The reference run, and why it is owed

`scripts/gold-logits.py` loaded the BF16 checkpoint in float32 on the CPU lane after both score runs had finished. The
weight load reached 412 of 427 tensors and then went silent for 40 minutes with the lane at 45% I/O wait. The
container's server stopped answering, which reads as the container's memory limit thrashing on the mapped weights and
the page cache. The run was ended by restarting the container.

Two changes come before the rerun:

- `gold-logits.py` now projects only the 300 scored rows (`logits_to_keep`). That turns about 4.5 GB of logits,
  and a log-softmax copy of them, into 0.3 GB.
- The rerun loads the weights straight to float32 and prints a progress line per stage, so a stall is visible.

The rerun is scheduled for a lane slot that does not block the lab's other work.
