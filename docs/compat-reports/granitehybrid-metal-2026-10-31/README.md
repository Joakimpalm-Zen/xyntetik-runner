# Dense Mamba-2 hybrids on Metal (R4.10), the D1 row, 2026-10-31

This PR moves the default placement of a dense `granitehybrid` or `nemotron_h` model on a Metal build from the CPU
route to a new Metal route. The Metal route is within reduction-order rounding of the CPU, not bit-identical, so the
owner's D1 rule applies.

- Model: `granite-4.0-h-micro-Q4_K_M.gguf` from ibm-granite/granite-4.0-h-micro-GGUF at revision
  dc1dd2585fac18a78001c677d33ef8a7bbb7eb68, sha256 bcc78b9b25450101d1ad90d4b9a264e1bac892f534dfb76066f4eec792fdf023.
- Reference: transformers `GraniteMoeHybridForCausalLM`, ibm-granite/granite-4.0-h-micro at revision d5f01a3ea75f,
  float32 on the CPU.
- Corpus: `tests/fixtures/gold-corpus-2k.txt`, 2,000 positions (`--max-positions 2000 --stride 2`).
- Side a: the CPU route (`--gpu off`). Side b: the Metal route (`--gpu auto`, all 40 layers on the device). Both are
  score files from the Metal test host, `RUNNER_SCORE_CHUNKED=1 runner --score 20 -c 8192`, so both took the batched
  prefill route (`score_paths`: keep and keep).

| side | mean KL vs gold | max KL | top-1 agreement | margin-qualified top-1 |
|---|---|---|---|---|
| a, CPU route | 0.091788 | 1.3402 | 84.35% | 94.35% |
| b, Metal route | 0.091761 | 1.3397 | 84.30% | 94.35% |

- Mean KL, b against a: 0.9997x, inside the 1.01x bar.
- Paired Wilcoxon on per-position KL: z 0.27, p 0.79, n 2,000.
- McNemar on the 1,433 margin-qualified positions: 0 discordant either way, p 1.0.
- No position disagrees by more than 1 nat.
- Verdict: "not distinguishable". D1 passes, so the Metal route can be the default.

The mean KL is the Q4_K_M file's own distance from float32; the CPU side reproduces the golden pass of 2026-09-07
(margin-qualified top-1 94.36% at Q4_K_M). The full report is `gold-logits-cpu-vs-metal.json` beside this file.
