# Dense qwen35 on Metal (R4.10), the D1 row, 2026-10-31

This PR moves the default placement of a dense `qwen35` model on a Metal build from the CPU route to a new Metal route.
The Metal route is within reduction-order rounding of the CPU, not bit-identical, so the owner's D1 rule applies: the
new default must not be worse than the old one against the publisher's float32 reference.

- Model: `Qwen3.5-0.8B-Q8_0.gguf` from ggml-org/Qwen3.5-0.8B-GGUF at revision
  807dfd9a6fadbf703207aac3bfb0b2b3040d077c, sha256 37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f.
- Reference: transformers `Qwen3_5ForCausalLM`, Qwen/Qwen3.5-0.8B at revision 2fc06364715b, float32 on the CPU.
- Corpus: `tests/fixtures/gold-corpus-2k.txt`, 2,000 positions (`--max-positions 2000 --stride 2`).
- Side a: the CPU route (`--gpu off`). Side b: the Metal route (`--gpu auto`, all 24 layers on the device). Both are
  score files from the Metal test host, `RUNNER_SCORE_CHUNKED=1 runner --score 20 -c 8192`, so both took the batched
  prefill route (`score_paths`: keep and keep), and the score guard accepted them as a route comparison.

| side | mean KL vs gold | max KL | top-1 agreement | margin-qualified top-1 |
|---|---|---|---|---|
| a, CPU route | 0.0032175 | 0.05709 | 98.35% | 100.0% |
| b, Metal route | 0.0032237 | 0.05708 | 98.35% | 100.0% |

- Mean KL, b against a: 1.0019x, inside the 1.01x bar.
- Paired Wilcoxon on per-position KL: z -1.03, p 0.30, n 2,000. Not worse at p < 0.05.
- McNemar on the 1,411 margin-qualified positions: 0 discordant either way, p 1.0.
- No position disagrees by more than 1 nat.
- Verdict from `scripts/gold-logits.py`: "not distinguishable". D1 passes, so the Metal route can be the default.

The full report is `gold-logits-cpu-vs-metal.json` beside this file.
