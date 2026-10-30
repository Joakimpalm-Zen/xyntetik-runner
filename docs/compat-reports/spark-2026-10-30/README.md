# Spark X2.5 and X2.6: the architecture checked against llama.cpp, float32 reference owed (2026-10-30)

**Unflattering first: no fidelity number exists yet for any Spark file.** The float32 references against XHToken's
BF16 checkpoints, X2.6-4B and X2.5-4B, are booked on the lab CPU lane. What this record holds is an agreement check
between two engines on one file. It shows the new architecture is implemented as llama.cpp implements it, which is
not the same as being right.

## Agreement: Runner against llama.cpp on the same file

| Runner vs llama.cpp, `Spark-X2.6-1.7B-Q8_0.gguf` | Positions | Mean KLD | Top-1 | Margin-qualified top-1 | Top-8 overlap |
|---|---|---|---|---|---|
| raw completions, both on the CPU | 200 | **0.0065** | 96.0% | 100% | 0.964 |

- **File:** the publisher's X2.6 1.7B Q8_0, the family's smallest member, from `XHToken/Spark-X2.6-1.7B-GGUF` at
  revision `cdc3d889`. It shares `modeling_spark.py`, the tokenizer and the template with the 4B files.
- **Runner** ran on the CPU (`--gpu off`), at the commit of this record.
- **llama.cpp** was mainline at `36a0e2f`, built CPU-only (`GGML_METAL=OFF`) on the same laptop, serving the same
  file.
- **Script:** `scripts/kld-compare-raw.py`, word-by-word teacher forcing over `tests/fixtures/mixed-corpus.txt` at
  stride 2, KLD over the union of the two top-20 heads.

One known source of difference: Runner computes the MLP's GELU in the exact erf form the publisher's modeling code
uses (`ACT2FN["gelu"]`), while ggml's GELU is the tanh approximation. The float32 reference will say which side that
puts nearer the publisher's model.

## What else was checked

- **Tokenizer:** the three Spark tokenizer.json files, X2.5-4B, X2.6-4B and X2.6-1.7B, are byte-identical. Every file
  reads 0 of 721 corpus strings differently.
- **Templates:** `spark` and `spark25`, 23 of 23 cases each in text and tokens, against XHToken's own templates.
- **Chat:** on the X2.6 1.7B Q8_0 it answers a one-word question correctly and works a small time sum, with its
  reasoning split out.

## Files

- `agreement-runner-vs-llamacpp-x2.6-1.7b-q8_0.json`: the 200 positions, per position.
