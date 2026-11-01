# Spark X2.6 and X2.5: both 4B files against the float32 reference, architecture agreement check (2026-10-30)

**Unflattering first: XHToken's Q4_K_M fails the fidelity bar on both models.** Against XHToken's BF16 checkpoints
computed in float32, `Spark-X2.6-4B-Q4_K_M.gguf` reads a mean KLD of **0.113** and `Spark-X2.5-4B-Q4_K_M.gguf` reads
**0.1136**, against a bar of 0.05, each with 87.0% top-1. On the margin-qualified positions X2.6 reads 93.6% and X2.5
95.6%, both under the 97% bar. XHToken's Q8_0 files read **0.0132** (X2.6) and **0.0044** (X2.5), each with 98.7% top-1
and 100% margin-qualified, so the gap is the Q4_K_M files' quantization, not the arithmetic: the Q8_0 files are the
fidelity files. The llama.cpp column (2026-11-01) is below: on the X2.6 Q8_0 it reads the lower mean, from one
position where the file's Q8_0 rounding tips a massive activation.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| X2.6-4B Q8_0 | 300 | **0.0132** | 0.0005 | 2.37 | 98.7% | 100% (236 positions) |
| X2.6-4B Q4_K_M | 300 | **0.113** (bar 0.05) | 0.034 | 4.72 | 87.0% | 93.6% (bar 97%) |
| X2.5-4B Q8_0 | 300 | **0.0044** | 0.0005 | 0.67 | 98.7% | 100% (229 positions) |
| X2.5-4B Q4_K_M | 300 | **0.1136** (bar 0.05) | 0.062 | 2.81 | 87.0% | 95.6% (bar 97%) |

The paired tests say the Q8_0 is the closer file on both models: McNemar exact p 6.1e-05 (X2.6) and 0.002 (X2.5) on the
margin-qualified positions, Wilcoxon on per-position KLD p below 1e-48 on both.

By depth (mean KLD, top-1 and margin-qualified top-1, 100 positions each):

| Token positions | X2.6 Q8_0 | X2.6 Q4_K_M | X2.5 Q8_0 | X2.5 Q4_K_M |
|---|---|---|---|---|
| 8 to 1,493 | 0.0045, 99%, 100% | 0.109, 87%, 92.7% | 0.0089, 99%, 100% | 0.152, 85%, 96.2% |
| 1,508 to 2,993 | 0.0326, 98%, 100% | 0.155, 84%, 90.7% | 0.0021, 98%, 100% | 0.106, 85%, 94.4% |
| 3,008 to 4,493 | 0.0024, 99%, 100% | 0.075, 90%, 97.5% | 0.0022, 99%, 100% | 0.083, 91%, 96.2% |

- **X2.6 Q4_K_M worst position:** 1,613, KLD 4.72, where the reference's top-1 is ` tiny` and the file's is ` gg`.
- **X2.5 Q4_K_M worst position:** 938, KLD 2.81, where the reference's top-1 is `3` and the file's is `W`.
- **X2.6 Q8_0 worst position:** 1,538, KLD 2.37, on ` a`, which both keep as top-1. It carries most of the middle third's
  0.0326; the X2.6 Q8_0's median is 0.0005, the same as X2.5's.

This settles the GELU question the agreement check below left open only in part: Runner's exact-erf GELU on the
publisher's own files reads 0.0044 to 0.0132 against the publisher's float32 model, inside the bar. The tanh form was
not measured against the reference.

### Setup

- **References:** `XHToken/Spark-X2.6-4B` at revision `d1b5f260` and `XHToken/Spark-X2.5-4B` at `0bcb3567`, the
  publisher's own modeling code (`Spark2_5ForCausalLM`, `trust_remote_code`), float32 on the lab CPU lane (16 cores, no
  GPU), by `scripts/gold-logits.py` from `model/mimo-v2.6-9b` at `fcd461ef` (the scored rows only). The remote code
  needs transformers 4.57.1 (its `_tied_weights_keys` is a list, which 5.x refuses), so both ran under transformers
  4.57.1 and torch 2.8.0. Another tenant ran about 5 CPUs on the same lane through the runs. The lab volume reads about
  8 MB/s uncached, so each BF16 checkpoint (7.8 GB) was read into the page cache first (about 19 minutes each); each
  forward over 4,509 tokens took about 70 seconds.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. The reference and all
  four Runner sides see the same token ids at all 300 positions, with no dropped mass.
- **Runner sides:** score files from `RUNNER_SCORE_CHUNKED=1 runner --score --score-top 20 -c 9216 -t 16 --gpu off` on
  the same lane at `9575e306`, this branch's head before its rebase onto main. At the rebased head the X2.6 Q4_K_M's
  smoke reply on the Metal laptop's CPU is still byte-identical to its pin.
- **Files:** from `XHToken/Spark-X2.6-4B-GGUF` at revision `0e8aa39a`: `Spark-X2.6-4B-Q8_0.gguf` (4,375,021,824 bytes,
  sha256 `145c2aa6...31d58b81`) and `Spark-X2.6-4B-Q4_K_M.gguf` (3,543,441,312 bytes, sha256 `91cb2811...d7f5cbd9`); from
  `XHToken/Spark-X2.5-4B-GGUF` at revision `9826e0be`: `Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes, sha256
  `5c2c3c19...25d9dea2`) and `Spark-X2.5-4B-Q4_K_M.gguf` (2,600,224,352 bytes, sha256 `adfcfa19...40109428`).

## llama.cpp column (2026-11-01)

**Unflattering first: on the X2.6 Q8_0, llama.cpp reads the lower mean KLD, 0.0086 against Runner's 0.0132.** One
position carries the difference (below). llama.cpp mainline `dde879bc` (`llama-server`, CPU build, x86-64 with
AVX-512, flash attention off, `-c 9216`), against the same float32 references at the same 300 positions with
`scripts/gold-logits.py --endpoint-ids` (each position sent as the reference's own token ids), the reference under
transformers 4.57.1 as above.

| File | Engine | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| X2.6-4B Q8_0 | llama.cpp, CPU | 300 | **0.0086** | 0.0010 | 0.734 | 98.00% | 100.00% |
| X2.6-4B Q4_K_M | llama.cpp, CPU | 300 | **0.1232** | 0.0357 | 4.570 | 85.00% | 91.53% |

- **X2.6 Q8_0:** paired per position, Runner is the closer (Wilcoxon z 9.0, p 2e-19, llama.cpp minus Runner); no
  margin-qualified top-1 discordance. Runner's mean is one position: at 1,538 Runner reads KLD 2.37 and llama.cpp 0.10.
  Without it, Runner reads 0.0053 and llama.cpp 0.0083.
- **X2.6 Q4_K_M:** Runner 0.113 against llama.cpp 0.123, not distinguishable per position (Wilcoxon p 0.074; McNemar
  6 to 1 for Runner, p 0.125).
- **The text path:** sending each prefix as text gives the same numbers as the ids here (0.0086 and 0.123): the
  publisher's tokenizer adds no BOS, so the files' missing `add_bos_token` changes nothing for this family.

Why the two engines differ, one numbered sentence each:

1. llama.cpp's CPU dot products quantize the activations to 8 bits before the integer dot, and Runner multiplies the
   dequantized weights by float32 activations.
2. At position 1,538 the file's Q8_0 rounding tips the model into a massive activation in layer 19's FFN (output norm
   48 against the float32 model's 9.4): the publisher's model with the same weights rounded to Q8_0 gives Runner's
   distribution, so the 2.37 is the file's, and llama.cpp's 0.10 there is its 8-bit activation rounding landing on the
   other side of the same edge. The edge is in the Q8_0 file at that context, and a llama.cpp user meets it too: its
   own step-by-step path (one token at a time from its cache) lands on Runner's side (KL 0.013 against Runner's row).
   R4.28; the material (the activation trace, transformers against itself, the Q8_0-rounded reference) is in shade
   `research/runner-engine/r4.28-spark-row1538-2026-11-01/`.

Files: `gold-logits-llamacpp-x2.6-q8_0.json`, `gold-logits-llamacpp-x2.6-q4_k_m.json` (the column);
`*-text.json` beside them (the text path).

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
uses (`ACT2FN["gelu"]`), while ggml's GELU is the tanh approximation. The float32 references above measure Runner's side
only.

## What else was checked

- **Tokenizer:** the three Spark tokenizer.json files, X2.5-4B, X2.6-4B and X2.6-1.7B, are byte-identical. Every file
  reads 0 of 721 corpus strings differently.
- **Templates:** `spark` and `spark25`, 23 of 23 cases each in text and tokens, against XHToken's own templates.
- **Chat:** on the X2.6 1.7B Q8_0 it answers a one-word question correctly and works a small time sum, with its
  reasoning split out.

## Files

- `gold-logits-cpu-x2.6-q8-q4.json` and `gold-logits-cpu-x2.5-q8-q4.json`: the reference runs, both files each, per
  position (host paths replaced by file and checkpoint names).
- `agreement-runner-vs-llamacpp-x2.6-1.7b-q8_0.json`: the 200 positions, per position.
