# flint: the publisher's Q4_1 against another engine, and the architecture against transformers on a random model (2026-10-30)

**This is agreement, not fidelity.** RecursiveSI's flint is 39.8B parameters, so a float32 reference on the publisher's
checkpoint (about 160 GB) is not feasible on any machine this project runs. Two things are measured instead, and neither
says how close the Q4_1 is to the model RecursiveSI trained. First, on the publisher's Q4_1 file, Runner against mainline
llama.cpp, on the same file and the same positions. Second, the architecture's arithmetic against transformers' own
`FlintForCausalLM` on a small converted model with random weights.

## The publisher's Q4_1, Runner against llama.cpp

| Runner vs mainline llama.cpp 36a0e2f, `flint-int4-q4_1.gguf` | Positions | Mean KLD | Top-1 | Margin-qualified top-1 | Top-8 overlap |
|---|---|---|---|---|---|
| raw completions, both on the CPU | 200 | **0.028** | 90.0% | 99.0% | 0.924 |

- **File:** `RecursiveSI/flint-int4-gguf` at revision `9c471510`, `flint-int4-q4_1.gguf` (25,992,966,976 bytes, sha256
  `43c07bac...08ee1a4b`).
- **Runner:** on the lab CPU lane (16 cores, the GPU hidden), `--gpu off`, at `b37860d9`, the flint branch with the x86-64
  sigmoid fix (#473) applied. Before that fix this file answered `!` to everything on x86-64: the router of layer 46
  reaches -270, and the plain sigmoid returned NaN there.
- **llama.cpp:** mainline at `36a0e2f`, CPU build (`GGML_CUDA=OFF`, `GGML_NATIVE=ON`), serving the same file with
  `--no-repack`.
- **Script:** `scripts/kld-compare-raw.py`, word-by-word teacher forcing over `tests/fixtures/mixed-corpus.txt` at stride
  2, KLD over the union of the two top-20 heads. Median KLD 0.013; the largest, 0.83, sits on a position where llama.cpp's
  top two are 0.0065 nats apart.
- **Compat report** on the same lane at the same commit: load, tokenizer (0 of 721) and chat pass, the report complete
  (`docs/compat-reports/1.7.1-2026-10-30-linux-x86_64-flint-int4-q4_1.json`).

The two engines read the same Q4_1 weights differently in one place: llama.cpp's CPU dot products quantize the
activations to 8 bits for Q4_1, and Runner dequantizes the weights to float32. Which of the two sits nearer the trained
model cannot be told without the float32 reference.

## The architecture against transformers, on a random model

This checks the architecture's arithmetic another way: a small flint with random weights, built in transformers' own
`FlintForCausalLM` (transformers 5.22, which ships the model natively), converted with llama.cpp's converter (36a0e2f) to
an F32 GGUF, and scored by Runner on the CPU at the same token ids.

### The model

`make_toy.py` takes the publisher's config.json at `2a247be0` and shrinks it: hidden 128, 4 heads over 2 KV heads,
head_dim 32, 8 blocks (`full, full, sliding, sliding, sliding, full, sliding, full`), the first two dense, then 8 experts
top 3 by sigmoid score with a 64-wide shared expert, a sliding window of 7. The real tokenizer and vocabulary (150,144
tokens) are kept. Weights are seeded (`torch.manual_seed(0)`) at an initializer range of 0.25, so attention and routing
are decisive rather than near uniform. Sinks are drawn at std 2 and the norms at 1 ± 0.3, so neither sits at its
trivial value. 39.7M parameters, 200 tokens of `tests/fixtures/gold-corpus-2k.txt`.

### Result

| Comparison | Positions | Top-1 | Max abs log-prob difference on Runner's top-20 ids |
|---|---|---|---|
| Runner (CPU, F32 weights, f16 KV cache) vs transformers in float32 | 199 | 99.5% | 0.838 |
| Runner vs transformers with K and V rounded to f16, as Runner caches them | 199 | **100%** | **7.5e-4** |

Runner's KV cache is f16 (`--kv` offers nothing wider), and its rounding is what separates the first row. With K
rounded after rope on the sliding blocks, K rounded after its norm on the NoPE blocks and V rounded on every block
(`f16kv_probe.py`), transformers and Runner agree to 7.5e-4 in log probability at every position. The first row's one
large difference, position 73, is a near tie in expert selection that the rounding tips. It is gone in the second row.

### The check can see each feature

`ablations.json`: how far transformers' own output moves against the unchanged model when one flint feature is taken
away. Each is far above Runner's residual, so a missing or misplaced feature in Runner would show. The margin-qualified
set is the 91 positions where the unchanged model's top-1 leads its second by at least 0.5 nats.

| Change in transformers | Mean KL | Top-1 kept | Margin-qualified top-1 kept |
|---|---|---|---|
| sinks set to -1e4 | 0.655 | 45.0% | 56.0% |
| rope on every block, the full ones too | 5.38 | 3.0% | 4.4% |
| no sliding window | 0.377 | 48.0% | 59.3% |
| routing weights not renormalised | 1.366 | 16.0% | 20.9% |

A window one wider or one narrower than the declared 7 moves top-1 agreement with Runner from 99.5% to 72.4%, so the
window is read with the reference's width.

### Reproduce

```
python make_toy.py OUT_DIR PUBLISHER_FILES_DIR tests/fixtures/gold-corpus-2k.txt   # config.json + tokenizer from RecursiveSI/flint at 2a247be0
python llama.cpp/convert_hf_to_gguf.py OUT_DIR --outtype f32 --outfile toy-f32.gguf
runner -m toy-f32.gguf --score --score-top 20 -f text.txt -c 512 --gpu off > runner.score.json   # text.txt: the decoded 200 ids
python compare.py OUT_DIR/reference.json runner.score.json
python f16kv_probe.py OUT_DIR runner.score.json
```

The KLD that `compare.py` also prints, 0.0011 mean, is over the union of two top-20 heads with a floor for ids one
head lacks. On a random model whose heads hold many near-equal values, that floor dominates, so the log-probability
difference is the figure reported.
