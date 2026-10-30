# LFM2.5-1.2B-Instruct and LFM2.5-350M: the publisher's files against the float32 references (2026-10-30)

**Unflattering first: every 4-bit file LiquidAI publishes for these two models fails the fidelity bar, in Runner and in
llama.cpp alike.** Against the BF16 parent computed in float32, the 1.2B-Instruct's Q4_K_M reads a mean KLD of
**0.115** against a bar of 0.05, with 94.7% margin-qualified top-1 against a bar of 97%. llama.cpp reads the same file
at **0.119**, and the two engines are not distinguishable (Wilcoxon p 0.16). The 350M's Q4_K_M reads **0.874**, and
llama.cpp reads **0.924**. The publisher's quantization-aware QAD-Q4_0 files fail too: 0.128 for the 1.2B and 0.483
for the 350M, with llama.cpp at 0.132 and 0.524. Each Q8_0 is close to its parent: **0.0042** and 99.0% top-1 for the
1.2B, **0.0081** and 95.3% for the 350M, both 100% margin-qualified. So the gaps are the files' quantization, not the
arithmetic, and the Q8_0s are the fidelity files.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| 1.2B-Instruct Q8_0 | 300 | **0.0042** | 0.0007 | 0.050 | 99.0% | 100% (227 positions) |
| 1.2B-Instruct Q4_K_M | 300 | **0.115** (bar 0.05) | 0.083 | 0.679 | 85.7% | 94.7% (bar 97%) |
| 1.2B-Instruct Q4_K_M, llama.cpp | 300 | 0.119 | | 0.862 | 84.3% | 93.0% |
| 1.2B-Instruct QAD-Q4_0 | 300 | 0.128 | | 5.08 | 84.0% | 93.4% |
| 1.2B-Instruct QAD-Q4_0, llama.cpp | 300 | 0.132 | | 5.11 | 82.3% | 93.0% |
| 350M Q8_0 | 300 | **0.0081** | 0.0043 | 0.059 | 95.3% | 100% (226 positions) |
| 350M Q4_K_M | 300 | **0.874** (bar 0.05) | 0.591 | 8.19 | 54.3% | 61.9% (bar 97%) |
| 350M Q4_K_M, llama.cpp | 300 | 0.924 | | 8.48 | 53.0% | 61.1% |
| 350M QAD-Q4_0 | 300 | 0.483 | | 3.80 | 67.7% | 76.5% |
| 350M QAD-Q4_0, llama.cpp | 300 | 0.524 | | 4.48 | 65.0% | 74.8% |

On each file the Q8_0 is the closer of the publisher's two (Wilcoxon on per-position KLD p below 1e-50; McNemar on
the margin-qualified positions p 0.0005 for the 1.2B and 3e-26 for the 350M). Runner reads each Q4 file a little closer
to the reference than llama.cpp does; on the 1.2B Q4_K_M the difference is not significant.

By depth (mean KLD, top-1 and margin-qualified top-1, 100 positions each):

| Token positions | 1.2B Q8_0 | 1.2B Q4_K_M | 350M Q8_0 | 350M Q4_K_M |
|---|---|---|---|---|
| 8 to 1,493 | 0.0040, 99%, 100% | 0.137, 80%, 88.0% | 0.0075, 96%, 100% | 0.720, 59%, 66.2% |
| 1,508 to 2,993 | 0.0042, 98%, 100% | 0.115, 88%, 97.4% | 0.0096, 95%, 100% | 1.010, 52%, 59.2% |
| 3,008 to 4,493 | 0.0043, 100%, 100% | 0.093, 89%, 98.6% | 0.0072, 95%, 100% | 0.893, 52%, 60.3% |

### A fact, asked directly

The gap shows in an answer as well as in the KLD. Asked "In three sentences, describe the planet closest to the Sun."
through its own template (temperature 0, the CPU), the 350M's Q4_K_M answers "The planet closest to the Sun is Earth".
The Q8_0 answers "The planet closest to the Sun is Mercury". The 350M rows' smoke asks for the planets in order instead,
which both files get right.

### The QAD files

LiquidAI ships each QAD-Q4_0 beside the float32 weights it was quantized from (`qad/model.safetensors`, after
quantization-aware distillation). Against that source the 350M's QAD-Q4_0 reads 0.528, no closer than against the
original model, and the original Q8_0 reads 0.457 against the same source. The QAD weights are trained for their Q4_0
form, not to run unquantized (LiquidAI's `qad/README.md` says as much), so the original model is the reference these
files are measured against, as the table does.

### Setup

- **References:** `LiquidAI/LFM2.5-1.2B-Instruct` at `0f604ada3f76` and `LiquidAI/LFM2.5-350M` at `9e6c6ccf47cd`,
  loaded by transformers 5.22 (`Lfm2ForCausalLM`, its reference short convolution) with torch 2.15, float32 on the CPU of
  the Metal laptop (Apple M1, 8 GB), by `scripts/gold-logits.py` (the `logits_to_keep` version on `model/mimo-v2.6-9b`).
  The 1.2B's forward over 4,510 tokens took 90 seconds at a 5.9 GB peak; the 350M's 12 seconds at 2.4 GB.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. The tokenizer
  prepends `<|startoftext|>`, as Runner does, and the reference and every side see the same token ids at all 300
  positions, with no dropped mass.
- **Runner sides:** score files from `runner --score --score-top 20 -c 9216 --gpu off` at `d66f0f7a`, the architecture
  commit before the branch was rebased onto main (it is `67e75e8e` there, the same source).
- **llama.cpp sides:** `llama-server` at `3bb63cb`, `-ngl 0`, the same files, read by the same script.
- **Files:** from `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` at `8ed288026e23` (Q8_0 sha256 `f6b981dc...5a30d26a`, Q4_K_M
  `b1b3de11...6b19b4f5`, QAD-Q4_0) and `LiquidAI/LFM2.5-350M-GGUF` at `657e078c9408` (QAD-Q4_0 sha256
  `3d10b6ab...a0e05b1d`, as its `qad/provenance.json` records it).

## The MoE layers against transformers, on a converted random model

`lfm2moe` (LFM2.5-24B-A2B, 8B-A1B) has no row here; its routed experts are checked the way flint's were. `make_toy.py`
takes LFM2.5-24B-A2B's config.json and shrinks it: hidden 128, 4 heads over 2 KV heads, 8 blocks
(`conv, conv, attention, conv, conv, attention, conv, attention`), the first two dense, then 8 experts, top 3 by sigmoid
score with a selection bias, renormalised. The tokenizer is the 1.2B-Instruct's. Weights are seeded at an initializer
range of 0.25, with the selection bias drawn at std 0.3 and the norms at 1 ± 0.3, so neither sits at its trivial value.
transformers' own `Lfm2MoeForCausalLM` saves it in float32, llama.cpp's converter (`36a0e2f`) writes an F32 GGUF, and
Runner scores it on the CPU at the same 200 token ids.

| Comparison | Positions | Mean KLD | Top-1 | Margin-qualified top-1 | Max abs log-prob difference on Runner's top-20 ids |
|---|---|---|---|---|---|
| Runner (CPU, F32 weights, f16 KV cache) vs transformers in float32 | 199 | 2.3e-9 | **100%** | 100% (94 positions) | **5.2e-4** |

Each feature moves transformers' own output by far more when it is taken away (`ablations.json`, against the
unchanged model; the margin-qualified set is the 95 positions where the unchanged model's top-1 leads its second by at
least 0.5 nats):

| Change in transformers | Mean KL | Top-1 kept | Margin-qualified top-1 kept |
|---|---|---|---|
| selection bias set to zero | 0.123 | 62.0% | 85.3% |
| routing weights not renormalised | 0.550 | 36.5% | 48.4% |
| convolution sees the current position only | 6.71 | 0.5% | 0% |

## Reproduce

```
python gold-logits.py --hf LFM2.5-1.2B-Instruct --corpus tests/fixtures/gold-corpus-2k.txt \
  --score-a Q8_0.score.json --score-b Q4_K_M.score.json --max-positions 300 --stride 15 --min-prefix 8
python make_toy.py OUT_DIR LFM2.5-24B-A2B-config-dir LFM2.5-1.2B-Instruct tests/fixtures/gold-corpus-2k.txt
python llama.cpp/convert_hf_to_gguf.py OUT_DIR --outtype f32 --outfile toy-f32.gguf
runner -m toy-f32.gguf --score --score-top 20 -f text.txt -c 512 --gpu off > runner.score.json
python compare.py OUT_DIR/reference.json runner.score.json
```

## Files

- `gold-logits-1.2b-q8-q4.json`, `gold-logits-350m-q8-q4.json`: the reference runs, both publisher files, per position.
- `runner-vs-llamacpp-1.2b-q4_k_m.json`, `runner-vs-llamacpp-350m-q4_k_m.json`: Runner and llama.cpp on each Q4_K_M.
- `qad-q4_0-1.2b-runner-vs-llamacpp.json`, `qad-q4_0-350m-runner-vs-llamacpp.json`: the same for the QAD-Q4_0 files.
- `qad-q4_0-350m-against-its-own-parent.json`: the 350M's QAD-Q4_0 and Q8_0 against the QAD float32 source.
- `make_toy.py`, `compare.py`, `ablations.json`: the converted random lfm2moe.

Host paths in the JSON files are replaced by checkpoint and file names.
