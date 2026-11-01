# MiniCPM5-2B: both publisher files against the float32 reference (2026-10-29)

**Unflattering first: the publisher's Q4_K_M fails the fidelity bar.** Against openbmb's BF16 parent computed in float32,
`MiniCPM5-2B-Q4_K_M.gguf` reads a mean KLD of **0.102** and a top-1 of **86.0%**, against a bar of 0.05 and 97%
margin-qualified top-1. The Q8_0 of the same parent on the same build reads 0.0033 and 98.0%, so the gap is the file's
quantization, not the arithmetic. The Q8_0 is the fidelity file. The llama.cpp column (2026-11-01) is below, with Runner the closer on both files.

## Setup

- **Reference:** `openbmb/MiniCPM5-2B` at revision `f97400052a43`, transformers' own llama code
  (`LlamaForCausalLM`), float32 on the CPU of another host (16 cores), `scripts/gold-logits.py`.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493; the reference
  and both Runner sides see the same 8,042 token ids, BOS included (one special prefix token).
- **Runner sides:** score files from `runner --score --score-top 20 -c 9216` on the Metal laptop (Apple M1, 8 GB),
  matched to the reference by token id at all 300 positions, with no dropped mass.
- **Files:** `MiniCPM5-2B-Q8_0.gguf` (sha256 `c5415f89...cfaba078`) and `MiniCPM5-2B-Q4_K_M.gguf` (sha256
  `ec2d5801...386602fd`), both from `openbmb/MiniCPM5-2B-GGUF` at revision `2079a22f3bea`.

## Against the reference

| File | Positions | Mean KLD | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0033** | 0.050 | 98.0% | 100% |
| Q4_K_M | 300 | **0.102** | 1.419 | 86.0% | 94.6% |

The paired test in `gold-logits-metal-q8-q4.json` says the Q8_0 is the closer of the two: McNemar exact p 0.001 on 205
margin-qualified positions, Wilcoxon on per-position KLD p below 1e-50.

By depth (mean KLD and top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,500 | 0.0037, 100% | 0.099, 88% |
| 1,500 to 3,000 | 0.0036, 98% | 0.094, 86% |
| 3,000 to 4,493 | 0.0026, 96% | 0.113, 84% |

- **Q4_K_M worst position:** 4,403, KLD 1.42, where the reference's top-1 is `/un` and the file's is `">`.
- **Q8_0 worst position:** 368, KLD 0.050, on a newline both keep as top-1.

## llama.cpp column (2026-11-01)

llama.cpp mainline `dde879bc` (`llama-server`, CPU build, x86-64 with AVX-512, flash attention off, `-c 9216`), measured against the same float32 reference at the same 300 positions with `scripts/gold-logits.py --endpoint-ids`: each position is sent as the reference's own token ids, the special prefix included, so both engines see the context the reference saw.

| File | Engine | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| Q8_0 | llama.cpp, CPU | 300 | **0.0050** | 0.0016 | 0.045 | 98.00% | 100.00% |
| Q4_K_M | llama.cpp, CPU | 300 | **0.1079** | 0.0720 | 1.162 | 84.67% | 93.66% |

- **Q8_0:** Runner 0.0033 against llama.cpp 0.0050; paired per-position KLD (llama.cpp minus Runner) Wilcoxon z 10.7, p 1e-26; margin-qualified top-1 discordant positions 0 for Runner, 0 for llama.cpp.
- **Q4_K_M:** Runner 0.1020 against llama.cpp 0.1079; paired per-position KLD (llama.cpp minus Runner) Wilcoxon z 4.7, p 2.3e-06; margin-qualified top-1 discordant positions 4 for Runner, 2 for llama.cpp.

Why the two engines differ, one numbered sentence each:

1. llama.cpp's CPU dot products quantize the activations to 8 bits before the integer dot, and Runner keeps the activations in float32 against the dequantized weights, which is the closer of the two to the float32 reference here.
2. The GGUFs declare `add_bos_token = false` (the converter's copy of a flag for the slow tokenizer), so llama.cpp prepends no BOS to a text prompt, while the publisher's tokenizer.json prepends `<s>` to every encode and Runner follows it: a llama.cpp user who sends text to this file gets a context without BOS, which reads 0.0401 mean KLD on the Q8_0 and 0.120 on the Q4_K_M against the reference (`*-text.json`), against the column's 0.0050 and 0.108 with it.

Files: `gold-logits-llamacpp-q8_0.json`, `gold-logits-llamacpp-q4_k_m.json` (the column); `gold-logits-llamacpp-q8_0-text.json`, `gold-logits-llamacpp-q4_k_m-text.json` (the text-path observation in 2).

## Quant bar: Q4_K_M against Q8_0

| Comparison | Positions | Mean KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|
| Q4_K_M vs Q8_0, from the two score files | 300 | **0.098** (bar 0.05) | 85.3% | 95.0% (bar 97%) |

The house script for this row, `scripts/kld-compare.py`, serves both files at once. On the 8 GB laptop one server was
swapped out entirely and the run wrote nothing in 79 minutes, so it was stopped. The row above is computed from the two
score files instead, at the reference's 300 positions; the method is recorded in `quant-bar-q4_k_m-vs-q8_0.json`. It
agrees with the reference column: the file is about 0.1 from both its parent and its own Q8_0.

## Files

- `gold-logits-metal-q8-q4.json`: the reference run, both sides, per position.
- `quant-bar-q4_k_m-vs-q8_0.json`: the quant bar, per position, with its method.
