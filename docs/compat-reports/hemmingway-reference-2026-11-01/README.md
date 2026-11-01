# Hemmingway-1 and Hemmingway-1.5: four files against the float32 reference (2026-11-01)

**Unflattering first: the Hemmingway-1 Q4_K_M fails the fidelity bar.** Against HemmingwayAI's BF16 weights computed in
float32, `Hemmingway-1.Q4_K_M.gguf` reads a mean KLD of **0.248** and a margin-qualified top-1 of **91.6%** (bar: 0.05
and 97%). The same file scored on the CPU reads 0.248 and 91.6% too, so the distance is in the file, not in either
backend. The other three files pass: Hemmingway-1 Q8_0 at 0.0106, Hemmingway-1.5 Q4_K_M at 0.0203 and Hemmingway-1.5
Q8_0 at 0.0021.

This is a validation record only. Both models are CC-BY-NC-4.0. No file is redistributed here and nothing is derived
from them. No llama.cpp column was run.

## Setup

- **References:** `HemmingwayAI/Hemmingway-1` at `faba57469705` and `HemmingwayAI/Hemmingway-1.5` at `2f66c2569b99`,
  transformers 5.16.1 (`Qwen3_5ForCausalLM`, 64 layers), torch 2.14.0 on the CPU, float32 arithmetic on the BF16
  weights cast at use per decoder block (`scripts/gold-logits.py --upcast-at-use`). Hemmingway-1's weights were
  uploaded once (2026-09-20) and every later commit changes only the card or the licence, so the reference holds the
  weights its GGUFs were converted from (their metadata names the repo by its earlier name, `Altworld/Hemmingway-1`, which has the same history).
- **Positions:** 2,000 at stride 1, positions 8 to 2,007 of `tests/fixtures/gold-corpus-2k.txt` (8,107 tokens).
  Every side is matched to the reference by token id at all 2,000 positions, with no dropped mass.
- **Runner sides:** score files from `RUNNER_SCORE_CHUNKED=1 runner --score 20 -f CORPUS -c 8192`, the batched
  prefill route.
  - CUDA: commit `f4a1c212` (runner 1.9.0) on an RTX PRO 6000 Max-Q in a MIG 1g.24gb partition, all 64 layers on the
    device.
  - CPU: commit `65e31f06` (runner 1.10.0) on x86-64 with AVX-512, 24 threads, with no GPU visible.
- **Files**, each checked against the Hub's LFS digest:

| File | Repo at revision | Size | sha256 |
|---|---|---|---|
| `Hemmingway-1.Q4_K_M.gguf` | `mradermacher/Hemmingway-1-GGUF` at `99b95619d459` | 16,810,715,360 | `d8a77565...0d785dbc` |
| `Hemmingway-1.Q8_0.gguf` | same | 29,047,085,280 | `10b56634...6fac1809` |
| `HemmingwayAI_Hemmingway-1.5-Q4_K_M.gguf` | `bartowski/HemmingwayAI_Hemmingway-1.5-GGUF` at `4d91b1bc0502` | 17,442,400,608 | `69b4d499...0a316c4a16` |
| `HemmingwayAI_Hemmingway-1.5-Q8_0.gguf` | same | 29,116,389,728 | `8b3c689c...fa32834b4e` |

## Against the reference

| File | Backend | Mean KLD | Median | p99 | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| Hemmingway-1 Q4_K_M | CUDA | **0.248** | 0.033 | 4.52 | 8.51 | 84.5% | 91.6% |
| Hemmingway-1 Q4_K_M | CPU | **0.248** | 0.033 | 4.52 | 8.49 | 84.55% | 91.66% |
| Hemmingway-1 Q8_0 | CPU | **0.0106** | 0.0003 | 0.171 | 4.04 | 98.15% | 99.54% |
| Hemmingway-1.5 Q4_K_M | CUDA | **0.0203** | 0.0087 | 0.161 | 2.49 | 95.55% | 99.72% |
| Hemmingway-1.5 Q4_K_M | CPU | **0.0203** | 0.0087 | 0.161 | 2.48 | 95.6% | 99.72% |
| Hemmingway-1.5 Q8_0 | CPU | **0.0021** | 0.0002 | 0.024 | 0.67 | 99.1% | 99.93% |

- **CPU against CUDA on the same Q4_K_M file:** "not distinguishable" for both models (Hemmingway-1: 1 discordant
  margin-qualified position of 1,523; Hemmingway-1.5: none of 1,449).
- **Q8_0 against Q4_K_M:** the Q8_0 is the closer for both models. For Hemmingway-1, McNemar exact p 7.5e-37 (121
  discordant positions, all for the Q8_0). For Hemmingway-1.5, the per-position KLD Wilcoxon gives p below 1e-300, and
  McNemar p 0.375 (4 to 1), because the Q4_K_M already agrees on the top-1 almost everywhere.

By depth (mean KLD and top-1, about 667 positions each):

| Token positions | H-1 Q4_K_M | H-1 Q8_0 | H-1.5 Q4_K_M | H-1.5 Q8_0 |
|---|---|---|---|---|
| 8 to 674 | 0.178, 85.8% | 0.0104, 98.8% | 0.0227, 95.4% | 0.0022, 98.8% |
| 675 to 1,341 | 0.316, 83.7% | 0.0096, 98.1% | 0.0188, 96.4% | 0.0023, 99.3% |
| 1,342 to 2,007 | 0.250, 84.1% | 0.0118, 97.6% | 0.0195, 94.9% | 0.0017, 99.2% |

- **Hemmingway-1 Q4_K_M worst position:** 1,768, KLD 8.51. The reference's top-1 is `ent`, the file's is `oll`.
- **Hemmingway-1 Q8_0 worst position:** 1,824, KLD 4.04. The reference's top-1 is ` October`, the file's is
  ` korrekt`.

## Why Hemmingway-1's files sit further out

The two models share the architecture and the tokenizer, but their files were quantized differently. Hemmingway-1's
files come from an older conversion, and Hemmingway-1.5's from a newer one with an importance matrix:

| Tensors | H-1 Q4_K_M | H-1.5 Q4_K_M | H-1 Q8_0 | H-1.5 Q8_0 |
|---|---|---|---|---|
| `ssm_alpha`, `ssm_beta` (the Gated DeltaNet gates, 48 layers) | Q4_K | F32 | Q8_0 | F32 |
| `ssm_out` | Q4_K | Q6_K (36), Q8_0 (12) | Q8_0 | Q8_0 |
| `attn_k`, `attn_v`, `attn_output` (full-attention layers) | mostly Q4_K | mostly Q8_0 | Q8_0 | Q8_0 |
| importance matrix | none | 496 entries | n/a | n/a |

The gates set each recurrent layer's decay and write strength at every token, so their error carries forward through
the recurrent state. The two Q8_0 files' tensor types differ only in those 96 gate tensors, and the Hemmingway-1 Q8_0
reads 5 times the KLD (the models are different fine-tunes, so the weights themselves differ too). This is consistent with the gates being the cause, but it is not proven here: testing
it would mean building a variant file, which the licence and this item exclude.

## Tokenizer, template and chat path

- **Tokenizer:** the Runner's token ids equal the reference tokenizer's at every scored position for all four files,
  and `gold-logits.py` stops on any mismatch.
- **Template:** both models' `chat_template.jinja` and the template inside all four GGUFs are byte-identical to
  `Qwen/Qwen3.8-27B`'s (sha256 `c3cf9e34...f9d7a81041`). That template is the `qwen38` family's oracle in
  `make template-conformance`.
- **Chat path** (CPU, Q8_0, `--serve`, greedy): with thinking off, both models answer "The capital of France is
  Paris." in 7 tokens and stop. With thinking on, both write their reasoning (228 and 200 characters) into
  `reasoning_content`, then the same answer, and stop.
- **Greedy smokes** (raw completion): CUDA Q4_K_M and CPU Q8_0 of both models answer "The capital of France is
  Paris."

## Files

- `h1-gold-logits-cuda-q4_k_m-cpu-q8_0.json` and `h15-gold-logits-cuda-q4_k_m-cpu-q8_0.json`: the reference runs with
  side a = CUDA Q4_K_M and side b = CPU Q8_0, per position.
- `h1-gold-logits-cpu-q4_k_m-cuda-q4_k_m.json` and `h15-gold-logits-cpu-q4_k_m-cuda-q4_k_m.json`: the same file on
  both backends, side a = CPU and side b = CUDA, per position.
