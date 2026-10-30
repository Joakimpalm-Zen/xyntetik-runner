# LFM2.5-8B-A1B: the publisher's files against the float32 reference (2026-10-30)

**Unflattering first: LiquidAI's Q4_K_M fails the fidelity bar by far, and the Q8_0 passes it narrowly, with an open
question about why.** Against the BF16 parent computed in float32, the Q4_K_M reads a mean KLD of **0.237** against a
bar of 0.05, with 85.4% margin-qualified top-1 against a bar of 97%. The Q8_0 reads **0.0443** and 97.6%, inside the
bar. That is about ten times the dense LFM2.5 Q8_0 files (1.2B-Instruct 0.0042, 350M 0.0081;
[their record](../lfm2.5-reference-2026-10-30/README.md)) and above Mellum2.1's MoE Q8_0 (0.0187), and its median is
0.0124 where the 1.2B's is 0.0007. Until the two checks below say whether the gap is the file's quantization or
Runner's `lfm2moe` arithmetic, the Q8_0 is not named the fidelity file.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0443** | 0.0124 | 0.959 | 93.7% | 97.6% (212 positions) |
| Q4_K_M | 300 | **0.237** (bar 0.05) | 0.134 | 2.05 | 75.3% | 85.4% (bar 97%) |

The Q8_0 is the closer of the two (McNemar on the margin-qualified positions: 28 discordant for the Q8_0, 2 for the
Q4_K_M, exact p below 1e-6; Wilcoxon on per-position KLD p 8e-46).

By depth (mean KLD and top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,493 | 0.0589, 91% | 0.270, 74% |
| 1,508 to 2,993 | 0.0490, 93% | 0.284, 73% |
| 3,008 to 4,493 | 0.0249, 97% | 0.158, 79% |

The Q8_0's error is spread rather than concentrated: 166 of the 300 positions read above 0.01, 30 above 0.1, and the
ten largest carry 40% of the total. Router flips on near-tied expert scores (32 sigmoid-routed experts, four used per
token) would concentrate the error at a few positions; a spread this wide, and falling with depth, is not yet
explained.

## Pending: file or engine

- **Runner on the publisher's BF16 GGUF** (`LFM2.5-8B-A1B-BF16.gguf` in the same repository revision) against the same
  float32 gold. About 0.001 places the 0.0443 in the Q8_0 file; near 0.04 places it in Runner's `lfm2moe` path, which is
  then root-caused before any LFM2.5 MoE row merges.
- **Another engine on the same Q8_0**, scored on the same corpus as a second side against the same gold.

## Setup

- **Reference:** `LiquidAI/LFM2.5-8B-A1B` at `5dd22602c2e9`, loaded by transformers 5.16.1 (`Lfm2MoeForCausalLM`) with
  torch 2.14.0, float32 on the lab CPU lane (16 cores), by `scripts/gold-logits.py`. The forward over 4,509 tokens took
  about 30 seconds once the weights were on local disk.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. The tokenizer
  prepends `<|startoftext|>`, so the reference's sequence starts with it too (`special_prefix_forced`); every position
  matched by id. On Runner's tokenizer before runner #479 the corpus read differently at token 94 (" läsa"), and the
  comparison refused to run; these scores were taken on the fixed tokenizer.
- **Files:** `LiquidAI/LFM2.5-8B-A1B-GGUF` at `49c14831`: Q8_0 (8.4 GiB), Q4_K_M (4.8 GiB, sha256 4923ec14...).
  Runner at `bee195dd` (`model/lfm2.5-8b-a1b`), `--score --score-top 20 -c 9216 -t 16 --gpu off`, each file read from
  a local-disk copy.
- **Data:** `gold-logits-8b-a1b-q8-q4.json` (summary and the 300 positions).
