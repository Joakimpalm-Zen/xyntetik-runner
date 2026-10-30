# LFM2.5-8B-A1B: the publisher's files against the float32 reference (2026-10-30)

**Unflattering first: LiquidAI's Q4_K_M fails the fidelity bar by far, and the Q8_0 passes it narrowly; the Q8_0's
gap is the file's quantization, not Runner's arithmetic.** Against the BF16 parent computed in float32, the Q4_K_M reads
a mean KLD of **0.237** against a bar of 0.05, with 85.4% margin-qualified top-1 against a bar of 97%. The Q8_0 reads
**0.0443** and 97.6%, inside the bar but about ten times the dense LFM2.5 Q8_0 files (1.2B-Instruct 0.0042, 350M
0.0081; [their record](../lfm2.5-reference-2026-10-30/README.md)). Two checks place that gap in the file. Runner on the
publisher's BF16 GGUF reads **0.0078** (median 0.00034, 100% margin-qualified). llama.cpp reads the same Q8_0 at
**0.0572** with 96.7% margin-qualified top-1, which fails the bar, further off than Runner. The Q8_0 is the fidelity
file of the two the publisher quantized.

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
ten largest carry 40% of the total.

## File or engine

One run against the same float32 gold, 300 positions:

| Side | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| Runner, the publisher's BF16 GGUF | **0.0078** | 0.00034 | 0.183 | 97.3% | 100% (212 positions) |
| llama.cpp (`36a0e2f`, `-ngl 0`), the same Q8_0 | **0.0572** (bar 0.05) | 0.0169 | 0.902 | 90.0% | 96.7% (bar 97%) |
| Runner, the same Q8_0 (above) | 0.0443 | 0.0124 | 0.959 | 93.7% | 97.6% |

- **The gap is the file's.** On the exact weights Runner is near-exact at the median, and the other engine reads the
  Q8_0 further from the reference than Runner does, outside the bar. Position by position the two engines' Q8_0
  errors go together (Spearman 0.82): the file's quantization, read the same way by both.
- **Runner's BF16 tail.** Its mean is held up by a few positions: the top 20 carry 66% of the total, 52 read above
  0.01. They are positions where both Q8_0 sides diverge too (Spearman 0.47 and 0.52 with them), mostly below the
  median gold margin (1.05). That is consistent with near-tied expert choices (32 sigmoid-routed experts, four per
  token) turning the f16 KV cache's and the accumulation order's rounding into a different choice; it is not yet
  measured. Pending: a dense baseline, Runner on LFM2.5-1.2B-Instruct's BF16 GGUF against its float32 gold, which gives
  Runner's floor where no routing exists.

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
- **Checks:** `LFM2.5-8B-A1B-BF16.gguf` from the same repository revision, scored by Runner at `b846a2bf` (the same code
  as `bee195dd`); llama.cpp's `llama-server` at `36a0e2f`, `-ngl 0 -c 8192 -t 16`, on the same Q8_0, queried by the same
  script. Each file was fetched from the Hub to local disk and read there.
- **Data:** `gold-logits-8b-a1b-q8-q4.json` (the two publisher quantizations) and
  `gold-logits-8b-a1b-bf16-vs-other-engine-q8.json` (the two checks), each the summary and its 300 positions.
