# LFM2.5-24B-A2B: the publisher's files against the float32 reference (2026-10-30)

**Unflattering first: LiquidAI's Q4_K_M fails the fidelity bar, and the Q8_0 passes it.** Against the BF16 parent with
float32 arithmetic, the Q4_K_M reads a mean KLD of **0.144** against a bar of 0.05, with 90.5% margin-qualified top-1
against a bar of 97%. The Q8_0 reads **0.0297** and **99.1%**, inside the bar on both counts. The Q8_0 is the fidelity
file of the two the publisher quantized.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0297** | 0.0142 | 0.513 | 91.3% | 99.1% (231 positions) |
| Q4_K_M | 300 | **0.144** (bar 0.05) | 0.097 | 1.29 | 84.0% | 90.5% (bar 97%) |

The Q8_0 is the closer of the two (McNemar on the margin-qualified positions: 21 discordant for the Q8_0, 1 for the
Q4_K_M, exact p 1.1e-5; Wilcoxon on per-position KLD p 4e-50).

By depth (mean KLD and top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,493 | 0.0418, 85% | 0.164, 79% |
| 1,508 to 2,993 | 0.0240, 94% | 0.141, 83% |
| 3,008 to 4,493 | 0.0232, 95% | 0.127, 90% |

The Q8_0's error is spread like the 8B-A1B's: 177 of the 300 positions read above 0.01, 17 above 0.1, and the ten
largest carry 26% of the total.

## Why no file-or-engine check here

The 8B-A1B's record ([LFM2.5-8B-A1B](../lfm2.5-8b-a1b-reference-2026-10-30/README.md)) placed its Q8_0's gap in the
file: Runner on the publisher's BF16 GGUF read 0.0078 (median 0.00034, 100% margin-qualified), and another engine read
the same Q8_0 further off than Runner. The 24B-A2B runs the same `lfm2moe` path in Runner, and this Q8_0 sits well
inside the bar (0.0297 against 0.05, 99.1% against 97%), so neither check was repeated for it: no BF16 GGUF score
(44.7 GB) and no second engine.

## Setup

- **Reference:** `LiquidAI/LFM2.5-24B-A2B` at `3fcfdbfa1a1b`, loaded by transformers 5.16.1 (`Lfm2MoeForCausalLM`) with
  torch 2.14.0 on the lab CPU lane (16 cores), by `scripts/gold-logits.py --upcast-at-use`: the weights stay at their
  stored bfloat16 and each decoder block's are cast to float32 for its arithmetic. The run's own line: "float32
  arithmetic on weights stored at torch.bfloat16, cast at use per decoder block (40 blocks, config num_hidden_layers 40;
  43 units hooked)". The forward over 4,510 tokens took 84 seconds once the weights were on local disk.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. The tokenizer
  prepends `<|startoftext|>`, so the reference's sequence starts with it too (`special_prefix_forced`); every position
  matched by id. The tokenizer is byte-identical to the 8B-A1B's, and the files tokenize the corpus as it does.
- **Files:** `LiquidAI/LFM2.5-24B-A2B-GGUF` at `0912d331`: Q8_0 (23.7 GiB), Q4_K_M (13.5 GiB, sha256 d3334405...).
  Runner at `025fd5ce` (`model/lfm2.5-24b-a2b`), `--score --score-top 20 -c 9216 -t 16 --gpu off`, each file fetched
  from the Hub to local disk and read there.
- **Data:** `gold-logits-24b-a2b-q8-q4.json` (the summary and its 300 positions).
