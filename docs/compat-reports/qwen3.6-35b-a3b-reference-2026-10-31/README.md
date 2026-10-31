# Qwen3.6-35B-A3B: bartowski's files against Qwen's float32 reference (2026-10-31)

**Unflattering first: the Q4_K_M passes the fidelity bar by a hair.** Against Qwen's BF16 parent with float32
arithmetic, bartowski's standard Q4_K_M reads a mean KLD of **0.0475** against a bar of 0.05, with **97.2%**
margin-qualified top-1 against a bar of 97%. Both margins are under three hundredths of their bars, so a different
corpus could move it either way. The Q8_0 reads **0.0349** and **99.1%**, and is the closer file of the two.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0349** | 0.0042 | 5.42 | 94.7% | 99.1% (214 positions) |
| Q4_K_M | 300 | **0.0475** (bar 0.05) | 0.0202 | 1.36 | 89.0% | 97.2% (bar 97%) |

The Q8_0 is the closer of the two (Wilcoxon on per-position KLD p 1.5e-43; McNemar on the margin-qualified positions
4 discordant for the Q8_0, 0 for the Q4_K_M, exact p 0.125).

By depth (mean KLD and top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,493 | 0.0632, 97% | 0.0486, 95% |
| 1,508 to 2,993 | 0.0320, 93% | 0.0559, 85% |
| 3,008 to 4,493 | 0.0094, 94% | 0.0380, 87% |

## Where the Q8_0's error sits

The two files spread their error differently. The Q4_K_M's is broad: 197 of 300 positions read above 0.01, 24 above
0.1, and the ten largest carry 37% of the total. The Q8_0's is concentrated: 90 positions above 0.01, 4 above 0.1,
and two positions carry 71% of the total. Without those two, its mean would be 0.0102.

Those two positions are the file's, not Runner's. Another engine (llama.cpp 36a0e2f, one slot, flash attention
off) read the same Q8_0 at the same 300 positions against the same reference:

| Engine on the Q8_0 | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| Runner | **0.0349** | 0.0042 | 5.42 | 94.7% | 99.1% (214 positions) |
| llama.cpp | **0.0618** | 0.0050 | 7.50 | 94.0% | 98.1% |

At position 1,133, where the reference splits between two tokens (margin 0.25) and the file puts 98% on one, the two
engines read 5.42 and 5.37; at 2,363 they read 1.98 and 0.80. The two engines' per-position errors rank alike
(Spearman 0.82), so the spikes come with the file. Runner is the closer engine overall (Wilcoxon p 0.035; McNemar 2
discordant positions to 0, p 0.5). The other engine's own largest, 7.50 at position 3,608, is a position where the
reference is certain (margin 8.56) and Runner reads 0.000; that one is the other engine's, not the file's.

## The NextN head

bartowski's Q4_K_M carries Qwen's NextN predictor block (block 40: attention, the trunk's MoE FFN and the
`nextn.*` tensors); the separate `mtp-*.gguf` in the same repository is an extra. Runner drafts from it with `--mtp`,
and greedy output is byte-identical with and without it on a 64-token reply. On the lab CPU lane it is not faster:

| Decoding | Drafts accepted | Generation |
|---|---|---|
| plain | | 26.4 tok/s |
| `--mtp --draft-k 1` | 31 of 33 (1.94 tokens a round) | 20.1 tok/s |
| `--mtp --draft-k 3` | 43 of 60 (3.20 tokens a round) | 16.5 tok/s |

The head's acceptance is high; why verifying its drafts costs more than it saves on the CPU was not measured here.

## Setup

- **Reference:** `Qwen/Qwen3.6-35B-A3B` at `995ad96eacd98c81ed38be0c5b274b04031597b0`, the publisher's checkpoint and
  the anchor (bartowski's files are quantized from it). Loaded by transformers 5.16.1 (`Qwen3_5MoeForCausalLM`, the
  text model of the multimodal checkpoint, checked beforehand to load the composite checkpoint with no missing keys
  and logits identical to the multimodal class) with torch 2.14.0 on the lab CPU lane (16 cores), by
  `scripts/gold-logits.py --upcast-at-use`: the weights stay at their stored bfloat16 and each decoder block's are
  cast to float32 for its arithmetic. The run's own line: "float32 arithmetic on weights stored at torch.bfloat16,
  cast at use per decoder block (40 blocks, config num_hidden_layers 40; 43 units hooked)". The forward over 4,509
  tokens took 403 seconds; the weights were memory-mapped (peak resident 63.7 GiB, almost all file pages; the
  container's anonymous memory peaked at 5.2 GiB under its 80 GiB cap).
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. Qwen's tokenizer
  prepends nothing; every position matched by id.
- **Files:** `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` at `5c2410d7`: Q8_0 (37.8 GB, sha256 bc57de09...), Q4_K_M
  (22.3 GB, sha256 b46fedd3...). The standard Q4_K_M with the most downloads: the other large publisher of this
  model ships only a dynamic Q4_K_M. Runner at `a9eaca5f` (`model/qwen35moe`),
  `--score --score-top 20 -c 9216 -t 16 --gpu off`, each file fetched from the Hub to local disk and read there.
- **Tokenizer and template:** 0 of 721 corpus strings and 0 of 247,091 vocabulary entries differ from Qwen's
  tokenizer.json at 995ad96e; the GGUF's chat template is the repository's `chat_template.jinja` byte for byte, and
  renders 23 of 23 conformance cases identical in text and tokens (template family `qwen36`).
- **Data:** `gold-logits-q8-q4.json` (the two files, the summary and its 300 positions) and
  `gold-logits-q8-runner-vs-other-engine.json` (Runner's Q8_0 score file beside llama-server on the same
  file, the reference recomputed in the same run).
