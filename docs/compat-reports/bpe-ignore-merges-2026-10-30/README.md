# BPE whole-word rule (`ignore_merges`), 2026-10-30

The LFM2.5-8B-A1B float32 reference refused to compare at token 94 of the
gold corpus: Runner read " läsa" as " l" + "äs" + "a", where the publisher's
tokenizer reads one token. The publisher's tokenizer.json sets
`"ignore_merges": true`: a pre-token that is itself a vocabulary entry is
that token, whatever the merges would build (tiktoken's rule). Runner's BPE
always merged, and the merges do not always reach the whole word. The GGUF
carries no key for the flag, so Runner now reads it from the split's name
(`tokenizer.ggml.pre`), each family taken from its publisher's file.

## Which publishers set it

Every plain-BPE tokenizer.json behind a row, and the pre name its GGUF
writes:

| Family (reference) | GGUF pre | `ignore_merges` |
|---|---|---|
| Llama 3.2 (unsloth/Llama-3.2-3B-Instruct at 006f5dcd, tokenizer.json sha256 6b9e4e7f...; the meta-llama repository is gated here) | llama-bpe | true |
| Llama 4 (unsloth/Llama-4-Scout-17B-16E-Instruct at afd8e498), Muse-Glimmer-30B (meta-models at cec94fcf) | llama4 | true |
| gpt-oss-20b (openai at 6cee5e81) | gpt-4o | true |
| Mistral-Nemo (mistralai at 04d8a905), Apertus-8B (swiss-ai at b946d404) | tekken | true |
| Nemotron 3.5 Lightning (nvidia BF16 at a9904d24) | pixtral | true |
| LFM2.5-8B-A1B (LiquidAI at 5dd22602; 24B-A2B's tokenizer.json is byte-identical) | lfm2 | true |
| LFM2.5-1.2B-Instruct (0f604ada), LFM2.5-350M (9e6c6ccf) | lfm2 | false |
| granite 4.1 and 4.2, Qwen 2.5/3/3.8, MiMo, Ornith, MiniCPM5, Spark, K2-Horizon, Mellum, Trinity, flint, SmolLM2 | their own | false or absent |

Nemotron 3.5 Bolt E4B is a `gemma4` vocabulary, which no split reads.

"lfm2" names both LFM2.5 vocabularies, and the rule is on for the name. On
the 65,536-token vocabulary that changes nothing: over its 64,400 model
entries, the only ones its merges do not build whole are its 507 special
and reserved tokens (`<|im_start|>`, `<|reserved_N|>`), and no split yields
those as a pre-token. Counted with the publisher's own `tokenizers` model
(`model.tokenize(entry)` with the flag off), the entries the merges do not
reach whole are: LFM2.5-1.2B 507 (all special), LFM2.5-8B-A1B 2,464,
Llama 3.2 588, the Tekken vocabularies 1,000 (all their `<SPECIAL_N>`
placeholders), gpt-oss, Llama 4 and Muse-Glimmer 0.

## Before and after

`scripts/difftok.py` against each reference above: the 721-string corpus,
the gold corpus (`tests/fixtures/gold-corpus-2k.txt`, whole and as its 106
paragraphs) and, new with this change, `--vocab`: every model-vocabulary
entry decoded to text. Header-only GGUFs (the bytes up to the end of the
tensor directory) were read for the files not held here: ggml-org's
Lightning Q4_0 (8cd08217) and gpt-oss MXFP4 (15dcfc7e), meta-models'
Muse-Glimmer (b44f1072), bartowski's Mistral-Nemo Q4_K_M (a2dd64a0),
unsloth's Apertus Q4_K_M (61c32783) and Llama-4-Scout Q4_K_M (72a6853f).

| File | 721 before | 721 after | gold before | gold after | vocab before | vocab after |
|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct Q4_K_M | 1 | 0 | 0 | 0 | 674 / 126,638 | 86 |
| LFM2.5-8B-A1B Q4_K_M | 0 | 0 | 7 / 107 | 0 | 2,567 / 123,331 | 108 |
| Nemotron 3.5 Lightning | 0 | 0 | 0 | 0 | 202 / 128,634 | 202 |
| Mistral-Nemo | 0 | 0 | 0 | 0 | 202 / 128,634 | 202 |
| Apertus-8B | 0 | 0 | 0 | 0 | 202 / 128,634 | 202 |
| gpt-oss-20b | 0 | 0 | 0 | 0 | 106 / 198,412 | 106 |
| Muse-Glimmer-30B | 0 | 0 | 0 | 0 | 47 / 198,160 | 47 |
| Llama-4-Scout | 0 | 0 | 0 | 0 | 47 / 198,160 | 47 |
| LFM2.5-1.2B-Instruct Q8_0 and Q4_K_M | 0 | 0 | 0 | 0 | 11 / 63,097 | 11 |
| LFM2.5-350M Q8_0 and Q4_K_M | 0 | 0 | 0 | 0 | 11 / 63,097 | 11 |
| granite-4.1-3b Q8_0 | 0 | 0 | 0 | 0 | 37 / 99,473 | 37 |
| MiniCPM5-2B Q8_0 | 0 | 0 | 0 | 0 | 80 / 129,037 | 80 |
| Spark-X2.6-4B Q4_K_M | 0 | 0 | 0 | 0 | 144 / 129,029 | 144 |
| K2-Horizon-1B Q8_0 | 0 | 0 | 0 | 0 | 16 / 63,019 | 16 |
| SmolLM2-135M Q8_0 | 0 | 0 | 0 | 0 | 0 / 48,820 | 0 |

Every string the rule changes now matches the publisher. The ones still
different are not merge cases (the rule changes none of them): they come
from the split's character classes, which read full-width punctuation and
U+FEFF as letters ("：“" and "）\n" are one pre-token to the publishers, two
to Runner). That is a separate change. The three distinct tier-smoke
prompts tokenize the same before and after on every file above.

The LFM2.5-1.2B and 350M rows now declare `expect_vocab_divergences: 11`,
so the compatibility matrix gates both directions on the real files: the
8B's row (on its admission branch) declares its own count.

## Reproduce

```
make difftok
python3 scripts/difftok.py --gguf models/LFM2.5-8B-A1B-Q4_K_M.gguf \
    --ref LiquidAI/LFM2.5-8B-A1B --ref-revision 5dd22602c2e9f6a097b1de4c4efe0658b605015c --vocab
```
