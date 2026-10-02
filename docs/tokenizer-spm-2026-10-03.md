# The four SentencePiece "divergences", read against the publishers' own models

Measured 2026-10-03 on the Apple M1, runner `decided-batch`, over the 721
strings of `tests/fixtures/tokenizer-corpus.txt`. The compatibility program
listed four SentencePiece families as diverging from their references:
Mistral 7B v0.3 (44 strings), Phi-3.5 mini (2), Salamandra 7B (16) and Lucie
7B (190 to 259). Those references are the publishers' Hugging Face
`tokenizer.json` files, read by the `tokenizers` library. Three of the four
publishers also ship the SentencePiece model the model was trained with, and
that is the one this note compares against as well.

| family | runner vs `tokenizer.json` | runner vs the SentencePiece model | `tokenizer.json` vs the SentencePiece model |
|---|---|---|---|
| Mistral 7B Instruct v0.3 (`tokenizer.model.v3`) | 44 | 7, all special-token markers | 51 |
| Phi-3.5 mini instruct (`tokenizer.model`) | 2 | 10, all special-token markers | 10 |
| Salamandra 7B instruct (`tokenizer.model`) | 16 | 7, all special-token markers | 19 |
| Lucie 7B Instruct v1.1 | 190 | no SentencePiece model published | n/a |

"Special-token markers" are strings such as `<s>`, `[INST]` or
`not a <s> real tag`: the runner reads the marker as the special token (as a
chat template needs), the SentencePiece library reads it as text. On every
other string the runner produces exactly the SentencePiece model's ids.

## What this means

- **Mistral, Phi-3.5, Salamandra: no runner defect.** The runner tokenizes as
  the model's own SentencePiece model does. The differences are between the
  publisher's two artifacts. Mistral's `tokenizer.json` uses a Metaspace
  pre-tokenizer with `prepend_scheme: "first"`, which adds the leading "▁"
  only when the text does not already begin with a space; SentencePiece adds
  it always (`" hello"` is `[▁, ▁hello]` in SentencePiece and in the runner,
  `[▁hello]` in `tokenizer.json`). Matching `tokenizer.json` here would move
  the runner AWAY from the tokenizer the model was trained with.
- **Lucie: a real gap.** Lucie publishes only `tokenizer.json`, whose
  normalizer is a fixed sequence the runner does not implement: NFC, drop
  `\r` and NUL, insert a space after a newline, a tab and eighteen opening
  brackets and quotes when a space or a word character follows, prepend a
  space, map spaces and U+00A0 to "▁". The GGUF declares `pre: default`, so
  nothing in the file identifies this normalizer; implementing it needs a
  detection rule (the model name or a vocabulary fingerprint) and NFC tables.

## What changed (owner decision, 2026-10-03)

The Mistral v0.3, Phi-3.5 and Salamandra references are now captured from
the SentencePiece model and gate at 0 of 721 (7, 10 and 7 special-marker rows
reported apart); the `tokenizer.json` captures are kept as an informational
second reference. Lucie is deferred until there is demand.

## How it was measured

`difftok` (the harness behind `scripts/difftok.py`) now opens a GGUF
header-only, so each family's vocabulary was read from the first 8 MB of a
public GGUF (ranged download): bartowski Mistral v0.3 and Phi-3.5 Q4_K_M,
cstr Salamandra Q4_K_M-f32, OpenLLM-France Lucie Q4_K_M. The `tokenizer.json`
ids are the committed captures in `tests/compatibility/tokenizer-references/`
(corpus digest checked); the SentencePiece ids come from the `sentencepiece`
library on each publisher's model file.
