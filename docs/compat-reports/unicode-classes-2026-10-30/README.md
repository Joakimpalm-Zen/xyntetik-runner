# Character classes and splits, 2026-10-30

`scripts/difftok.py --vocab` (runner #479) compares Runner's tokenizer with
the publisher's over every entry of the vocabulary decoded to text. After the
whole-word rule, every family still differed on some entries (Llama 3.2 on
86, LFM2.5-8B-A1B on 108, Qwen3 on 128, MiMo on 1,176), and none of them were
merge cases. Four causes, each fixed here:

1. **The character classes.** The splits are regexes over `\p{L}` (and its
   Lu, Ll, Lt, Lm, Lo), `\p{M}`, `\p{N}` (and Nd), `\p{P}`, `\p{S}` and `\s`.
   Runner read them from ranges written by hand, with every code point above
   U+007F a letter unless a listed block said otherwise: full-width
   punctuation ("："), U+FEFF, the Arabic-Indic digits, most combining marks
   and private use were letters, and case was known for Latin, Greek and
   Cyrillic only. `src/unicode_classes.h` is now generated
   (`scripts/gen-unicode-classes.py`) from python's Unicode 16.0.0 database,
   checked code point by code point from U+0080 against tokenizers 0.23.3's
   own regex engine; the two agree on every one of the eleven classes.
2. **The case alternatives** (Tekken, Llama 4, gpt-oss): in
   `[U]*[L]+` the greedy `U*` gives characters back until an `L` can follow,
   so " 亚洲AV" is " 亚洲" then "AV" (two caseless letters, which are in both
   classes, then two capitals). Runner took the whole run as the second
   alternative. The alternatives are now tried in the regex's order, each
   with the optional leading character and then without it.
3. **afmoe's and spark2_5's punctuation and digits** (Trinity, Spark): the
   lead of a letter run is `[^\r\n\p{L}\p{P}\p{S}]`, so a format character
   (U+FEFF, U+00AD, ZWNJ, U+180E) leads one; Runner read every non-letter as
   punctuation. Trinity's digit passes group `\p{Nd}` by three, so "1²" is
   "1" then "²". Text no alternative matches (a run of U+200B, of control
   characters) is one piece, as the Isolated split keeps each gap; Runner
   cut it per character. Trinity's script runs are one per group of its
   alternation, Thai as the whole block U+0E00..U+0E7F; Runner had
   U+0E40..U+0E7F and one group for all scripts.
4. **Long merges.** A merge key past 512 bytes was skipped, so no token
   longer than about 255 bytes in byte-level form was ever built: MiniCPM5's
   run of 160 spaces came out as two 128-space tokens. Such keys are now
   built on the heap.

## Before and after

`scripts/difftok.py` against each publisher's tokenizer.json (references
and header-only GGUF sources as in
[the whole-word rule](../bpe-ignore-merges-2026-10-30/README.md), plus
Qwen/Qwen3-4B-GGUF, unsloth's Qwen3.8-27B and Qwen3-Coder-30B, bartowski's
MiMo-V2.6, ornith-ai's Ornith-1.0-9B, arcee-ai's Trinity-Nano-Preview Q8_0,
JetBrains' Mellum2.1, RecursiveSI's flint and ibm-granite's granite-4.2-8b
GGUFs). Cells are strings that differ, before -> after; one number means
unchanged. The vocabulary column was taken before runner #480, so each
vocabulary's U+0000 entry is not in it; with #480 it is compared too and
matches (Llama 3.2 0 of 126,639, the LFM2.5 rows 0 of 63,098 and 123,332).

| File | 721 corpus | gold (107) | NFC (13,264) | every vocabulary entry |
|---|---|---|---|---|
| Llama-3.2-3B (llama-bpe) | 0 | 0 | 0 | 86 -> 0 of 126,638 |
| LFM2.5-8B-A1B (lfm2) | 0 | 0 | 2 -> 0 | 108 -> 0 of 123,331 |
| LFM2.5-1.2B (lfm2) | 0 | 0 | 2 -> 0 | 11 -> 0 of 63,097 |
| LFM2.5-350M (lfm2) | 0 | 0 | 2 -> 0 | 11 -> 0 of 63,097 |
| Nemotron-3.5-Lightning (pixtral) | 0 | 0 | 0 | 202 -> 0 of 128,634 |
| Mistral-Nemo (tekken) | 0 | 0 | 0 | 202 -> 0 of 128,634 |
| Apertus-8B (tekken) | 0 | 0 | 0 | 202 -> 0 of 128,634 |
| gpt-oss-20b (gpt-4o) | 0 | 0 | 0 | 106 -> 0 of 198,412 |
| Muse-Glimmer-30B (llama4) | 0 | 0 | 0 | 47 -> 0 of 198,160 |
| Llama-4-Scout (llama4) | 0 | 0 | 0 | 47 -> 0 of 198,160 |
| granite-4.1-3b (dbrx) | 0 | 0 | 0 | 37 -> 0 of 99,473 |
| granite-4.2-8b (granite-docling) | 0 | 2 -> 0 | 0 | 31 -> 0 of 99,473 |
| Qwen3-4B (qwen2) | 0 | 0 | 0 | 128 -> 0 of 150,185 |
| Qwen3-Coder-30B (qwen2) | 0 | 0 | 0 | 128 -> 0 of 150,185 |
| Qwen3.8-27B (qwen35) | 0 | 0 | 0 | 41 -> 0 of 247,090 |
| MiMo-V2.6-9B (qwen35 file, qwen2 split) | 0 | 0 | 0 | 1176 -> 0 of 247,090 |
| Ornith-1.0-9B (qwen35) | 3 | 0 | 1 | 5544 -> 5503 of 247,090 |
| Trinity-Nano (afmoe) | 0 | 0 | 0 | 143 -> 0 of 198,753 |
| Mellum2.1 (mellum2) | 0 | 0 | 2 -> 0 | 29 -> 0 of 97,298 |
| flint (flint) | 0 | 0 | 0 | 53 -> 0 of 148,839 |
| MiniCPM5-2B (minicpm5) | 0 | 0 | 2 -> 0 | 80 -> 0 of 129,037 |
| Spark-X2.6-4B (spark2_5) | 0 | 0 | 0 | 144 -> 0 of 129,029 |
| K2-Horizon-1B (k2-horizon) | 0 | 0 | 0 | 16 -> 0 of 63,019 |
| SmolLM2-135M (smollm) | 0 | 0 | 0 | 0 of 48,820 |

Ornith-1.0-9B's remaining strings are its own, already declared on its row
(3 of 721): its tokenizer.json carries the qwen2 split (`\p{L}+`), while its
GGUF declares qwen35, which Runner follows. Against Qwen/Qwen3.8-27B's
tokenizer.json (the qwen35 split over the same vocabulary) the Ornith file
differs on none of its 247,090 entries.

The rows that declare `expect_vocab_divergences` (LFM2.5-1.2B and 350M,
both files each) now declare 0. Tokenizing 2.2 million characters of mixed
text (the gold corpus, the NFC corpus, random CJK) takes the same time
before and after on Llama 3.2, Qwen3 and Mistral-Nemo vocabularies (0.24
to 0.38 s per run, the table lookup a binary search over 2,822 ranges).

## Reproduce

```
python3 scripts/gen-unicode-classes.py > src/unicode_classes.h   # python 3.14 (Unicode 16.0.0)
make difftok
python3 scripts/difftok.py --gguf models/Qwen3-4B-Q4_K_M.gguf --ref Qwen/Qwen3-4B --vocab
```

