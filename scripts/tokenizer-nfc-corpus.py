#!/usr/bin/env python3
"""Write tests/fixtures/tokenizer-nfc-corpus.txt: strings whose NFC differs
from their bytes, for the normalizer the Qwen tokenizers declare.

    scripts/tokenizer-nfc-corpus.py > tests/fixtures/tokenizer-nfc-corpus.txt

One JSON string per line (the difftok corpus format): every code point with
a canonical decomposition, written decomposed inside a short context, then
hand-picked sequences (marks out of canonical order, conjoining Hangul jamo,
composition exclusions, singletons, Indic, Thai, Hebrew and Vietnamese text).
Run it through scripts/difftok.py against a Qwen tokenizer.json.
"""
import json
import sys
import unicodedata

out = []
for cp in range(0x110000):
    if 0xD800 <= cp <= 0xDFFF:
        continue
    c = chr(cp)
    d = unicodedata.normalize("NFD", c)
    if d != c:
        out.append(f"x {d}y {c}")
out += [
    "ą́ ą́ q̣̇ q̣̇",          # canonical order
    "각 한글",          # conjoining jamo
    "क़ क़ ড় ଡ଼",                    # composition exclusions
    "Å Ω K 豈 ̀ ̈́",                         # singletons
    "हिन्दी ภาษาไทย",
    "שָׁלוֹם שָׁלוֹם",
    "Việt Nam tiếng Việt",
    "ୋ ொ ೀ ൊ",              # starter + starter pairs
    "́́a é́́ ̀",                          # leading and stacked marks
    "café café CAFÉ naı̈ve",
    "\U0001d15e \U0001d160 \U0001f600️ \U0001f469‍\U0001f4bb",
]
for s in out:
    sys.stdout.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"{len(out)} strings", file=sys.stderr)
