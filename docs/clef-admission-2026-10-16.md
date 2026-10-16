# Clef admission, 2026-10-16

What was admitted: GGUF files with `general.architecture` `clef`
(Cloudflare/clef, Cloudflare/clef-flash, and every quantization of them),
the backbone as `qwen35` and the joint schema head, served at
`POST /v1/systemone` and `--systemone FILE`. Text input only.

## The reference

The publisher's own code, not another engine's port of it:
`joint_schema_model.py` from
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) at revision
`0b331204bb13fbd2ca93a64df1956f5b55478ce5`, sha256
`0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`
(`encode_record`, `JointSchemaHead`, `ClefModel`, `systemone`).
`scripts/clef-gold.py` refuses any other revision of the file unless told
otherwise.

Reference configuration: torch 2.14.0+cpu, transformers 5.16.1, float32,
CPU, attention `sdpa`. The host has no torchvision, which the publisher's
`load_release_model` needs for its image and video processor, so the script
runs that function's own steps with `AutoTokenizer` in the processor's place
(text-only records reach the processor for its tokenizer alone). The model,
the head and every computation are the publisher's.

llama.cpp's Clef conversion (`conversion/clef.py`, `src/models/clef.cpp` at
`2f729cfa`) was read for the GGUF tensor layout only, which is the layout
every public Clef GGUF has.

## Gold member

A small Clef: the Qwen3.5-0.8B backbone with a joint head (hidden 1024,
width 64, 2 routing blocks, 1 decoder block, 4 heads, feedforward 128,
float32 weights), converted by llama.cpp's converter to BF16
(sha256 `4017841b88d92d6fe1c893c3f23e20f113b1dc92d4ac042222b3ad455f74629a`).
The converter stores every head matrix in BF16 except the type embedding,
which it keeps in F32; `--head-bf16` gives the reference the same weights.

Records: `tests/fixtures/clef-records.jsonl`, 15 requests chosen to break an
encoder: the model card's two examples, noul criteria overrides with a null,
choice keys out of code point order and non-ASCII keys with non-string
descriptions, a state of python spelling traps (an integer past 2^63, `-0`,
`-0.0`, `1e-05`, `1e+16`, `1E3`, `2.50e+2`, control characters, U+2028, an
emoji), instructions that are an object, `0`, `false` and `""`,
special-token text inside the state, six questions of mixed types, a
3,789-token state, whitespace-only, empty, numeric and null states, question
ids with spaces and non-ASCII, and text with combining marks.

| Runner build and placement | input ids | spans | max \|dlogit\| | max \|dprob\| | answers byte-equal after rounding |
|---|---|---|---|---|---|
| arm64 CPU (macOS) | 15/15 | 15/15 | 1.4e-4 | 1.9e-5 | 27/29 |
| x86-64 CPU (Linux, 4 threads) | 15/15 | 15/15 | 1.4e-4 | 1.8e-5 | 25/29 |
| CUDA, full offload (RTX PRO 6000, MIG 1g.24gb) | 15/15 | 15/15 | 1.5e-4 | 1.7e-5 | 25/29 |

Logits are about 2 in magnitude, so 1.5e-4 is about 1e-4 relative. Every
answer that differs after rounding differs by exactly one unit in the fourth
decimal. CUDA against the CPU on the same host: max |dlogit| 3.2e-5, the
same size as arm64 CPU against x86-64 CPU.

## What the gold run found

1. **Tokenizer split for Clef.** Cloudflare's `tokenizer.json` carries the
   qwen2 regex (`\p{L}+`), and transformers builds that regex for Clef; the
   publisher's encoder splits a Devanagari word at every mark. Qwen's own
   `tokenizer.json` carries `[\p{L}\p{M}]+`. llama.cpp's converter writes
   `qwen35` into every Clef GGUF, so a `clef` file's declared `qwen35` is
   now read as the qwen2 split it was trained with. Expected divergence:
   text with combining marks tokenizes differently from llama.cpp on Clef
   files.
2. **NFC.** Qwen's tokenizers (Qwen2 through Qwen3.8) declare an NFC
   normalizer and Runner never applied it: `cafe` plus U+0301 was two tokens
   (one, ` café`, in the reference). Runner now normalizes text between
   special tokens on the qwen2 and qwen35 tokenizers, with tables generated
   from Unicode 16.0 and cross-checked code point by code point against the
   `tokenizers` package (0.23.2), following it on the 21 decompositions and
   120 combining classes it does not know yet. `difftok` over the standard
   corpus and `tests/fixtures/tokenizer-nfc-corpus.txt` (13,264 strings):

   | GGUF | reference `tokenizer.json` | standard | NFC corpus |
   |---|---|---|---|
   | small Clef (`clef`) | Cloudflare/clef 0b331204 | 0/721 | 0/13264 |
   | Qwen3.5-0.8B Q8_0 (`qwen35`) | Qwen/Qwen3.5-0.8B | 0/721 | 0/13264 |
   | Qwen3-0.6B BF16 (`qwen2`) | Qwen/Qwen3-8B | 0/721 | 0/13264 |

3. **A reference-side error, caught and fixed.** The first reference run
   rounded the type embedding to BF16 as well; the GGUF keeps it in F32.
   With the rounding matched, max |dlogit| fell from 5.0e-4 to 1.4e-4.
4. **transformers and Qwen3.5.** transformers 5.16 builds the qwen2 regex
   even for Qwen's own Qwen3.5 `tokenizer.json`. A transformers-based
   reference on a Qwen3.5 model therefore disagrees with that file, and with
   Runner, on text with combining marks; Runner follows the file.

## A real Clef

bartowski's `Cloudflare_clef-IQ3_M.gguf` (sha256
`4977cdedb8784ed414ae03e17ff0ee33f1be1d6bc89370114ebcbc6c5f818975`), the
model card's two examples, CPU: the invoice is `overdue` at 0.9934 and the
total above 1000 USD at 0.9936; the checkout message goes to `technical` at
0.9033, urgency `Today` at 0.9162 (score 1.8884), service down at 0.8711.
This shows the trained head binds and decides; it is not a fidelity
measurement of the quantized file.

## Not covered yet

- Images and video: the vision encoder (the mmproj file), the image
  processor and 3-D rotary positions for image tokens. Refused with `400`.
- A float32 gold run on a released checkpoint (Clef-Flash 9B or Clef 27B)
  against its BF16 GGUF.
- Metal: the qwen35 backbone runs on the CPU there today, as for every
  qwen35 model; the hidden-row copy for a fully offloaded Metal model is in
  place but not exercised by this family.

## Reproduce

```bash
# reference (torch, transformers)
python3 scripts/clef-gold.py reference --model CLEF_DIR --code CLEF_DIR/joint_schema_model.py \
    --records tests/fixtures/clef-records.jsonl --out ref.jsonl --head-bf16
# runner
python3 scripts/clef-gold.py requests --records tests/fixtures/clef-records.jsonl --out req.jsonl
./runner -m CLEF.gguf --systemone req.jsonl > run.jsonl
python3 scripts/clef-gold.py compare --ref ref.jsonl --runner run.jsonl
# tokenizer
python3 scripts/difftok.py --gguf CLEF.gguf --ref CLEF_DIR/tokenizer.json \
    --corpus tests/fixtures/tokenizer-nfc-corpus.txt
```
