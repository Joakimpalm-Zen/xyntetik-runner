# Inkling 2 Mini: the admission record (R4.48.6, 2026-11-22)

**Text path only.** This page is the evidence for two `inkling2_loop` files: unsloth's Mainline_Support
`Inkling-2-Mini-UD-Q8_K_XL` (the publisher's NVFP4 and MXFP8 weights, bit for bit, in llama.cpp's layout) and
`Inkling-2-Mini-UD-Q4_K_XL` (the same NVFP4 experts, Q8_0 elsewhere). Measured on the CPU route of a Linux x86_64 host
with CUDA, which is borrowed hardware (docs/device-evidence.json, `linux-x86_64-cuda`).

**Limits first.**

- CPU only. Inkling 2 Mini runs on its own engine (`src/inkling2.c`), a serial float32 order with no GPU path; the
  architecture row says so and a GPU request runs on the CPU. CUDA and Metal are not run; the roster declares
  them as limits.
- Speed, 16 threads on this host: about 0.34 s a token to prefill and 1.1 s a token to decode. RAM at an 8K context
  (`--info fit`): 32.2 GiB for the exact file, 27.3 GiB for Q4_K_XL (the mapped file, the dense tensors decoded to
  float32 once, the cache, a 4 GiB decoded-expert cache).
- No speculative decoding, shared prefix cache or LoRA on this architecture; a rewind re-prefills from the start.
  `--info adapter` refuses both directions with a sentence.
- The vision tower is not loaded (llama.cpp keeps it in a separate mmproj file).
- The positions scored: 1991 per corpus (the reference rows hold 2000 ids; the house's other admissions read 2000).

## The files

| file | repository and revision | sha256 | bytes |
|---|---|---|---|
| `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` | `unsloth/Inkling-2-Mini-GGUF` at `45fd08764f2b0bf16b03a5483a80e5c7aaec8772` | `2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68` | 22867452896 |
| `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` | the same | `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3` | 17629760736 |

Upstream: `thinkingmachines/Inkling-2-Mini` at `b85c3cfa615fc413cc9af3379f491b963f3b586a`,
`model-00001-of-00001.safetensors` sha256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`.

What the exact file holds (R4.48.5 gate (a), every tensor against the checkpoint, by range reads and local files):
the 40 NVFP4 expert tensors of layers 0 to 19 at code level (the publisher's scale bytes verbatim, the codes
repacked to ggml's nibble order, the w13 rows de-interleaved gate first), their 40 per-expert `.scale` factors equal
to amax / denominator in float32, the MXFP8 layers 20 to 23 as BF16 equal to E4M3 times 2^(s - 127), and every
other tensor under llama.cpp's converter's mapping and transforms: zero differing elements in 744 tensors. Q4_K_XL
keeps the same 40 NVFP4 expert tensors and their factors; its 146 dense tensors and the experts of layers 20 to 23
are Q8_0.

## The anchor

The publisher's implementation in float32, computed alone in one process: transformers pull request 51050 at
`6173a2f679ff4ed393afc499aefc569da7e68bc1` (the merge, `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`, holds the same
model code), torch 2.14.0+cpu, eager attention, grouped_mm experts, through the R4.48 dequantizing store (each
parameter kept in a storage form and upcast to the float32 the loader would give at use; peak 57.7 GB resident).
transformers at the merge does not load this checkpoint in float32 within the box: it skips the checkpoint's
`modelopt` quantizer and would materialize about 105 GB in float32, and `--upcast-at-use` would round the
dequantized experts to BF16. The rows (`ref-chat` sha256 `f6861dc1c4b43ef7b98f2e54e59f30fcbbc02a4163a65aebea5c65e3d8c28908`,
`ref-raw` `3bce48da8d1c760ade703e1c7b258f6e92806186cbda99eada0c0611fd903457`) were restored on this box bit for
bit equal to the 2026-11-21 record.

Corpora: raw, `tests/fixtures/gold-corpus-2k.txt` (sha256 `56e9fcba...`); chat, the R4.48 read's 2000 chat ids
through the publisher's template, written as text (`chat-corpus.txt` sha256 `98c78169...`, which tokenizes back to
those ids through the publisher's tokenizer, transformers' and Runner's).

## Numerical: scripts/gold-logits.py, the exact file

`--gold-rows`/`--gold-ids` (the reference rows above), 1991 positions, stride 1, top 20, token ids matched by id at
every position, no gold mass dropped. Side A is Runner's `--score` (the chunked prefill route); side B is llama.cpp
56eea9e4's CPU rows on the same ids (`logits_dump`: one sequence, every row, 16 threads, flash attention auto, F16 KV).

| corpus | engine | mean KL | max KL | top-1 | margin-qualified top-1 |
|---|---|---|---|---|---|
| raw | **Runner** | **0.001259** | 0.0637 | 99.40% | 100.00% (1508/1508) |
| raw | llama.cpp | 0.007811 | 0.311 | 96.89% | 99.67% (1503/1508) |
| chat | **Runner** | **1.28e-06** | 0.00084 | 100.00% | 100.00% (1761/1761) |
| chat | llama.cpp | 0.003746 | 1.134 | 98.64% | 99.94% (1760/1761) |

Paired over the positions: KL Wilcoxon z 34.74 (raw) and 38.64 (chat), Runner closer. Summaries:
`gold-raw.summary.json`, `gold-chat.summary.json`.

Beside it, the full-row read (KL over all 200058 ids, 2000 positions, the R4.48 method): Runner 5.44e-07 (chat) and
5.71e-04 (raw); llama.cpp 3.93e-03 and 4.80e-03 (`llamacpp-q8_k_xl-read-*.full.json`). The reference against
itself across one instruction-set step (AVX2 against AVX512, raw): 5.31e-04. Why llama.cpp's column is wider:
its CPU NVFP4 product quantizes the expert inputs to Q8_0 (0.59% per product, measured at op level in R4.48.5),
which the loop amplifies.

## The engine on the files: what Runner runs is the step-1 forward

- **Gate (i).** The forward on the GGUF equals the step-1 forward on the publisher's safetensors bit for bit: the
  2000-id chat and raw reads' rows hash to the recorded `cbab5e99...` and `f6a913d6...`. Every tensor and expert the
  GGUF source serves equals the safetensors provider's (`test-inkling2-gguf.txt`: 645 tensors and 6144 experts, 0
  differing elements; three wrong decodes fail it). An NVFP4 expert is `(E2M1 x UE4M3) x factor`, two float32
  multiplies in the reference loader's order, not the factor after the product.
- **The runner.** `runner --score` on the exact file gives the read harness's rows on the same ids, byte for byte; the
  sliding applications' K/V held for the window only gives the same rows (a control that drops one admitted key
  changes exactly the rows it should); any prompt split gives the same rows.
- **Gate (ii).** First light 64 of 64 against the reference; the reads' top-1 as step 1 (0 and 12 differences).
- **Q4_K_XL.** Its Q8_0 blocks decode through Runner's `dequant_row`, bit-equal to ggml's own dequantizer on 643074
  sampled blocks and 20000 random ones (`dequant-gate-q4_k_xl.txt`). Its rows are the reference order on dequantized
  weights:

| corpus | engine | mean KL (full rows) | top-1 | margin-qualified |
|---|---|---|---|---|
| first light | Runner | 2.17e-03 | 64/64 | |
| first light | llama.cpp | 6.92e-03 | 63/64 | |
| chat | Runner | 4.19e-03 [1.74e-03, 8.60e-03] | 1969/2000 | 1768/1769 |
| chat | llama.cpp | 4.18e-03 [2.72e-03, 5.97e-03] | 1968/2000 | 1768/1769 |
| raw | Runner | 5.49e-03 [4.36e-03, 7.09e-03] | 1936/2000 | 1512/1514 |
| raw | llama.cpp | 7.27e-03 [5.82e-03, 9.12e-03] | 1937/2000 | 1511/1514 |

  On a quantized file the file's own loss dominates; Runner's large advantage is on the exact file.
- **Gate (iii).** `tests/test_inkling2_hardening.py`: a tiny complete file loads and answers; five damaged copies are
  each refused with a sentence. `tests/fuzz/fuzz_inkling_open.c` fuzzes the header and tensor-table checks.

## Protocol

The tokenizer, chat renderer, tool protocol and reasoning transitions are R4.48.13's (its own record). On this
file: Runner's tokenizer gives the publisher's ids on the whole of gold-corpus-2k (8543 ids, equal), and
`--info tools` answers `{"tool_family":"inkling2","native_tool_protocol":true,"template":"inkling2"}` with no
fallback.

**The tool protocol on the real weights.** `scripts/tool-protocol-check.py` against `runner --serve` on the CPU,
the shipped generic preset, `-c 4096`, 400 tokens a turn, 2026-11-23: on both files cases A (one call), B (prose
and parallel calls) and C (prose and a call) pass buffered and streamed, `finish_reason` `tool_calls`, arguments
valid for the schema, no tool framing in `content`
(`docs/cross-family-remedy-evidence/tool-protocol-inkling2-mini-ud-q8_k_xl-lab-slice.json` and `-q4_k_xl-`). Both
rows list `tools` on this evidence. Those runs were served on one thread (the defect fixed in this change, below);
the rows do not depend on the thread count, so the verdicts stand.

**Thinking.** This model's template has no `enable_thinking`; its control is `reasoning_effort` (default 0.9, and
`"none"` is 0), and at `"none"` the model still reasons before it answers. Measured on the exact file with the
smoke's prompt at temperature 0: the default request and `reasoning_effort` `"none"` at 48 tokens, and `"none"` at
160, each ended at the limit with no content (reasoning 204, 192 and 701 bytes). llama.cpp maps `enable_thinking`
false to `reasoning_effort` `"none"` for this model (its pull request 32763, merged 2026-11-21, in the pinned
56eea9e4); Runner does not yet (R4.48.16). The smoke therefore sends both fields and gives the turn 600 tokens.

**Threads.** The rows do not depend on the thread count, measured with the pool's own record (`ink2_par_stats`:
the loops that really split over more than one thread, and the threads the last one ran on), on the exact file:

| path | count | loops split | last pool | rows sha256 |
|---|---|---|---|---|
| read harness, 256 chat ids, chunks of 64 | 16 | 360896 | 16 | `6863122f22cc8d43...` |
| the same | 4 | 360896 | 4 | the same |
| the same | 1 | 0 | 1 | the same |
| `runner --serve`, one chat turn's 30 rows | 2 | 42414 | 2 | `c091fc871d19caf3...` |
| the same, the default count | 13 | 42414 | 13 | the same |

A served slot is loaded with a one-thread pool and the server swaps the shared pool in afterwards; the engine now
takes the count at use, so a served forward runs on the server's threads (before this change, on one).
`tests/test_inkling2_threads.c` holds it on the tiny file: a swapped pool runs at its size with one thread's bits,
two slots at once (pools of 4 and 2) give the same bits, and under ThreadSanitizer the engine's pool is clean.

## Repeatable: the validation smoke

`scripts/tier-smoke.py` on `linux-x86_64-cuda`, the CPU cell: the smoke's prompt (the eight planets, one per line,
a sentence each), temperature 0, the rows' request `{"enable_thinking": false, "reasoning_effort": "none"}`,
`n_predict` 600, `ctx` 1024 (`tests/compatibility/tier-smoke.json`). The pins, from a clean build of 00da8540:

| row | bytes | finish | tokens | names Mercury | pin |
|---|---|---|---|---|---|
| `inkling-2-mini-ud-q8_k_xl` | 791 | stop | 490 | yes | `45c2a9751c8cce53...` |
| `inkling-2-mini-ud-q4_k_xl` | 799 | stop | 491 | yes | `c713db472dde292f...` |

The records, from a clean build of 20f2eb87 (the served thread count fixed, 13 threads): both replies equal their
pins byte for byte and name Mercury (`inkling-2-mini-ud-q8_k_xl` in 620.63 s, `inkling-2-mini-ud-q4_k_xl` in
623.27 s, 2026-11-23), written into `docs/model-tiers.json` by `--record`. `--info adapter` on both files: training and adapter serving are not supported, with the sentences
"the LoRA backward does not admit inkling2_loop yet: its engine (src/inkling2.c) is a forward only" and "--lora
does not cover inkling2_loop yet: its engine (src/inkling2.c) has no adapter hooks".

## Commands

```sh
# the reference rows (the R4.48 dequantizing store, one process, 4 threads)
ref_firstlight.py read real CKPT_DIR --map inkling2-map.txt --mode patched --ids IDS --n 2000 --threads 4 --out ref-C
# Runner's side
RUNNER_SCORE_CHUNKED=1 runner -m Inkling-2-Mini-UD-Q8_K_XL.gguf_file --score --score-top 20 -f CORPUS \
    --score-tokens 2000 -c 4096 -b 64 -t 16 --gpu off --score-out score-C.json
# llama.cpp's side, its own rows on the same ids
logits_dump Inkling-2-Mini-UD-Q8_K_XL.gguf IDS llama-C.logits.f32 16
# the scorer
python scripts/gold-logits.py --hf CKPT_DIR --corpus CORPUS --gold-rows ref-C.logits.f32 --gold-ids IDS \
    --score-a score-C.json --rows-b llama-C.logits.f32 --max-positions 1991 --stride 1 --top-n 20
```

The per-position records (`gold-*.json`, `score-*.json`) stay outside the repository; their sha256 are in
`uncommitted-records.sha256`.
