# Granite-4.2-30B: the R4.29 gate row (lab, 2026-11-04)

**Unflattering first: IBM's Q4_K_M, the row's file, fails the bar on both engines, by far.** Against the
publisher's float32 reference, Runner reads mean KLD 0.139 and margin-qualified top-1 93.9%; llama.cpp reads 0.151 and
92.5%. The bar is 0.05 and 97%. The file itself is the distance, not an engine: Runner's Q4_K_M against Runner's own
Q8_0 at the same positions reads top-20 KL 0.111 with 83.7% top-1 agreement (`q4_k_m-vs-q8_0-runner.json`).

**IBM's Q8_0 passes on both engines and is the row's fidelity file**, at 31.1 GB:

| File | Engine | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 (the fidelity file) | Runner | **0.0041** | 0.0004 | 0.501 | 97.67% | 99.56% |
| Q8_0 | llama.cpp | 0.0241 | 0.0017 | 3.345 | 95.00% | 99.12% |
| Q4_K_M (the row's file) | Runner | **0.1393** | 0.0504 | 3.522 | 83.00% | 93.86% |
| Q4_K_M | llama.cpp | 0.1512 | 0.0562 | 5.022 | 80.67% | 92.54% |

The method is the house's: 300 positions at stride 15 (8 to 4,493) of `tests/fixtures/gold-corpus-2k.txt`, every one
matched by token id, with no dropped mass on either side. Margin-qualified positions are the 228 where the reference's
top-1 leads by more than 0.5 nats.

- **Runner against llama.cpp, Q8_0:** Runner is the closer at 270 of 300 positions. Paired per-position KLD has
  Wilcoxon z 12.31, p 8e-35. Margin-qualified discordant positions: 2 for Runner, 1 for llama.cpp (McNemar p 1.0).
- **Runner against llama.cpp, Q4_K_M:** Runner is again the closer, z 3.18, p 0.0014. Discordant positions: 5 for
  Runner, 2 for llama.cpp (p 0.45).
- **Depth split** (mean KLD / margin-qualified top-1; 50, 83 and 167 positions):

  | Positions | Q8_0 Runner | Q8_0 llama.cpp | Q4_K_M Runner | Q4_K_M llama.cpp |
  |---|---|---|---|---|
  | 8-749 | 0.0022 / 100% | 0.0140 / 96.9% | 0.1136 / 96.9% | 0.1227 / 93.8% |
  | 750-1,999 | 0.0092 / 98.5% | 0.0562 / 98.5% | 0.2048 / 90.8% | 0.2466 / 89.2% |
  | 2,000-4,493 | 0.0021 / 100% | 0.0112 / 100% | 0.1145 / 94.7% | 0.1123 / 93.9% |

  The Q4_K_M fails the bar in every range; both engines read their worst range at positions 750 to 1,999.
- **Full-corpus PPL (Runner, 8,434 scored tokens):** Q4_K_M 15.851, Q8_0 15.000.

## The reference

The reference is IBM's `GraniteForCausalLM`, run through transformers 5.16.1 with float32 arithmetic on the BF16
weights, cast at use per decoder block (`scripts/gold-logits.py --upcast-at-use`, CPU, 64 blocks). The parent is
`ibm-granite/granite-4.2-30b` at 9e668ce1c538. The tokenizer adds no special prefix (Runner's `n_special` is 0), so no
BOS is forced. Granite is dense attention, with no recurrent state and no router, so neither of row 1's reference
questions (the Mamba-2 scan, the dt floor) applies.

## Why the columns differ, one numbered sentence each

1. llama.cpp's CPU dot products quantize the activations to 8 bits (Q8_K blocks against the K-quant types, Q8_0
   blocks against Q8_0), while Runner's default CPU route dots float32 activations against the weights (`cpu_act`
   f32; the int8 routes are opt-in and were off). This is the one arithmetic difference between the columns that we
   know of. It fits Runner being the closer at 270 of 300 Q8_0 positions, but no ablation here isolates it.
2. llama.cpp's three largest Q8_0 readings (3.34 at position 1,178, 1.08 at 2,963, 0.38 at 503) fall on near-ties of
   the reference ("md" against "m" at margin 0.55), positions where both engines' Q4_K_M also read 0.86 to 3.15.
   They are fragile positions that the extra activation error tips, not a separate fault; without those three,
   llama.cpp's Q8_0 reads 0.0082.
3. llama.cpp's server cuts each top-k token's text to its longest valid UTF-8 prefix (`validate_utf8` in
   `completion_token_output::to_json`), so the 202 vocabulary ids that end in a partial character read as " ", where
   the reference's decode writes U+FFFD. gold-logits counted 9 such ids (Q4_K_M) and 8 (Q8_0), every one explained in
   `llamacpp-spelling-*.json`. The column matches by id, so they change no number.
4. The column sends token ids (`--endpoint-ids`), so llama.cpp's own tokenizer is not exercised by these numbers.

## The gate row

- **Files**, each checked against the Hub's LFS digest as it streamed into memory (sha256 and sizes in
  `FILES.sha256`, which also lists the parent's files): `ibm-granite/granite-4.2-30b-GGUF` at 27b350a791e8.
  - `granite-4.2-30b-Q4_K_M.gguf`: f299dace85d77ee0..., 17,721,455,328 bytes
  - `granite-4.2-30b-Q8_0.gguf`: 005b0933353e9ba2..., 31,111,705,312 bytes
- **Tokenizer:** the differential reads 0 of 99,474 vocabulary entries and 0 of 721 corpus segments
  (`difftok-*.log`).
- **Template:** both GGUFs' `tokenizer.chat_template` is byte-identical to the publisher's `chat_template.jinja`
  (sha256 f0ba43f79b3c..., 9,013 bytes; `template-gguf.log`).
  - Conformance runs as family `granite42` (`template-conformance.py --family granite42 --require-tokens`): 24 of 24
    cases conform, and 24 of 24 are token-identical (`template-conformance.log`).
  - That family's oracle is `ibm-granite/granite-4.2-3b`, whose `chat_template.jinja`, `tokenizer.json`
    (883975314d58...) and `tokenizer_config.json` (2007f26df166...) are byte-identical to the 30B's. Their sha256
    values are listed in `FILES.sha256`.
- **Tool gate (Q4_K_M, CUDA on the MIG slice)** (`tool-gate.json`, `tool-gate-b-temperature0.json`):
  - A: pass, buffered and streamed.
  - B: pass, both, and also at temperature 0.
  - C: buffered pass; streamed fail. The streamed run spent its whole 400-token budget in reasoning at temperature
    1.0 (finish_reason `length`, no content, no call), so the gate's budget ran out before any call was parsed. The
    parser was never reached.
- **Tier smoke:** both files pinned and recorded at 8b2729f, each by the file's sha256.
  - Q4_K_M on CUDA (`tier-smoke-cuda-*.json`): pin file `granite-4.2-30b.linux-x86_64-cuda.cuda.txt`, sha256
    eeb1b390d202e1b0..., 149 bytes, 48 tokens. The fact was found, and the record matched it (5.73 s).
  - Q8_0 on the CPU backend of the same machine (`tier-smoke-q8_0-cpu-*.json`; it serves with `--gpu off`, the spec
    the other linux CPU pins use: `-n 48`, ctx 512, `-b 64`). Pin file
    `granite-4.2-30b-q8_0.linux-x86_64-cuda.cpu.txt`, sha256 f9609a2de5001...; the reply lists the eight planets in
    order. The fact was found, and the record matched it (17.3 s).
- **Adapters:** `--info adapter` reports serving supported (CUDA device kernels) and training supported (host CPU)
  (`info-adapter.json`).
- **Roster and manifest rows as recorded, one row per file:**
  - `granite-4.2-30b` (the Q4_K_M, below the bar; its text points at the Q8_0 row): `model-tiers-row.json`,
    `tier-smoke-manifest-row.json`.
  - `granite-4.2-30b-q8_0` (the fidelity file): `model-tiers-row-q8_0.json`, `tier-smoke-manifest-row-q8_0.json`.
  - The tier and the row texts are the coordinator's call (Tier 2; the Q8_0 named as the fidelity file).

## Setup

- **Runner:** 8b2729f (1.12.0), built with clang.
  - The score files ran on the CPU (x86-64 with AVX-512, 40 threads) on the chunked route:
    `RUNNER_SCORE_CHUNKED=1 runner --score 20 -c 9216 -t 40 --gpu off` (528 forwards of at most 16 rows).
  - The tool gate and the tier smoke ran on CUDA (MIG slice, 1g.24gb, sm_120).
- **llama.cpp:** 73a43d1, `llama-server` (CPU build, flash attention off, `-c 9216 -t 40 -tb 40`), with the GGUF held in
  memory.
- **Reference:** transformers 5.16.1, torch 2.14.0 (CPU), float32 at use.
  - The parent was held in memory for every pass, because this lab's disk rereads at 6-11 MB/s and its page cache is
    reclaimed from outside.
  - `goldref-ownmem.py` clones the reference's weights into process memory after the load, which changes no value.

## Not measured

- The Q8_0 on CUDA. At 31.1 GB it is larger than the slice's 24 GB, so its cell would be a partial offload; the tool
  gate and the smoke ran on the Q4_K_M.
- Metal (the Apple Silicon laptop's).
- Context beyond the gold window's 4,509 tokens.
- Task quality.
