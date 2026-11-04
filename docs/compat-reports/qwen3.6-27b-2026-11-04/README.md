# Qwen3.6-27B: the R4.29 gate row (lab, 2026-11-04)

**Unflattering first.**

- **No publisher GGUF exists.** These are ggml-org's conversions: `ggml-org/Qwen3.6-27B-GGUF` at 1132d039, made with
  llama.cpp b15ca93 from `Qwen/Qwen3.6-27B` at 6a9e13bd (the repo's `.src_sha`), `--no-mtp`.
- **The Q4_K_M fails the bar on both engines.** Runner reads mean KLD 0.129 and margin-qualified top-1 95.9%;
  llama.cpp reads 0.154 and 93.8%. The bar is 0.05 and 97%.
- **The Q4_K_M's distance is the file itself, and it grows with depth** (0.072, then 0.109, then 0.156 on Runner).
  Against Runner's own Q8_0 at the same positions it reads top-20 KL 0.101 with 89.3% top-1 agreement
  (`q4_k_m-vs-q8_0-runner.json`).
- **llama.cpp fails the bar on the Q8_0 too,** at mean KLD 0.0626.

**ggml-org's Q8_0 passes on Runner and is the row's fidelity file**, at 28.6 GB:

| File | Engine | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 (the fidelity file) | Runner | **0.0273** | 0.0004 | 5.744 | 98.00% | 99.59% |
| Q8_0 | llama.cpp | 0.0626 | 0.0009 | 5.876 | 96.33% | 98.34% |
| Q4_K_M | Runner | **0.1288** | 0.0155 | 4.923 | 89.33% | 95.85% |
| Q4_K_M | llama.cpp | 0.1539 | 0.0178 | 6.034 | 87.00% | 93.78% |

The method is the house's: 300 positions at stride 15 (8 to 4,493) of `tests/fixtures/gold-corpus-2k.txt`, every one
matched by token id, with no dropped mass on either side. Margin-qualified positions are the 241 where the reference's
top-1 leads by more than 0.5 nats.

- **Runner against llama.cpp, Q8_0:** Runner is the closer at 239 of 300 positions. Paired per-position KLD has
  Wilcoxon z 10.21, p 1.9e-24. Margin-qualified discordant positions: 3 for Runner, 0 for llama.cpp (McNemar p 0.25).
- **Runner against llama.cpp, Q4_K_M:** Runner is the closer at 180 of 300 positions, z 3.36, p 0.0008. Discordant
  positions: 6 for Runner, 1 for llama.cpp (p 0.125).
- **One position carries most of Runner's Q8_0 mean.** Position 1,238 reads 5.744 (sentence 2 below). Without it,
  Runner's Q8_0 reads 0.0082 and llama.cpp's 0.0432.
- **Depth split** (mean KLD / margin-qualified top-1; 50, 83 and 167 positions):

  | Positions | Q8_0 Runner | Q8_0 llama.cpp | Q4_K_M Runner | Q4_K_M llama.cpp |
  |---|---|---|---|---|
  | 8-749 | 0.0186 / 97.4% | 0.0329 / 97.4% | 0.0723 / 97.4% | 0.0900 / 94.9% |
  | 750-1,999 | 0.0705 / 100% (1,238's 5.74 is in here) | 0.0763 / 100% | 0.1089 / 97.0% | 0.1311 / 97.0% |
  | 2,000-4,493 | 0.0084 / 100% | 0.0647 / 97.8% | 0.1557 / 94.9% | 0.1843 / 91.9% |

- **Full-corpus PPL (Runner, 8,106 scored tokens):** Q4_K_M 56.04, Q8_0 53.03.
  - That is high next to the other rows, but it is the model on raw text, not the engine. The Q8_0's median KLD
    against the reference is 0.0004.

## The reference

- **The model:** Qwen's `Qwen3_5ForConditionalGeneration` checkpoint. gold-logits reads its language model
  (`Qwen3_5ForCausalLM`, 64 blocks, every fourth one full attention, head_dim 256).
- **The run:** transformers 5.16.1, with float32 arithmetic on the BF16 weights, cast at use per decoder block
  (`scripts/gold-logits.py --upcast-at-use`, CPU). The parent is `Qwen/Qwen3.6-27B` at 6a9e13bd6fc8.
- **The prefix:** the tokenizer adds no special prefix (Runner's `n_special` is 0).
- **The Gated DeltaNet scan was checked first,** the lesson of row 1. On a tiny random qwen35 model with
  slow-decaying heads, transformers 5.16.1's full forward matches its own step decode to 2.7e-7 at every range, across
  chunk boundaries (5.14.1 also passes).

## Why the columns differ, one numbered sentence each

1. llama.cpp's CPU dot products quantize the activations to 8 bits, while Runner's default CPU route dots float32
   activations against the weights. This is the one arithmetic difference between the columns that we know of. It fits
   Runner being the closer at 239 of 300 Q8_0 positions, and llama.cpp's Q8_0 gap opening at depth (0.0647 against
   0.0084 past position 2,000) on a model whose DeltaNet state carries any error forward. No ablation here isolates
   it.
2. Position 1,238 is a near-tie of the reference that every quantized column moves the same way, not an engine
   fault.
   - The reference reads " prefix" (margin 0.41); all four columns read " pre", for KL 4.9 to 5.9.
   - The corpus's actual next token, " [", is far from both.
   - Runner's own Q8_0 gives " pre" 0.54 and its Q4_K_M 0.9987, so the quantized weights move this position a lot.
   - No BF16-file arm was run to tell whether quantization alone moves it.
3. llama.cpp's server cuts each top-k token's text to its longest valid UTF-8 prefix (`validate_utf8`), so 181
   vocabulary ids read as a prefix where the reference's decode writes U+FFFD. gold-logits flagged 2 such ids
   (Q4_K_M) and 1 (Q8_0), every one explained in `llamacpp-spelling-*.json`. The column matches by id, so they
   change no number.
4. The column sends token ids (`--endpoint-ids`), so llama.cpp's own tokenizer is not exercised by these numbers.

## The gate row

- **Files**, each checked against the Hub's LFS digest as it streamed into memory (sha256 and sizes in
  `FILES.sha256`, which also lists the parent's files): `ggml-org/Qwen3.6-27B-GGUF` at 1132d03976d8.
  - `Qwen3.6-27B-Q4_K_M.gguf`: 65b753ea835627f7..., 19,095,766,304 bytes. ggml-org's recipe keeps `attn_*` and
    `ssm_*` at Q8_0 and puts `ffn_*` at Q4_K; Runner's route reads it as Q4_K 54%, Q8_0 40%, Q6_K 5%.
  - `Qwen3.6-27B-Q8_0.gguf`: 73f8260284708ed7..., 28,595,762,464 bytes.
- **Tokenizer:** the differential reads 0 of 247,091 vocabulary entries and 0 of 721 corpus segments
  (`difftok-*.log`).
- **Template:** both GGUFs' `tokenizer.chat_template` is byte-identical to the publisher's `chat_template.jinja`
  (sha256 e84f32a23fdd..., 7,764 bytes; `template-gguf.log`). Runner's `template_detect` picks its qwen36 renderer for
  it (by the template's `preserve_thinking` test).
  - The conformance gate reads 23 of 23 compared cases conformant and 23 of 23 token-identical
    (`template-conformance.log`). One case is not expressible: the template itself refuses a mid-history system
    message.
  - The gate ran `template-conformance.py --require-tokens` with one family registered for this run
    (`template-conformance-family.py`). Its oracle is Qwen3.6-27B's own template, byte-identical to the committed
    `qwen36-35b-a3b` family's. Its tokenizer is the 27B's own Q4_K_M.
- **Tool gate (Q4_K_M, CUDA on the MIG slice)** (`tool-gate.json`, `tool-gate-b-temperature0.json`):
  - A: pass, buffered and streamed.
  - B: pass, both, and also at temperature 0.
  - C: buffered pass; streamed fail. The streamed run spent its whole 400-token budget in reasoning at temperature
    1.0 (the qwen36 preset; finish_reason `length`, no content, no call). That is the gate's budget, not the parser,
    the same as row 2's C.
- **Tier smoke:** both files pinned and recorded at 7567c94, each by the file's sha256.
  - Q4_K_M on CUDA (`tier-smoke-q4_k_m-cuda-*.json`), 6.0 s.
  - Q8_0 on the CPU backend of the same machine (`tier-smoke-q8_0-cpu-*.json`; `--gpu off`, the spec the other linux
    CPU pins use), 15.0 s.
  - Both wrote the same 231 bytes, 48 tokens (sha256 877a19216b6da899...), and the fact was found in both.
- **Adapters:** `--info adapter` reports serving supported (CUDA device kernels) and training supported (host CPU)
  (`info-adapter.json`).
- **Roster and manifest rows as recorded, one row per file:**
  - `qwen3.6-27b-q4_k_m`: `model-tiers-row-q4_k_m.json`, `tier-smoke-manifest-row-q4_k_m.json`.
  - `qwen3.6-27b-q8_0` (the fidelity file): `model-tiers-row-q8_0.json`, `tier-smoke-manifest-row-q8_0.json`.
  - The tier and the row texts are the coordinator's call (Tier 2; the Q8_0 named as the fidelity file; both rows say
    the GGUF is ggml-org's conversion, llama.cpp b15ca93, since no publisher file exists).

## Setup

- **Runner:** main 7567c94 (1.12.0), built with clang.
  - The score files ran on the CPU (x86-64 with AVX-512, 40 threads) on the chunked route:
    `RUNNER_SCORE_CHUNKED=1 runner --score 20 -c 9216 -t 40 --gpu off` (507 forwards of at most 16 rows).
  - The tool gate and the Q4_K_M smoke ran on CUDA (MIG slice, 1g.24gb, sm_120).
- **llama.cpp:** 73a43d1, `llama-server` (CPU build, flash attention off, `-c 9216 -t 40 -tb 40`), with the GGUF held in
  memory.
- **Reference:** transformers 5.16.1, torch 2.14.0 (CPU), float32 at use.
  - The 55.6 GB parent was held in memory for every pass.
  - `goldref-ownmem.py` clones the reference's weights into process memory after the load, which changes no value.
  - Peak resident set: 111 GB.

## Not measured

- The Q8_0 on CUDA. At 28.6 GB it is larger than the slice's 24 GB; its cell is the CPU.
- Vision (the `mmproj` files), and the MTP and dflash heads.
- Metal (the Apple Silicon laptop's).
- The `RUNNER_CUDA_ATTN_TC_WIDE` route (#574) on this head_dim-256 model; the row ran the default.
- Context beyond the gold window's 4,509 tokens.
- Task quality.
