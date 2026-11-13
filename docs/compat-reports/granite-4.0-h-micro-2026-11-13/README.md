# Granite 4.0 H Micro: IBM's and unsloth's files against IBM's float32 reference (2026-11-13)

Granite 4.0 H Micro is IBM's 3B chat model on the granitehybrid architecture: 40 layers, 36 of them Mamba-2 and 4
attention layers without positional encoding, no experts, hidden 2,048, a tied output head, vocabulary 100,352. The
Granite 4 h-series had no file with a cell, only its architecture placeholder. Two rows join: IBM's own Q8_0 and
unsloth's Q4_K_M.

**What is not flattering, first:**
- **IBM's own Q4_K_M, the file most people fetch, is not admitted.** It reads 0.0854 ± 0.0048 mean KLD (bootstrap 95%
  0.0763 to 0.0953, wholly over 0.05) and 94.85% margin-qualified top-1, failing both bars. The loss is the file's,
  not Runner's: on the first window, llama.cpp 73a43d1 reads the same file at 0.0888 and 94.97%, against Runner's
  0.0805 and 95.48%. Both rows say so first in their limitations. The file keeps its existing compat entry (load
  and chat), whose notes now carry these figures.
- **unsloth's Q4_K_M passes on its point estimate only.** It reads 0.0490 ± 0.0048 and 98.28% margin-qualified. Its
  bootstrap 95% interval, 0.0407 to 0.0596, crosses the bar of 0.05, and its second window alone reads 0.0553. It was
  the one further file read, and it was declared before the read: a row only if both point estimates pass, and no
  third file whatever the result. The Q8_0 is the fidelity file.
- **No training and no adapters.** `--adapt-info` refuses both for Mamba-2 hybrids, so the support page marks Train n/a
  and there is no LoRA smoke.
- **No Metal cell yet.**
- **Scope:** two windows over one corpus, CPU score files, all inside 4,096 tokens. Task quality is not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), for each file on its own.

## The files and the reference

- **Reference:** `ibm-granite/granite-4.0-h-micro` at `d5f01a3ea75f088947be3aae039f4ad52837dfde` (not gated; licence
  apache-2.0). Held whole; both safetensors shards equal the Hub's LFS records. Stored in bfloat16.
- **Reference path:** transformers 5.16.1, the native GraniteMoeHybridForCausalLM (no remote code),
  `scripts/gold-logits.py --upcast-at-use` (float32 arithmetic, cast at use per decoder block). transformers 4.57.1 and
  5.16.1 read the tokenizer identically (7,869 corpus tokens).
- **Files**, each sha256 checked against the Hub's LFS record on download and as held:
  - IBM, `ibm-granite/granite-4.0-h-micro-GGUF` at `dc1dd2585fac18a78001c677d33ef8a7bbb7eb68`, no importance matrix:
    - Q8_0 `granite-4.0-h-micro-Q8_0.gguf`, 3,397,676,704 bytes, sha256
      `a009111abf2865b7aad1e66326a6c772cddc29bccd22898f470292068b27bb59` (the row);
    - Q4_K_M `granite-4.0-h-micro-Q4_K_M.gguf`, 1,942,564,512 bytes, sha256 `bcc78b9b…` (measured, not admitted).
  - unsloth, `unsloth/granite-4.0-h-micro-GGUF` at `0577b1ba05f77dc2b7f554cf427a69dd0c8f18af`, with unsloth's
    importance matrix: Q4_K_M `granite-4.0-h-micro-Q4_K_M.gguf`, 1,942,564,800 bytes, sha256
    `1f21ba20ca249f6de1b7fcc436d95bba80a11d40060532bf6eb0c6a9e54e5d28` (the row). Its embedded chat template is
    byte-equal to IBM's. Its file name is IBM's, so its manifest path is `models/unsloth/`.
- **Windows:** two windows of 272 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, from position 8 and
  from 15. Token-id keyed: every position matched by id, no token spelling mismatches, no dropped reference mass.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,096 tokens, `-c 4096`) from the clean release-recipe STRICT build of main e39220e6 (runner 1.18.0, gcc 13.3.0,
  sha256 `e7fa5f48…`). The attention layers carry no positional encoding, and no score log names YaRN. Between that
  main and c408c957, this branch's base, the engine change is StableLM 2's default-system entry, which Granite's template
  does not reach; score files do not use templates.

## Against the reference

| File | Window | Mean KLD | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| **IBM Q8_0** | 1 | 0.0024 | 0.024 | 99.26% | 100% |
| | 2 | 0.0026 | 0.041 | 98.90% | 100% |
| | **pooled 544** | **0.0025 ± 0.0002** | 0.041 | 99.08% | **100% (408/408)** |
| **unsloth Q4_K_M** | 1 | 0.0426 | 0.505 | 93.01% | 98.99% |
| | 2 | 0.0553 | 2.070 | 90.81% | 97.61% |
| | **pooled 544** | **0.0490 ± 0.0048** | 2.070 | 91.91% | **98.28% (401/408)** |
| IBM Q4_K_M (not admitted) | 1 | 0.0805 | 0.647 | 87.50% | 95.48% |
| | 2 | 0.0904 | 1.340 | 86.40% | 94.26% |
| | **pooled 544** | **0.0854 ± 0.0048** | 1.340 | 86.95% | **94.85% (387/408)** |
| IBM Q4_K_M on llama.cpp 73a43d1 | 1 | 0.0888 | 0.772 | 86.40% | 94.97% |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0021 to 0.0029 for the IBM Q8_0, 0.0407 to 0.0596 for the unsloth Q4_K_M (38% of
the resampled means over 0.05) and 0.0763 to 0.0953 for the IBM Q4_K_M. The pooled medians are 0.0005, 0.0274 and
0.0559.

The llama.cpp row is llama-server at `-c 4096` on the CPU, sent the reference's token ids, beside Runner's score file on
the same positions (`gold-q4_k_m-w1-llama.json`).

Per position: `gold-q8_0-w1.json`, `gold-q8_0-w2.json`, `gold-unsloth-q4_k_m-w1.json`,
`gold-unsloth-q4_k_m-w2.json`, `gold-q4_k_m-w1.json`, `gold-q4_k_m-w2.json`, `gold-q4_k_m-w1-llama.json` (FILES.sha256).

## The tokenizer

- difftok against IBM's tokenizer.json at d5f01a3e: 0 of 721 corpus strings and 0 of 99,474 vocabulary entries differ,
  on all three files. Tokenizer pre `dbrx`.
- The reference ids are captured in
  `tests/compatibility/tokenizer-references/granite-4.0-h-micro.tokenizer-json.json`.

## The template

- Runner picks its `granite4` renderer. The template is byte-equal to granite-4.0-h-small's (sha256 `9524df67`), so the
  existing `granite40h` conformance family covers it. It was 21/21 in text before any change, the publisher's default
  system turn included, so there is no template fix here.
- The one change: h-micro's Q4_K_M joins that family's tokenizer files, so the token comparison runs on this host too.
  It is 21/21 in tokens.

## What the file resolves to on load

Architecture granitehybrid, template granite4, the native qwen_json tool protocol, sampling preset generic. CUDA runs
all 40 layers on the device.

## The cells

On a CUDA MIG slice host (1g.24gb, sm_120), the CPUs shared with other work, from the clean release-recipe STRICT build
of main c408c957, this branch's base (runner sha256 `4a475478…`, provenance commit c408c957, dirty false), built before
the rows were added (R6.67).
- **tier-smoke**, `--pin` then `--record`, on the CPU and CUDA cells of both rows: all four pass with the fact found.
  CPU and CUDA give the same reply bytes for each file (IBM Q8_0 `e2b4044c`, unsloth Q4_K_M `3ad3cc77`).
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks`,
  `1.18.0-2026-11-13-linux-x86_64-granite-4.0-h-micro.json`): complete, every declared check passes.
  - IBM Q8_0 and unsloth Q4_K_M (new entries): load, tokenizer (0 of 721 against the captured reference ids; 0 of
    99,474 vocabulary entries), cpu_cuda (9 of 9 identical each, `cpu_cuda/granite-4.0-h-micro-q8_0.json` and
    `cpu_cuda/granite-4.0-h-micro-q4_k_m-unsloth.json`) and chat.
  - IBM Q4_K_M (the entry already in the manifest, its checks unchanged): load and chat. Its notes now carry the
    conformance result and its fidelity figure in place of the out-of-date "default system prompt not sent yet".
  The files were held in memory for the run; a lab wrapper kept their links, and the sha256 check read the held bytes.
