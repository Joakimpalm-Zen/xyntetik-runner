# SmolLM2: bartowski's files and the publisher's own against HuggingFaceTB's BF16 checkpoints in float32 (2026-11-13)

SmolLM2 is HuggingFaceTB's family of small instruction models on the llama architecture. The family had no file with
a cell. Three rows join: bartowski's Q8_0 of the 135M (the file CI already runs), and bartowski's Q8_0 and Q4_K_M of
the 1.7B. The publisher's own GGUF of the 1.7B, and bartowski's Q4_K_M of the 135M, were measured and not admitted.

**What is not flattering, first:**
- **The publisher's own file fails.** HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF carries one file, a q4_k_m with no
  importance matrix. Pooled over 595 positions it reads 0.1021 ± 0.0047 mean KLD and 94.38% margin-qualified top-1,
  against a bar of 0.05 and 97%. It is the file, not Runner: llama.cpp 73a43d1 reads the same file on the same 300
  positions at 0.1248 / 92.31%, against Runner's 0.1032 / 92.76%. Not admitted.
- **The 1.7B Q4_K_M sits on the bar.** bartowski's Q4_K_M (importance matrix) reads 0.0494 ± 0.0023 (bootstrap 95%
  interval 0.0450 to 0.0542; 39% of resamples over 0.05), both windows under the bar (0.0493 and 0.0495), with 98.59%
  margin-qualified top-1. Admitted on the same footing as Apertus-8B-Instruct-2509's Q4_K_M; the Q8_0 is the fidelity
  file of the pair.
- **The 135M Q4_K_M fails.** bartowski's Q4_K_M of the 135M reads 0.0623 ± 0.0030 (bootstrap 95% 0.0567 to 0.0685,
  wholly over the bar), 97.96% margin-qualified. Not admitted.
- **No published file carries the architecture string "smollm".** Every SmolLM2 GGUF declares `llama` (of the 300
  most-downloaded SmolLM GGUF repositories on the Hub, 247 are llama, 52 smollm3, 0 smollm). The rows are llama rows
  in the SmolLM family, and the old architecture placeholder row (arch-smollm) goes; the name in `src/arch.h` is left
  for its own item.
- **transformers 5.16.1 reads this tokenizer differently from its own file.** It loads SmolLM2's tokenizer.json as
  GPT2Tokenizer with a plain ByteLevel pre-tokenizer, dropping the Digits(individual_digits) step the file declares:
  a double line break before a digit becomes two tokens instead of one, 8,911 tokens over the gold corpus against the
  file's 8,909. The reference therefore ran on transformers 4.57.1, which reads the file as written. Runner reads it
  as the file and the `tokenizers` library do.
- **Scope:** two windows over one corpus, CPU score files. Task quality is not measured; Metal cells come from the
  Mac.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%). The 1.7B conversions were a second read, declared after the publisher's file
failed: bartowski's Q8_0 first, and its Q4_K_M only if the Q8_0 passed, no other files.

## The files and the references

- **References**, held whole, every LFS file's sha256 equal to the Hub's record (apache-2.0, not gated):
  - `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
  - `HuggingFaceTB/SmolLM2-1.7B-Instruct` at `31b70e2e869a7173562077fd711b654946d38674`.
  - The two tokenizer.json files are byte-identical, and so are the two chat templates.
- **Reference path:** transformers 4.57.1, LlamaForCausalLM, `scripts/gold-logits.py --upcast-at-use`: float32
  arithmetic on the stored bfloat16, cast at use per decoder block.
- **Files**, each sha256 checked against the Hub's LFS record on download and as held:
  - `bartowski/SmolLM2-135M-Instruct-GGUF` at `09816acd5d99df7be770d85ea30822623dab342c`:
    - Q8_0 `SmolLM2-135M-Instruct-Q8_0.gguf`, 144,811,360 bytes, sha256
      `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83`. The file `download-model.sh` fetches for CI;
      it has been in the repository since its first file commit (9a7c2ed6, 2024-10-31). unsloth's Q8_0 (revision
      9e6855bc, 144,811,072 bytes) carries the same tensor data, 272 of 272 tensors byte-equal; only the metadata
      differs (`bartowski-vs-unsloth-tensors.txt`).
    - Q4_K_M `SmolLM2-135M-Instruct-Q4_K_M.gguf`, 105,454,432 bytes, sha256 `2e8040ce…` (measured, not admitted).
      unsloth's Q4_K_M is a different conversion: 206 of 272 tensors differ in data. Not measured.
  - `bartowski/SmolLM2-1.7B-Instruct-GGUF` at `1f03464768bfcc0319fc50da8ff5fb20b6417ba2`:
    - Q8_0 `SmolLM2-1.7B-Instruct-Q8_0.gguf`, 1,820,414,944 bytes, sha256
      `0c6e8955788b1253f418c354a4bdc4cf507b8cfe49c48bb10c7c58ae713cfa2a`;
    - Q4_K_M `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`, 1,055,609,824 bytes, sha256
      `77665ea4815999596525c636fbeb56ba8b080b46ae85efef4f0d986a139834d7`.
  - `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF` at `2d4a76a30b4af41ecd395c35725ac11688d4cfe4`:
    `smollm2-1.7b-instruct-q4_k_m.gguf`, 1,055,609,536 bytes, sha256 `decd2598…` (measured, not admitted).
  - bartowski's headers name general.license apache-2.0 and bartowski's importance matrix (calibration_datav3, 141
    chunks), which a Q8_0 does not use. The publisher's file names none.
- **Windows:** two windows at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, the first from position 8 (8 to
  4493, 299 positions), the second from 15 (15 to 4500, 296 positions), so no position is shared. Token-id keyed: every
  position matched by id, no token spelling mismatches, no dropped reference mass.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,520 tokens) from the clean release-recipe STRICT build of main 4934c359 (gcc 13.3.0, runner sha256 `3eddc1c0…`).

## Against the references

| File | Window | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| **135M Q8_0** (bartowski) | 1 | 0.0071 | 0.0018 | 0.082 | 99.00% | 100% (193/193) |
| | 2 | 0.0077 | 0.0018 | 0.056 | 97.97% | 100% (199/199) |
| | **pooled 595** | **0.0074 ± 0.0005** | 0.0018 | 0.082 | 98.49% | **100% (392/392)** |
| 135M Q4_K_M (bartowski, not admitted) | 1 | 0.0621 | 0.0461 | 0.857 | 89.63% | 97.93% (189/193) |
| | 2 | 0.0625 | 0.0415 | 0.554 | 87.84% | 97.99% (195/199) |
| | **pooled 595** | **0.0623 ± 0.0030** | 0.0436 | 0.857 | 88.74% | **97.96% (384/392)** |
| **1.7B Q8_0** (bartowski) | 1 | 0.0038 | 0.0008 | 0.043 | 98.33% | 100% (220/220) |
| | 2 | 0.0045 | 0.0008 | 0.073 | 96.96% | 100% (207/207) |
| | **pooled 595** | **0.0041 ± 0.0003** | 0.0008 | 0.073 | 97.65% | **100% (427/427)** |
| **1.7B Q4_K_M** (bartowski) | 1 | 0.0493 | 0.0341 | 0.376 | 90.97% | 97.73% (215/220) |
| | 2 | 0.0495 | 0.0290 | 0.345 | 88.18% | 99.52% (206/207) |
| | **pooled 595** | **0.0494 ± 0.0023** | 0.0306 | 0.376 | 89.58% | **98.59% (421/427)** |
| 1.7B q4_k_m (the publisher's, not admitted) | 1 | 0.1033 | 0.0764 | 1.112 | 84.62% | 92.73% (204/220) |
| | 2 | 0.1008 | 0.0701 | 0.997 | 84.80% | 96.14% (199/207) |
| | **pooled 595** | **0.1021 ± 0.0047** | 0.0737 | 1.112 | 84.71% | **94.38% (403/427)** |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0066 to 0.0083 (135M Q8_0), 0.0567 to 0.0685 (135M Q4_K_M, all over 0.05), 0.0035
to 0.0048 (1.7B Q8_0), 0.0450 to 0.0542 (1.7B Q4_K_M, 39% over 0.05) and 0.0932 to 0.1113 (the publisher's file).
Both treat the positions as independent draws; they come from one text.

The publisher's file on llama.cpp 73a43d1 (llama-server, CPU, the reference's token ids sent, window 1):
`gold-1.7b-publisher-q4_k_m-w1-runner-vs-llamacpp.json`, side b 0.1248 mean KLD, 85.00% top-1, 92.31%
margin-qualified, beside Runner's side a on the same positions.

For scale, the golden pass of 2026-09-07 read the same 135M Q8_0 (sha256 `5a139571`) at 1,983 positions served:
Runner 0.0080 / 100% margin-qualified, llama.cpp 0.0164 / 99.92% (`docs/golden-pass-evidence/gold-2k`).

Per position: the `gold-*.json` files (FILES.sha256).

## The tokenizer

- difftok against HuggingFaceTB's tokenizer.json: 0 of 721 corpus strings and 0 of 48,821 vocabulary entries differ,
  on all five files. Tokenizer pre `smollm`: digits one by one, as the file's Digits(individual_digits) step.
- The reference ids are captured in `tests/compatibility/tokenizer-references/smollm2-135m.tokenizer-json.json` and
  `smollm2-1.7b.tokenizer-json.json` (identical ids; one per reference checkpoint).

## The template

- Runner picks its `chatml` renderer. `scripts/template-conformance.py --family smollm2` against the publisher's
  chat template (sha256 `872be49d`, the same in both checkpoints): 16 cases compared, 16 identical in text and tokens.

## What each file resolves to on load

All three admitted files, through `--tool-info` and the load line on the CPU: architecture llama, tokenizer pre
smollm, template chatml, tool family qwen_json (Runner's generic JSON tool protocol; the publisher's template declares
no tool format), sampling preset smollm2 (temperature 0.2, top_p 0.9, repeat_penalty 1.1), chosen by general.name.
Nothing these rows run goes through Runner's `smollm` architecture entry.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work, from the clean release-recipe STRICT
build of main 4934c359 (runner sha256 `3eddc1c0c18be4b76dd9d2d4bfbe5b0142c97d833a6ccf634987e3ace06f303f`, provenance
commit 4934c359, dirty false), built before the rows were added (R6.67).
- **tier-smoke**, `scripts/tier-smoke.py --row <row> --model <file> --backend cpu|cuda --device-class
  linux-x86_64-cuda`, `--pin` then `--record`: all six cells pass with the fact found. Every file runs all of its
  layers on the device. CPU and CUDA replies are byte-identical for each file: 135M Q8_0 `5f570ac8`, 1.7B Q8_0
  `3891fd01`, 1.7B Q4_K_M `96763253`.
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks --require-complete`,
  `1.17.0-2026-11-13-linux-x86_64-smollm2.json`): all three files pass load, tokenizer (0 of 721 against the captured
  reference ids), chat ("What is 2+2?" answered and stopped) and cpu_cuda (9 of 9 prompts identical, none tolerated as
  a near tie, `RUNNER_CUDA_TC=0`, `RUNNER_CUDA_ROUTE=reference`; `cpu_cuda/smollm2-*.json`). The files were held in
  memory for the run; the compat script resolves a model path, which a memory-held file's link does not survive, so a
  lab wrapper kept the link for those paths only. The sha256 check read the held bytes.

## Training and adapters

On the CPU, each admitted file: `runner -m <file> --train <8 prompt/completion pairs> --train-steps 4 --lora-rank 8`
runs four steps with finite losses that fall on the repeated examples (135M 0.85 to 0.68, 1.7B Q8_0 1.09 to 0.94,
1.7B Q4_K_M 1.21 to 1.06) and writes an adapter, which loads beside its base with `--lora` (210 adapted projections on
the 135M, 168 on the 1.7B) and generates. `--adapt-info`: training and adapter serving supported.
