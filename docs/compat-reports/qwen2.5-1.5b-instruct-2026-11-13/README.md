# Qwen2.5-1.5B-Instruct: the publisher's files and bartowski's against Qwen's BF16 checkpoint in float32 (2026-11-13)

Qwen2.5-1.5B-Instruct is Qwen's instruction model on the qwen2 architecture. The family had no file with a cell.
Two rows join: bartowski's Q8_0, the fidelity file, and the publisher's own q8_0, the file most people download. Both
Q4_K_M conversions were measured and not admitted.

**What is not flattering, first:**
- **The publisher's q8_0 reads nine times the KLD of bartowski's Q8_0 of the same checkpoint.** 0.0297 ± 0.0012 mean
  KLD against 0.0033 ± 0.0003. It passes the bar (0.05, 99.75% margin-qualified), and it is the file, not Runner:
  llama.cpp 73a43d1 reads the same file at 0.0324 on window 1, against Runner's 0.0307.
- **Why, as far as the tensors show.** The publisher's GGUF is not a straight conversion of the checkpoint it names.
  Each layer norm is the checkpoint's times a per-channel factor (mean 1.54 to 1.74 in layers 0, 10 and 27), and the
  q_proj that reads it is divided by the same factor: norm times q_proj equals the checkpoint's within 0.4 to 0.6%,
  Q8_0's own error. The file also carries an untied output.weight (339 tensors against 338). This is consistent with
  an equalization pass before quantizing: in exact arithmetic the same model, but the rescaled weights quantize worse.
  The publisher does not document it that I found. A Q8_0 converted here from the checkpoint with llama.cpp's own
  converter reads 0.0033 (its BF16 0.00011), and bartowski's Q8_0 is byte-identical to that conversion in all 338
  tensors. The publisher's 0.5B q8_0 shows the same rescaled norms and q_proj against its checkpoint, which fits the
  0.047 the golden pass read on Qwen2.5-0.5B served.
- **Neither Q4_K_M is admitted.** The publisher's q4_k_m fails both bars (0.0798 ± 0.0036, 96.95% margin-qualified;
  llama.cpp 0.0931 on window 1). bartowski's Q4_K_M with its importance matrix reads 0.0516 ± 0.0025 (bootstrap 95%
  0.0470 to 0.0568, 73% of resamples over 0.05), 97.97% margin-qualified: its KLD point estimate is over the bar.
- **The backends give two different replies to the tier-smoke prompt on bartowski's file.** They diverge after about
  thirty tokens ("has its unique" against "has unique"). The CPU (at prompt batch 1, 64 and 512, and with the chunked
  prefill alike) and CUDA with its tensor-core route off (`RUNNER_CUDA_TC=0`, with or without
  `RUNNER_CUDA_ROUTE=reference`) give `c4a53e20`; CUDA's default route and Metal give `e82cf027`. So it is the
  tensor-core route that takes the other branch at a near tie, not the CPU. Each cell is pinned on its own backend and
  passes. compat's cpu_cuda check, on the reference route, is 9 of 9 identical for both files. The publisher's file
  gives one reply on all three backends.
- **The two rows' files have the same name in different case** (`Qwen2.5-1.5B-Instruct-Q8_0.gguf` and
  `qwen2.5-1.5b-instruct-q8_0.gguf`), which is one file on a case-insensitive file system (macOS, Windows). The compat
  manifest keeps bartowski's under `models/bartowski/`; anyone holding both in one folder there loses one.
- **Scope:** two windows over one corpus, CPU score files. Task quality is not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), and a failing publisher q4_k_m stops the run for a decision. The second
read (bartowski's Q4_K_M, the lab conversion, bartowski's Q8_0) was declared after it.

## The files and the reference

- **Reference:** `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (apache-2.0, not gated),
  held whole; its model.safetensors (3,087,467,144 bytes, sha256 `dd924a11…`) equals the Hub's LFS record, and it is
  the only model.safetensors the repository has ever had.
- **Reference path:** transformers 5.16.1, Qwen2ForCausalLM, `scripts/gold-logits.py --upcast-at-use`: float32
  arithmetic on the stored bfloat16, cast at use per decoder block. transformers 5.16.1 and 4.57.1 both read this
  tokenizer.json as written (8,008 corpus tokens each, equal to the `tokenizers` library's).
- **Files**, each sha256 checked against the Hub's LFS record on download and as held:
  - `bartowski/Qwen2.5-1.5B-Instruct-GGUF` at `9eadc66189c7641e1ddd226b8267a9119b2ce2d4`:
    - Q8_0 `Qwen2.5-1.5B-Instruct-Q8_0.gguf`, 1,646,573,312 bytes, sha256
      `7185d306cf45956c8c017cd0d3b05ecc6bc18b3ea8eb5c240dce40e87563db7f` (the first row);
    - Q4_K_M `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`, 986,048,768 bytes, sha256 `1adf0b11…` (measured, not admitted).
  - `Qwen/Qwen2.5-1.5B-Instruct-GGUF` at `91cad51170dc346986eccefdc2dd33a9da36ead9` (its only version of these files):
    - q8_0 `qwen2.5-1.5b-instruct-q8_0.gguf`, 1,894,532,128 bytes, sha256
      `d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8` (the second row);
    - q4_k_m `qwen2.5-1.5b-instruct-q4_k_m.gguf`, 1,117,320,736 bytes, sha256 `6a1a2eb6…` (measured, not admitted).
  - The lab conversion: llama.cpp 73a43d1's `convert_hf_to_gguf.py --outtype bf16` of the reference, then
    `llama-quantize … Q8_0`. A lab instrument, not a file of any row.
- **Windows:** two windows of 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, the first from
  position 8 (8 to 4493), the second from 15 (15 to 4500), so no position is shared. Token-id keyed: every position
  matched by id, no token spelling mismatches, no dropped reference mass.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,520 tokens) from the clean release-recipe STRICT build of main c00c19c4 (runner 1.18.0, gcc 13.3.0, sha256
  `9600cd1b…`). bartowski's Q8_0 and the lab Q8_0 give equal score files (tokens, logprobs and top lists).

## Against the reference

| File | Window | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| **Q8_0, bartowski** | 1 | 0.0032 | 0.0006 | 0.053 | 98.00% | 100% (198/198) |
| | 2 | 0.0034 | 0.0006 | 0.060 | 98.67% | 100% (196/196) |
| | **pooled 600** | **0.0033 ± 0.0003** | 0.0006 | 0.060 | 98.33% | **100% (394/394)** |
| **q8_0, the publisher's** | 1 | 0.0307 | 0.0212 | 0.167 | 91.33% | 100% (198/198) |
| | 2 | 0.0287 | 0.0192 | 0.214 | 92.00% | 99.49% (195/196) |
| | **pooled 600** | **0.0297 ± 0.0012** | 0.0203 | 0.214 | 91.67% | **99.75% (393/394)** |
| q4_k_m, the publisher's (not admitted) | 1 | 0.0846 | 0.0599 | 0.656 | 86.67% | 97.47% (193/198) |
| | 2 | 0.0750 | 0.0538 | 0.945 | 86.67% | 96.43% (189/196) |
| | **pooled 600** | **0.0798 ± 0.0036** | 0.0569 | 0.945 | 86.67% | **96.95% (382/394)** |
| Q4_K_M, bartowski (not admitted) | 1 | 0.0556 | 0.0391 | 0.919 | 86.00% | 96.97% (192/198) |
| | 2 | 0.0476 | 0.0370 | 0.431 | 90.00% | 98.98% (194/196) |
| | **pooled 600** | **0.0516 ± 0.0025** | 0.0380 | 0.919 | 88.00% | **97.97% (386/394)** |
| BF16, the lab conversion | pooled 600 | 0.0001 ± 0.0001 | 0.0000 | 0.020 | 99.67% | 100% (394/394) |
| Q8_0, the lab conversion | pooled 600 | 0.0033 ± 0.0003 | 0.0006 | 0.060 | 98.33% | 100% (394/394) |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0028 to 0.0039 (bartowski's Q8_0), 0.0273 to 0.0321 (the publisher's q8_0), 0.0731
to 0.0871 (the publisher's q4_k_m, all over 0.05) and 0.0470 to 0.0568 (bartowski's Q4_K_M, 73% over 0.05). Both
treat the positions as independent draws; they come from one text.

On llama.cpp 73a43d1 (llama-server, CPU, the reference's token ids sent, window 1), beside Runner's side a on the same
positions: the publisher's q8_0 0.0324 / 93.00% top-1 / 100% margin-qualified, its q4_k_m 0.0931 / 86.00% / 97.47%
(`gold-*-publisher-w1-runner-vs-llamacpp.json`).

Per position: the `gold-*.json` files (FILES.sha256). The tensor comparison: `publisher-rescaling.txt`.

## The tokenizer

- difftok against Qwen's tokenizer.json at 989aa798: 0 of 721 corpus strings and 0 of 150,186 vocabulary entries
  differ, on all four files. Tokenizer pre `qwen2`. tokenizer.json, vocab.json and merges.txt are the same blobs as
  Qwen2.5-7B-Instruct's.
- The reference ids are captured in `tests/compatibility/tokenizer-references/qwen2.5-1.5b-instruct.tokenizer-json.json`.

## The template

- Runner picks its `chatml` renderer. The checkpoint's chat template is byte-identical to Qwen2.5-7B-Instruct's (sha256
  `cd8e9439`), which the conformance family `chatml` tests against; the 1.5B q8_0 joins the family's tokenizer list.
  `scripts/template-conformance.py --family chatml`: 21 cases compared, 21 identical in text and tokens, with this
  file's tokenizer.

## What each file resolves to on load

Both rows' files, through `--tool-info` and the load line on the CPU: architecture qwen2, tokenizer pre qwen2,
template chatml, tool family qwen_json, sampling preset qwen2.5 (temperature 0.7, top_p 0.8, top_k 20,
repeat_penalty 1.05).

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work, from the clean release-recipe STRICT
build of main c00c19c4 (runner sha256 `9600cd1b06513e37…`, provenance commit c00c19c4, dirty false), built before the
rows were added (R6.67).
- **tier-smoke**, `scripts/tier-smoke.py --row <row> --model <file> --backend cpu|cuda --device-class
  linux-x86_64-cuda`, `--pin` then `--record`: all four cells pass with the fact found, every layer on the device.
  The manifest's default prompt drew a 62-byte reply from both files, under the 96-byte minimum; both rows ask the
  paragraph prompt the LFM2.5-350M rows ask (80 tokens). Replies: the publisher's q8_0 `0308e3d8` on both backends;
  bartowski's Q8_0 `c4a53e20` on the CPU and `e82cf027` on CUDA (above).
- **Metal**, on an 8 GB Apple M1 from a clean STRICT build of the rows' first commit: both rows pinned, compared and
  recorded. The publisher's q8_0 replies `0308e3d8`, as on the CPU and CUDA; bartowski's Q8_0 replies `e82cf027`, as on
  CUDA's default route. The two files were kept in separate folders there, since their names differ only in case.
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks --require-complete`,
  `1.18.0-2026-11-13-linux-x86_64-qwen2.5-1.5b-instruct.json`): both files pass load, tokenizer (0 of 721 against the
  captured reference ids), chat ("What is 2+2?" answered and stopped) and cpu_cuda (9 of 9 prompts identical, none
  tolerated as a near tie, `RUNNER_CUDA_TC=0`, `RUNNER_CUDA_ROUTE=reference`; `cpu_cuda/qwen2.5-1.5b-instruct-*.json`).
  The files were held in memory for the run; a lab wrapper kept a memory-held file's link for those paths only, and
  the sha256 check read the held bytes.

## Training and adapters

On the CPU, each row's file: `runner -m <file> --train <8 prompt/completion pairs> --train-steps 4 --lora-rank 8`
runs four steps with finite losses that fall on the repeated examples (0.14 to 0.04 on both files) and writes an
adapter, which loads beside its base with `--lora` (196 adapted projections) and generates. `--adapt-info`: training
and adapter serving supported.
