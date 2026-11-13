# Apertus-8B-Instruct-2509: bartowski's files against swiss-ai's BF16 checkpoint in float32 (2026-11-12)

Apertus-8B-Instruct-2509 is swiss-ai's instruction model on the apertus architecture (an ungated FFN with the xIELU
activation). swiss-ai publishes no GGUF. The two rows are bartowski's Q8_0 and Q4_K_M, a third party's conversion with
an importance matrix, pinned by sha256 and repository revision.

**What is not flattering, first:**
- **The Q4_K_M sits on the bar.** Pooled over 600 positions it reads 0.0492 ± 0.0024 mean KLD (standard error;
  bootstrap 95% interval 0.0447 to 0.0540) against a bar of 0.05. One of its two windows of 300 reads 0.0526, over the
  bar; the other reads 0.0459. A different conversion of the same model (sha256 `87e7f0eb`, 128 bytes smaller) read
  0.0507 at 2,000 positions in the golden pass (`docs/golden-pass-evidence/gold-2k`, Runner's side). The Q8_0 is the
  fidelity file: 0.0054 ± 0.0003.
- **The second window was declared after the first was read.** The first window's Q4_K_M reading (0.0459, 92% of the
  bar) was the reason to read a second; the rule was set before it: report both windows and the pooled 600, open the
  claim with the pooled number and the worse window, and read no third window. Pooled at or under 0.05 passes.
- **Training is refused** at load: "training does not cover this model yet: FFN activation not in the backward
  (gpt-oss/apertus)". Serving an adapter is supported.
- **Metal is not measured on this file:** the load says "gpu: 'apertus' FFN activation is not on the metal backend
  yet" and runs on the CPU.
- **The template inside the GGUF is the publisher's earlier one.** Runner does not render it (below); an engine that
  renders a file's own template gets the earlier behaviour.
- **Scope:** two windows over one corpus, CPU score files. Task quality and the tool protocol on the real checkpoint
  are not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), and the smallest passing file is the row's.

## The files and the reference

- **Reference:** `swiss-ai/Apertus-8B-Instruct-2509` at `b946d40447b2b597999b9c86d44bee0b452c919f` (apache-2.0,
  base_model swiss-ai/Apertus-8B-2509, not gated), held whole; every file's sha256 equals the Hub's LFS record.
- **Reference path:** transformers 5.16.1, ApertusForCausalLM, `scripts/gold-logits.py --upcast-at-use`: float32
  arithmetic on the stored bfloat16, cast at use per decoder block (32 blocks).
- **Files:** `bartowski/swiss-ai_Apertus-8B-Instruct-2509-GGUF` at `d2ffbcc459d0a1bccb2097aca8bb825974f7678c`.
  - Q8_0: `swiss-ai_Apertus-8B-Instruct-2509-Q8_0.gguf`, 8,565,372,288 bytes, sha256
    `68a9d338961e9cc47362649e703fd67b4f4eb616a16d97b9c2c458aea4e3db7a`.
  - Q4_K_M: `swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M.gguf`, 5,057,885,568 bytes, sha256
    `5901007f15aec9aabeea401cbe65559671166fc851bbdf8c26f56c2b61e9cd1d`. The same bytes as at the repository's earlier
    revision a5d17423, which `tests/compatibility/models.json` named before this record.
  - Each file's sha256 was checked against the Hub's LFS record on download and again as it was held. The header
    names general.license apache-2.0, the pre-tokenizer tekken, and bartowski's importance matrix
    (calibration_datav5, 822 chunks).
- **Windows:** two windows of 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, the first starting
  at position 8 (8 to 4493), the second at 15 (15 to 4500), so no position is shared. Token-id keyed: every position
  matched by id, no token spelling mismatches, no dropped reference mass. 204 and 215 positions are margin-qualified.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route)
  from a release-recipe build (gcc 13.3.0, the release workflow's x86-64 CFLAGS on make's command line). The first
  window read 4,510-token files (`-c 4510`), the second 4,520-token files: the second window needs 4,517 reference
  tokens. The 4,520-token files equal the 4,510-token files on every one of the first 4,510 positions (tokens,
  logprobs, argmax and top lists), so both windows read the same numbers.

## Against the reference

| File | Bytes | Window | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| **Q8_0** | 8,565,372,288 | 1 (from 8) | 0.0050 | 0.0015 | 0.047 | 97.33% | 100% (204/204) |
| | | 2 (from 15) | 0.0057 | 0.0019 | 0.052 | 96.67% | 100% (215/215) |
| | | **pooled 600** | **0.0054 ± 0.0003** | 0.0017 | 0.052 | 97.00% | **100% (419/419)** |
| **Q4_K_M** | 5,057,885,568 | 1 (from 8) | 0.0459 | 0.0290 | 0.317 | 89.00% | 98.04% (200/204) |
| | | 2 (from 15) | 0.0526 | 0.0342 | 0.534 | 88.33% | 98.14% (211/215) |
| | | **pooled 600** | **0.0492 ± 0.0024** | 0.0312 | 0.534 | 88.67% | **98.09% (411/419)** |

± is the standard error of the pooled mean over the 600 per-position values. A percentile bootstrap of the pooled
mean (10,000 resamples, seed 20261112) gives 0.0048 to 0.0060 for the Q8_0 and 0.0447 to 0.0540 for the Q4_K_M; 36%
of the Q4_K_M's resamples are over 0.05. Both treat the 600 positions as independent draws; they come from one
text.

Per position: `gold-q8_0-w1.json`, `gold-q8_0-w2.json`, `gold-q4_k_m-w1.json`, `gold-q4_k_m-w2.json`
(FILES.sha256).

For scale, the golden pass of 2026-09-07 and 2026-10-25 read a different Q4_K_M conversion of this model (sha256
`87e7f0eb`, 5,057,885,440 bytes) at 2,000 positions against float32: Runner 0.0507 / 98.1% margin-qualified,
llama.cpp 0.0773 / 96.4%.

## The tokenizer

- difftok against swiss-ai's tokenizer.json at b946d404: 0 of 721 corpus strings and 0 of 128,635 vocabulary entries
  differ on both files (1,000 added and 1,437 partial-UTF-8 entries left out of the vocabulary pass).
- The 1,000 added tokens are equal to the publisher's, token 73 included (`<SPECIAL_73>`; b946d404 reverted an
  earlier `<|image|>` there). BOS 1, EOS 68 (`<|assistant_end|>`).
- The reference ids are captured in
  `tests/compatibility/tokenizer-references/apertus-8b-instruct-2509.tokenizer-json.json`.

## The template

- Runner picks its `apertus` renderer (the file's template carries `<|assistant_start|>`), which follows the
  publisher's chat_template.jinja at b946d404 (14,601 bytes, sha256 `56f72c27`).
- `scripts/template-conformance.py --family apertus --require-tokens`, with the family's tokenizer read from the Q4_K_M:
  20 cases compared, 20 identical in text and tokens. One case is not expressible: system-mid-history, which the
  reference refuses ("Invalid message role").
- The template inside both GGUFs (13,991 bytes, sha256 `00067a2e`) is the publisher's from before cdb3e4f4
  (2025-11-14, "Add support for images and fix end of assistant issue"); bartowski's repository dates from 2025-10-03. It lacks
  the image token and the rule that closes a finished assistant turn with `<|assistant_end|>` unless the turn waits
  for tool outputs. Runner does not render it, so a user of these files through Runner is not affected; an engine
  that renders the file's own template gets the earlier behaviour.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work, from a release-recipe build of main
d431c978 (gcc 13.3.0, runner sha256 `63a9af5d231bedeb3717cd483b563933f651e3634dd966aaf9b51de4915fcfed`). The same
build reproduces both files' 4,520-token score files byte for byte (the fidelity was read from main ba49206f's build).
The pins were taken with a build that does not yet contain these rows: Runner compiles its roster table
(`src/model_roster.h`) from the JSON that holds the pins, so a pin can only come from a binary built before its
row; the replies do not depend on that table. The pins predate R6.67 (main 562d8b74), which stamps a build's commit
into the binary and takes a pin's commit from it: they were taken with the release-recipe build of main d431c978
above, each names commit d431c978 and that binary's sha256, and they were not re-taken when the row was rebased onto
562d8b74.
- **tier-smoke**, `scripts/tier-smoke.py --row <row> --model <file> --backend cpu|cuda --device-class
  linux-x86_64-cuda`, `--pin` then `--record` (a compare run against the new pin): all four cells pass with the fact
  found. Both files run all 32 layers on the device.
  - Q8_0: the CPU and CUDA replies are byte-identical (sha256 `a519428f`).
  - Q4_K_M: the CPU and CUDA replies are byte-identical (sha256 `5fd562b8`).
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks --require-complete`,
  `1.17.0-2026-11-12-linux-x86_64-apertus-8b-instruct-2509.json`): both files pass load, tokenizer (0 of 128,635
  vocabulary entries), chat ("What is 2+2?" answered and stopped) and cpu_cuda (9 of 9 prompts identical, none
  tolerated as a near tie, `RUNNER_CUDA_TC=0`, `RUNNER_CUDA_ROUTE=reference`; `cpu_cuda/apertus-8b-instruct-2509-*.json`).
  The files were held in memory for the run; the compat script resolves a model path, which a memory-held file's link
  does not survive, so a lab wrapper kept the link for those two paths only. The sha256 check read the held bytes.
- **Metal:** not a cell. On an 8 GB Apple Silicon Mac at main ba49206f the Q4_K_M loads with "gpu: 'apertus' FFN
  activation is not on the metal backend yet" and runs on the CPU, paging (5.1 GB of weights, 2.4 GB of RAM free;
  0.11 tokens a second), with a sensible answer. The Metal port of the activation is not started.

## Training and adapters

- `runner train` on the Q4_K_M: exit 1 at load, "error: training does not cover this model yet: FFN activation not in
  the backward (gpt-oss/apertus)". The xIELU activation has no backward.
- `--adapt-info`: adapter_serving supported (device kernels cuda), training not supported with the same reason.
