# NVIDIA Nemotron 3.5 Super VL-120B-A12B: Runner against llama.cpp on the same bytes (2026-11-10)

Nemotron 3.5 Super VL-120B-A12B is NVIDIA's 120B-A12B Mamba-2 / attention / LatentMoE hybrid, with a vision tower.
NVIDIA publishes no GGUF of it. The row's file is ggml-org's automatic conversion at Q4_0, and only its text path
is read here.

**What is not flattering, first:**
- **This is agreement, not fidelity.** NVIDIA's BF16 parent is 241.5 GB, about 483 GB in float32. It cannot be held
  for a float32 reference on the lab host, so no publisher reference was read.
  - The anchor is llama.cpp 36a0e2f on the same held bytes: the owner's pick for this item.
  - The numbers below say how closely two engines agree on one file, not how close either is to NVIDIA's model.
- **The prompt pass is slow on the CPU.** About 6,000 tokens took 651.6 s against llama.cpp's 50.2 s on the same
  40 shared CPUs: 13 times as long.
  - The whole 8,103-token corpus as one prompt took 655.8 s and 717.0 s in two runs at the default batch (-b 64,
    about 12 tokens/s), and 321.6 s to 360.4 s at -b 512. llama.cpp took 60.3 s for the same prompt.
  - The routed experts are 82% of the -b 64 time and 69% at -b 512 (RUNNER_DEBUG_TIME).
  - The cause is a separate item (R3.57), with its own kill. This release states the figure as a limit.
- **CPU only.** LatentMoE and the router (sigmoid gating with a selection bias) have no device path. CUDA and Metal
  refuse the file by name, and it runs on the CPU.
- **Two positions of the agreement run are not clean,** and both reproduce on a rerun:
  - one failed to score (llama.cpp gave no logprobs there) and is outside the 300;
  - one is the only margin-qualified miss, at a flat position.
- **Scope:** one 300-position window over one corpus. Text only: the vision tower and the MTP head ggml-org ships
  beside the file are not loaded. Task quality and the tool protocol were not measured.

The kills were declared before each read (the lab's record, R4.27.4.4):
- the loader probe;
- the existing Nemotron pins byte-identical;
- the tokenizer;
- the template;
- agreement with llama.cpp at mean KLD at most 0.05 and margin-qualified top-1 at least 97%.

## The file and the model

- **Parent:** `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` at `fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe`
  (licence openmdw-1.1, model_type nemotron_h_omni). Its text model is `llm_config`.
- **File:** `ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF` at `99144d1d`,
  `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf`: 71,933,775,840 bytes, sha256 `502276c53b8e0173...`, checked
  against the Hub's LFS record as it was held in memory.
- **Layout, the header against the parent's `llm_config`:** every field below agrees.
  - 88 blocks in the same order: 40 Mamba-2, 40 MoE, 8 attention.
  - MoE: 512 experts top 22, expert width 2,688, a shared expert 5,376 wide.
  - LatentMoE width 1,024, embedding width 4,096.
  - Router weight scale 5.0, weights renormalised.
- **LatentMoE,** as llama.cpp 36a0e2f's nemotron-h-moe graph runs it and Runner now does:
  - the router and the shared expert read the full-width normed input;
  - the routed experts read its projection through `ffn_latent_down` (4,096 to 1,024), and their weighted sum goes
    back through `ffn_latent_up`.

## The tokenizer and the template

- **Tokenizer:** difftok against the parent's tokenizer.json at fb0ee0bc. 0 of 721 corpus strings and 0 of 128,635
  vocabulary entries differ.
- **Template:** the file's template is the parent's chat_template.jinja byte for byte (12,927 bytes, sha256
  `1369e90b`).
  - Runner picks the `nemotron35` renderer and qwen3_xml tools.
  - A new conformance family, `nemotron35-super`, has the file's own template as its oracle: 24 of 24 cases are
    identical in text and in tokens.

## Agreement with llama.cpp

`scripts/kld-compare-raw.py` over `tests/fixtures/gold-corpus-2k.txt`, 300 positions at stride 15, top-20 on both
sides, tie band 0.5 nats (`agree-runner-vs-llamacpp.json`). Side a is Runner, side b is llama.cpp, KL(a || b).

| | mean KLD | median | p95 | max | top-1 | margin-qualified top-1 | top-8 overlap |
|---|---|---|---|---|---|---|---|
| Runner against llama.cpp | 0.0136 | 0.0031 | 0.0309 | 1.28 | 96.00% | 99.67% | 0.949 |

- Of the 12 positions where the two engines pick different tokens, 11 sit inside the tie band.
- **The run reproduces.** The scorer was run again on fresh servers: the same mean KLD and agreement figures, the
  same failed and missed positions, and every position's KLD equal to four decimals. Neither engine's prefix reuse
  changes its answers.
- **The failed position, i 3510, is llama.cpp's.** The next token there begins an emoji.
  - Runner answers with its distribution. Its top token is a partial UTF-8 sequence, logprob -0.008.
  - llama-server returns the sampled text without logprobs, as it does whenever the sampled token is an incomplete
    multibyte sequence, so the scorer cannot score the position.
- **The one margin-qualified miss, i 3585, is a real disagreement at a flat position.** The prefix ends "Sphinx of
  black quartz, judge my".
  - Runner's top token is " voice" (logprob -2.43); llama.cpp's is " friend" (-1.59).
  - Neither ranks the corpus's " vow" in its top 20.
  - Queried fresh, without any earlier request, it reads the same. Its cause is not examined.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work.
- **CPU:** tier-smoke pinned, then a compare run matched and recorded the cell (fact found).
  - The reply is 221 bytes, sha256 `01849dbc`.
  - The run was on the branch's release-recipe build, server `-c 512 -b 64 --gpu off`.
- **CUDA:** a declared limit. `cuda_admit` says "LatentMoE (experts at width 1024 between ffn_latent_down and
  ffn_latent_up) has no device path, running on CPU", and the router has no device kernel either.
- **Metal:** a declared limit. Metal refuses LatentMoE by name, and no Mac that holds the file has run it.

## Setup

- **Runner:** branch r4.27.4.4-nemotron35-latent (R4.27.4.4 S1), built by the release recipe: gcc 13.3.0, release.yml's
  x86-64 CFLAGS on make's command line. `--serve --gpu off -t 40 -c 9216`.
- **llama.cpp:** 36a0e2f, `llama-server` (CPU build, GGML_NATIVE, flash attention off, its default weight repack,
  `-c 9216 -t 40 -tb 40 -np 1`), on the same held path.
- **The file** was held in memory from the Hub for each pass. This host's disk rereads at 6 to 11 MB/s.

## Not measured

- A float32 reference (the parent does not fit).
- The CUDA and Metal cells.
- The tool protocol, and task quality.
- Context beyond the window's longest prefix, about 6,000 tokens.
