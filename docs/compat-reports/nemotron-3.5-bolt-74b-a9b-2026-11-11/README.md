# NVIDIA Nemotron 3.5 Bolt-74B-A9B: Runner against llama.cpp on the same bytes (2026-11-11)

Nemotron 3.5 Bolt-74B-A9B is NVIDIA's 74B-A9B Mamba-2 / attention / LatentMoE hybrid, with a vision projector. Its
MoE blocks differ from one another: each has its own expert count, expert width and top-k. The row's file is
NVIDIA's own 24GB GGUF, and only its text path is read here.

**What is not flattering, first:**
- **This is agreement, not fidelity.** NVIDIA's BF16 parent is 149.8 GB, about 300 GB in float32. It cannot be held
  for a float32 reference on the lab host, so no publisher reference was read.
  - The anchor is llama.cpp 36a0e2f on the same held bytes, as for Nemotron 3.5 Super.
  - The numbers below say how closely two engines agree on one file, not how close either is to NVIDIA's model.
- **Raw top-1 agreement is 96.33%.** 11 of the 300 positions pick different tokens. Every one sits inside the
  0.5-nat tie band on both sides, so margin-qualified top-1 is 100%.
- **The file's purpose is not met.** NVIDIA names it for a 24 GB card. LatentMoE, the per-layer expert widths and the
  router (sigmoid gating with a selection bias) have no device path, so CUDA and Metal refuse the file by name, and it
  runs on the CPU.
- **The prompt pass is slow on the CPU.** A cold 4,470-word prompt (about 6,000 tokens) took 494.6 s against llama.cpp's
  65.9 s on the same 40 shared CPUs: 7.5 times as long. This is one request each, from the timing probe before the
  agreement run, with the host's load at 30 to 68 from other work. The cause is R3.57, a separate item.
- **One position failed to score** and is outside the 300: i 3510, the position where Nemotron 3.5 Super's run also
  failed (llama-server answers a partial UTF-8 token without logprobs). Which side answered without logprobs was not
  re-queried for this file.
- **Scope:** one 300-position window over one corpus. Text only: the vision projector NVIDIA ships beside the file is
  not loaded. Task quality and the tool protocol were not measured.

The kills were declared before each read (the lab's record, R4.27.4.4 S2):
- the loader probe on the real header;
- a heterogeneous toy against its zero-padded twin;
- the existing Nemotron pins byte-identical;
- the tokenizer;
- the template;
- agreement with llama.cpp at mean KLD at most 0.05 and margin-qualified top-1 at least 97%.

## The file and the model

- **Parent:** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` at `e09a4cedb1787db1760375ee5539f222cae0c672` (licence
  openmdw-1.1, model_type nemotron_h_omni; its text model is `llm_config`, nemotron_h_puzzle).
- **File:** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF` at `e0fecb69`, `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`:
  22,018,326,528 bytes, sha256 `91d63363e782c8a3...`, checked against the Hub's LFS record as it was held in memory.
  - The routed and shared experts are IQ2_XXS.
  - The Mamba-2 projections are NVFP4 with per-tensor scales.
  - Attention, the latent projections, the embeddings and the head are Q8_0.
- **Layout, the header against the parent's `llm_config`:** every field below agrees, block by block.
  - 88 blocks in the same order: 40 Mamba-2, 40 MoE, 8 attention.
  - Each MoE block's expert count (240, 304, 368, 432 or 496), width (1,536 to 2,560) and top-k (9) equal its
    `block_configs` entry. The header carries the count as the router's shape, and the width and top-k as per-layer
    arrays.
  - LatentMoE width 1,024, embedding width 4,096, a shared expert 5,376 wide.
  - Router: sigmoid, weight scale 5.0, weights renormalised.
- **What Runner needed:** it read `expert_feed_forward_length` and `expert_used_count` as scalars, got 0, and refused
  the file. Now each block runs its experts at its own width, and the widest sizes the scratch. A top-k that differs
  between blocks is refused by name: no published file varies it.

## The tokenizer and the template

- **Tokenizer:** difftok against the parent's tokenizer.json at e09a4ced. 0 of 721 corpus strings and 0 of 128,635
  vocabulary entries differ.
- **Template:** the file's template is the parent's chat_template.jinja byte for byte (12,514 bytes, sha256
  `6521afc7`).
  - Runner picks the `nemotron35` renderer.
  - A new conformance family, `nemotron35-bolt`, has the file's own template as its oracle: 24 of 24 cases are
    identical in text and in tokens.

## Agreement with llama.cpp

`scripts/kld-compare-raw.py` over `tests/fixtures/gold-corpus-2k.txt`, 300 positions at stride 15, top-20 on both
sides, tie band 0.5 nats (`agree-runner-vs-llamacpp.json`). Side a is Runner, side b is llama.cpp, KL(a || b).

| | mean KLD | median | p99 | max | top-1 | margin-qualified top-1 | top-8 overlap |
|---|---|---|---|---|---|---|---|
| Runner against llama.cpp | 0.0072 | 0.0026 | 0.066 | 0.163 | 96.33% | 100.00% | 0.967 |

- The 11 positions where the two engines pick different tokens all sit inside the tie band. The widest margin among
  them is 0.224 nats.
- Six positions read KLD above 0.05: i 255, 330, 2550, 3495 (the maximum), 3675 and 3750.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work.
- **CPU:** tier-smoke pinned, then a compare run matched and recorded the cell (fact found).
  - The reply is 229 bytes, sha256 `79e9b26e`.
  - The run was on the branch's release-recipe build, server `-c 512 -b 64 --gpu off`.
- **CUDA:** a declared limit. `cuda_admit` says "LatentMoE (experts at width 1024 between ffn_latent_down and
  ffn_latent_up) has no device path, running on CPU". The per-layer widths and the router have no device path either.
- **Metal:** a declared limit. Metal refuses LatentMoE by name.

## Setup

- **Runner:** branch r4.27.4.4-s2-bolt (R4.27.4.4 S2), built by the release recipe: gcc 13.3.0, release.yml's x86-64
  CFLAGS on make's command line. `--serve --gpu off -t 40 -c 9216`.
- **llama.cpp:** 36a0e2f, `llama-server` (CPU build, flash attention off, `-c 9216 -t 40 -tb 40 -np 1`), on the same
  held path.
- **The file** was held in memory from the Hub for the pass.

## Not measured

- A float32 reference (the parent does not fit).
- The CUDA and Metal cells.
- The tool protocol, and task quality.
- Context beyond the window's longest prefix, about 6,000 tokens.
