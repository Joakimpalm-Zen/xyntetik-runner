# Nemotron 3.5 Lightning-30B-A3B: the R4.29 gate row (lab, 2026-11-03)

**Unflattering first: Runner computed this architecture wrong until today.** It routed nemotron_h_moe's experts with a
softmax. The GGUF names no `expert_gating_func`, and the publisher's router is a sigmoid. On identical contexts the
Q8_0 read PPL 18.31 against llama.cpp's 11.73, with 63% top-1 agreement. The fix is commit 86d63171 (PR #575;
measured here as 590bab5b, the same change before its rebase). After it, Runner reads PPL 11.83 against llama.cpp's
11.73 on those contexts, with 95.8% top-1 agreement.

With the fix, both files pass the bar against the publisher's float32 reference at the model's declared dt limit:

| File | Engine | Mean KLD | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| BF16 GGUF (arithmetic only) | Runner | **0.0055** | 0.119 | 96.33% | 100.00% |
| Q8_0 | Runner | **0.0146** | 0.155 | 92.33% | 99.05% |
| Q8_0 | llama.cpp | 0.0173 | 0.254 | 93.33% | 100.00% |
| Q4_0 (the row's file) | Runner | **0.0476** | 0.584 | 89.33% | 97.62% |
| Q4_0 | llama.cpp | 0.0499 | 0.583 | 90.33% | 99.05% |

The bar is mean KLD ≤ 0.05 and margin-qualified top-1 ≥ 97%; the Q4_0 clears it narrowly. 300 positions at stride 15
(8 to 4,493) of `tests/fixtures/gold-corpus-2k.txt`, every one matched by token id, with no dropped mass.

- **Runner against llama.cpp, Q8_0:** Runner is the closer. Paired per-position KLD (llama.cpp minus Runner) has
  Wilcoxon z 2.55, p 0.011. Margin-qualified discordant positions: 0 for Runner, 2 for llama.cpp (McNemar p 0.5).
- **Runner against llama.cpp, Q4_0:** again Runner is the closer, z 2.73, p 0.006. Discordant positions: 0 for Runner,
  3 for llama.cpp (p 0.25).
- **BF16 against Q8_0, both Runner:** the BF16 file is the closer (p 6e-33), as quantization predicts.

## The reference, two columns

The publisher ships no modeling code. The reference is transformers' NemotronH in float32 on the BF16 weights, cast at
use per decoder block (`scripts/gold-logits.py --upcast-at-use`, CPU). The parent is
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at a9904d24bcc1. Two corrections to that reference are recorded here.

1. **transformers 5.14.1's naive Mamba-2 scan is wrong across chunks.** On a tiny random NemotronH with slow-decaying
   heads, its full forward departs from its own step decode only after the first chunk: 2.7e-7 inside chunk 0,
   4-6e-5 after. The 5.16.1 scan (`mamba2_chunk_scan`, rewritten) holds 2.5e-7 throughout. On this model, 5.14.1 read
   mean KL 1.475 against llama.cpp at positions 2000-3108. The reference here is therefore transformers **5.16.1**.
   On the real weights, its chunked scan matches a float64 sequential recurrence to 2e-7 at positions up to 1000
   (Mamba layers 2 and 21; `layer-probe-chunked-vs-sequential.json`).
2. **The dt floor.** The 5.16.1 mixer clamps dt at `time_step_min` (0.001) on both its kernel and fallback paths:
   `self.time_step_limit = (config.time_step_min, inf)`. The model's config declares the limit separately.
   transformers' own `NemotronHConfig` has `time_step_limit` defaulting to `(0.0, inf)`, Lightning's config.json sets
   none, and `time_step_min` is Mamba's dt initialisation range. Runner and llama.cpp follow the declared limit, with
   no floor. The table above is the declared-limit column (floor 0), the anchor. With the mixer's clamp, Runner reads
   0.246 on the BF16 file and 0.252 on the Q8_0, with margin-qualified top-1 84.8% and 84.3%
   (`gold-logits-bf16-q8_0-runner-configured-floor.json`). The clamp reads an initialisation range as a limit.
   - The empirical line (`reference-floor-tiebreak.json`): over the corpus's first 4,509 tokens, the reference at
     floor 0 predicts the text better in every range. NLL is 2.351 against 2.377 for the clamp at positions 0-128,
     3.199 against 3.238 at 128-2000, 2.590 against 2.722 at 2000-4508, and 2.836 against 2.926 overall
     (PPL 17.05 against 18.66).

## Where the engines and the reference part

A per-layer comparison (`RUNNER_ACT_TRACE` against `scripts/act-trace-hf.py`, Runner on the BF16 GGUF at floor 0)
found no op that computes differently:

- Positions 64-130 are within 4.4e-4 relative L2 at all 52 layers.
- From there the distance grows with sequence position, mostly at the Mamba-2 layers: layers 0-1 sit at 5e-6 and
  layer 2 at 7e-5, uniformly across positions 937-1000.
- Runner's serial Mamba-2 step matches the reference's math op for op. The GGUFs keep every small SSM tensor
  (`ssm_a`, `ssm_d`, dt bias, conv, norm) in F32.
- Runner's `-b 1` and `-b 64` traces are bit-identical, so its CPU prefill is the sequential recurrence.

The growth is long-memory heads integrating rounding differences. Then near-tie top-k flips follow: the expert sets
differ at 10% of (position, MoE layer) pairs over positions 0-1001, none before position 143, and the count grows with
depth (`routing-flips.json`). This is the effect R3.25.1 pins on Qwen3-30B-A3B.

A metric note, so nobody reads 0.10 against 0.010 again:

- "KL 0.10 at long range" compared the BF16 reference with llama.cpp on the Q8_0 file, over the full vocabulary.
- "0.010" compared Runner with llama.cpp on the same Q8_0 file, over llama.cpp's top-20 only.

These are different metrics on different file pairs. This record's table uses one method throughout: gold-logits'
top-20 KLD against the reference.

## Why the columns differ, one numbered sentence each

1. llama.cpp's CPU dot products quantize the activations to 8 bits for its Q8_0 and Q4_0 kernels, and Runner keeps
   them in float32 against the dequantized weights, which is the closer of the two to the float32 reference on both
   files.
2. llama.cpp (73a43d1) tokenizes the text `[INST]` inside the corpus as plain text (`[`, `IN`, `ST`, `]`), while the
   publisher's tokenizer and Runner read it as the special token; the column sends token ids (`--endpoint-ids`), so it
   is unaffected, but a llama.cpp user who sends such text gets other tokens.
3. Both engines route with the sigmoid now, llama.cpp by a hardcoded default for this architecture and Runner since
   86d63171.

## The gate row

- **Files**, each checked against the Hub's LFS digest (sha256 and sizes in `FILES.sha256`, which also lists the
  BF16 GGUF and the parent's 30 files): `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` at 8cd082170ed3.
  - `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`: 61f87e75974e4b53..., 18,898,091,584 bytes
  - `...-Q8_0.gguf`: d4cc4c9fffaa356d..., 33,585,495,616 bytes
- **Tokenizer:** the differential reads 0 of 128,635 vocabulary entries and 0 of 721 corpus segments
  (`difftok-*.log`). Template conformance is 24 of 24 token-identical against the publisher's `chat_template.jinja`;
  the GGUF's template is that file (`template-conformance.log`).
- **Tool gate (fixed build, Q4_0, CPU)** (`tool-gate.json`, `tool-gate-b-temperature0.json`):
  - A: pass, buffered and streamed.
  - B: pass, both, and also at temperature 0. Before the fix, B failed with a three-opener envelope; that output is
    R2.10's parser fixture, a fair parser case that Lightning no longer needs.
  - C: streamed pass; buffered fail ("no prose beside the call").
- **Tier smoke:** pinned and recorded on the CPU backend at 590bab5b (`tier-smoke-cpu-*.json`). The pin file is
  `nemotron-3.5-lightning-30b-a3b-q4_0.linux-x86_64-cuda.cpu.txt`, sha256 4635ac8b266740e8...; the reply names the
  eight planets in order. The earlier pin, 8be9125de896, came from the softmax build and is withdrawn.
- **CUDA:** refused by name. `cuda_admit` has no device kernel for a sigmoid-plus-bias router (`k_moe_route` is
  softmax-only), so the row runs on the CPU on CUDA hosts. The roster's cuda cell is a limit.
- **Adapters:** `--info adapter` reports serving unsupported ("--lora does not cover Mamba-2 hybrid architectures yet")
  and training unsupported ("recurrent architecture (Mamba-2)") (`info-adapter.json`).
- **Roster and manifest rows as recorded:** `model-tiers-row.json` and `tier-smoke-manifest-row.json`. The row's tier
  is the coordinator's call.

## Setup

- **Runner:** 590bab5b, the content of 86d63171, built with clang 22.1.8. Runs were CPU only, x86-64 with AVX-512,
  40 threads, with score files from `runner --score 20 -c 9216` on the solo route.
- **llama.cpp:** 73a43d1, `llama-server` (CPU build, flash attention off, `-c 9216 -t 40 -tb 40`), with the GGUF held in
  memory.
- **Reference:** transformers 5.16.1, torch 2.14.0 (CPU), float32 at use, the dt floor as stated per column.
  - The parent was held in memory for every pass, because this lab's disk rereads at 6-11 MB/s and its page cache is
    reclaimed from outside.
  - `goldref-ownmem.py` clones the reference's weights into process memory after the load. That changes no value:
    a plain and a wrapped load gave bitwise-equal logits on Qwen3-0.6B.

## Not measured

- The CUDA and Metal cells (no CUDA path for this router; Metal is the Mac worker's).
- Context beyond the gold window's 4,509 tokens.
- Task quality.
