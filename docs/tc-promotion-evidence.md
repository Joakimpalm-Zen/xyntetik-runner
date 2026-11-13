# Tensor-core promotion evidence

The measurement rows behind each tensor-core and tolerance route that runs by default on CUDA. They were written as comments in `src/cuda.c` beside the code that promotes each route; the code now states the rule and this page keeps the rows, in the words they were recorded in, dates included. [kernel-routes.md](kernel-routes.md) states each route's obligation.

A row here is a record of what was measured when the route was promoted. It is not re-run per release.

## Promoted (type, architecture) combinations: the dense list

The comment above `tc_codebook` and `tc_promoted`.

```text
Promoted (type, arch) combos, every row measured by test_tc_tol on real
weights (datacenter MIG slice, 2026-07-29; table in the TC spec). Dense archs
only, and only those with a gate row: llama 0.003% of range / phi3 0.004%
/ gemma4 0.012% / qwen3 dense, mistral, gemma3, smollm per the same run , 
all 0 top-1 flips in 64 teacher-forced positions. qwen3moe PASSED its
gate too (0.216%, one near-tie) but stays opt-in by owner decision: MoE
routing amplifies fp16 noise ~86x over dense, and the promotion decision
deliberately covers the dense family first. Unmeasured archs are absent,
not implied; an arch joins with its own row (qwen35 and granite later,
muse-glimmer and qwen2 on 2026-10-02: see tc_promoted).
The codebook family, promoted 2026-09-14 (see tc_promoted below).
```

## MoE families whose tolerance route is their default

The comment above `moe_tc_family`.

```text
MoE families measured against the publisher's float32 model (2,000
positions, the owner's rule of 2026-10-20: a tolerance route becomes a
family's default only when its mean KL is within 1.01x and neither a
paired Wilcoxon on per-position KL nor McNemar on margin-qualified top-1
finds it worse than the current route at p < 0.05). They run as named
routes, so a record names them and --verify pins them, and a record made
before them (no field, the reference set) replays without them.
  moe-tc       qwen3moe  Qwen3-30B-A3B Q4_K_M: KL 0.09167 vs 0.09157,
                         Wilcoxon p 0.36, McNemar p 0.48
               gpt-oss   gpt-oss-20b MXFP4: KL 0.08142 vs 0.08312 (closer,
                         p 1e-5), McNemar p 0.61
  moe-grouped  qwen3moe  with moe-tc, scored chunked
                         (RUNNER_SCORE_CHUNKED=1, 64-row chunks by
                         RUNNER_SPEC_BATCH=64, so its batched expert
                         GEMMs ran; RUNNER_SPEC_BATCH alone scores solo
                         and runs none): KL 0.09178 vs 0.09170, Wilcoxon
                         p 0.83, McNemar p 1.0; Q4_K / Q6_K experts;
                         prefill 2.6-3.9x the fused path's at 512 / 2K
               not gemma4: gemma-4-26B-A4B-it, a plain Q4_0 of the
                         float32 conversion, chat-formatted text (that
                         file's greedy replies, 2,000 positions): KL
                         0.30527 vs 0.29527, 1.034x, over the 1.01x bound
                         (Wilcoxon p 0.54, McNemar p 0.48); on raw text,
                         where it reads ~2.2 nats from its reference,
                         2.22047 vs 2.22727. Prefill would have been 4.9x
                         / 2.4x at 512 / 8K: RUNNER_MOE_GROUPED=1 or a
                         route pin runs it (2026-10-21)
(TC alone on qwen3moe was once kept opt-in because its greedy text left
the scalar route's at token 20 on a free-running check; measured against
the reference instead, it reads no further from it.)
```

## Per-type rows inside `tc_promoted`: Q6_K, Q8_0, Q4_0, the codebook family, NVFP4

```text
Q6_K joined 2026-08-08: the profile showed it was 26% of prefill running
on the SCALAR path, because attn_v/ffn_down are Q6_K in every Q4_K_M.

Q8_0 joined 2026-08-09, and it is the strongest row in this table:
test_tc_tol on Qwen3-4B-Q8_0 reported logits BIT-IDENTICAL across 504
counted TC dispatches, not "within tolerance", identical. Prefill went
113.72 -> 475.71 tok/s (4.18x) on the same model and prompt.

Bit-identity is why this is promoted for the whole trusted arch list on
one model's evidence, where a tolerance-based row would not be. The arch
list exists because architectures differ in how much they AMPLIFY
numerical difference, MoE routing amplifies fp16 noise ~86x over dense,
which is why qwen3moe stays opt-in. With zero difference to amplify, that
sensitivity cannot bite. The kernel is exact here because int8 weights
convert to fp16 without loss and the accumulation order matches the
scalar path; that is a property of the kernel, not of the model.

RESOLVED 2026-08-13b, and the caveat's own instruction applies: the
sm_120 confirmation was run and Q8_0 is NOT bit-identical there. It is
merely near-identical, so this row is hereby DEMOTED from "exact" to
tolerance-gated like every other row:

  smollm  SmolLM2-135M-Q8_0        0/64 flips, 8e-5 of logit range
  phi3    Phi-4-mini-Q8_0          0/64,        2e-5
  granite granite-4.1-3b-Q8_0      0/64,        5e-5

The old sm_86 bit-identity reading is also retired for a second reason:
it was measured 2026-08-09, one day AFTER TC_N widened to 64, on the
kernel that computed 16 of its 64 columns. That kernel reported
bit-identity on phi3 and gemma4 while producing different free-running
text at -b 64, so "BIT-IDENTICAL" was not evidence of a correct kernel
and cannot carry a promotion on its own. Promotion for the whole arch
list now rests on three tolerance rows across three archs plus the
free-running arm in test_tc_tol, not on an exactness argument.
Q4_0 joined 2026-08-13, and only because the bug above it was fixed
first. Its TC kernel shares TC_GEMM_32B with Q8_0, which until that day
computed 16 of its 64 token columns and published uninitialised shared
memory for the rest; Q4_0 forced on scored 28/64 to 59/64 top-1 flips.
With the macro fixed, six models across four archs gate clean at
N_BATCH=64, 0/64 flips every row, mean|dlogit| 3e-5 to 8e-5 of range:

  qwen3   Qwen3-0.6B-q4_0, Qwen3-1.7B-q4_0, dense-control-stage1-q4_0
  phi3    Phi-4-mini-instruct-q4_0
  smollm  SmolLM2-135M requantised to q4_0
  llama   Llama-3.2-3B-Q4_K_M requantised to q4_0 (Q4_K + Q4_0 mix)

This is the largest prefill row in the table because Q4_0 had no GEMM
at all on the promoted path: pp512 goes 57.8 -> 977.8 tok/s on
Qwen3-1.7B (16.9x) and 21.2 -> 445.4 on Phi-4-mini (21x). gemma4,
mistral and gemma3 have no q4_0 gate row of their own and inherit the
arch list below, as Q6_K and Q8_0 did; they are the rows to measure
next, not rows this evidence covers.
The codebook family joined 2026-09-14 (docs/cuda-iq-tensorcore-2026-09-14.md,
evidence files beside it), each kernel a device twin of the CPU
decoder staged to fp16 with the token columns scaled to fp16's range.
Forced against the scalar path with test_tc_tol (64 teacher-forced
positions, then free-running greedy at batch 64 / ctx 4096), every row
0 flips unless noted, every free-running arm token-identical:

  granite-4.2-3b requantized to each of the nine formats, all layers
  on the device, datacenter MIG slice: IQ4_XS 4e-5 of range, IQ4_NL 5e-5,
  IQ3_XXS 6e-5, IQ3_S 9e-5, IQ2_S recipe (IQ2_XS+IQ3_S) 1.1e-4 with one
  near-tie, IQ2_XS 1.2e-4, IQ2_XXS 1.3e-4, IQ1_M 3.4e-4, IQ1_S 9.0e-4
  with two near-ties; IQ3_S, IQ3_XXS, IQ2_XXS and IQ1_M the same on an
  RTX 3070.
  IQ3_S on every other arch of the list, same protocol, the Linux CUDA host:
  llama 9e-5, qwen3 2.4e-4, smollm 3e-5, mistral 4e-5, phi3 2.4e-4
  (two near-ties; this is the model whose activations overflow fp16
  without the column scaling), gemma3 6e-5, gemma4 9e-5; IQ2_XXS on
  llama 4e-5, qwen3 1.0e-4 (one near-tie), smollm 1.0e-4.
  qwen35: the GSQ-RCO Qwen3.8 27B mixed file, 40 of 64 layers on the
  slice, 9 of 9 greedy outputs identical at 64 tokens forced on
  against forced off.
  the llama-quantize fixtures of all nine formats on both device
  families: 3e-5 to 7e-5 of range; compute-sanitizer memcheck,
  racecheck, synccheck and initcheck clean on each (RTX 3070).

Prefill on the RTX 3070 (5 warmed runs, median, spread under 1.1%):
25 tok/s on the scalar path to 277-495 tok/s across the formats at
118 to 1892 tokens; the scalar path was the generic warp-per-row
matvec applied to 64 columns, the family had no batched GEMM at all.
The plan's gate (10% median gain over five runs, no 5% regression on
the path's workloads) is met by an order of magnitude on both device
families.

The four formats above were re-gated the same day when their kernels
took the same column scaling (Step 6 of the same doc): 25 real-model
rows across every arch of the list and both device families, 1e-5 to
6e-5 of range, at most one near-tie flip, free-running identical,
and the scaling costs under 2% of prefill on the RTX 3070.
NVFP4 PROMOTED 2026-10-31 (R4.9.5, #508): the float32-reference row on
Qwen3.5-4B NVFP4 (2,000 positions against the publisher's model, chunked
so the GEMMs ran) put the dequantizing tensor-core route at 0.9997x the
scalar route's mean KL, Wilcoxon with TC closer (p 0.040), no McNemar
discordance. RUNNER_CUDA_TC_NVFP4=0 holds it off; =1 forces it on any
architecture, the trusted list below deciding otherwise.
```

## The six further K-quant and legacy types

```text
The tc_kq_type six joined 2026-10-21 on float32-reference rows (owner
rule D1: 2,000 positions against the publisher's model, mean KL within
1.01x of the scalar route's and neither a paired Wilcoxon nor McNemar
on margin-qualified top-1 worse at p < 0.05; the datacenter MIG slice,
chunked scoring in 64-row chunks, RUNNER_SCORE_CHUNKED=1 with
RUNNER_SPEC_BATCH=64, so the GEMMs ran), Qwen3-4B quantized to each type:
  Q5_K_M  0.02327 vs 0.02329  Wilcoxon p 0.019, TC closer
  Q3_K_M  0.27870 vs 0.27869  p 0.50
  Q2_K    0.91540 vs 0.91524  p 0.81
  Q4_1    0.11899 vs 0.11908  p 0.002, TC closer
  Q5_0    0.03736 vs 0.03731  p 0.997
  Q5_1    0.03338 vs 0.03338  p 0.025, TC closer
and the two types most files carry on a second family, Llama-3.2-3B:
  Q5_K_M  0.02395 vs 0.02398  p 0.14
  Q3_K_M  0.16235 vs 0.16226  p 0.997
Their stagers equal dequant_row bit for bit on the host
(test-kq-stage) and test-cuda-tc-kq holds each on the TC, TC2 and
compute_80 kernels. qwen4exp keeps them on the scalar GEMM: its
admission (below) covers what the TC prefill touched on its release
file, which these types did not, and that file is too large for the
page cache of the one host that holds it, so its row is not measured.
```

## The trusted architectures

```text
granite joined 2026-08-13 with a gate row for EVERY promoted type on
its own weights, the only arch here that has one, because it was
the arch that showed what the missing coverage costs: granite-4.1-8b
Q4_0 ran prefill at 8.0 tok/s on a fully offloaded GPU, slower than
the same model on CPU, purely because no TC combo admitted it.
  Q4_0        granite-4.1-3b-q4_0            0/64, 5e-5 of range
  Q8_0        granite-4.1-3b-Q8_0            0/64, 5e-5
  Q4_K/Q6_K   granite-3.3-8b-instruct-Q4_K_M 0/64, 3e-5
gemma4's 31B QAT Q4_0 row was RE-GATED against the fixed kernel on
2026-08-13b and holds: 0/64 flips, 6e-5 of range, 60/60 offload. The
pre-fix "820 dispatches bit-identical" reading it was promoted on is
withdrawn, reproduced from a rebuilt pre-fix binary, that verdict is a
false pass (same binary, same model, free-running divergence at -b 64).
The promotion survives; the evidence behind it did not.

qwen35 PROMOTED 2026-08-21 on four rows across two checkpoints and both
promoted types (Linux CUDA host, N_BATCH=64, all 0/64 flips, free-running
batch-64 greedy token-identical):

  Qwen3.5-4B    Q4_K_M  mean|dlogit| 0.000518 = 2e-5 of range
  Qwen3.5-4B    Q8_0    mean|dlogit| 0.000366 = 1e-5 of range
  Qwen3.5-0.8B  Q4_K_M  mean|dlogit| 0.000400 = 1e-5 of range
  Qwen3.5-0.8B  Q8_0    mean|dlogit| 0.000516 = 2e-5 of range

The rows gate the whole model (teacher-forced + free-running), so
whatever subset of qwen35's projections the TC prefill actually
reaches, attention, MLP and the recurrent-layer projections alike , 
is inside the measured envelope; the scan kernels themselves have no
GEMM shape and are untouched.
muse-glimmer and qwen2 joined 2026-10-02, each on its own test_tc_tol
rows (the datacenter MIG slice, real weights, 64 teacher-forced positions, 0
top-1 flips and a token-identical free-running arm on every row):

  muse-glimmer  Muse-Glimmer-30B IQ3_XXS   IQ3_XXS/IQ3_S/Q4_K/Q6_K  0.00005 of range
  muse-glimmer  Muse-Glimmer-30B kquant    Q4_K/Q6_K                0.00005
  muse-glimmer  Kvist-14B a12 Q8_0         Q8_0                     0.00001
  qwen2         Qwen2.5-7B Q4_K_M          Q4_K/Q6_K                0.00007

What forced the question: unpromoted, a codebook i-quant has only the
scalar batched kernel, and Muse-Glimmer-30B IQ3_XXS prefilled at 2.60
tok/s on the device against 24.0 on the host's 16 threads (slower than
its own decode, 3.67): a 3,300-token prompt took over fifteen minutes
on a GPU. With the tensor-core GEMM it is 85.3 tok/s; the K-quant file
went 34.9 -> 116.9. Neither arch is MoE (muse-glimmer's gates are
per-layer sigmoid gates, not routing), so the amplification that keeps
qwen3moe opt-in does not apply.

apertus, nemotron_h and granitehybrid joined the same day, on the same
gate (0 flips of 64 and token-identical free-running output on each):

  apertus        Apertus-8B Q4_K_M            Q4_K/Q6_K   0.00003 of range
  nemotron_h     Nemotron-Nano-9B-v2 Q4_K_M   Q8_0/Q4_K   0.00001
  granitehybrid  granite-4.0-h-micro Q4_K_M   Q4_K/Q6_K   0.00005

qwen3moe was run for the record and stays opt-in: Qwen3-30B-A3B keep96
passes the teacher-forced bounds (2 flips of 64, 0.00250 of range, some
fifty times the dense rows) and FAILS the free-running arm, diverging
at token 20. That is the MoE amplification the owner's decision was
about, now measured. afmoe has no row: its router has no device kernel
and the model runs on the CPU.
qwen4exp PROMOTED 2026-10-06 (experimental family) on the release
file Qwen3.8-Flash-Next GSQ-RCO IQ3_S with the experts on the host
(--cpu-moe, the only placement whose dense half fits the 24 GB slice):
forced on against forced off, token-identical on the five
kernel-verify prompts at 32 and at 64 tokens, and byte-identical
greedy-32 on the 481-token prompt and greedy-64 on the 3,675-token
prompt (docs/qwen4exp-admission-evidence/). The promotion covers what
the TC prefill touched there: of the dense projections (Q5_K, Q3_K,
IQ4_XS, BF16 mixers) only IQ4_XS took it; Q5_K and Q3_K have
tensor-core kernels since 2026-10-21, promoted elsewhere but held off
here (above) until a row on this file exists, RUNNER_CUDA_TC=1 forcing
them.
With expert layers on the device the codebook
expert GEMMs would join the TC path under the same arch flag; that
placement was measured only through the eager route today, so a
device-expert row is still owed before it is called covered (R4.26.8).
```

## Architectures promoted for the types their row exercised

```text
R3.39.2 (a): promoted for exactly the types its row's file exercised, on
a float32-reference row (2,000 positions of gold-corpus-2k, chunked
prefill, TC against the default route) and a speed row.
  spark2_5  XHToken Spark-X2.5-4B Q4_K_M (Q4_K, Q6_K), against
            XHToken/Spark-X2.5-4B@0bcb3567 in float32 (transformers 4.57.1,
            the version its code targets): mean KLD ratio 0.9995, CI
            [0.9981, 1.0007], Wilcoxon p 0.115; margin-qualified top-1
            95.01% against 94.95%; prefill 203 -> 1,927 tok/s (9.48x)
  clef      ggml-org Clef-Flash Q4_K_M (Q8_0, Q4_K), against
            Cloudflare/clef-flash@fb4417d3 in float32: mean KLD ratio
            1.0001, CI [0.9991, 1.0011], Wilcoxon p 0.452; margin-qualified
            top-1 95.13% against 94.99%, McNemar 2 to 0 for the tensor
            cores (p 0.5); prefill 102 -> 1,566 tok/s (15.4x). Within the
            obligation below (mean KL within 1.01x, not worse at p < 0.05).
k2-horizon and stablelm run on the CPU on a CUDA build (grouped RMSNorm,
LayerNorm), so they have no device route to promote.
```

## Tiled prefill attention

The comment above `attn_tile512_on`.

```text
RUNNER_CUDA_ATTN_TILE: the tiled prefill attention (k_attn_tile*, C2
step 1), on by default since its float32-reference rows (2,000 positions
at 4,000-6,000 tokens of context; Qwen3-4B, Mistral-7B, and gpt-oss-20b
with its sinks and alternating window) were never further from the
reference than k_attn's; 0 keeps k_attn. A tolerance route: recorded as
attn-tile, and --gpu-route reference runs k_attn.
RUNNER_CUDA_ATTN_TILE512: the tiled prefill attention also takes head
widths 257-512 (gemma-4's global layers, k_attn_tile8), on by default since
its long-context rows passed the owner's rule against k_attn (2026-10-22,
the chat corpus at positions 4,000 to 6,000: gemma-4-26B-A4B 0.31995 vs
0.32108, Wilcoxon p 0.72; the dense gemma-4-E2B 0.14722 vs 0.14714, p
0.41); 0 keeps k_attn for those layers.
```

## Tensor-core prefill attention

The comment above `attn_tc_family`.

```text
RUNNER_CUDA_ATTN_TC: the tensor-core prefill attention (k_attn_tc, C2
step 2) for fp16 caches, ahead of the tiled kernel. With Q and P as single
halves its Mistral-7B float32-reference row read 0.19% further from the
reference than k_attn's (Wilcoxon p 0.014); the kernel carries both in two
halves (X23b), which reads the same as k_attn there (p 0.60).
On by default for the families whose long-context rows (positions 4,000 to
6,000 against the publisher's float32 model) passed the owner's rule
against the current route, attn-tile (2026-10-21): llama (Llama-3.2-3B
0.03833 vs 0.03832, Wilcoxon p 0.76; Mistral-7B 0.02069 vs 0.02068, p
0.43), qwen2 (Qwen2.5-3B 0.03644 vs 0.03643, p 0.47), qwen3 (Qwen3-4B
0.11199 vs 0.11197, p 0.14) and granite (granite-4.2-8b 0.01106 vs
0.01126, p 0.87), about 3x the tile's prefill at 8K; and (2026-10-22)
qwen3moe (Qwen3-30B-A3B 0.07584 vs 0.07580, p 0.86) and phi3
(Phi-3.5-mini 0.07912 vs 0.07912, p 0.12; Phi-4-mini 0.13998 vs 0.13996,
p 0.21), whose published 262144 window over a 131072 context is no window
(bind_swa_default). gpt-oss failed its row (0.04275 vs 0.04205, p 0.017;
with only its full-attention layers on the tensor cores, 0.04065 vs
0.04039, p 3e-18) and the windowed files (gemma, a windowed phi3 or llama
file) have none, so they keep the tile; 1 / 0 force it on / off everywhere.
```

## Wide-head tensor-core prefill attention

The comment above `tcw_mla_vlo`.

```text
RUNNER_CUDA_ATTN_TC_WIDE: the tensor-core prefill attention for head
widths 256 to 512 (k_attn_tcw, R3.6.3.1), where k_attn_tc does not reach
and the tile's 8- and 16-row kernels run on the CUDA cores. On by default
for gemma-4's E-series (gemma4 with per-layer embeddings: Nemotron 3.5
Bolt E4B, gemma-4-E2B/E4B), whose 512-wide global layers it takes from
k_attn_tile8 (owner decision 2026-11-03), and for qwen35 at head width 256
(Qwen3.5/3.6/3.8: every fourth layer full attention; 1.82x prefill at 16K
on Qwen3.8-27B with the D1 rule passing, R3.6.3.2), and for deepseek2's
absorbed row through k_attn_tcw_mla (R3.32: 1.05x / 1.11x the naive form's
prefill at 4K / 8K on DeepSeek-V2-Lite-Chat Q6_K, and its D1 read against
the publisher's float32 model passing, mean KL 0.986x); 0 keeps the tile
there, 1 forces it on every model it admits.
The absorbed MLA row k_attn_tcw_mla takes (R3.32), as the column its latent
starts at, or -1: one table (V is the K row, model_kv_v_shared), a rope
slice of whole 8-wide groups, quarters of the row that are whole k16 steps,
and a latent of whole 128-wide quarters up to 512 (DeepSeek-V2-Lite: 576 =
64 + 512), within the shared memory the kernel was granted (AW_SMEM(576)).
```
