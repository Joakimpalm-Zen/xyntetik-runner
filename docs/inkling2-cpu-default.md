# Inkling 2 Mini's fast CPU default

**Limits first:** on the measured 16-thread CPU path, prefill at 512 tokens
is 52.7 tok/s against llama.cpp's 75.5 (0.70x); decode is 11.77 against
12.154 tok/s (0.968x). The fast route follows the publisher's greedy sequence
on 99 of 100 Claim 1 prompts, not all 100. These are dated measurements on
one file and build, not task-quality scores or measurements of a future
release artifact. The CUDA route is a separate [opt-in](inkling2-cuda.md).

## File, build and reference

R4.48.24 run 3 completed on 2026-11-25 and measured
`7bf8b64db20c5dc1ff2b706607517c8e42d04940`, after
the CPU fast route restored scalar `ink2_swiglu`. The prescribed native
build and its STRICT build were byte-identical. The measurements below are
not claimed for a different compiler recipe or instruction set.

The file is `unsloth/Inkling-2-Mini-GGUF` at
`45fd08764f2b0bf16b03a5483a80e5c7aaec8772`,
`Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`, 22,867,452,896 bytes,
SHA256 `2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68`.
The publisher checkpoint is `thinkingmachines/Inkling-2-Mini` at
`b85c3cfa615fc413cc9af3379f491b963f3b586a`. The reference runs the publisher's
transformers model code at `6173a2f6` (identical model code at `046bdb8`),
CPU float32, eager attention, torch 2.14.0+cpu and four threads. Weights are
dequantized/cast to float32 at use; the publisher's activation quantizer is
off. Runner's fast and reference routes use one binary, 16 threads.

## The decision and the earlier failures

Run 3 passed the bar fixed before its release-build reads: on each of two
fresh 2,000-position reads, fast mean KL must be at most twice reference
mean KL, and margin-qualified top-1 must be 100% (publisher margin above
0.5 nats). The 8-bit activation control must fail on both reads. The greedy
count must not be significantly lower by the predeclared paired rule;
thread identity, a decode speed floor of 0.5x llama.cpp, file smokes and
protocol/route checks must also pass. The verdict was **PASS**, making fast
the CPU default. `RUNNER_INKLING2_ROUTE=reference` keeps the serial reference
route; training loads use that reference route.

The earlier R4.48.19 bar still failed on both of its reads: fast chat mean
KL 8.906e-06 against the publisher's one-ISA-step envelope 6.908e-13, raw
3.620e-12 against 1.379e-12, with 100% margin-qualified top-1 on both. The
reference route also failed that envelope. That result is not replaced.
R4.48.24 used a new pre-registered bar and fresh reads. Its runs 1 and 2
then failed the chat rule; run 3 passed after restoring scalar SwiGLU.
Do not compare mean KL across those different corpora as an improvement.

| Fresh read / route | Mean KL vs publisher float32 [95% interval] | Plain top-1 | Margin-qualified top-1 |
|---|---|---|---|
| Chat, fast | 3.311354e-05 [5.136e-06, 8.251e-05] | 1999/2000 | 1707/1707 |
| Chat, reference | 3.311425e-05 [5.136e-06, 8.251e-05] | 1999/2000 | 1707/1707 |
| Raw, fast | 2.438582e-12 [2.105e-12, 2.798e-12] | 2000/2000 | 1659/1659 |
| Raw, reference | 1.116652e-11 [9.936e-12, 1.242e-11] | 2000/2000 | 1659/1659 |

The live 8-bit activation control failed both: chat mean KL 1.556592e-02,
1697/1707 margin-qualified; raw 2.916316e-03, 1656/1659. No non-finite row
was reported. The fast chat rows were byte-identical at 1, 4 and 16 threads;
observed pool counters were serial at 1 and split over the requested 4/16.

## Claim 1: greedy sequence identity

Teacher-forced verification v2 checks every generated token, including the
stop token, against the publisher's greedy next token under that exact
prefix. It is not a human correctness assessment. The primary set has 25
chat, 25 code, 25 arithmetic and 25 short-tool-use prompts, up to 256 tokens.
Prompt encoding, completeness, exact ties and the verifier's step-0 control
are checked separately.

Fast follows 99/100 publisher sequences; reference follows 98/100. The
paired table is 98 both, one fast-only, zero reference-only and one neither.
The difference is +1 percentage point, Newcombe 95% interval
[-3.18, +5.78] percentage points; exact McNemar p = 1.0: a statistical tie,
not evidence that fast is better. The fast miss is c1_090 at token 88.
The short-tool-use reference category was regenerated after its first-five
check changed; the other categories retained their checked reference set.
No encoding discrepancy, incomplete sequence, exact-tie position or head
shape flag was reported. All three control sequences passed.

## Speed and the final rebase

The timed row's system load ranged from 21.43 to 71.59; fast-t16's block
had the highest mean, 66.52, against llama-t16's 48.48. Engines ran in
blocks, not interleaved repetitions. The counted attempt was the first
accepted residency run after refusals; all six start gates read 1.0
residency. The reference-t1 mid-row residency fell to 0.8166; that row did
not decide the 16-thread decode rule. The speed comparator was llama.cpp
`56eea9e4`. The 0.968x decode result passed the predeclared 0.5x floor; it is
not a faster-than-llama.cpp claim.

On the final rebased source at `1fb25712`, four fast row captures and the
100 Claim 1 answers' deterministic fields reproduced run 3. The 16-thread
decode reproduction was 12.05 against 12.217 tok/s (0.986x). Wall time and
signed receipts were excluded from the answer comparison. This rebase
check used the same native recipe; release-artifact gates remain separate.

The standalone Claim 2 (alone versus batched answers) and Claim 3 (the
model's reasoning loops) results are not inferred from these checks.
