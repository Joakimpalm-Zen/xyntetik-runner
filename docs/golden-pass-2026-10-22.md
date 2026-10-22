# The golden pass on the 1.5.0 release, 2026-10-22

The README's fidelity claim rests on one measurement: each family's smallest
member scored against the publisher's own implementation in float32, with
llama.cpp scored beside Runner on the same file. This page reruns that
measurement on the published 1.5.0 release and corrects the claim to what
it supports.

Same corpus (`tests/fixtures/gold-corpus-2k.txt`), same 2,000 positions,
same llama.cpp revision (73a43d1; flash attention off at bf16, on at the
served quantization) and the same 22 rows as the 2026-09-07 rerun, plus
Apertus-8B at bf16, whose 09-07 row failed to start. Runner is the
published `runner-linux-x86_64` asset of 1.5.0, checked against the
release's SHA256SUMS, on the CPU. Each row is also scored against the
published 1.4.2 asset, so the release's own effect on fidelity is visible.
Summaries: [`golden-pass-evidence/gold-2k/summaries-2026-10-22.json`](golden-pass-evidence/gold-2k/summaries-2026-10-22.json);
the script that ran: [`gold1022-run.sh`](golden-pass-evidence/gold-2k/gold1022-run.sh).

## The rows where 1.5.0 does not read closer

They come first, because they are the rows the claim has to survive.
Mean KL is the divergence from the reference's distribution, averaged
over the positions (lower is closer). Margin-qualified top-1 is the
"chosen token" measure: at positions where the reference's top token leads
its second by more than 0.5 nats of log-probability, the share where the
engine picks the same token. McNemar tests that measure, Wilcoxon the paired
per-position KL.

| row | 1.5.0 mean KL | llama.cpp mean KL | 1.5.0 margin-qualified top-1 | llama.cpp margin-qualified top-1 | McNemar p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| gemma-4-E2B, served (q4_0) | 0.8126 | 0.7910 | 69.44% | 68.88% | 0.37 | 1.8e-09 | llama.cpp closer |
| Qwen3-0.6B, served | 0.3041 | 0.3046 | 81.33% | 81.25% | 1 | 0.096 | not distinguishable |
| Trinity Nano, bf16 | 0.0460 | 0.0453 | 98.60% | 98.60% | 1 | 0.28 | not distinguishable |
| Trinity Nano, served | 0.0626 | 0.0610 | 97.43% | 97.80% | 0.51 | 0.0023 | 1.5.0 closer, behind on the chosen token |

**gemma-4-E2B, served.** llama.cpp reads closer, as it did on 09-07
(0.8134 against 0.8016). The 09-07 files were no longer on the host, so
the bf16 file was converted again from the publisher's checkpoint with the
same llama.cpp checkout and the served file written from it by Runner's
`--quantize --quant q4_0`; both engines read that one file. It is q4_0
throughout, the token embedding and the per-layer embedding tables
included, and it is the most damaged file in the set: the reference's
chosen token survives at 69% of positions. At bf16 Runner is the closer
engine (mean KL 4.1e-5 against 5.7e-4), so the architecture itself is
implemented faithfully.

The cause is that llama.cpp computes a different model than the file
holds. To show it, each engine was scored a second time against the
file's own model: the publisher's float32 forward with every one of the
278 quantized text-model matrices replaced by the file's values,
dequantized exactly
([`goldref-bf16.py`](golden-pass-evidence/gold-2k/goldref-bf16.py), mode
`gguf`; same corpus and 2,000 positions). Against that model Runner reads
a mean KL of 0.000069 with the chosen token kept at 100.00% of positions.
llama.cpp reads 0.0629 with flash attention off and 0.0625 with it on
(the served row's setting), the chosen token kept at 90%, and at 49
qualified positions it picks another token where Runner never does
(McNemar p 4e-15). On this one heavily damaged file, llama.cpp's departure
from the file happens to land nearer the publisher's model; Runner reads
the file as written. The departure is
consistent with the engines' known arithmetic difference (llama.cpp
rounds activations to 8 bits before its quantized dot products; Runner
keeps them in float32), but it was not isolated further. Keeping one group
of tensors in bf16 at a time agrees: the gap moves in both directions
(Runner 1.005x llama.cpp's mean KL with the token embedding unquantized,
1.072x with the attention matrices, 0.993x with the per-layer embeddings,
0.990x with the FFN matrices), which rules out a single faulty part of
Runner.

**Trinity Nano.** A mixture-of-experts model. On the served file five
positions carry 193% of llama.cpp's mean-KL advantage; with the 1% most
divergent positions set aside Runner is the closer engine (mean difference
-0.0020), it is further at 918 of the 2,000 positions, and the median
position favours it. That is the signature of expert routing flipping at a
handful of positions near a tie, which the mean is sensitive to and the
Wilcoxon test is not. The bf16 row was not distinguishable on 09-07 either.

**Qwen3-0.6B, served.** 1.5.0 reads exactly as 1.4.2 did (0.3041); the
09-07 verdict was "Runner closer" against a llama.cpp reading of 0.3061,
and the corrected scorer (below) reads llama.cpp at 0.3046 on the same
file, inside the test's reach.

## What the README said, and what the 09-07 table supported

The README said Runner was closer than llama.cpp on 21 of 22 rows and level
or ahead on the reference's chosen token on 20 of 22. The 09-07 table
supports 20 of 22 by its own Wilcoxon verdicts (gemma-4-E2B served read
llama.cpp closer, Trinity Nano bf16 not distinguishable), not 21; the
chosen-token count, 20 of 22, was right.

The scorer itself also had a defect, fixed in
[#354](https://github.com/Joakimpalm-Zen/xyntetik-runner/pull/354) before
this rerun: it read llama.cpp's top-1 from the token its server returned,
which is chosen after the server's default repeat penalty, not from the
distribution itself. That moved llama.cpp's chosen-token column (and
nothing in the KL column). Every number on this page uses the fixed
scorer.

## The 1.5.0 rows against llama.cpp

Over all 23 rows Runner 1.5.0 is closer to the publisher's reference by
the Wilcoxon verdict on 20, level or ahead on the reference's chosen token
on 22, and has the lower mean KL on 20. On the 22 rows the 09-07 table
measured: 19 closer, 21 level or ahead.

| row | 1.5.0 mean KL | llama.cpp mean KL | 1.5.0 margin-qualified top-1 | llama.cpp margin-qualified top-1 | McNemar p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B, served | 0.3041 | 0.3046 | 81.33% | 81.25% | 1 | 0.096 | not distinguishable |
| Trinity Nano, bf16 | 0.0460 | 0.0453 | 98.60% | 98.60% | 1 | 0.28 | not distinguishable |
| Trinity Nano, served | 0.0626 | 0.0610 | 97.43% | 97.80% | 0.51 | 0.0023 | 1.5.0 closer |
| gemma-4-E2B, served | 0.8126 | 0.7910 | 69.44% | 68.88% | 0.37 | 1.8e-09 | llama.cpp closer |
| SmolLM2-135M, bf16 | 1.4e-4 | 1.7e-3 | 100.00% | 100.00% | 1 | 1.1e-306 | 1.5.0 closer |
| SmolLM2-135M, served | 7.5e-3 | 0.0147 | 100.00% | 100.00% | 1 | 6.1e-219 | 1.5.0 closer |
| Qwen2.5-0.5B, bf16 | 2.3e-4 | 1.4e-3 | 100.00% | 100.00% | 1 | 1.4e-299 | 1.5.0 closer |
| Qwen2.5-0.5B, served | 0.0471 | 0.0509 | 98.96% | 98.81% | 0.75 | 1.5e-46 | 1.5.0 closer |
| Qwen3-0.6B, bf16 | 1.2e-4 | 7.3e-4 | 100.00% | 100.00% | 1 | 4.3e-311 | 1.5.0 closer |
| Qwen3.5-0.8B, bf16 | 2.8e-5 | 5.6e-4 | 100.00% | 100.00% | 1 | 8.5e-322 | 1.5.0 closer |
| Qwen3.5-0.8B, served | 0.0632 | 0.0690 | 97.71% | 96.98% | 0.052 | 6.9e-38 | 1.5.0 closer |
| StableLM 2 1.6B, bf16 | 2.3e-4 | 8.9e-4 | 100.00% | 100.00% | 1 | 1.5e-286 | 1.5.0 closer |
| granite-4.0-h-micro, bf16 | 8.3e-5 | 7.5e-4 | 100.00% | 100.00% | 1 | 8.7e-320 | 1.5.0 closer |
| granite-4.0-h-micro, served | 0.0956 | 0.1039 | 94.36% | 94.36% | 1 | 2.9e-36 | 1.5.0 closer |
| granite-4.2-3B, bf16 | 1.1e-4 | 7.0e-4 | 100.00% | 99.93% | 1 | 1.1e-301 | 1.5.0 closer |
| granite-4.2-3B, served | 0.1623 | 0.1702 | 91.25% | 90.97% | 0.63 | 2.8e-17 | 1.5.0 closer |
| gemma-3-4B, bf16 | 4.3e-5 | 4.6e-4 | 100.00% | 100.00% | 1 | 2.7e-312 | 1.5.0 closer |
| gemma-3-4B, served | 0.0670 | 0.0772 | 96.19% | 95.48% | 0.071 | 3.7e-33 | 1.5.0 closer |
| gemma-4-E2B, bf16 | 4.1e-5 | 5.7e-4 | 100.00% | 100.00% | 1 | 5.4e-322 | 1.5.0 closer |
| Qwen3-4B, bf16 | 4.2e-5 | 3.6e-4 | 100.00% | 100.00% | 1 | 4e-318 | 1.5.0 closer |
| Qwen3-4B, served | 0.1399 | 0.1455 | 92.08% | 91.89% | 0.66 | 5.3e-11 | 1.5.0 closer |
| Apertus-8B, bf16 | 1.1e-4 | 7.4e-4 | 100.00% | 100.00% | 1 | 1.8e-304 | 1.5.0 closer |
| Apertus-8B, served | 0.0508 | 0.0773 | 98.10% | 96.42% | 0.00043 | 2e-165 | 1.5.0 closer |

## The 1.5.0 rows against 1.4.2

The release's work on routes and kernels left fidelity where it was, to
within 0.1% of mean KL. On 16 of the 23 rows 1.5.0 and 1.4.2 give identical answers at every
position: every bf16 row and the served q4_0 and Q8_0 files. The seven
rows that differ are exactly the Q4_K_M files, and on them mean KL agrees
within 0.0001 and the chosen token within 0.1 point. In 1.5.0 the CPU's
K-quant dot products keep one accumulator per row, which adds the same
float32 terms in a different order (3a8499fe; bisected on another host's
Q4_K_M file to this commit alone). That reordering moves the per-position
KL by tiny amounts in either direction, and at 2,000 positions the paired
test can resolve shifts that small: four of
the seven rows read "1.4.2 closer" by a Wilcoxon p between 0.002 and 0.049
at equal mean KL. A same-compiler pair (the 1.5.0 tree, and the same tree
with only the reordering reverted) measures the reordering alone, on the
first 9,500 characters of the same corpus. Its effect is real at this
power and its sign depends on the file: the reordered build reads closer
on granite-4.0-h-micro (p 0.034), further on Apertus-8B (mean KL 0.05077
against 0.05072, p 0.0036), and not distinguishable on granite-4.2-3B and
gemma-3-4B. In size it is at most 0.1% of mean KL, and on none of the four
files did it change the chosen token at a single qualified position.

| row | 1.5.0 mean KL | 1.4.2 mean KL | 1.5.0 margin-qualified top-1 | 1.4.2 margin-qualified top-1 | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|
| SmolLM2-135M, bf16 | 1.4e-4 | 1.4e-4 | 100.00% | 100.00% |  | identical at every position |
| SmolLM2-135M, served | 7.5e-3 | 7.5e-3 | 100.00% | 100.00% |  | identical at every position |
| Qwen2.5-0.5B, bf16 | 2.3e-4 | 2.3e-4 | 100.00% | 100.00% |  | identical at every position |
| Qwen2.5-0.5B, served | 0.0471 | 0.0472 | 98.96% | 98.96% | 0.6 | not distinguishable |
| Qwen3-0.6B, bf16 | 1.2e-4 | 1.2e-4 | 100.00% | 100.00% |  | identical at every position |
| Qwen3-0.6B, served | 0.3041 | 0.3041 | 81.33% | 81.33% |  | identical at every position |
| Qwen3.5-0.8B, bf16 | 2.8e-5 | 2.8e-5 | 100.00% | 100.00% |  | identical at every position |
| Qwen3.5-0.8B, served | 0.0632 | 0.0632 | 97.71% | 97.71% | 1 | not distinguishable |
| StableLM 2 1.6B, bf16 | 2.3e-4 | 2.3e-4 | 100.00% | 100.00% |  | identical at every position |
| granite-4.0-h-micro, bf16 | 8.3e-5 | 8.3e-5 | 100.00% | 100.00% |  | identical at every position |
| granite-4.0-h-micro, served | 0.0956 | 0.0956 | 94.36% | 94.36% | 0.013 | 1.4.2 closer |
| granite-4.2-3B, bf16 | 1.1e-4 | 1.1e-4 | 100.00% | 100.00% |  | identical at every position |
| granite-4.2-3B, served | 0.1623 | 0.1623 | 91.25% | 91.18% | 0.049 | 1.4.2 closer |
| Trinity Nano, bf16 | 0.0460 | 0.0460 | 98.60% | 98.60% |  | identical at every position |
| Trinity Nano, served | 0.0626 | 0.0626 | 97.43% | 97.43% |  | identical at every position |
| gemma-3-4B, bf16 | 4.3e-5 | 4.3e-5 | 100.00% | 100.00% |  | identical at every position |
| gemma-3-4B, served | 0.0670 | 0.0670 | 96.19% | 96.19% | 0.0064 | 1.4.2 closer |
| gemma-4-E2B, bf16 | 4.1e-5 | 4.1e-5 | 100.00% | 100.00% |  | identical at every position |
| gemma-4-E2B, served | 0.8126 | 0.8126 | 69.44% | 69.44% |  | identical at every position |
| Qwen3-4B, bf16 | 4.2e-5 | 4.2e-5 | 100.00% | 100.00% |  | identical at every position |
| Qwen3-4B, served | 0.1399 | 0.1399 | 92.08% | 92.08% | 0.11 | not distinguishable |
| Apertus-8B, bf16 | 1.1e-4 | 1.1e-4 | 100.00% | 100.00% |  | identical at every position |
| Apertus-8B, served | 0.0508 | 0.0507 | 98.10% | 98.10% | 0.0019 | 1.4.2 closer |

## Method notes

- The reference is transformers 5.16 in float32 on the CPU. For Apertus-8B
  the host's memory limit on the day (32 GiB) could not hold the float32
  model beside two servers, so its reference was held as the checkpoint's
  bfloat16 weights, widened to float32 at the moment of use (every other
  parameter float32;
  [`goldref-bf16.py`](golden-pass-evidence/gold-2k/goldref-bf16.py)).
  The checkpoint is bfloat16, so the widened values are the float32
  model's values; checked bitwise on a truncated model (192
  positions, identical logits) and on the full model (its bf16 row read the
  same summary in both passes, one with each reference).
- The scorer has matched tokens by id since after the 09-07 rerun (09-07
  matched by spelling), so this page compares verdicts with 09-07, not
  absolute numbers.
- Each pass starts fresh servers; Runner's answer for a row is identical
  across the two passes wherever both scored the same positions (the
  llama.cpp side dropped one position on two rows of the first pass).
