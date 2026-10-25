# Golden Pass: the 1.6.0 source, local Clang build, 2026-10-25

The README's fidelity claim rests on one measurement: each family's smallest member scored against the publisher's own implementation in float32, with llama.cpp scored beside Runner on the same file. This page reruns it on 1.6.0. Same corpus (`tests/fixtures/gold-corpus-2k.txt`), same 2,000 requested positions, same scorer (`scripts/gold-logits.py`, PR #354), same llama.cpp revision (73a43d1; flash attention off at bf16, on at the served quantization) and the same 23 rows as 2026-10-22, on the CPU. Runner is a clang build of `33c2a08f`, the v1.6.0 merge. Each row is also scored against the published 1.5.0 asset, the binary the 10-22 page scored, so the combined source/build difference is visible. Summaries: [`golden-pass-evidence/gold-2k/summaries-2026-10-25.json`](golden-pass-evidence/gold-2k/summaries-2026-10-25.json); the script that ran: [`gold1025-run.sh`](golden-pass-evidence/gold-2k/gold1025-run.sh).

## The row that moved against llama.cpp

It comes first, because it changes the README's count. On 10-22 Runner read closer to the reference than llama.cpp on 20 of 23 rows by the Wilcoxon verdict; on this local Clang build of 1.6.0 it is **19 of 23**. The row is **Trinity Nano, served** (Q8_0, a mixture-of-experts model): 1.6.0 reads mean KL 0.0645 against llama.cpp's 0.0610 (10-22: 1.5.0 0.0626, "1.5.0 closer" at p 0.0023), now "not distinguishable" (Wilcoxon p 0.14). On the chosen token it stays behind llama.cpp, as on 10-22 (97.65% against 97.80%). Against 1.5.0 itself the row reads "1.5.0 closer" (p 0.011), mean KL +3.0%. These are different build configurations; see the controlled comparison below.

Over all 23 rows 1.6.0 is closer than llama.cpp on 19, level or ahead on the reference's chosen token on 22, and has the lower mean KL on 20.

## The 1.5.0 asset as an instrument check

The published 1.5.0 asset, scored again beside 1.6.0, reproduces its own 10-22 mean KL to the last digit on all 23 rows. On 4 of them the raw summaries differ only because they cover different positions: in the pass against llama.cpp, llama.cpp's side drops one position on Qwen2.5-0.5B (position 722, both tiers) and granite-4.2-3B (813 at bf16, 347 served), on 10-22 as on 1.6.0, so the 10-22 means are over 1,999 or 1,983 positions and the pass against 1.5.0 is over 2,000 or 1,984. Over the common positions the two readings are equal. The per-position asset and llama.cpp readings match 10-22 exactly; see `golden-pass-evidence/gold-2k/position-audit.json`.

## The 1.6.0 rows against 1.5.0

On the 11 served rows 1.6.0 and 1.5.0 are not distinguishable on 8; 1.6.0 reads closer on 2 and 1.5.0 on 1 (Trinity Nano, above). On 10-22 every bf16 row was identical at every position between 1.5.0 and 1.4.2. On 1.6.0 none of the 12 bf16 rows is: 4 read "1.5.0 closer" by the Wilcoxon test and 8 are not distinguishable. The four significant bf16 differences are at roughly the 1e-4 scale of mean KL, with the reference's chosen token kept at every qualified position on those four rows. These are different build configurations; see the controlled comparison below.

## The 1.6.0 rows against llama.cpp

| row | 1.6.0 mean KL | llama.cpp mean KL | 1.6.0 margin-qualified top-1 | llama.cpp margin-qualified top-1 | McNemar p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M, bf16 | 1.4e-4 | 0.0017 | 100.00% | 100.00% | 1 | 4.8e-312 | 1.6.0 closer |
| SmolLM2-135M, served | 0.0075 | 0.0147 | 100.00% | 100.00% | 1 | 5.8e-218 | 1.6.0 closer |
| Qwen2.5-0.5B, bf16 | 1.5e-4 | 0.0014 | 100.00% | 100.00% | 1 | 2.5e-306 | 1.6.0 closer |
| Qwen2.5-0.5B, served | 0.0471 | 0.0509 | 98.96% | 98.81% | 0.75 | 4.7e-47 | 1.6.0 closer |
| Qwen3-0.6B, bf16 | 1.3e-4 | 7.3e-4 | 100.00% | 100.00% | 1 | 1.6e-306 | 1.6.0 closer |
| Qwen3-0.6B, served | 0.3040 | 0.3046 | 81.33% | 81.25% | 1 | 0.096 | not distinguishable |
| Qwen3.5-0.8B, bf16 | 2.7e-5 | 5.6e-4 | 100.00% | 100.00% | 1 | 4.9e-323 | 1.6.0 closer |
| Qwen3.5-0.8B, served | 0.0632 | 0.0690 | 97.71% | 96.98% | 0.052 | 5.1e-38 | 1.6.0 closer |
| StableLM 2 1.6B, bf16 | 2.4e-4 | 8.9e-4 | 100.00% | 100.00% | 1 | 9e-288 | 1.6.0 closer |
| granite-4.0-h-micro, bf16 | 8.2e-5 | 7.5e-4 | 100.00% | 100.00% | 1 | 4.7e-321 | 1.6.0 closer |
| granite-4.0-h-micro, served | 0.0956 | 0.1039 | 94.36% | 94.36% | 1 | 3.2e-36 | 1.6.0 closer |
| granite-4.2-3B, bf16 | 1.1e-4 | 7.0e-4 | 100.00% | 99.93% | 1 | 3.1e-302 | 1.6.0 closer |
| granite-4.2-3B, served | 0.1623 | 0.1702 | 91.25% | 90.97% | 0.63 | 2e-17 | 1.6.0 closer |
| Trinity Nano, bf16 | 0.0463 | 0.0453 | 98.68% | 98.60% | 1 | 0.29 | not distinguishable |
| Trinity Nano, served | 0.0645 | 0.0610 | 97.65% | 97.80% | 0.86 | 0.14 | not distinguishable |
| gemma-3-4B, bf16 | 2.9e-5 | 4.6e-4 | 100.00% | 100.00% | 1 | 5e-313 | 1.6.0 closer |
| gemma-3-4B, served | 0.0670 | 0.0772 | 96.19% | 95.48% | 0.071 | 3.4e-33 | 1.6.0 closer |
| gemma-4-E2B, bf16 | 4.4e-5 | 5.7e-4 | 100.00% | 100.00% | 1 | 1.2e-319 | 1.6.0 closer |
| gemma-4-E2B, served | 0.8128 | 0.7910 | 69.44% | 68.88% | 0.37 | 1.2e-9 | llama.cpp closer |
| Qwen3-4B, bf16 | 3.9e-5 | 3.6e-4 | 100.00% | 100.00% | 1 | 4.6e-321 | 1.6.0 closer |
| Qwen3-4B, served | 0.1399 | 0.1455 | 92.08% | 91.89% | 0.66 | 3.2e-11 | 1.6.0 closer |
| Apertus-8B, bf16 | 9.2e-5 | 7.4e-4 | 100.00% | 100.00% | 1 | 1.1e-309 | 1.6.0 closer |
| Apertus-8B, served | 0.0507 | 0.0773 | 98.10% | 96.42% | 0.00043 | 7.2e-166 | 1.6.0 closer |

## The 1.6.0 rows against 1.5.0

| row | 1.6.0 mean KL | 1.5.0 mean KL | 1.6.0 margin-qualified top-1 | 1.5.0 margin-qualified top-1 | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|
| SmolLM2-135M, bf16 | 1.4e-4 | 1.4e-4 | 100.00% | 100.00% | 0.13 | not distinguishable |
| SmolLM2-135M, served | 0.0075 | 0.0075 | 100.00% | 100.00% | 0.44 | not distinguishable |
| Qwen2.5-0.5B, bf16 | 1.5e-4 | 2.3e-4 | 100.00% | 100.00% | 0.85 | not distinguishable |
| Qwen2.5-0.5B, served | 0.0471 | 0.0471 | 98.96% | 98.96% | 0.58 | not distinguishable |
| Qwen3-0.6B, bf16 | 1.3e-4 | 1.2e-4 | 100.00% | 100.00% | 0.017 | 1.5.0 closer |
| Qwen3-0.6B, served | 0.3040 | 0.3041 | 81.33% | 81.33% | 0.98 | not distinguishable |
| Qwen3.5-0.8B, bf16 | 2.7e-5 | 2.8e-5 | 100.00% | 100.00% | 0.9 | not distinguishable |
| Qwen3.5-0.8B, served | 0.0632 | 0.0632 | 97.71% | 97.71% | 0.26 | not distinguishable |
| StableLM 2 1.6B, bf16 | 2.4e-4 | 2.3e-4 | 100.00% | 100.00% | 0.51 | not distinguishable |
| granite-4.0-h-micro, bf16 | 8.2e-5 | 8.3e-5 | 100.00% | 100.00% | 0.053 | not distinguishable |
| granite-4.0-h-micro, served | 0.0956 | 0.0956 | 94.36% | 94.36% | 0.0066 | 1.6.0 closer |
| granite-4.2-3B, bf16 | 1.1e-4 | 1.1e-4 | 100.00% | 100.00% | 0.0022 | 1.5.0 closer |
| granite-4.2-3B, served | 0.1623 | 0.1623 | 91.25% | 91.25% | 0.62 | not distinguishable |
| Trinity Nano, bf16 | 0.0463 | 0.0460 | 98.68% | 98.60% | 0.7 | not distinguishable |
| Trinity Nano, served | 0.0645 | 0.0626 | 97.65% | 97.43% | 0.011 | 1.5.0 closer |
| gemma-3-4B, bf16 | 2.9e-5 | 4.3e-5 | 100.00% | 100.00% | 0.0011 | 1.5.0 closer |
| gemma-3-4B, served | 0.0670 | 0.0670 | 96.19% | 96.19% | 0.012 | 1.6.0 closer |
| gemma-4-E2B, bf16 | 4.4e-5 | 4.1e-5 | 100.00% | 100.00% | 0.52 | not distinguishable |
| gemma-4-E2B, served | 0.8128 | 0.8126 | 69.44% | 69.44% | 0.41 | not distinguishable |
| Qwen3-4B, bf16 | 3.9e-5 | 4.2e-5 | 100.00% | 100.00% | 0.12 | not distinguishable |
| Qwen3-4B, served | 0.1399 | 0.1399 | 92.08% | 92.08% | 0.36 | not distinguishable |
| Apertus-8B, bf16 | 9.2e-5 | 1.1e-4 | 100.00% | 100.00% | 0.0013 | 1.5.0 closer |
| Apertus-8B, served | 0.0507 | 0.0508 | 98.10% | 98.10% | 0.076 | not distinguishable |

## Method notes

- Host: the lab box's CPU route (`--gpu off`), every server at `-t 16 -c 4096`, fresh servers per pass, the GPU hidden from every server. No CPU pinning (the host's cores are shared). Files read from local NVMe copies, each checked against the sha256 of its read.
- gemma-4-E2B is scored on the 10-22 pair itself (the bf16 converted by llama.cpp 73a43d1 and the q4_0 that 1.5.0 wrote from it), so its rows compare files, not conversions.
- The reference is transformers 5.16.1 in float32 on the CPU; for Apertus-8B, bfloat16 storage widened to float32 at use (`goldref-bf16.py`), as on 10-22.
- The published 1.6.0 asset was not scored: it does not start on this host (Ubuntu 24.04, glibc 2.39). It is built with GCC 15.2 and binds `sqrtf` at GLIBC_2.43; the 1.5.0 asset needs at most GLIBC_2.38 (GCC 13.3). The page scores a clang build of `33c2a08f` instead.
- The pass was paused once at a row boundary for a training run on the same host and resumed; finished rows were kept, and the remaining row ran on fresh servers as every row does.
- Side a (1.6.0) gives the same answer at every position in both passes of every row. On the four rows where llama.cpp drops a position, the passes' summaries differ by that one position, not by any answer.

## Controlled build findings and release status

A controlled comparison now confirms that **compiler and CPU target each change the score** on Qwen3-0.6B bf16 and Trinity Nano Q8_0. The original Clang native / Conda-sysroot build and published GCC x86-64-v3 build changed multiple factors. With GCC 13.3 and release flags held fixed, v1.5.0 and v1.6.0 give byte-identical complete score JSON on both tested corpus prefixes (1,529 and 1,438 positions). A local v1.5.0 release build is byte-identical to the published asset itself. These probes support build sensitivity and no source regression on those inputs; they do not prove all 23 rows source-equivalent or isolate the full 20-to-19 shift to compiler alone. The earlier bisect sampled eight commits in a 198-commit interval, not all 198; its bad endpoint tested identical, so its printed culprit is invalid. [Build controls, results and limitations](golden-pass-evidence/gold-2k/toolchain/README.md).

The **published 1.6.0 Linux asset has a confirmed compatibility regression**, requiring GLIBC_2.43 instead of the prior GLIBC_2.38 ceiling. A release-level fix pins Ubuntu 24.04 and checks imported symbol versions before packaging. A local GCC 13.3 / x86-64-v3 rebuild of the v1.6.0 tag passes startup and fixture inference under GLIBC 2.38 and 2.39. This is not a published replacement. The README count remains 20/23 only for the measured published v1.5.0 binary, 19/23 for this measured Clang build; the released v1.6.0 count is unknown. The five-row follow-up requested in the existing review is blocked on a corrected published asset; no v1.6.1 asset was publicly available at this check. No merge or release publication was performed.

## Source traceability

[Run provenance](golden-pass-evidence/gold-2k/run-provenance.json) records the exact source, binary/scorer/corpus hashes and reference package versions; [input-files.json](golden-pass-evidence/gold-2k/input-files.json) records read-back hashes of the staged model and HF files. The archived run script fetched main at start; it actually pinned `33c2a08fee8fbd08dd564396466c7c532df12cf2`. Replays must use that commit, not a future main.

Every table cell is generated from the matching summary in [summaries-2026-10-25.json](golden-pass-evidence/gold-2k/summaries-2026-10-25.json). Gemma E2B uses the explicit `gemma-4-e2b-it-forcebos-{bf16,served}` keys in both passes. [source-files.json](golden-pass-evidence/gold-2k/source-files.json) maps all 46 original result names and SHA256s to archived per-position JSON; only local paths/endpoints were sanitized, and numeric results were preserved. [position-audit.json](golden-pass-evidence/gold-2k/position-audit.json) checks every matched position against the October 22 raw files and verifies side-a repeatability across the two new passes. Scientific notation uses `1.4e-4`. No rows or unfavorable values were removed. [Document generator](golden-pass-evidence/gold-2k/make-doc-1025.py).
