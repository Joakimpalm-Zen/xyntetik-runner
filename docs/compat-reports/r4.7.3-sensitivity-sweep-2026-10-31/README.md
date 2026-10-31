# R4.7.3 follow-up: a sensitivity-ordered type plan against the hand rule at the same size, 2026-10-31

#505 added `--quant-error` and `scripts/type-plan-by-sensitivity.py`, which demote the expert groups that add the
least imatrix-weighted error per saved byte. This sweep sets that planner against the hand rule, "demote the deepest
blocks", at two sizes, on Qwen3-30B-A3B.

**Setup**
- Source file: Qwen3-30B-A3B Q8_0, rewritten with `--type-plan` and the imatrix.
- Every expert tensor is Q4_0, except K of the 48 blocks' expert groups, which are Q3_K.
- Plans (beside this file as `plan-k*-*.json`):
  - `deep` demotes blocks 48-K to 47;
  - `sens` demotes the K groups the planner picks. It picked blocks 0 to K-1 at both sizes.
- The two files of each K are the same size to the byte: 17,507,120,640 at K=6 and 16,704,960,000 at K=16.
- Measurement: chunked score files on the CPU (`RUNNER_SCORE_CHUNKED=1 --score 20`, `tests/fixtures/gold-corpus-2k.txt`)
  against transformers' float32 Qwen/Qwen3-30B-A3B @ad44e777bcd1, bf16 weights upcast at use (`--upcast-at-use`), at
  2,000 positions.

| K | plan | mean KL vs gold | max KL | margin-qualified top-1 |
|---|---|---|---|---|
| 6 | deep (hand rule) | 0.03310 | 1.301 | 98.82% |
| 6 | sensitivity | **0.02671** | **0.970** | 99.14% |
| 16 | deep (hand rule) | 0.03770 | 1.887 | 98.95% |
| 16 | sensitivity | **0.03102** | **1.217** | 98.82% |

**Readings**
- **K=6:** the sensitivity plan is closer by a paired Wilcoxon on per-position KL, z -12.7 (p 6e-37), 19% lower mean
  KL. McNemar is 8 : 13 (deep only : sensitivity only), p 0.38.
- **K=16:** again closer, z -12.2 (p 2e-34), 18% lower mean KL. McNemar is 15 : 13, p 0.85.
- **Top-1:** not distinguishable at either size.
- **What it means for a smaller file:** at the same bytes, the measured plan beats "demote the deepest". On this
  model the error per saved byte is lowest in the shallow blocks, which is the opposite of the rule's premise.
- **A note on the run:** K=6's comparison was OOM-killed on the first try, when both K values loaded the 30B
  reference at once, and was rerun alone. The scores are the original files.
