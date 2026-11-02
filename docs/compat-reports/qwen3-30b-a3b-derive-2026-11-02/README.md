# Qwen3-30B-A3B made by `runner derive`, against Qwen's float32 model (2026-11-02)

**Unflattering first: the derive file is slightly further from the float32 model than our hand-built selective file.**
Its mean KLD is 0.0290 against 0.0279 (paired Wilcoxon p 0.02), and it is 0.88 GB smaller. Margin-qualified top-1 is a
tie: 99.01% against 98.82%, 13 against 10 discordant positions (McNemar p 0.68). Qwen's official Q4_K_M fails the
bar in the same window: 0.0909 mean KLD and 94.80% margin-qualified top-1, against a bar of 0.05 and 97%.

## The file

`runner derive -m Qwen3-30B-A3B-Q8_0.gguf --target-size 17106040320 --high q4_0 --low q3_k --match _exps --imatrix
Qwen_Qwen3-30B-A3B.imatrix` on Qwen's own Q8_0, with bartowski's importance matrix. The planner demoted 11 of 48 expert
groups to q3_k, saving 882,376,704 bytes. Its plan is R4.7.3's `plan-sens.json`, and the file is R4.7.3's K=11 file byte
for byte: 17,106,040,320 bytes, sha256 `9f71dfa02ffaddc0dd8219b054823df8f4c883f528d87d4b5904f3220fc3dffd`. derive wrote
its card from its record (`runner derive --card-from`).

## Against the reference

Reference: `Qwen/Qwen3-30B-A3B` at `ad44e777`, float32 arithmetic with the BF16 weights upcast at use
(`scripts/gold-logits.py --upcast-at-use`), on the CPU. 2,000 positions at stride 2 over
`tests/fixtures/gold-corpus-2k.txt`, all three files matched to the reference by token id at every position. Runner's
sides are solo `--score --score-top 20` files on the CPU (runner at 2c54cf27).

| File | Size | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 | Positions above 0.1 |
|---|---|---|---|---|---|---|---|
| Qwen's official Q4_K_M | 18,556,685,824 | **0.0909** (fails) | 0.0467 | 1.53 | 85.65% | 94.80% (fails) | 570 |
| our selective file, by hand | 17,988,417,024 | **0.0279** | 0.0128 | 3.22 | 93.00% | 98.82% | 90 |
| `runner derive` | 17,106,040,320 | **0.0290** | 0.0134 | 1.18 | 92.90% | 99.01% | 109 |

- **derive against the official Q4_K_M:** closer by the paired per-position KLD (Wilcoxon p 2e-235) and on the
  margin-qualified top-1 (73 positions to 9, McNemar p 1e-13).
- **derive against the selective file:** the selective file is closer on mean KLD (p 0.02); margin-qualified top-1 is a tie.
- R4.7.3 read 0.0292 for the same bytes in the same window. Its Runner side was scored on another route (the record
  does not name it); this record's sides are solo scores.

## Files

- `three-way.json`: the numbers the README's result graphic is drawn from (`scripts/readme-svg-derive.py`).
- `gold-logits-derive.json`: derive's reference run, per position.
- `gold-logits-official-q4_k_m-selective.json`: the other two files' reference run, per position.
