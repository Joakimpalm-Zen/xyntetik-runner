# R3.29 D1 result: one-pass `k_attn_dec_gqa` against the two-pass kernel on Qwen3-0.6B (lab, 2026-11-03)

**Verdict: passes the attn-dec rule (2026-10-20).** Side a is main 853e478c (the two-pass kernel), side b is branch
`r3.29/cuda-decode-attn` at ff613375 (one-pass). Over 2,000 positions (4,000 to 5,999) against the publisher's
float32 reference:

| Rule | a (two-pass) | b (one-pass) | Reading | Result |
|---|---|---|---|---|
| Mean KL at most 1.01x | 0.0055037 | 0.0055070 | b/a = 1.0006 | pass |
| Paired Wilcoxon on per-position KL not worse at p < 0.05 | | | z 0.14, p 0.890 | pass |
| McNemar on margin-qualified top-1 not worse | 1,432 of 1,432 | 1,432 of 1,432 | discordant 0 / 0, p 1.0 | pass |

The scorer's verdict is "not distinguishable". Top-1 agreement with the reference is 97.50% (a) and 97.45% (b), and
the max KL is 0.123 on both. No position differs between the sides by more than a nat. All 2,000 positions are
matched by token id, with no dropped mass and no special prefix token: the score files and the publisher's
tokenizer agree on all 8,008 ids.

## How it was read

- **Reference:** `Qwen/Qwen3-0.6B` at c1899de289a0, `Qwen3ForCausalLM` loaded in float32 (no `--upcast-at-use`, no
  offload). Its `model.safetensors` (sha256 f47f71177f32bcd1..., 1,503,300,328 bytes) was checked against the Hub's LFS
  digest as it was fetched. The 6,001-token forward took 29 s, and the process peaked at 7.3 GiB resident.
- **Environment:** python 3.12.3, torch 2.11.0+cu130 (CPU), transformers 5.14.1, tokenizers 0.22.2; CPU only, x86-64.
- **Scorer:** `scripts/gold-logits.py` from runner main 853e478c (sha256 b9f1e484958ceab5...), corpus
  `tests/fixtures/gold-corpus-2k.txt` (sha256 56e9fcba658f72ae...).
- **Command,** exactly as in the README:

      python scripts/gold-logits.py --hf <Qwen3-0.6B at c1899de2> --corpus tests/fixtures/gold-corpus-2k.txt \
        --score-a score-main-853e478c.json --score-b score-r329-ff613375.json --solo-ok \
        --min-prefix 4000 --max-positions 2000 --stride 1 --threads 4 --out gold-main-vs-r329.json

- Both sides are solo score files of one model file, so `--solo-ok` names the deliberate decode-route comparison
  (`score_paths`: a solo, b solo).

## Files

- `gold-main-vs-r329.json`: the scorer's output. The parent's directory is replaced by the repo at its revision, and
  the score files' directory by their base names.
