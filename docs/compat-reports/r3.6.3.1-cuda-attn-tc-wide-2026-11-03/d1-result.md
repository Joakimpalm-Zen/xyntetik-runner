# R3.6.3.1 D1 result: `k_attn_tcw` against the tile on Nemotron 3.5 Bolt E4B (lab, 2026-11-03)

**Verdict: passes the attention routes' rule (2026-10-20).** Side a is main 340fffa0 (the tile), side b is branch
`r3.6.3.1/cuda-attn-tc-wide` at 1a2d85ee (`k_attn_tcw`). Over 2,000 positions (4,000 to 5,999) against the
publisher's float32 reference:

| Rule | a (tile) | b (`k_attn_tcw`) | Reading | Result |
|---|---|---|---|---|
| Mean KL at most 1.01x | 0.14269 | 0.14220 | b/a = 0.9965 | pass |
| Paired Wilcoxon on per-position KL not worse at p < 0.05 | | | z 1.32, p 0.188 | pass |
| McNemar on margin-qualified top-1 not worse | 1,607 of 1,718 | 1,605 of 1,718 | discordant 2 (a only) / 0 (b only), p 0.5 | pass |

The scorer's verdict is "not distinguishable". Top-1 agreement with the reference is 87.6% on both sides; the max KL
is 5.58 (a) and 5.60 (b), and one position differs between the sides by more than a nat (a the worse there). All
2,000 positions are matched by token id, with no dropped mass. The mean KL of 0.14 on both sides is the Q4_K_M file
against its float32 parent; the read compares the two kernels, both on that file.

## How it was read

- **Reference:** `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` at 49843eed8db1, loaded in float32 (it fits, so there is no
  `--upcast-at-use`): `Gemma4ForConditionalGeneration`, driven with text. The index maps 2,076 tensors to
  `model-00001-of-00001.safetensors` (sha256 15006dbbfc99dbde..., 15,882,494,698 bytes) and 54 to
  `model-frozen-shared-kv.safetensors` (sha256 32aaa38402af5a51..., 110,118,048 bytes), and both were loaded.
  Every file was checked against the Hub's LFS digest as it was fetched.
- **Environment:** python 3.12.3, torch 2.11.0+cu130 (CPU), transformers 5.14.1, tokenizers 0.22.2; CPU only, x86-64.
- **Scorer:** `scripts/gold-logits.py` from runner main 340fffa0 (sha256 b9f1e484958ceab5...), corpus
  `tests/fixtures/gold-corpus-2k.txt` (sha256 56e9fcba658f72ae...).
- **Command** (the README's, with one addition):

      python scripts/gold-logits.py --hf <Bolt-E4B-BF16 at 49843eed> --corpus tests/fixtures/gold-corpus-2k.txt \
        --score-a score-main-340fffa0.json --score-b score-r3631-1a2d85ee.json --force-bos \
        --min-prefix 4000 --max-positions 2000 --stride 1 --out gold-main-vs-r3631.json

- **`--force-bos`:** the score files' token ids start with the GGUF's BOS (id 2), and the publisher's loaded
  tokenizer adds none to a bare text. Without the flag the scorer stops at index 0 (file `[2, 6974, 496, 9813]`,
  reference `[6974, 496, 9813, 236764]`). gold-logits.py documents the same case on Gemma 4 E2B (2026-09-07).
  With it, the reference sees the same prefix as both sides (one special prefix token, forced); the forward ran over
  6,002 tokens.
- Both sides were scored chunked (`paths.keep` on a and b), so `--solo-ok` was neither needed nor given.

## Files

- `gold-main-vs-r3631.json`: the scorer's output. The parent's directory is replaced by the repo at its revision,
  and the score files' directory by their base names.
