# R4.32: Qwen3.8 GSQ-RCO IQ3_S against Qwen's float32 reference

**Finding: the published `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf` reads
0.41 mean KLD against Qwen's float32 reference, 8x the fidelity bar.** It is
a learned derivative of its parent, not a faithful quantization of it. So
is ISTA-DASLab's unmodified GSQ-RCO IQ3_S it was made from. Our scale
recovery moved the file toward the parent. No file is withdrawn. The roster
row's claim and anchor now lead with the reference reading, and by the
owner's decision of 2026-11-04 the row moves to Tier 2, maintained on
report, and stays published. The Hub card edits are in this directory as
text; the owner approved them, and they were pushed to the Hub from the
coordinator's machine after the merge (2026-11-04 17:19Z): the 27B card at
revision `a796aa06`, the Flash-Next card at revision `d4731ed9`.

## The reads

Reference: `Qwen/Qwen3.8-27B` at revision `1d4bf0f2` in float32. Two
windows: the house window, and positions 4000-5999 ("deep").

| | house: mean KLD | house: margin-qualified top-1 | deep: mean KLD | deep: margin-qualified top-1 | corpus perplexity |
|---|---:|---:|---:|---:|---:|
| fresh BF16 GGUF of the parent (engine check) | 0.00002 | 100% | 0.00003 | 100% | 73.29 |
| ISTA-DASLab IQ3_S, unmodified (`64b53b64`) | 0.5616 | 82.48% | 0.4718 | 86.36% | 53.48 |
| **recovered IQ3_S, published (`873df48d`)** | **0.4119** | **85.04%** | **0.4084** | **86.12%** | **21.54** |

- **Recovered against source:** the recovered file is closer to the
  reference than the source in both windows by KLD (Wilcoxon p 1.7e-10 in
  the house window, 0.0017 deep).
- **The BF16 row:** it shows no engine or reference gap.
- **The fidelity bar:** mean KLD at most 0.05 and margin-qualified top-1 at
  least 97%. Both IQ3_S files are far outside it.
- **Perplexity:** both IQ3_S files predict the corpus better than the
  parent does, so neither is a noisier copy of the parent's distribution.
  GSQ fits codes and scales jointly on calibration data (an imatrix of 1000
  chunks of 4096 tokens). ISTA-DASLab reports Wikitext-2 perplexity 7.05 for
  BF16 and 7.07 for the IQ3_S.
- **The earlier anchor, kept as what it is:** the card's 0.0450 / 97.80% was
  measured against the project's BF16 GGUF on 500 short held-out Wikitext-2
  prefixes. It is margin-qualified with a 0.5-nat tie band; plain top-1 is
  87.20%. It holds for those positions and does not describe the file on
  longer or other text.

## The lab's record (R4.32 section, as delivered)

The recovered file's source, ISTA-DASLab's unmodified `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`
(`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` at d562806d, sha256 64b53b64c7aa39f2...), was scored by Runner on the CPU,
chunked. It was read against the same float32 reference, side a the source, side b the recovered file's `=0` score
file. The files are prefixed `r432-`.

| Window | Source: mean / median / max / top-1 / margin-qualified | Recovered: mean / median / max / top-1 / margin-qualified | Paired |
|---|---|---|---|
| House, 300 at stride 15 | 0.5616 / 0.0812 / 10.77 / 74.33% / 82.48% | 0.4119 / 0.0720 / 7.46 / 77.67% / 85.04% | recovered closer, Wilcoxon z -6.39, p 1.7e-10; McNemar 6 / 12, p 0.24 |
| 4,000 to 5,999 | 0.4718 / 0.0403 / 10.86 / 80.75% / 86.36% | 0.4084 / 0.0433 / 8.96 / 80.40% / 86.12% | recovered closer, z -3.14, p 0.0017; McNemar 39 / 35, p 0.73 |

- **The house window by depth,** source against recovered: 0.536 against 0.344, 0.400 against 0.203, and 0.650
  against 0.536.
- **Gold-corpus PPL:** the BF16 parent 73.29, the source 53.48, the recovered file 21.54. The top-1 rates are 0.460,
  0.481 and 0.495.
- **The reading:**
  - The unmodified GSQ-RCO file is already 0.47 to 0.56 from Qwen's reference on this corpus. It predicts the corpus
    better than its parent, so it is a learned derivative (GSQ-RCO is a PTQ fitted on calibration data), not a
    faithful quantization of the parent.
  - The scale recovery moved the file significantly closer to the parent in both windows, the direction the card
    claims against its source.
  - The PPL fall from 53.5 to 21.5 is larger than the KL change. Whether the recovery's training text
    (Wikitext-103 and SmolTalk) also moved the file toward generic text is not measured here.

The BF16 engine check (`bf16diag-`) is Runner scoring a BF16 GGUF of the same parent on the CPU, chunked. It was
converted by llama.cpp 73a43d1 (`--outtype bf16 --no-mtp`): 53,808,281,952 bytes, sha256 22caa62cf4d378e4....
Against the same float32 reference it reads 0.00002 (house) and 0.00003 (deep) mean KLD, with 100% margin-qualified
top-1 in both windows. Paired against the recovered IQ3_S it is the closer in both windows (Wilcoxon z 15.0 and
38.7; McNemar 35/0 and 231/0).

## Files

Every file here is the lab's, by name and sha256 (`SHA256SUMS` in this directory; the same digests as the
lab package's own SHA256SUMS).

| File | What |
|---|---|
| `r432-score-source.cpu.json`, `r432-score-source.cpu.log` | Runner `--score` of ISTA-DASLab's unmodified IQ3_S (CPU, chunked) |
| `r432-gold-house.json`, `r432-gold-d1window.json` | `gold-logits.py`: float32 reference, side a the source, side b the recovered file, two windows |
| `r432-r432.log` | the run's log: in-memory sha256 checks against the Hub's LFS digests, exits |
| `bf16diag-gold-house.json`, `bf16diag-gold-d1window.json` | the engine check: the BF16 GGUF of the parent against the same reference, side b the recovered IQ3_S |

## Provenance (settled before the reads)

Both files are quantizations of the public parents. Each parent has one
weight set on the Hub, and the files' unquantized tensors match it bit for
bit (norms equal to the parent's bf16 + 1; ssm, router and hyper-connection
tensors equal up to the converter's head reorder).

- **27B:** `Qwen/Qwen3.8-27B@1d4bf0f2`. The 18 weight files are unchanged
  since the 2026-08-13 upload, and the later commits are card, license,
  tokenizer and template. The recovery changed only the fp16 `d`/`dmin`
  fields of 399 tensors of ISTA's file `64b53b64...`. Its teacher was a BF16
  GGUF converted from the same revision.
- **Flash-Next:** `Qwen/Qwen3.8-Flash-Next@de4b8e4d`, with one weight set
  across all three commits. The project's 4-part set carries ISTA's trunk
  bytes unchanged, and the MTP head was built from `de4b8e4d`. The trunk's
  own reference read is pending.

## What this changes

| Where | Change | State |
|---|---|---|
| `docs/model-tiers.json`, row `qwen3.8-27b-gsq-rco-iq3_s-recovered` | claim, anchor, limitations, upstream revision, evidence; tier2 with a transition entry and a published exception (owner decision 2026-11-04) | applied in this PR |
| `docs/model-tiers.json`, row `qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head` | the limit sentence as its limitations; tier unchanged, its own reference read pending | applied in this PR |
| `docs/model-support.md` | re-rendered from the roster | applied in this PR |
| `MANUAL.md`, published-files list | the 0.0450 sentence replaced by the reference reading | applied in this PR |
| Hub card `Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF` | [hub-card-27b.md](hub-card-27b.md) | text only; pushed from the coordinator's machine after the merge, revision `a796aa06` |
| Hub card `Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF` | [hub-card-flash-next.md](hub-card-flash-next.md) | text only; pushed from the coordinator's machine after the merge, revision `d4731ed9` |
