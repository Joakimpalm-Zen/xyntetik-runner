# Measure a quantized file against its publisher's reference

This page is for anyone who publishes quantized GGUF files and wants to say,
with a number, how far a file is from the model it was made from. The
reference is the publisher's own checkpoint run in float32. The instrument is
Runner's scorer, which reads every position of a text in one pass. The tool
in the middle is `scripts/gold-logits.py`.

The result is two numbers per file, both measured against the reference:

- **Mean KL divergence** over the scored positions, in nats: how far the
  file's next-token distribution is from the reference's, on average.
- **Margin-qualified top-1**: of the positions where the reference has a
  clear favourite (its top two tokens more than 0.5 nats apart), the share
  where the file picks that favourite too. Positions where the reference is
  itself undecided are left out, because there a different pick is not an
  error.

The bar this project uses to call a file faithful is **mean KL at most 0.05
and margin-qualified top-1 at least 97%**. A file can be measured without
adopting that bar; the JSON keeps every position.

## What the instrument adds by itself

A measurement is only as good as the instrument's own error. Scored on BF16
files, which hold the reference's own weights, Runner 1.6.0 read a mean KL
between 2.7e-5 and 2.4e-4 against the float32 reference on 11 of 12 models,
with margin-qualified top-1 at 100% on all eleven. That is the arithmetic
alone: two to three orders of magnitude below the 0.05 bar. The twelfth,
Trinity Nano, read 0.046 at BF16, and another engine read 0.045 on the same
file, so that gap belongs to the file or the checkpoint, not to one engine
([golden pass of 2026-10-25](golden-pass-2026-10-25.md)). Score the BF16
file beside your quantized ones (step 3): if the BF16 file does not read near
zero, fix that before reading anything else.

## Steps

You need the publisher's checkpoint (safetensors), Python with `torch` and
`transformers`, a Runner binary, and the GGUF files to measure. The
reference runs on the CPU; its memory is about four bytes per parameter, or
two with `--upcast-at-use` (below).

**1. The reference, pinned.** Download the publisher's checkpoint at an
exact revision, so the measurement names what it was measured against:

```sh
hf download ORG/MODEL --revision COMMIT_SHA --local-dir ref/
```

**2. A corpus.** `tests/fixtures/gold-corpus-2k.txt` is the corpus this
project's reports use: mixed prose, code and lists, about 4,500 tokens on
most tokenizers. Any text works; use the same one for every file you compare.

**3. One scoring pass per file.** Runner reads the text teacher-forced and
writes each position's 20 most likely tokens:

```sh
./runner -m MODEL-BF16.gguf   --score 20 -f tests/fixtures/gold-corpus-2k.txt \
         -c 8192 --gpu off > bf16.score.json
./runner -m MODEL-Q4_K_M.gguf --score 20 -f tests/fixtures/gold-corpus-2k.txt \
         -c 8192 --gpu off > q4_k_m.score.json
```

Keep `-c` within the model's training context: past it Runner applies YaRN
scaling (and says so), and the measurement would then be of the scaling.

**4. Score against the reference.** Two files per run:

```sh
python3 scripts/gold-logits.py --hf ref/ \
    --corpus tests/fixtures/gold-corpus-2k.txt \
    --score-a bf16.score.json --score-b q4_k_m.score.json \
    --max-positions 2000 --stride 2 --out gold-bf16-q4_k_m.json
```

The reference runs one forward pass over the whole text and the script reads
2,000 positions, every second token. Use 2,000: at a few hundred positions
nearly every gap between two good files is noise (this project learned it
when a rerun at 2,000 positions overturned a 100-position table:
[golden pass of 2026-09-07](golden-pass-2026-09-07.md)). 300
positions (`--max-positions 300 --stride 15`) is a cheaper screen for a large
model, but at 300 the paired test cannot tell small gaps apart. Add `--upcast-at-use` when the checkpoint is
stored in BF16 and float32 would not fit: the weights stay at BF16 and each
decoder block's are cast to float32 for its arithmetic.

The run stops if a score file's token ids differ from the reference
tokenizer's ids for the same text. That is a finding about the GGUF's
tokenizer, not a measurement problem: the file would read text differently
from the model it claims to be.

**5. Read the summary.** Per side, `gold-bf16-q4_k_m.json` gives
`mean_kld_vs_gold`, `max_kld_vs_gold`, `top1_agreement_pct` and
`margin_qualified_top1_pct`, then a paired comparison of the two sides
(Wilcoxon on per-position KL, McNemar on the margin-qualified positions)
that says whether their difference is distinguishable from noise at this
corpus size. Every position is in `rows`.

## Reporting it

The report this project writes for a file leads with the result that is
least flattering to it, then gives the table, the reference's exact
revision, the file's repository revision and sha256, the corpus, the
positions and the Runner version. [The Qwen3.6-35B-A3B report](compat-reports/qwen3.6-35b-a3b-reference-2026-10-31/README.md)
is one example: its Q4_K_M passes the bar by under three hundredths on both
numbers, and it says so in its first sentence.

## What this does not measure

KL and top-1 agreement are about the next-token distribution on one text.
They do not measure task quality: a file inside the bar can still lose on a
benchmark, and a file outside it can still answer a given question well.
They also say nothing about the chat template or the tokenizer's handling of
text outside the corpus; check those separately.
