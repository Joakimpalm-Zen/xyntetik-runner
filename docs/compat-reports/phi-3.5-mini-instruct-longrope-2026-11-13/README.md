# Phi-3.5-mini-instruct: which LongRoPE factor set, and where (2026-11-13)

Phi-3.5-mini-instruct carries two LongRoPE factor sets, short and long, with an original context of 4,096 tokens. The
publisher's reference code picks the set per sequence: the long factors only once the sequence itself passes 4,096
tokens. Runner and llama.cpp pick the set from the requested context (`-c`) and apply it from the first token. This
record measures that difference from both sides, on one file.

**The finding.** Inside 4,096 tokens, at `-c 8192`, both servers read about fifty times further from the reference than
at `-c 4096`, because they use the long factors where the reference uses the short ones. Past 4,096 tokens, where the
reference uses the long factors too, Runner at `-c 8192` reads as close as it does inside 4,096 at `-c 4096`. So
Runner's long-factor arithmetic agrees with the reference; what differs is only the choice of the set per sequence.

**Scope:** one file (the Q8_0), one corpus, CPU on every side. The inside-4,096 reads are one window of 272
positions, the past-4,096 read one window of 114 positions. Task quality is not measured.

## Setup

- **File:** bartowski's `Phi-3.5-mini-instruct-Q8_0.gguf` from `bartowski/Phi-3.5-mini-instruct-GGUF` at
  `6d70da17e749a471ccb62ade694486011a75cda3`, 4,061,222,688 bytes, sha256
  `76fbf02f6fe92af57dbd818409bc8a0240026f1f3609bb405c3be94c973fb823` (checked against the Hub's LFS record).
- **Reference:** `microsoft/Phi-3.5-mini-instruct` at `2fe192450127e6a83f7441aef6e3ca586c338b77`, transformers
  5.16.1, the native Phi3ForCausalLM (no remote code), `scripts/gold-logits.py --upcast-at-use` (bfloat16 storage,
  float32 arithmetic at use per decoder block), on the CPU. The reference runs one forward over the prefix up to the
  last scored position, so its factor set follows that length: short for the inside-4,096 windows (4,089 tokens),
  long for the past-4,096 window (6,011 tokens).
- **Corpus:** `tests/fixtures/gold-corpus-2k.txt`. Positions are matched by token id; no spelling mismatches and no
  dropped reference mass on any side.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill
  route). Every load line names the set (`rope: phi3 LongRoPE short factors` or `long factors`, `mscale 1.1902`),
  and no score log names YaRN.
  - Inside-4,096 reads: 4,096-token score files, from the clean release-recipe STRICT build of main e9cb9c07 (runner
    1.18.0, gcc 13.3.0).
  - Past-4,096 read: a 6,200-token score file at `-c 8192`, from the clean release-recipe STRICT build of main
    e39220e6 (runner 1.18.0, gcc 13.3.0). Between the two mains the only source change is the generated roster
    header.
- **llama.cpp:** llama-server at commit 73a43d1, CPU (`-ngl 0`), one slot, sent the reference's token ids
  (`--endpoint-ids`), at the context named in each row.

## The three reads

| Window | Positions | Reference set | Server | Context | Server set | Mean KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|---|
| inside 4,096 | 8 to 4,073, stride 15 (272) | short | Runner | `-c 4096` | short | 0.0042 | 98.90% | 100% |
| | | | llama.cpp | `-c 4096` | short | 0.0073 | 96.32% | 100% |
| inside 4,096 | 8 to 4,073, stride 15 (272) | short | Runner | `-c 8192` | long | **0.2052** | 79.04% | 88.54% |
| | | | llama.cpp | `-c 8192` | long | **0.2071** | 79.78% | 89.06% |
| past 4,096 | 4,300 to 5,995, stride 15 (114) | long | Runner | `-c 8192` | long | **0.0045 ± 0.0007** | 99.12% | 100% |
| | | | llama.cpp | `-c 8192` | long | 0.0252 ± 0.0028 | 94.74% | 98.81% |

± is the standard error of the mean over the window's positions. For the past-4,096 window, a percentile bootstrap
(10,000 resamples, seed 0) gives 0.0032 to 0.0059 for Runner and 0.0200 to 0.0308 for llama.cpp; the medians are
0.00095 and 0.01415, the maxima 0.034 and 0.138.

llama.cpp reads further from the reference past 4,096 than inside it at `-c 4096` (0.0252 against 0.0073). This
record does not look into why.

## Files

- `gold-q8_0-past4096.json`: the past-4,096 window, Runner at `-c 8192` (side a) and llama.cpp at `-c 8192`
  (side b), per position.
- The inside-4,096 reads are in the Phi-3.5 roster record, `docs/compat-reports/phi-3.5-mini-instruct-2026-11-13/`:
  `gold-q8_0-w1.json` (Runner at `-c 4096`), `gold-q8_0-w1-c8192.json` (Runner at `-c 8192`),
  `gold-q8_0-w1-llama-c4096.json` and `gold-q8_0-w1-llama-c8192.json` (Runner at `-c 4096` beside llama.cpp at each
  context).
- The past-4,096 read was run by the lab script `g1.sh`. It does the following:
  1. writes the Runner score file (6,200 tokens, `-c 8192`);
  2. starts llama-server at `-c 8192`;
  3. runs `gold-logits.py --upcast-at-use --score-a <score file> --endpoint-b <llama-server> --endpoint-ids
     --min-prefix 4300 --stride 15 --max-positions 114 --top-n 20`.
- Cost: on a CUDA MIG slice host (1g.24gb, sm_120), with CPUs shared with other work and 40 threads, Runner's score
  file took 197 s and the reference forward plus scoring 311 s, at a peak resident set of 13.3 GiB.
- `FILES.sha256`: the sha256 of the gold file.
