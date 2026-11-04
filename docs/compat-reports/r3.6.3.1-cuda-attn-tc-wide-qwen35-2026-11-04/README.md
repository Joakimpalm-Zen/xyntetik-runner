# R3.6.3.2: the wide-head tensor-core prefill attention on by default for qwen35 at head width 256

**Decision: `RUNNER_CUDA_ATTN_TC_WIDE`'s default list gains qwen35 at head width 256** (Qwen3.5, 3.6 and 3.8: one
layer in four is full attention). The route detail is unchanged: `attn_wide=on` in the route record, `=0` keeps the
tile, `=1` forces it on every model the kernel admits. The gemma-4 E-series default from R3.6.3.1 (#574) stays.

**Kill numbers, stated before the read and both met:**

| Condition | Needed | Read on Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered | Result |
|---|---|---|---|
| Prefill gain at 16K, medians of five interleaved rounds | at least 1.2x | **1.820x** (329.6 against 181.1 tok/s) | met |
| D1, the attention routes' rule (2026-10-20): mean KL at most 1.01x | at most 1.01x | **0.9998x** (0.408320 against 0.408390) | met |
| D1: paired Wilcoxon on per-position KL not worse at p < 0.05 | not worse | **z 0.23, p 0.82** | met |
| D1: McNemar on margin-qualified top-1 not worse | not worse | 0 / 0 discordant of 1,664 | met |

**Scope:**
- **Measured:** one qwen35 file (Qwen3.8-27B, head width 256), on one CUDA MIG slice (sm_120).
- **Not yet measured:** the Flash-Next sibling (`qwen4exp`, a different architecture, not on this list); Metal; the
  RTX 3070 at these sizes; prompts past 16K; the MTP head under `=1`.
- **What the default keys on:** the measured width. A qwen35 file at another head width stays on the tile unless
  forced, and a 256-wide model of another architecture is not admitted by default. `tests/test_attn_wide_repeat.py`
  pins all three on CUDA.
- **The file's distance from its parent:** the file itself reads about 0.41 mean KLD from its float32 parent (the
  diagnostic below). That is a property of the GSQ-RCO file (R4.32, PR #606). It is not the route's: D1 compares `=0`
  against `=1` on the same file, and the two read the same.
- **The tier-smoke caveat** below is closed on main by R1.24 (#603): the six 1.5.1 CUDA pins' specs now carry their
  rows' `backend_args`.

The lab's record follows. Its R4.32 section is in the R4.32 report (PR #606) with that read's files. The two
`bf16diag-gold-*` files the diagnostic cites are there too. The machine name in the diagnostic section is replaced
by the host's role.

---

## #574 evidence row: `RUNNER_CUDA_ATTN_TC_WIDE` on Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered (qwen35, lab, 2026-11-04)

**16K prefill: 1.820x with `k_attn_tcw` on** (329.6 against 181.1 tok/s, 16,576 tokens). The D1 read passes the
attention routes' rule, so this file meets both of the booking's widening conditions for qwen35: at least 1.2x, and
D1 passing.

**Unflattering, and separate from the verdict:** the file itself sits about 0.41 mean KLD from its float32 parent on
the gold corpus, about 8x the house bar, in both windows read today. Runner on a BF16 GGUF of the same parent reads
0.00002 against the same reference, so the distance is the file's, not the engine's or the reference's (R4.32;
"The diagnostic" below).

### Prefill A/B (the MIG slice, runner main 7567c94)

- **Method:** `runner -m FILE --bench-json -f PROMPT -c CTX -n 1 -b 512 --gpu 99 -t 40`. One discarded warm-up per
  size, then five rounds of `=0` and `=1` interleaved, the order alternating by round. The GGUF was held in memory
  throughout.
- **Prompts:** cut from `tests/fixtures/gold-corpus-2k.txt` (`prompt-4k.txt`, `prompt-16k.txt`).
- **Route record:** it names `attn_wide=on` on every `=1` run and on no `=0` run.

| Prompt | `=0` prompt tok/s (5 runs) | `=1` prompt tok/s (5 runs) | Gain (medians) | Prompt seconds, medians |
|---|---|---|---|---|
| 16,576 tokens | 181.1-181.2, median 181.1 | 329.5-329.7, median 329.6 | **1.820x** | 91.51 against 50.29 |
| 4,005 tokens | 306.5-306.6, median 306.6 | 377.3-377.5, median 377.4 | 1.231x | 13.06 against 10.61 |

Decode is unchanged (17.2-18.1 tok/s on both). The full-attention layers are one in four, and the gain grows with
the prompt because the attention share grows with it.

### gpu-identity

`tests/test_gpu_identity.c` compares the CPU with CUDA at logit precision over 24 positions (5,959,680 logits). Both
settings are ok:

| Setting | Mean \|Δlogit\| | As a share of the mean logit range (limit 1e-3) | Worst |
|---|---|---|---|
| `=1` | 0.000237 | 6.79e-6 | 0.00589 |
| `=0` | 0.000225 | 6.44e-6 | 0.00463 |

Files: `gpu-identity-tcw*.log`.

### The tier smoke against the pin

The file matches the row's pin with `=1` and with `=0` (7.8 s each, fact found; `tier-smoke-lab-tcw*.json`). One
caveat on how:

- Main's `tests/compatibility/tier-smoke.json` gives the row `backend_args {cuda: [--gpu, 38]}`. But its pin's
  `spec`, pinned 2026-10-24 at 1.5.1, predates that field, so the tool refuses to compare (`tier-smoke-tcw1.log`).
- The roster's `ran` line shows the pin was made with `--gpu 38`.
- The comparison ran on a lab copy of the manifest whose pin spec records it (`tier-smoke-lab-pin-entry.json`).
- Main needs the field added to that entry, or a retire and re-pin.

### D1 (the attention routes' rule, 2026-10-20)

Two chunked score files on the slice (`RUNNER_SCORE_CHUNKED=1 RUNNER_SPEC_BATCH=64`, `-c 8448`; PPL 21.542 at `=0`,
21.540 at `=1`). They were read against `Qwen/Qwen3.8-27B` at 1d4bf0f2 in float32 arithmetic (`--upcast-at-use`,
transformers 5.16.1), positions 4,000 to 5,999 at stride 1, every one matched by id, no dropped mass
(`gold-tcw0-vs-tcw1.json`).

| Rule | a (`=0`) | b (`=1`) | Reading | Result |
|---|---|---|---|---|
| Mean KL at most 1.01x | 0.408390 | 0.408320 | b/a = 0.9998 | pass |
| Paired Wilcoxon on per-position KL not worse at p < 0.05 | | | z 0.23, p 0.82 | pass |
| McNemar on margin-qualified top-1 not worse | | | 0 / 0 discordant of 1,664 | pass |

### The diagnostic: the 0.41 is the file's (R4.32)

The same float32 reference was read against Runner on a BF16 GGUF of the same parent. It was converted into memory by
llama.cpp 73a43d1 (`--outtype bf16 --no-mtp`): 53,808,281,952 bytes, sha256 22caa62cf4d378e4.... These are not the
bytes of the lab's `Qwen3.8-27B-BF16-backbone.gguf` (f50473e0), the release card's anchor. Runner scored it on the
CPU, chunked. Side b is the IQ3_S's `=0` file. The diagnostic's files are prefixed `bf16diag-`.

| Window | BF16 GGUF: mean / max / top-1 / margin-qualified | IQ3_S: mean / median / max / top-1 / margin-qualified |
|---|---|---|
| House, 300 at stride 15 (8 to 4,493) | 0.00002 / 0.002 / 99.67% / 100% | 0.4119 / 0.0720 / 7.463 / 77.67% / 85.04% |
| 4,000 to 5,999 at stride 1 | 0.00003 / 0.013 / 99.95% / 100% | 0.4084 / 0.0433 / 8.961 / 80.40% / 86.12% |

- **The paired tests:** in the house window the BF16 file is the closer, Wilcoxon z 15.0 and McNemar 35/0. In the
  deep window, z 38.7 and McNemar 231/0.
- **The house window by depth** (IQ3_S): 0.344 (8-749), 0.203 (750-1,999), 0.536 (2,000-4,493).
- **The reading:**
  - Runner and transformers' float32 path agree on Qwen3.8 to 2e-5, so there is no reference or engine gap.
  - The recovered IQ3_S sits about 0.41 from its parent on this corpus at every depth.
  - The card's 0.0450 / 97.80% is a different measurement: a BF16-GGUF anchor (Runner on both sides), 500 short
    Wikitext-2 prefixes.
  - The 2026-10-10 artifact baseline (runner 1.2.1) read this file at 0.688 against that backbone on
    `tests/fixtures/mixed-corpus.txt`, in line with today's reading.
- **Unexplained, and so not claimed:** corpus PPL is 73.29 for the BF16 parent against 21.54 for the IQ3_S.
  - The top-1 rate is 0.460 against 0.495.
  - A faithful 3.4-bit quantization reads worse than its parent, not 3.4 times better.
  - Whether the GSQ-RCO file was quantized from this revision was then established on the Windows CUDA host: it was
    (R4.32, `docs/compat-reports/r4.32-qwen38-iq3s-reference-2026-11-04`).

### Files

`FILES.sha256` lists the GGUF (Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF at 7b10f923, sha256
873df48dd3ecf205..., 11,771,546,784 bytes), the parent's 18 files at 1d4bf0f2 and the BF16 conversion. Each was
checked against the Hub's LFS digest as it streamed into memory. `SHA256SUMS` covers every file here.

### Not measured

- Metal and the 3070.
- Prompts past 16K.
- The MTP head (`Qwen3.8-27B-MTP-Q8_0.gguf`) under `=1`.
