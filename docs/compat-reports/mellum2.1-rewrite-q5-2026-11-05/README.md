# Mellum 2.1: Runner's own Q5-band rewrite against the float32 reference (2026-11-05)

**What is not flattering, first:**
- The file Runner writes is 31.4 MB larger than bartowski's Q5_K_S, and its mean KLD is 0.0004 higher.
- Plain top-1 is 94.0%.
- It is under 9 GB by only 5.2 MB.
- Its first window (positions 8 to 749) reads a mean KLD of 0.0419, the closest of its three windows to the 0.05
  bar. B5's same window read 0.0497.
- It is one 300-position window over one corpus.
- The three new types are written without an importance matrix.
- The file was published on 2026-11-05 as [Joakimpalm-Zen/Mellum2.1-12B-A2.5B-Thinking-selective-Q8_0-expQ5_K-GGUF](https://huggingface.co/Joakimpalm-Zen/Mellum2.1-12B-A2.5B-Thinking-selective-Q8_0-expQ5_K-GGUF) (the same bytes, sha256 `8a145881...`, under a descriptive name). No roster row is claimed here; the row is its own item.

What passes is the recipe the Q5-band report found, written by Runner itself (R4.33):
`runner --rewrite --type-plan` from the BF16 parent writes 8,994,830,528 bytes, reads a mean KLD of **0.0293** and a
margin-qualified top-1 of **98.74%** against JetBrains' BF16 parent in float32. That is inside the bar of 0.05 and 97%.

The kill was declared before the read: the bar itself, so missing either number ended the claim, and there was no
retuning. A fallback plan (`attn_q.weight` at q6_k) was declared too, for the case of the file reaching 9e9 bytes. It
was not needed.

## Setup

- **Reference:** `JetBrains/Mellum2.1-12B-A2.5B-Thinking` at revision `92ddae9fc766`, transformers' own mellum code in
  float32 (`scripts/gold-logits.py`, transformers 5.16.1), on the Linux CUDA host's CPUs.
- **Positions:** the window of the 2026-11-03 derive study and the Q5-band report: 300 positions at stride 15 over
  `tests/fixtures/gold-corpus-2k.txt` (sha256 `56e9fcba...`), positions 8 to 4,493, token-id keyed, no special
  prefix, no dropped mass. 239 positions are margin-qualified.
- **Parent:** the checkpoint converted to a BF16 GGUF by llama.cpp 73a43d1's `convert_hf_to_gguf.py`. That is
  `parent-bf16.gguf`, 24,311,969,984 bytes, sha256 `8053a907bde4b833...`, byte for byte the 2026-11-03 study's
  parent (conversion 74 s, load average 21 to 42).
- **The rewrite:** Runner at the R4.33 commit, `runner -m parent-bf16.gguf --rewrite R1.gguf --type-plan R1.json`
  on the host's CPUs (447 s, load average 25 to 31), with
  `{"default": "q8_0", "rules": [{"match": "ffn_gate_inp", "type": "keep"}, {"match": "_exps.weight", "type": "q5_k"}]}`.
  The 896-wide expert down projections cannot take Q5_K's 256-wide block, so they are written as Q5_1, its 32-block
  cousin (28 tensors, reported by the rewrite).
- **Output:** `R1.gguf`, 8,994,830,528 bytes, sha256
  `8a1458812a81e61229018184968ae954f0f943d31ad8c8cc8144326d73b76af6`. It was published on 2026-11-05 (see the
  top of this record), so a roster row can cite it without a new read. The
  header declares architecture mellum, apache-2.0, 28 blocks, 64 experts and 8 used. It has 339 tensors, the parent's
  339, and each tensor's type is the plan's.
- **Runner side:** `runner --score --score-top 20 --score-tokens 4510 -c 4510 --gpu 99` on a CUDA MIG slice
  (1g.24gb, sm_120), 42 s at load average 27 to 32. All 28 layers ran on the device, with no expert bank on the host.
  The 4,510 token ids are the 2026-11-03 parent score's.
- **The reference pass:** 291 s on the host's CPUs, at load average 27 to 43.

## Against the reference

| File | Bytes | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|---|
| **Runner R1 (this rewrite)** | **8,994,830,528** | **0.0293** | 0.0126 | 0.437 | 94.00% | **98.74%** | **pass** |
| bartowski Q5_K_S (B5, 2026-11-04) | 8,963,422,944 | 0.0289 | 0.0126 | 0.476 | 93.00% | 98.33% | pass |
| Runner derive D1 (2026-11-03) | 9,380,575,424 | 0.0983 | 0.0437 | 2.242 | 86.67% | 92.89% | fail |
| Publisher Q4_K_M (2026-11-03) | 8,071,295,264 | 0.1172 | 0.0476 | 3.663 | 88.67% | 92.89% | fail |

By depth (mean KLD / margin-qualified top-1; 50, 83 and 167 positions):

| File | 8 to 749 | 750 to 1,999 | 2,000 to 4,493 |
|---|---|---|---|
| Runner R1 | 0.0419 / 100% | 0.0325 / 100% | 0.0239 / 97.8% |
| B5 | 0.0497 / 97.4% | 0.0282 / 100% | 0.0230 / 97.8% |

## What the bytes bought

| File | Attention q | Attention k, v, output | Expert gate / up | Expert down (896 wide) | `token_embd` / `output` | Routers |
|---|---|---|---|---|---|---|
| Runner R1 | Q8_0 (28) | Q8_0 | Q5_K / Q5_K | Q5_1 | Q8_0 / Q8_0 | F32 |
| B5 | Q8_0 (17), Q6_K (6), Q5_K (5) | Q8_0 | Q5_K / Q5_K | Q5_1 | Q8_0 / Q8_0 | F32 |

The two files differ in the attention q projections, and B5 was made with an importance matrix. Runner's
importance-weighted writers cover q3_k and q4_0 only (R4.34).

## What this is not

This is not a benchmark claim, not a speed row, and not a claim about other text. It is one window over one corpus.
The 8.99 GB file is not for an 8 GB machine.

## The roster row (R4.33.4, 2026-11-06)

The published file became the row `mellum2.1-12b-a2.5b-thinking-selective-q8_0-expq5_k`. There was no new reference
pass: the numbers above are the published bytes' own.

- **The bytes:** streamed from the Hub at revision `4941f110` with the sha256 taken in the same pass: `8a1458812a81...`,
  the Hub's LFS record and the rewrite's output. Header: mellum, apache-2.0, 28 blocks, 64 experts, 8 used, 339
  tensors, each of the plan's type. Its template is JetBrains' `chat_template.jinja` at `92ddae9f` byte for byte
  (4,849 B).
- **Tokenizer:** `scripts/difftok.py` against JetBrains' tokenizer.json at `92ddae9f`: 0 of 721 corpus strings and
  0 of 97,302 vocabulary entries differ.
- **Template:** `--info tools` picks the `mellum` template. `template-conformance --family mellum`: 23 of 23 cases are
  identical in text and tokens.
- **tier-smoke:** the CPU and CUDA replies are byte-identical (`b08f1e19`), both with the expected fact, at runner
  `cf645fff` on main `0abf82f1`, on a CUDA MIG slice (1g.24gb, sm_120) and the host's CPUs.
- **Not run on this file:** the live tool gate, the CPU-vs-CUDA check, a Metal row, and an llama.cpp column.
