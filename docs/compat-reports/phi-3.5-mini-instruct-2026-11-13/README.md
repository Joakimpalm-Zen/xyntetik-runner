# Phi-3.5-mini-instruct: bartowski's files against Microsoft's float32 reference (2026-11-13)

Phi-3.5-mini-instruct is Microsoft's 3.8B chat model on the phi3 architecture (32 layers, hidden 3,072, 32 heads
without grouping, LongRoPE with an original context of 4,096). The family had no file with a cell. Two rows join:
bartowski's Q8_0 and Q4_K_M, from one repository at one revision. Microsoft publishes no GGUF of Phi-3.5.

**What is not flattering, first:**
- **Above a 4,096-token context the model reads markedly worse, on short prompts too.** Phi-3.5 carries two LongRoPE
  factor sets. Runner picks the set from the requested context (`-c`) and applies it from the first token, as
  llama.cpp does. The publisher's reference code applies the short factors until the sequence itself passes 4,096
  tokens. On the same 272 positions of a 4,096-token text, the Q8_0 reads 0.0042 mean KLD at `-c 4096` (short
  factors) and 0.2052 at `-c 8192` (long factors, 88.54% margin-qualified top-1, under the bar). llama.cpp 73a43d1
  reads 0.0073 and 0.2071 on the same positions. Runner's default context for this model is 4,096, which uses the
  short factors. Both rows state this first in their limitations. No fix is in this change.
- **The rows are measured inside 4,096 tokens only.** There is no reference read of a sequence that itself passes
  4,096, where both sides would use the long factors.
- **The Q4_K_M has less margin.** Its KLD bootstrap interval, 0.0375 to 0.0474, ends near the bar of 0.05, and its
  plain top-1 is 91.4%. The Q8_0 is the fidelity file.
- **The existing Q4_K_M compat entry declares greedy_reference and long_context**, which this run did not execute
  (see "The cells").
- **Scope:** two windows over one corpus, CPU score files. Task quality is not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), for each file on its own.

## The files and the reference

- **Reference:** `microsoft/Phi-3.5-mini-instruct` at `2fe192450127e6a83f7441aef6e3ca586c338b77` (not gated; licence
  mit), held whole. Both safetensors shards equal the Hub's LFS records. Stored in bfloat16.
- **Reference path:** transformers 5.16.1, the native Phi3ForCausalLM (no remote code), `scripts/gold-logits.py
  --upcast-at-use` (float32 arithmetic, cast at use per decoder block).
- **Files** from `bartowski/Phi-3.5-mini-instruct-GGUF` at `6d70da17e749a471ccb62ade694486011a75cda3`, each sha256
  checked against the Hub's LFS record on download and as held. Both were made with bartowski's importance matrix
  (Q8_0 does not use one):
  - Q8_0 `Phi-3.5-mini-instruct-Q8_0.gguf`, 4,061,222,688 bytes, sha256
    `76fbf02f6fe92af57dbd818409bc8a0240026f1f3609bb405c3be94c973fb823`;
  - Q4_K_M `Phi-3.5-mini-instruct-Q4_K_M.gguf`, 2,393,232,672 bytes, sha256
    `e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5`.
- **Windows:** two windows of 272 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, from position 8 and
  from 15, inside the 4,096-token context, where both sides use LongRoPE's short factors. Token-id keyed: every
  position matched by id, no token spelling mismatches, no dropped reference mass.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,096 tokens, `-c 4096`) from the clean release-recipe STRICT build of main e9cb9c07 (runner 1.18.0, gcc 13.3.0,
  sha256 `1cb3a908…`). Each score log names `phi3 LongRoPE short factors`. Between that main and e39220e6, the
  branch's base, the only source change is the generated roster header.

## Against the reference

| File | Window | Mean KLD | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| **Q8_0** | 1 | 0.0042 | 0.053 | 98.90% | 100% |
| | 2 | 0.0033 | 0.030 | 98.16% | 100% |
| | **pooled 544** | **0.0038 ± 0.0003** | 0.053 | 98.53% | **100% (389/389)** |
| **Q4_K_M** | 1 | 0.0424 | 0.696 | 91.91% | 98.96% |
| | 2 | 0.0421 | 0.445 | 90.81% | 98.98% |
| | **pooled 544** | **0.0423 ± 0.0025** | 0.696 | 91.36% | **98.97% (385/389)** |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0031 to 0.0044 for the Q8_0 and 0.0375 to 0.0474 for the Q4_K_M. The pooled
medians are 0.0005 and 0.0248.

Per position: `gold-q8_0-w1.json`, `gold-q8_0-w2.json`, `gold-q4_k_m-w1.json`, `gold-q4_k_m-w2.json` (FILES.sha256).

## Above 4,096: which factor set is applied

One extra read, on the Q8_0 and window 1 only (the same 272 positions of the same 4,096-token text; the reference is the
same file, with the short factors throughout, since the sequence never passes 4,096):

| Server | Context | Factor set | Mean KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| Runner | `-c 4096` | short | 0.0042 | 98.90% | 100% |
| Runner | `-c 8192` | long (`phi3 LongRoPE long factors, mscale 1.1902` on load) | **0.2052** | 79.04% | 88.54% |
| llama.cpp 73a43d1 llama-server | `-c 4096` | short | 0.0073 | 96.32% | 100% |
| llama.cpp 73a43d1 llama-server | `-c 8192` | long | **0.2071** | 79.78% | 89.06% |

So the gap comes from the choice of factor set, not from the file or from Runner's arithmetic: llama.cpp makes the same
choice and reads the same. The publisher's code switches per sequence, at 4,096 tokens. Runner and llama.cpp switch
per context. A Runner user who asks for more than 4,096 tokens of context gets the long factors on every prompt,
short ones included.

Files: `gold-q8_0-w1-c8192.json` (Runner at `-c 8192`), `gold-q8_0-w1-llama-c4096.json` and
`gold-q8_0-w1-llama-c8192.json` (Runner at `-c 4096` beside llama.cpp at each context).

## The tokenizer

- difftok against Microsoft's tokenizer.json at 2fe19245: 0 of 721 corpus strings and 0 of 31,865 vocabulary entries
  differ, on both files.
- The reference ids were already captured in `tests/compatibility/tokenizer-references/phi-3.5-mini-instruct.json`
  (gated) and `phi-3.5-mini-instruct.tokenizer-json.json` (informational).

## The template

- Runner picks its `phi3` renderer. The `phi3` conformance family against the publisher's chat template: 16 cases
  compared, 16 identical in text and tokens.

## What the file resolves to on load

Architecture phi3, template phi3, sampling preset phi3, and the generic tool protocol (Runner has no native tool format
for this model).

## The cells

On a CUDA MIG slice host (1g.24gb, sm_120), the CPUs shared with other work, from the clean release-recipe STRICT build
of main e39220e6, this branch's base (runner sha256 `e7fa5f48…`, provenance commit e39220e6, dirty false), built before
the rows were added (R6.67).
- **tier-smoke**, `--pin` then `--record`, on the CPU and CUDA cells of both rows: all four pass with the fact found.
  CPU and CUDA give the same reply bytes for each file (Q8_0 `2365a83e`, Q4_K_M `429ec866`). CUDA runs all 32 layers
  on the device.
- **LoRA smoke** on both files: `--train` on eight lines (two prompt/completion pairs, four times each), 4 steps at rank 8, then serving with
  `--lora`. Both train and serve (224 adapted projections); the adapted model completes "The capital of France is"
  with "Paris."
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks`,
  `1.18.0-2026-11-13-linux-x86_64-phi-3.5-mini-instruct.json`):
  - Q8_0 (a new entry): load, tokenizer (0 of 721 against the captured reference ids; 0 of 31,865 vocabulary
    entries), cpu_cuda (9 of 9 identical, `cpu_cuda/phi-3.5-mini-instruct-q8_0.json`) and chat pass; complete.
  - Q4_K_M (the entry already in the manifest, its checks and notes unchanged): load, tokenizer, cpu_cuda (9 of 9),
    chat and tool (the agent-torture matrix, 8 of 8 passed, `tool/phi-3.5-mini-instruct-q4_k_m.json`) pass.
    **Two declared checks did not run here**, so the entry and the report are not complete: greedy_reference (its
    reference is llama.cpp b10076, which this host does not hold; no other build was put in its place) and
    long_context (no runner for that check class on this host).
  The files were held in memory for the run; a lab wrapper kept their links, and the sha256 check read the held bytes.
