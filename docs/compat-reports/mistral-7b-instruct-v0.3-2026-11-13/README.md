# Mistral-7B-Instruct-v0.3: bartowski's files against Mistral AI's float32 reference (2026-11-13)

Mistral-7B-Instruct-v0.3 is Mistral AI's 7B chat model: 32 layers, hidden 4,096, 32 query heads over 8 KV heads,
FFN 14,336, rope base 1,000,000, vocabulary 32,768, trained context 32,768. Its GGUFs declare the `llama`
architecture. The Mistral family had no file with a cell, only the arch-mistral placeholder (no GGUF of this model
carries the `mistral` label). Two rows join: bartowski's Q8_0 and Q4_K_M, from one repository at one revision.
Mistral AI publishes no GGUF of this model.

**What is not flattering, first:**
- **The repository's two tokenizers disagree.** Mistral AI ships a SentencePiece model (`tokenizer.model`) and a
  `tokenizer.json`. Against the SentencePiece model (the gated reference), 0 of 721 corpus strings differ. Against
  tokenizer.json, 44 of 721 differ, 40 of them strings that begin with whitespace, where tokenizer.json drops or
  shifts one leading space. Clients that tokenize with tokenizer.json get different tokens from Runner on such
  strings. The fidelity corpus did not reach a differing spot: every scored position matched by id.
- **The template the files embed is not the one Runner serves.** These GGUFs embed an older template in the v0.1 form.
  Runner detects v0.3 by the vocabulary and renders the publisher's current v0.3 template (R4.22.7), which it matches
  exactly. Against the embedded template, the `mistral-file` conformance family records 7 known differences (the
  existing backlog, none new). A client that renders the file's own template sends a slightly different prompt from
  Runner's.
- **The Q8_0 is over an 8 GB machine** (7.70 GB): there is no Metal cell for it.
- **Window 2 is 271 positions, not 272.** With the BOS, the 4,096-token score file ends one token short of a 272nd
  position. The first attempt at 272 failed on that and was rerun at 271.
- **The existing Q4_K_M compat entry** (the same file) declares greedy_reference and long_context, which this host
  did not execute (see "The cells").
- **Scope:** two windows over one corpus, CPU score files, inside 4,096 tokens of a 32,768-token context. Task quality
  is not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), for each file on its own.

## The files and the reference

- **Reference:** `mistralai/Mistral-7B-Instruct-v0.3` at `c170c708c41dac9275d15a8fff4eca08d52bab71` (not gated;
  licence apache-2.0), the three HF-format safetensors shards (not `consolidated.safetensors`), each equal to the
  Hub's LFS record. Stored in bfloat16.
- **Reference path:** transformers 5.16.1, the native MistralForCausalLM (no remote code), `scripts/gold-logits.py
  --upcast-at-use --force-bos` (float32 arithmetic, cast at use per decoder block; the score files begin with the BOS).
- **Files** from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` at `61fd4167fff3ab01ee1cfe0da183fa27a944db48`, each sha256
  checked against the Hub's LFS record on download and as held, with bartowski's importance matrix (Q8_0 does not use
  one):
  - Q8_0 `Mistral-7B-Instruct-v0.3-Q8_0.gguf`, 7,702,565,088 bytes, sha256
    `404857e776114baada71a08ebd3bba79d721ec7fca99705e7e7b892ae8bc583f`;
  - Q4_K_M `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`, 4,372,812,000 bytes, sha256
    `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6` (the file the existing manifest entry
    already pins).
  - `lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF` carries byte-identical files (the same LFS sha256), so this
    measurement covers both repositories.
- **Windows:** two windows at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, from position 8 (272 positions) and
  from 15 (271), inside the 4,096-token file. Token-id keyed: every position matched by id, no token spelling
  mismatches, no dropped reference mass.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,096 tokens, `-c 4096`) from the clean release-recipe STRICT build of main 85b6bef8 (runner 1.18.0, gcc 13.3.0,
  sha256 `63e7ebc9…`). No score log names YaRN. Between that main and 5ac478c6, this branch's base, the one engine
  change is a Phi-3 LongRoPE load warning, which this model does not reach.

## Against the reference

| File | Window | Mean KLD | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| **Q8_0** | 1 (272) | 0.0015 | 0.029 | 98.16% | 100% |
| | 2 (271) | 0.0008 | 0.019 | 98.89% | 100% |
| | **pooled 543** | **0.0011 ± 0.0002** | 0.029 | 98.53% | **100% (391/391)** |
| **Q4_K_M** | 1 (272) | 0.0222 | 0.272 | 93.01% | 99.49% |
| | 2 (271) | 0.0216 | 0.176 | 94.46% | 98.97% |
| | **pooled 543** | **0.0219 ± 0.0013** | 0.272 | 93.74% | **99.23% (388/391)** |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0009 to 0.0014 for the Q8_0 and 0.0193 to 0.0245 for the Q4_K_M. The pooled
medians are 0.0001 and 0.0116.

Per position: the `gold-*.json` files (FILES.sha256).

## The tokenizer

- difftok against Mistral AI's SentencePiece model at c170c708 (the reference ids captured in
  `tests/compatibility/tokenizer-references/mistral-7b-instruct-v0.3.json`): 0 of 721 corpus strings differ, on both
  files. Tokenizer pre `default`.
- Against the repository's tokenizer.json (informational, as R6.2.2 recorded): 44 of 721 differ, 40 of them strings
  that begin with whitespace. Examples (Runner, which matches SentencePiece, then tokenizer.json):
  - `" "`: [1027] against [29473];
  - `" hello"`: [29473, 7080, 29477] against [7080, 29477];
  - `"  hello"`: [1027, 7080, 29477] against [29473, 7080, 29477];
  - `"   hello"`: [3055, 7080, 29477] against [1027, 7080, 29477].

## The template

- Runner picks its `mistral` renderer, the v0.3 form with the tool protocol (`[AVAILABLE_TOOLS]`, `[TOOL_CALLS]`).
  The `mistral` conformance family against the publisher's current template (sha256 `e16746b4`): 18 cases compared,
  18 identical in text and tokens, 3 not expressible by the family.
- The `mistral-file` family against the template these files embed (the v0.1 form, sha256 `26a59556`): 7 known
  differences (the backlog, 0 new, 0 changed), 9 not expressible. Runner does not serve this form for these files.
- No template change here. No other roster row uses the `mistral` template entry. Of the manifest entries,
  mistral-nemo-instruct-2407-q4_k_m does (its pinned GGUF's older template detects as the v0.3 form), along with the
  existing entry for this Q4_K_M.

## What the file resolves to on load

Architecture llama, template mistral, the native mistral_json tool protocol, sampling preset mistral. CUDA runs all 32
layers on the device.

## The cells

On a CUDA MIG slice host (1g.24gb, sm_120), the CPUs shared with other work, from the clean release-recipe STRICT build
of main 5ac478c6, this branch's base (runner sha256 `1bcf3986…`, provenance commit 5ac478c6, dirty false), built before
the rows were added (R6.67).
- **tier-smoke**, `--pin` then `--record`, on the CPU and CUDA cells of both rows: all four pass with the fact found.
  CPU and CUDA give the same reply bytes for each file (Q8_0 `e3c8beb8`, Q4_K_M `23d8e0c0`).
- **Metal, the Q4_K_M only**, on an Apple M1 with 8 GB, from a clean STRICT build of 6d4a4d1f (the row commit):
  tier-smoke `--pin` then `--record` passes with the fact found, and the reply is byte for byte the Linux reply
  (`23d8e0c0`). The Q8_0 does not fit that machine and has no Metal cell.
- **LoRA smoke** on both files: `--train` on eight lines (two prompt/completion pairs, four times each), 4 steps at
  rank 8, then serving with `--lora`. Both train and serve (224 adapted projections); the adapted model completes
  "The capital of France is" with "Paris."
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks`,
  `1.18.0-2026-11-13-linux-x86_64-mistral-7b-instruct-v0.3.json`):
  - Q8_0 (a new entry): load, tokenizer (0 of 721 against the SentencePiece reference ids; tokenizer.json 44 of 721,
    informational), cpu_cuda (9 of 9 identical, `cpu_cuda/mistral-7b-instruct-v0.3-q8_0.json`) and chat pass;
    complete.
  - Q4_K_M (the entry already in the manifest, its checks and notes unchanged): load, tokenizer, cpu_cuda (9 of 9),
    chat and tool (the agent-torture matrix, 8 of 8 passed, `tool/mistral-7b-instruct-v0.3-q4_k_m.json`) pass.
    **Two declared checks did not run here**, so the entry and the report are not complete: greedy_reference (its
    reference is llama.cpp b10076, which this host does not hold; no other build was put in its place) and
    long_context (no runner for that check class on this host).
  The files were held in memory for the run; a lab wrapper kept their links, and the sha256 check read the held bytes.
