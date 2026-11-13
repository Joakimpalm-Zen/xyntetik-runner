# StableLM 2 1.6B Chat: mradermacher's files against Stability AI's float32 checkpoint (2026-11-13)

StableLM 2 1.6B Chat is Stability AI's chat model on the stablelm architecture (LayerNorm, partial rotary, q/k/v
biases). The family had no file with a cell. One row joins: mradermacher's Q8_0 from the plain repository.
Stability AI publishes no GGUF of this size; its own GGUF repository is the 12B chat model's only.

**What is not flattering, first:**
- **CPU only, on every build.** Runner's GPU backends implement RMSNorm and this model normalises with LayerNorm, so
  a CUDA or Metal build serves it on the CPU and says so at load (the architecture table's `cpu_only` rule in
  `src/arch.h`). The support page marks CUDA and Metal n/a; there is no device result for this model. compat's
  cpu_cuda check is not declared, since a CUDA build's run would be the CPU compared with itself.
- **Training is refused** at load ("LayerNorm (stablelm)"). Serving an adapter is declared supported by
  `--adapt-info`; with training refused there is no adapter of this model to serve, so it was not exercised.
- **Neither Q4_K_M is admitted.** The plain repository's Q4_K_M reads 0.1114 ± 0.0077 mean KLD and 93.95%
  margin-qualified top-1; mradermacher's i1 Q4_K_M with an importance matrix reads 0.0662 ± 0.0037 and 97.37%, its
  KLD point estimate over the bar.
- **The first fidelity run was void and was redone.** It scored 4,520-token files with `-c 4520`, past the model's
  trained context of 4,096, so Runner applied YaRN (x1.10) while the reference ran plain rope; those numbers
  (0.0166 for the Q8_0) compared two different position encodings. The numbers below come from 4,096-token files,
  272 positions per window, all inside the trained context.
- **This PR changes Runner's prompt for StableLM 2.** Before it, a chat request without a system message got no system
  turn, 11 tokens short of the publisher's template, which writes `You are a helpful assistant.` on every such
  conversation. The default is now written (template_default_system, keyed on the template's own literal). No other
  roster file held on the box (49 of the roster's 89 file names) and no conformance family's template carries that
  literal; the whole conformance suite renders every other family's 741 cases as before.
- **Scope:** two windows over one corpus, CPU score files. Task quality is not measured.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%). If the plain Q4_K_M failed, one read of the i1 Q4_K_M, a row only if both
its point estimates pass; no further files.

## The files and the reference

- **Reference:** `stabilityai/stablelm-2-1_6b-chat` at `f3fe67057c2789ae1bb1fe42b038da99840d4f13` (not gated; the
  Stability AI Non-Commercial Research Community License, the model card says `license: other` and ships the text),
  held whole; both safetensors shards equal the Hub's LFS records. Stored in float32.
- **Reference path:** transformers 5.16.1, StableLmForCausalLM, `scripts/gold-logits.py --upcast-at-use` (a no-op on
  float32 storage). transformers 5.16.1 and 4.57.1 both read the tokenizer.json as written (8,050 corpus tokens).
- **Files**, each sha256 checked against the Hub's LFS record on download and as held:
  - `mradermacher/stablelm-2-1_6b-chat-GGUF` (the plain repository) at `037d59f734d37096f8317ddf9f732ceaa857f0af`:
    - Q8_0 `stablelm-2-1_6b-chat.Q8_0.gguf`, 1,751,882,848 bytes, sha256
      `70fa97127815cffc3e5dd004976c8986fca1011c74bb8fd581c784cdc00b9760` (the row);
    - Q4_K_M `stablelm-2-1_6b-chat.Q4_K_M.gguf`, 1,031,445,600 bytes, sha256 `255be8cc…` (measured, not admitted).
  - `mradermacher/stablelm-2-1_6b-chat-i1-GGUF` at `ee6516b749a5`, the most-downloaded conversion of this model:
    `stablelm-2-1_6b-chat.i1-Q4_K_M.gguf`, 1,031,445,856 bytes, sha256 `4c751644…` (measured, not admitted).
  - No file carries an importance matrix except the i1 one.
- **Windows:** two windows of 272 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, from position 8 (8
  to 4073) and from 15 (15 to 4080), inside the 4,096-token context. Token-id keyed: every position matched by id, no
  token spelling mismatches, no dropped reference mass.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,096 tokens, `-c 4096`) from the clean release-recipe STRICT build of main 36bfe3f6 (runner 1.18.0, gcc 13.3.0,
  sha256 `8035a284…`). No score log names YaRN.

## Against the reference

| File | Window | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| **Q8_0** | 1 | 0.0043 | 0.0008 | 0.062 | 97.79% | 100% (185/185) |
| | 2 | 0.0039 | 0.0008 | 0.028 | 98.53% | 100% (195/195) |
| | **pooled 544** | **0.0041 ± 0.0003** | 0.0008 | 0.062 | 98.16% | **100% (380/380)** |
| Q4_K_M (not admitted) | 1 | 0.1185 | 0.0775 | 2.988 | 82.35% | 95.68% (177/185) |
| | 2 | 0.1043 | 0.0658 | 1.457 | 81.25% | 92.31% (180/195) |
| | **pooled 544** | **0.1114 ± 0.0077** | 0.0687 | 2.988 | 81.80% | **93.95% (357/380)** |
| i1 Q4_K_M (not admitted) | 1 | 0.0705 | 0.0457 | 1.015 | 87.13% | 98.38% (182/185) |
| | 2 | 0.0619 | 0.0412 | 0.852 | 88.97% | 96.41% (188/195) |
| | **pooled 544** | **0.0662 ± 0.0037** | 0.0430 | 1.015 | 88.05% | **97.37% (370/380)** |

± is the standard error of the pooled mean over the per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 0) gives 0.0035 to 0.0047 for the Q8_0, 0.0974 to 0.1277 for the Q4_K_M and 0.0592 to 0.0739
for the i1 Q4_K_M, the last two wholly over 0.05.

Per position: the `gold-*.json` files (FILES.sha256).

## The tokenizer

- difftok against Stability AI's tokenizer.json at f3fe6705: 0 of 721 corpus strings and 0 of 99,474 vocabulary
  entries differ, on all three files. Tokenizer pre `stablelm2`.
- The reference ids are captured in
  `tests/compatibility/tokenizer-references/stablelm-2-1.6b-chat.tokenizer-json.json`.

## The template

- Runner picks its `chatml` renderer. A new conformance family `stablelm2` against the publisher's chat template (sha256
  `f02c5341`): 16 cases compared, 16 identical in text and tokens, after this PR's default-system entry (before it,
  10 of the 16 differed by the 11-token default system turn).

## What the file resolves to on load

Architecture stablelm, tokenizer pre stablelm2, template chatml, tool family qwen_json, sampling preset generic (there
is no StableLM preset), the CPU on every build.

## The cells

On the CPU of a CUDA MIG slice host (1g.24gb, sm_120), the CPUs shared with other work, from the clean release-recipe
STRICT build of dacf399e, this PR's first commit (runner sha256 `1d8d27cd…`, provenance commit dacf399e, dirty false),
built before the row was added (R6.67).
- **tier-smoke** on the CPU, `--pin` then `--record`: the cell passes with the fact found (reply `c29a76cd`). A CUDA
  build's run of the same row is the CPU's (tier-smoke refuses to record it as a CUDA cell: no layer was offloaded),
  so the row's CUDA and Metal cells are stated limits.
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks --require-complete`,
  `1.18.0-2026-11-13-linux-x86_64-stablelm-2-1.6b-chat.json`): load, tokenizer (0 of 721 against the captured
  reference ids) and chat pass. The chat check allows 400 tokens, as six other entries do: the model answers
  "What is 2+2?" with worked examples in several languages and ran past 200. The file was held in memory for the run;
  a lab wrapper kept its link, and the sha256 check read the held bytes.
