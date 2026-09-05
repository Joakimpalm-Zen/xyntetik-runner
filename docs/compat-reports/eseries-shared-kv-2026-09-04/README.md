# Gemma 4 E-series shared-KV exports: the evidence

Measured 2026-09-04 on an M1 MacBook Pro (8 GB, arm64, macOS), runner 0.4.7 +
the shared-KV loader fix, CPU path (`--gpu off`) throughout. The weights are
several times the free RAM on this box, so every real-model number here is
slow; none of them is inexact for that reason.

## The defect

A gemma-4 E-series layer at or past `block_count -
attention.shared_kv_layers` computes no K and no V: it attends over the cache
an earlier layer filled. Its `attn_k.weight`, `attn_v.weight` and
`attn_k_norm.weight` are therefore unreachable, and every quantized E-series
export published since the BF16 one leaves them out. E4B declares
`block_count: 42` and `shared_kv_layers: 18`, so layers 24..41 carry 14
tensors each instead of 17: 666 tensors where the BF16 export has 720.

The loader required all three on every attention layer, so runner 0.4.7
refused those files at load:

    error: missing tensor blk.24.attn_k.weight

Reproduced here on Google's own QAT release with a binary built from the
pre-fix source; the same binary rebuilt with the fix loads and generates.

## Gate 1: bit identity, 720-tensor vs 666-tensor, same weights

`scripts/gguf-drop-shared-kv.py` rewrote the local full-form
`gemma-4-E4B-it-Q4_K_M` (720 tensors, 5,405,170,144 bytes) into the compact
form (666 tensors, 5,367,021,952 bytes). Exactly 54 tensors were dropped, all
on layers 24..41, and every one of the 666 survivors was compared by SHA-256
against its source blob: 666/666 byte-identical, no shape or type change.

On the same file pair, the runner must therefore move no bit:

| run | positions | result |
|---|---|---|
| `--score` (default solo path, `-f` = first 180 bytes of `tests/fixtures/mixed-corpus.txt`) | 36 | score JSON byte-identical |
| `--score` with `RUNNER_SCORE_CHUNKED=1` (first 700 bytes of the same corpus) | 159 | score JSON byte-identical |
| greedy 24 tokens, `--temp 0`, `-p "The capital of France is"` | - | stdout byte-identical |

Absolute values (identical on both files): solo 36 positions `nll_mean`
4.7358444, `ppl` 113.959646, top1 15/36; chunked 159 positions `nll_mean`
3.98361059, `ppl` 53.7106113, top1 71/159 (0.446540881).

This is the anchor: the expected answer does not come from the runner
agreeing with itself under a change, it comes from the architecture, which
says those weights are never read.

## Gate 2: Google's official QAT Q4_0, as published

`google/gemma-4-E4B-it-qat-q4_0-gguf`, file `gemma-4-E4B_q4_0-it.gguf`,
5,154,941,280 bytes, sha256
`676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`.

- 666 tensors. Layers 0..23 carry 17 tensors each; layers 24..41 carry 14,
  missing exactly `attn_k.weight`, `attn_v.weight`, `attn_k_norm.weight`.
- Types: 342 Q4_0, 321 F32, 2 Q6_K, 1 F16.
- Load: **pass**. Chat smoke: **pass** ("2 + 2 equals **4**.").
  Ledger row: `docs/compat-reports/0.4.7-2026-09-04-macos-eseries-qat.json`.
- Greedy, `--temp 0`, 24 tokens, `-p "The capital of France is"`:
  "The capital of France is Paris." then end of text.
- `--score` (chunked, 159 positions, same corpus prefix as gate 1):
  `nll_mean` 3.53626308, `ppl` 34.3383595, top1 77/159 (0.48427673).

Classification: **loads and answers on the CPU path; not certified against a
second engine.** No llama.cpp comparison was run for this row, no CUDA, no
Metal. That is why the manifest row declares only `load` and `chat` and
carries no `reference`.

### The Q4_K_M comparison, and what it is not

A raw KLD of the QAT Q4_0 against the 720-tensor Q4_K_M would be the natural
sanity check that the two files are the same model.
`scripts/kld-compare-raw.py` needs both models resident at once (about 10.5
GB) and walks the corpus position by position through two HTTP servers; on an
8 GB box that is not a slow measurement, it is a different measurement. **It
was not run.** What is reported instead is the
teacher-forced per-position comparison the two `--score` runs above already
contain, over the identical 159 positions and identical tokenization:

- mean |Δ logprob| 0.847440, median 0.471420, max 6.602915
- `nll_mean` 3.983611 (Q4_K_M) vs 3.536263 (QAT Q4_0), a gap of 0.447348
- top1 71/159 vs 77/159

Two different quantizations of the same base disagree at this scale; the
number is here as a sanity check that both files are the same model, not as a
fidelity claim, and nothing gates on it.

## Gate 3: the fixture gate CI actually runs

CI has no multi-GB models. `scripts/make-test-model.py --drop-kv` builds the
same shape at fixture scale: `shared` omits the three tensors on every
shared-KV layer, a comma list of indices omits them on named layers. The
generator draws its random weights in the same order either way, so a
`--drop-kv` fixture's surviving tensors are byte-for-byte the full fixture's
and the identity assertion is a real comparison. Gates in
`tests/test_eseries.py`:

- `test_the_dropped_fixture_really_is_missing_those_tensors`
- `test_shared_kv_layers_may_omit_their_unread_kv_tensors`
- `test_dropping_them_changes_nothing_the_model_computes`
- `test_a_kv_owning_layer_missing_its_k_is_still_refused` (layers 0, 1, 2)

The last one is the boundary: below `kv_from_start` a layer computes its own
K, so a file without it is broken rather than compact and is still refused by
name.

## Files here

| file | what |
|---|---|
| `e4b-q4km-720.score-solo.json`, `e4b-q4km-666.score-solo.json` | gate 1, default solo scoring path, 36 positions |
| `e4b-q4km-720.score-chunked.json`, `e4b-q4km-666.score-chunked.json` | gate 1, chunked scoring path, 159 positions |
| `gemma-4-E4B_q4_0-it.score-chunked.json` | gate 2, the official QAT Q4_0 over the same 159 positions |
