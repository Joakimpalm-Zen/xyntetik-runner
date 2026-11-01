# Ministral 3 3B Instruct 2512: two files against the float32 reference (2026-11-01)

**Unflattering first: the Q4_K_M fails the fidelity bar.** Against Mistral AI's BF16 weights computed in float32,
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` reads a mean KLD of **0.0623** (bar 0.05), with margin-qualified top-1
at **99.52%** (bar 97%). A 3B model at four bits missing the mean-KLD bar is the expected shape. The Q8_0 passes at
**0.0174** and 100%, and is the fidelity file. Both are the publisher's own conversions.

The Q8_0 also holds across the context length where the architecture's attention temperature first departs from 1
(position 16,384): 0.0061 mean KLD and 100% top-1 over 246 positions from 15,000 to 19,980.

This is the reference measurement for the `mistral3` admission (R4.4). No llama.cpp column was run.

## Setup

- **Reference:** `mistralai/Ministral-3-3B-Instruct-2512` at `b35d4dfe56c1`, transformers 5.16.1
  (`Mistral3ForConditionalGeneration`, driven with text only), torch 2.14.0 on the CPU. The arithmetic is float32 on
  the BF16 weights cast at use per decoder block (`scripts/gold-logits.py --upcast-at-use`). A full float32 load
  (`dtype=float32`) stopped in `q_proj` with BF16 weights meeting float32 activations; casting at use gives the
  float32 arithmetic without that load.
- **BOS:** the GGUFs declare `add_bos_token` and Runner prepends `<s>`, while the publisher's loaded tokenizer adds
  nothing to a bare text. The reference therefore gets the same prefix with `--force-bos`, so the two are compared
  position for position. Without it, every row sits one token off.
- **Window:** 300 positions at stride 15, positions 8 to 4,493 of `tests/fixtures/gold-corpus-2k.txt` (4,510 tokens
  with BOS). Both files are matched to the reference by token id at all 300 positions, with no dropped mass; 207 are
  margin-qualified.
- **Long window:** the same corpus three times (20,002 tokens), 250 positions requested from 15,000 at stride 20. 246
  were scored; the other 4 are skipped by the harness because their decoded prefix does not re-tokenize to the same
  ids. Q8_0 only.
- **Runner sides:** score files from `RUNNER_SCORE_CHUNKED=1 runner --score 20 -f CORPUS` (`-c 9216` for the window,
  `-c 24576` for the long window), commit `ec6d3cf7` (the admission branch, runner 1.10.0), on x86-64 with AVX-512,
  24 threads, CPU only.
- **Files**, each checked against the Hub's LFS digest:

| File | Repo at revision | Size | sha256 |
|---|---|---|---|
| `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | `mistralai/Ministral-3-3B-Instruct-2512-GGUF` at `eb599d408350` | 2,147,023,008 | `9ed150d4...f93fc5f8` |
| `Ministral-3-3B-Instruct-2512-Q8_0.gguf` | same | 3,652,204,704 | `8c2b72eb...852413ef` |

## Against the reference

| File | Window | Positions | Mean KLD | Median | p99 | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 8 to 4,493 | 300 | **0.0623** | 0.0376 | 0.341 | 1.77 | 92.0% | 99.52% |
| Q8_0 | 8 to 4,493 | 300 | **0.0174** | 0.0073 | 0.079 | 0.87 | 94.33% | 100% |
| Q8_0 | 15,000 to 19,980 | 246 | **0.0061** | 0.0002 | 0.125 | 0.42 | 100% | 100% |

- **Q8_0 against Q4_K_M:** the Q8_0 is the closer. Per-position KLD Wilcoxon z -14.9, p 3e-50. McNemar p 1.0
  (discordant margin-qualified positions: 1, for the Q8_0), because the Q4_K_M already agrees on the top-1 almost
  everywhere it is margin-qualified. No position differs by more than a nat between the two.
- **Across 16,384:** the long window's 66 positions below 16,384 read 0.0157 mean KLD, and its 180 above read 0.0026,
  all 246 with the reference's top-1. Past `original_max_position_embeddings` the model scales each query by
  1 + beta x ln(1 + floor(pos / 16384)) (the Llama 4 attention temperature, on every layer, offset 0). Runner applies
  it there, and a missing or misplaced scale would show here as a step at 16,384, which these rows do not have.
- **Q4_K_M worst position:** 968, KLD 1.77. The reference's top-1 is `ng`, the file's is `ed`.

## Files here

- `gold-logits-cpu-q4_k_m-cpu-q8_0.json`: the window, both files (side a Q4_K_M, side b Q8_0).
- `gold-logits-long-cpu-q8_0.json`: the long window, Q8_0.

The host paths in both are replaced by the repo at its revision and the corpus's repository path.

## Not measured

- A llama.cpp column.
- The CUDA and Metal cells; the roster rows carry a CPU cell only.
- Vision: Runner loads the text path, and the vision tower is not loaded.
- Task quality.
