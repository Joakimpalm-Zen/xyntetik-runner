# K2-Horizon-7B: both publisher files against the float32 reference (2026-10-30)

**Unflattering first: the publisher's Q4_K_M fails the fidelity bar on both counts.** Against IFM's BF16 parent computed
in float32, `K2-Horizon-7B-Q4_K_M.gguf` reads a mean KLD of **0.1225** against a bar of 0.05, with a top-1 of **84.3%** and
a margin-qualified top-1 of 95.8% against a bar of 97%. The Q8_0 of the same parent reads **0.0159**, with 95.0% top-1
and 100% margin-qualified, so the gap is the Q4_K_M file's quantization, not the arithmetic: the Q8_0 is the fidelity
file. The llama.cpp column (2026-11-01) is below.

## Against the reference

| File | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0159** | 0.0056 | 0.335 | 95.0% | 100% (213 positions) |
| Q4_K_M | 300 | **0.1225** (bar 0.05) | 0.068 | 2.71 | 84.3% | 95.8% (bar 97%) |

The paired test in `gold-logits-cpu-q8-q4.json` says the Q8_0 is the closer of the two: McNemar exact p 0.004 on the 213
margin-qualified positions, Wilcoxon on per-position KLD p below 1e-49.

By depth (mean KLD, top-1 and margin-qualified top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,493 | 0.0135, 97%, 100% | 0.104, 86%, 95.8% |
| 1,508 to 2,993 | 0.0209, 96%, 100% | 0.125, 87%, 97.3% |
| 3,008 to 4,493 | 0.0135, 92%, 100% | 0.138, 80%, 94.0% |

- **Q4_K_M worst position:** 3,293, KLD 2.71, where the reference's top-1 is ` CUDA` and the file's is ` model`.
- **Q8_0 worst position:** 2,708, KLD 0.335, on ` model`, which both keep as top-1.

The 7B is the member that carries the grouped RMSNorm (`group_norm_groups` 4) the 0.9B does not, so this is that norm's
first check against the publisher's own code. At 0.0159 the Q8_0 is further from its parent than the 0.9B's Q8_0 is
from its own (0.0037). It holds 100% of the margin-qualified positions, so the larger figure sits on low-margin tokens.

### Setup

- **Reference:** `IFM/K2-Horizon-7B` at revision `4622f94a6632`, the publisher's own modeling code
  (`K2HorizonForCausalLM`, `trust_remote_code`), transformers 5.16.1 and torch 2.14.0, float32 on the lab CPU lane (16
  cores, no GPU), by `scripts/gold-logits.py` from `model/mimo-v2.6-9b` at `fcd461ef` (the scored rows only). Another
  tenant ran about 5 CPUs on the same lane through the run. The lab volume reads about 8 MB/s uncached, so the BF16
  checkpoint (17 GB) was read into the page cache first (39 minutes); the forward over 4,510 tokens took 72 seconds.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493. The 7B's tokenizer
  prepends BOS on a raw encode, as Runner does, so the reference and both Runner sides see the same token ids at all 300
  positions with no `--force-bos`, and no dropped mass.
- **Runner sides:** score files from `RUNNER_SCORE_CHUNKED=1 runner --score --score-top 20 -c 9216 -t 16 --gpu off` on
  the same lane at `5cc154e2`, this branch's head at the time.
- **Files:** `K2-Horizon-7B-Q8_0.gguf` (9,573,964,160 bytes, sha256 `5a98a289...b47809bf`) and `K2-Horizon-7B-Q4_K_M.gguf`
  (5,592,217,984 bytes, sha256 `eb89c15a...2ebdd28e4e`), both from `IFM/K2-Horizon-7B-GGUF` at revision `7a75dbda05e3`.

## llama.cpp column (2026-11-01)

llama.cpp mainline `dde879bc` (`llama-server`, CPU build, x86-64 with AVX-512, flash attention off, `-c 9216`), measured against the same float32 reference at the same 300 positions with `scripts/gold-logits.py --endpoint-ids`: each position is sent as the reference's own token ids, the special prefix included, so both engines see the context the reference saw.

| File | Engine | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| Q8_0 | llama.cpp, CPU | 300 | **0.0176** | 0.0071 | 0.389 | 93.67% | 100.00% |
| Q4_K_M | llama.cpp, CPU | 300 | **0.1284** | 0.0747 | 2.148 | 82.67% | 95.31% |

- **Q8_0:** Runner 0.0159 against llama.cpp 0.0176; paired per-position KLD (llama.cpp minus Runner) Wilcoxon z 4.3, p 1.6e-05; margin-qualified top-1 discordant positions 0 for Runner, 0 for llama.cpp.
- **Q4_K_M:** Runner 0.1225 against llama.cpp 0.1284; paired per-position KLD (llama.cpp minus Runner) Wilcoxon z 5.6, p 2e-08; margin-qualified top-1 discordant positions 2 for Runner, 1 for llama.cpp.

Why the two engines differ, in one numbered sentence, and one thing that does not:

1. llama.cpp's CPU dot products quantize the activations to 8 bits before the integer dot, and Runner keeps the activations in float32 against the dequantized weights, which is the closer of the two to the float32 reference here.
2. The BOS is not a difference here: unlike the 0.9B's, the 7B's GGUFs declare `add_bos_token = true`, so a llama.cpp text prompt carries the BOS the publisher's tokenizer prepends, and the text path reads the same as the ids (`*-text.json`).

Files: `gold-logits-llamacpp-q8_0.json`, `gold-logits-llamacpp-q4_k_m.json` (the column); `gold-logits-llamacpp-q8_0-text.json`, `gold-logits-llamacpp-q4_k_m-text.json` (the text-path observation in 2).

## Files

- `gold-logits-cpu-q8-q4.json`: the reference run, both sides, per position (host paths replaced by file and checkpoint
  names).
