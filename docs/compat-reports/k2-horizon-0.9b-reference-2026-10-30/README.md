# K2-Horizon-0.9B: both publisher files against the float32 reference (2026-10-30)

**Unflattering first: the publisher's Q4_K_M fails the fidelity bar.** Against IFM's BF16 parent computed in float32,
`K2-Horizon-1B-Q4_K_M.gguf` reads a mean KLD of **0.108** and a top-1 of **81.7%**, against a bar of 0.05 and 97%
margin-qualified top-1 (it reaches 97.1% there, so the mean fails, not the qualified top-1). The Q8_0 of the same parent
on the same build reads **0.0037** and 97.3%, with 100% margin-qualified, so the gap is the Q4_K_M file's quantization,
not the arithmetic. The Q8_0 is the fidelity file. The Q4_K_M is admitted as a Tier 2 row of its own with this
failure first in its claim, as MiniCPM5's Q4_K_M is: Runner runs the file people download, correctly. The llama.cpp column (2026-11-01) is below,
with Runner the closer on both files.

## Setup

- **Reference:** `IFM/K2-Horizon-0.9B` at revision `d50881f4f8dc`, the publisher's own modeling code
  (`K2HorizonForCausalLM`, `trust_remote_code`), transformers 5.22 and torch 2.15, float32 on the CPU of the Metal laptop
  (Apple M1, 8 GB), `scripts/gold-logits.py` (the `logits_to_keep` version on `model/mimo-v2.6-9b`). One forward over
  4,510 tokens took 73 seconds, with a peak footprint of 5.3 GB and no swapping.
- **Positions:** 300 at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, positions 8 to 4,493; the reference and both
  Runner sides see the same token ids, BOS included. The 0.9B's tokenizer.json adds no BOS on a raw encode. Its chat
  template writes one, and Runner prepends it on every encode, so the reference was given the same BOS (`--force-bos`).
- **Runner sides:** score files from `runner --score --score-top 20 -c 9216` on the Metal laptop at `78bc5b5b`, on the
  GPU, matched to the reference by token id at all 300 positions, with no dropped mass.
- **Files:** `K2-Horizon-1B-Q8_0.gguf` (sha256 `741fb9ad...965dd547`) and `K2-Horizon-1B-Q4_K_M.gguf` (sha256
  `78bff592...078c79f2`), both from `IFM/K2-Horizon-0.9B-GGUF` at revision `118965081cf1`.

## Against the reference

| File | Positions | Mean KLD | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|
| Q8_0 | 300 | **0.0037** | 0.035 | 97.3% | 100% |
| Q4_K_M | 300 | **0.108** | 0.722 | 81.7% | 97.1% |

The paired test in `gold-logits-metal-q8-q4.json` says the Q8_0 is the closer of the two: McNemar exact p 0.031 on 205
margin-qualified positions, Wilcoxon on per-position KLD p below 1e-50.

By depth (mean KLD and top-1, 100 positions each):

| Token positions | Q8_0 | Q4_K_M |
|---|---|---|
| 8 to 1,493 | 0.0033, 95% | 0.110, 77% |
| 1,508 to 2,993 | 0.0043, 99% | 0.093, 80% |
| 3,008 to 4,493 | 0.0036, 98% | 0.121, 88% |

- **Q4_K_M worst position:** 3,728, KLD 0.72, where the reference's top-1 is `in` and the file's is `the`.
- **Q8_0 worst position:** 3,503, KLD 0.035, on ` path`, which both keep as top-1.

The rope is part of what this checks. The 0.9B is YaRN x16 over 8,192 with `beta_fast` 128 and `beta_slow` 4, and Runner
reads those ends from the file in this change (they were 32 and 1 before). At 0.0037 the Q8_0 is as close to its parent
as MiniCPM5's Q8_0 is to its own (0.0033).

## llama.cpp column (2026-11-01)

llama.cpp mainline `dde879bc` (`llama-server`, CPU build, x86-64 with AVX-512, flash attention off, `-c 9216`), measured against the same float32 reference at the same 300 positions with `scripts/gold-logits.py --endpoint-ids`: each position is sent as the reference's own token ids, the special prefix included, so both engines see the context the reference saw.

| File | Engine | Positions | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| Q8_0 | llama.cpp, CPU | 300 | **0.0064** | 0.0021 | 0.083 | 97.00% | 100.00% |
| Q4_K_M | llama.cpp, CPU | 300 | **0.1174** | 0.0821 | 0.788 | 82.67% | 96.10% |

- **Q8_0:** Runner 0.0037 against llama.cpp 0.0064; paired per-position KLD (llama.cpp minus Runner) Wilcoxon z 11.4, p 7.3e-30; margin-qualified top-1 discordant positions 0 for Runner, 0 for llama.cpp.
- **Q4_K_M:** Runner 0.1078 against llama.cpp 0.1174; paired per-position KLD (llama.cpp minus Runner) Wilcoxon z 3.1, p 0.0017; margin-qualified top-1 discordant positions 2 for Runner, 0 for llama.cpp.

Why the two engines differ, one numbered sentence each:

1. llama.cpp's CPU dot products quantize the activations to 8 bits before the integer dot, and Runner keeps the activations in float32 against the dequantized weights, which is the closer of the two to the float32 reference here.
2. The GGUFs declare no `add_bos_token`, so llama.cpp prepends no BOS to a text prompt, while the publisher's tokenizer.json prepends one to every encode and Runner follows it: a llama.cpp user who sends text to this file gets a context without BOS, which reads 0.091 mean KLD on the Q8_0 and 0.195 on the Q4_K_M against the reference (`*-text.json`), against the column's 0.0064 and 0.117 with it.

Files: `gold-logits-llamacpp-q8_0.json`, `gold-logits-llamacpp-q4_k_m.json` (the column); `gold-logits-llamacpp-q8_0-text.json`, `gold-logits-llamacpp-q4_k_m-text.json` (the text-path observation in 2).

## Quant bar: Q4_K_M against Q8_0

| Comparison | Positions | Mean KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|
| Q4_K_M vs Q8_0, from the two score files | 300 | **0.098** (bar 0.05) | 83.3% | 97.6% (bar 97%) |

Computed from the two score files at the reference's 300 positions, because `scripts/kld-compare.py` serves both files at
once, which the 8 GB laptop cannot hold. The method is recorded in `quant-bar-q4_k_m-vs-q8_0.json`. It agrees with the
reference column: the file is about 0.1 from both its parent and its own Q8_0.

## Files

- `gold-logits-metal-q8-q4.json`: the reference run, both sides, per position.
- `quant-bar-q4_k_m-vs-q8_0.json`: the quant bar, per position, with its method.
