# Ornith-1.5-35B-A3B: ornith-ai's files against their BF16 parent in float32 (2026-11-05)

**What is not flattering, first:**
- **The floor:** the row's file is ornith-ai's own Q8_0. Its median KLD against the parent, 0.0154, is 3.7x
  Qwen3.6-35B-A3B's Q8_0 (0.0042) on the same base architecture by the same method.
- **The margin:** its mean, **0.0484**, passes the 0.05 bar by 0.0016, and its margin-qualified top-1, **97.50%**,
  passes 97% by half a point. Its middle window (positions 1,508 to 2,993) reads 0.0754 / 95.1% on its own, outside
  both bars.
- **The K-quant files:** ornith-ai's Q6_K (0.1033) and Q4_K_M (0.1710) miss. Another engine reads the same error
  spikes in them.
- **The file's size:** 37.8 GB, so on a 24 GB CUDA card it runs split, 26 of 40 blocks on the device.
- **Scope:** one 300-position window over one corpus. Text only: the vision projector is not loaded.

The kills were declared before each read: the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05
and margin-qualified top-1 at least 97%), and the smallest passing file is the row's.

## The files and the reference

- **Parent:** `ornith-ai/Ornith-1.5-35B-A3B` at `10fbf86fed7ecee4a061f8b499a618f46001cac1` (MIT), the publisher's
  checkpoint, held whole from the Hub (16 files, 71.9 GB).
- **Reference path:** transformers 5.16.1, the text model of the composite checkpoint, `scripts/gold-logits.py
  --upcast-at-use`: float32 arithmetic on the stored bfloat16, cast at use per decoder block (40 blocks).
- **Window:** 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, token-id keyed, every position
  matched by id. 240 positions are margin-qualified.
- **Files:** ornith-ai's own GGUF repository, `ornith-ai/Ornith-1.5-35B-A3B-GGUF` at `12393612`.
  - Q8_0: 37,802,149,280 bytes, sha256 `de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3`.
  - Q6_K: 29,208,731,392 bytes, sha256 `15d4658bbfc9c6034621729c15bbb50662c82b32a7ddd9624a1e545a74bdbb4b`.
  - Q4_K_M: 21,713,463,040 bytes, sha256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`.
  - Each file's sha256 was checked against the Hub's LFS record as it was held. The repository's README states no
    base revision, converter or importance matrix.
- **Runner:** main 973b2836 for the score files (`--score --score-top 20 --score-tokens 4510 -c 4510 --gpu off` on the
  host's CPUs), and main b3e87703 for the cells. The change between them is R3.32's deepseek2 CUDA prefill, which no
  CPU score here touches.

## Against the reference

| File | Bytes | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|---|
| **Q8_0** | 37,802,149,280 | **0.0484** | 0.0154 | 0.911 | 91.33% | **97.50%** | **pass** |
| Q6_K | 29,208,731,392 | 0.1033 | 0.0234 | 4.687 | 89.00% | 95.83% | fail |
| Q4_K_M | 21,713,463,040 | 0.1710 | 0.0600 | 4.717 | 83.67% | 92.08% | fail |

By depth (mean KLD / margin-qualified top-1, 100 positions each):

| File | 8 to 1,493 | 1,508 to 2,993 | 3,008 to 4,493 |
|---|---|---|---|
| Q8_0 | 0.0384 / 98.8% | 0.0754 / 95.1% | 0.0314 / 98.7% |
| Q6_K | 0.0961 / 97.6% | 0.1159 / 96.3% | 0.0978 / 93.5% |
| Q4_K_M | 0.1425 / 97.6% | 0.2449 / 87.7% | 0.1254 / 90.9% |

The Q5_K_M was not read. The declared order read the Q6_K before it, and a lower-bit file from the same publisher was
not expected to do better. That is not a reading.

## The K-quant files' spikes are the files'

- **The pattern:** the Q6_K and the Q4_K_M put their largest errors at the same positions, where the Q8_0 is small.
  - Position 338: Q6_K 4.69, Q4_K_M 4.72, Q8_0 0.71.
  - Position 2,333, where the reference's margin is 3.0: 2.94, 3.00 and 0.011.
- **Another engine:** llama.cpp 36a0e2f served the same Q6_K (CPU, one slot, flash attention off) and was read
  against the same reference at the same positions (`gold-q6_k-runner-vs-llamacpp.json`).

| Engine on the Q6_K | Mean KLD | Median | Max | Margin-qualified top-1 | Position 338 | Position 2,333 |
|---|---|---|---|---|---|---|
| Runner | 0.1033 | 0.0234 | 4.687 | 95.83% | 4.687 | 2.943 |
| llama.cpp | 0.1191 | 0.0313 | 5.474 | 93.75% | 5.474 | 3.707 |

- llama.cpp reads the same spikes, larger. Runner is the closer engine (Wilcoxon p 7e-6; McNemar 8 / 3, p 0.23).
- **The suspects:** 64 tensors are Q6_K in both K-quant files and Q8_0 in the Q8_0. Which one carries the spikes is
  not measured.
  - `output.weight`: 1;
  - `ffn_down_exps.weight`: 21 of 41 blocks;
  - `ffn_down_shexp.weight`: 21;
  - `attn_qkv.weight`: 14 of the 30 Gated DeltaNet blocks;
  - `attn_v.weight`: 7 of 11.

## The floor

ornith-ai's own BF16 GGUF (71.07 GB, sha256 `044d7f8b55580720...`), scored the same way and read against the same
parent beside the Q8_0 (`gold-bf16-vs-q8_0.json`):

| File | Mean KLD | Median | Max | Margin-qualified top-1 |
|---|---|---|---|---|
| BF16 GGUF | 0.0262 | 0.0057 | 0.823 | 98.33% |
| Q8_0 | 0.0484 | 0.0150 | 0.911 | 97.50% |

- **What a clean conversion would read:** a BF16 conversion of a parent should sit near zero. Runner on a BF16 GGUF
  of Qwen3.8-27B reads 0.00002 against its float32 parent (R4.32). This one sits at 0.026.
- **So:** about half the Q8_0's distance is already in the publisher's BF16 file, from the conversion or from the
  weights it was made from against the parent at `10fbf86f`. The rest is the Q8_0's quantization. The BF16 is the
  closer file (Wilcoxon p 2.3e-16).
- **No source revision is declared:** neither the file's header (no `general.source` or `general.base_model` keys)
  nor the GGUF repository's README names one. The model repository's last commit, the day before the GGUFs were
  uploaded, re-uploaded its last safetensors shard.
- **Not separated here:** a file cause from an engine cause on this model. Runner agrees with transformers on
  Qwen3.6-35B-A3B on the same architecture (its Q8_0's median is 0.0042).
- **Follow-ups, not run:** a tensor-level comparison of the BF16 GGUF against the parent, and which of the K-quant
  files' 64 shared Q6_K tensors carries their spikes.

## Tokenizer and template

- **Tokenizer:** 0 of 721 corpus strings and 0 of 247,091 vocabulary entries differ from the parent's
  tokenizer.json.
- **Three templates:**
  - the repository's `chat_template.jinja` (7,536 bytes): Ornith 1.0's with Qwen3.6's tool-argument rule;
  - `tokenizer_config.json`'s template: Qwen3.6-35B-A3B's byte for byte, the base model's left behind;
  - the GGUF's own (7,828 bytes, sha256 `f55f5293`, the same in the Q8_0 and the Q4_K_M): the repository's plus a
    merge of the first two system or developer messages.
- **Conformance:** the file's own template is the oracle (`ornith15-35b-a3b` in `scripts/template-conformance.py`,
  renderer `ornith`). 24 of 24 cases are identical in text and tokens.
  - A system or developer turn after the conversation has started is refused by Runner. The file's template drops it
    silently.
  - No house case carries a boolean or null tool argument, the one place 1.0's and 1.5's argument rules differ.

## The cells

- **tier-smoke:** the CPU and CUDA pins are byte-identical (`aec461cf`). Recorded on the lab CPU lane and a CUDA MIG
  slice (1g.24gb, sm_120).
- **gpu-identity** on the fitted split (26 of 40 blocks on the device):
  - routing pinned: 4.78e-6 of the logit range (limit 1e-3);
  - free routing: 0.00358 with 4 of 30,400 expert slots differing, reported, not graded.
- **compat_matrix:** load, tokenizer and chat pass; cpu_cuda `pass_margin_qualified`.

`FILES.sha256` lists every file here except this README.
