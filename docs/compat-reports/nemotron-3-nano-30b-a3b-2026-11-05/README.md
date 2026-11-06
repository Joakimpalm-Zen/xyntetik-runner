# NVIDIA Nemotron 3 Nano 30B-A3B: ggml-org's conversion against NVIDIA's BF16 parent in float32 (2026-11-05)

**What is not flattering, first:**
- **CPU only:** CUDA refuses the file by name. Its sigmoid-plus-bias MoE router has no device kernel, as for Nemotron
  3.5 Lightning.
- **The anchor:** it is the model's declared dt limit (floor 0), as for Lightning. With the reference's configured
  clamp (`time_step_min` 0.001 read as a floor), the same file reads a mean KLD of 0.137.
- **A third-party conversion:** the file is ggml-org's automatic conversion, since NVIDIA publishes no GGUF. Its
  converter revision is not declared.
- **The tool gate:** case B (prose and parallel calls) fails buffered and streamed, and case C fails buffered. Each
  stops on finish_reason `length` before a parsed call.
- **The first load test found a Runner bug:** `-v` crashed on a file whose first block is Mamba-2. It is fixed in the
  same change, and the load test read again on the fixed build.

The kills were declared before each read: the load and layout, the tokenizer, the template, then the owner's bar
(mean KLD at most 0.05 and margin-qualified top-1 at least 97%). The smallest passing file is the row's.

## The files and the reference

- **Parent:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`
  (nvidia-nemotron-open-model-license), BF16, 63.2 GB.
- **Reference path:** transformers 5.16.1's own `NemotronH` (the corrected Mamba-2 scan; the modeling code the
  repository ships is not used), float32 arithmetic on the stored bfloat16, cast at use per decoder block.
  `scripts/gold-logits.py --upcast-at-use`, the weights cloned into process memory.
- **Window:** 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, token-id keyed, every position
  matched by id, 202 margin-qualified.
- **File:** `ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF` at `fe801509a580ff84a5728d4a9104e3f6f7e9b0dd`,
  `NVIDIA-Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf`, 22,421,827,488 bytes, sha256
  `0f111a0d49777a2a0178758b1aee14e5365c495987499401ef3ebe87754322a8` (the Hub's LFS digest).
  - The repository's README: "automatically converted using https://github.com/ggml-org/convert".
  - Types: the expert banks Q5_0 (rows 2,688 and 1,856 wide, not multiples of 256); the shared expert, attention,
    `ssm_in`, `ssm_out` and `output` Q8_0; `attn_output` Q6_K; `token_embd` Q5_0; the router F32 with `exp_probs_b`.
- **Runner:** CPU score files, `--score --score-top 20 --score-tokens 4510 -c 4510 --gpu off`, at the fixed build.

## K0: the load and the layout

- **The layout:** the file's per-block kinds are the config's `hybrid_override_pattern` exactly. That is 52 blocks: 23
  Mamba-2, 23 MoE, 6 attention. 128 experts, 6 used, 1 shared; `expert_weights_norm` true and scale 2.5; SSM state
  128 in 8 groups. The file carries no `expert_gating_func`, and Runner routes `nemotron_h_moe` by sigmoid (#575).
- **The declared command:** `runner -m F -p "The capital of France is" -n 8 --temp 0 --gpu off -v`.
  - At main b3e87703 it printed the summary and exited 139. UBSan:
    `model.c: member access within null pointer of type 'struct gguf_tensor'`, in `runtime_report`. The "weight type"
    line read block 0's `wqkv` or `wq`, and this file's block 0 is Mamba-2.
  - After the fix (the first commit of this change) the same command exits 0. It prints "weight type Q8_0" (block
    0's `ssm_in`) and generates "The capital of France is Paris. ...", greedy, the same bytes twice.
  - The ASan/UBSan build of the fix runs the same command with no report.
  - `-v` on files whose first block is attention or a fused-qkv recurrent block is byte-identical before and after.

## K1 and K2: the tokenizer and the template

- **Tokenizer:** 0 of 721 corpus strings and 0 of 128,635 vocabulary entries differ from the parent's tokenizer.json.
- **Template:** the file's template is the repository's `chat_template.jinja` byte for byte (10,504 bytes, sha256
  `ab7813c3`). Under the renderer Runner picks for it (`nemotron35`, Lightning's), 24 of 24 conformance cases are
  identical in text and tokens (`nemotron3-nano` in `scripts/template-conformance.py`).

## K3: against the reference

| Column | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|
| **dt floor 0, the declared limit (graded)** | **0.0222** | 0.0116 | 1.140 | 93.67% | **99.50%** | **pass** |
| the configured clamp (beside, not graded) | 0.1366 | 0.0715 | 1.739 | 82.67% | 94.53% | |

The config declares `time_step_min` 0.001 and no `time_step_limit`. transformers 5.16.1's mixer clamps dt at
`time_step_min` anyway, which reads an initialisation range as a limit (the Lightning row's finding). Runner and
llama.cpp follow the declared limit.

## Beside K3, not graded

- **Another engine:** llama.cpp 73a43d1 (`llama-server`, CPU, one slot, flash attention off) read on the same file at
  floor 0, against the same reference:

| Engine | Mean KLD | Median | Margin-qualified top-1 |
|---|---|---|---|
| Runner | 0.0222 | 0.0116 | 99.50% |
| llama.cpp 73a43d1 | 0.0243 | 0.0128 | 99.50% |

  Runner is the closer engine (Wilcoxon p 8.7e-4; McNemar 0 / 0).
- **The tool gate** (`scripts/tool-protocol-check.py`, CPU, `tool-gate.json`):
  - case A (a single call) passes buffered and streamed;
  - case B fails buffered and streamed, and again at temperature 0 (`tool-gate-b-temp0.json`);
  - case C passes streamed and fails buffered;
  - every failure stops on finish_reason `length` before a parsed call.
- **`--info adapter`:** neither serving nor training is supported ("--lora does not cover Mamba-2 hybrid
  architectures yet").
- **CUDA:** "this model's MoE router (gating=2 groups=1 norm_w=1 scale=2.5) has no device kernel, running on CPU".
- **tier-smoke** on the CPU: pinned (`3c6bc992`) and recorded on the lab CPU lane.

`FILES.sha256` lists every file here except this README.
