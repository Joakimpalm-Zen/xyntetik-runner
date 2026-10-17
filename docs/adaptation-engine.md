# Scoring, adapters, training, and merge contracts

Runner trains a LoRA adapter through a frozen quantized GGUF and serves that
adapter beside the base. The CPU training forward shares the inference
primitives. CUDA/Metal serving uses device arithmetic with its own measured
numerical envelope; training on the CPU is not a claim that every device
forward is bit identical.

For commands, use [the complete adapter guide](lora.md), the
[training walkthrough](train-lora-on-quantized-gguf.md) and the
[manual](../MANUAL.md#adaptation). This page describes current
admission, ownership, numerical contracts, and the scoped public evidence.

## Scoring and attachment

`--score` teacher-forces raw text and emits token log probabilities, NLL,
and perplexity as `xyntetik.runner.score.v1`. The default uses scalar
forwards; `RUNNER_SCORE_CHUNKED=1` explicitly selects the batch contract.

`--lora` accepts adapter GGUFs with known projection names and matched
architecture/shapes/ranks. F32, F16, and BF16 adapter tensors load; quantized
adapter tensors refuse. The applied scale is trained alpha/r times
`--lora-scale`. A zero adapter or zero scale bypasses the delta exactly.
The adapter identity participates in prefix/cache identity.

Supported adapter serving runs on CPU, CUDA, and Metal, with the backend's
projection/layout refusals. An unavailable device adapter path refuses the
load rather than silently dropping the adapter. Per-request `--adapter`
sets use CPU hooks, borrow an adapter only for that request, and isolate
cached rows across adapter changes. Multi-sequence device decode declines
an attached-adapter combination its batch path does not implement.
Reload rechecks the adapter and retains its scale/signature policy; draft
models do not inherit the target's adapter.

## Training admission

`--train` and `--train-dpo` keep the base CPU-resident, use f16 KV, and
update only adapter parameters. Supported dense SiLU/GELU paths include
attention gates/head transforms, sliding windows, sandwich norms, and the
covered Gemma tied/shared-KV and per-layer-embedding layouts. Admission is
by implemented property, not by an old blanket dense-model list.

Attention-only adapters on supported sparse MoE also train: the backward
propagates through frozen softmax routing and frozen experts, including
selected-expert renormalization and supported non-renormalized weights.
Frozen parameters still affect the gradient to the attention adapter.
Expert adapters, unsupported gating/group/bias/shared-expert shapes,
Gemma dual-branch MoE, recurrent architectures, removed sublayers, attention
sinks, partial-dimension RoPE, and unimplemented backward properties refuse
by name. Derived-K (`RUNNER_TIEDV`), ring KV, and non-f16 training caches
retain their refusals. `--adapt-info` reports serving and training support
for the actual loaded model.

CUDA training assistance is opt-in with `RUNNER_TRAIN_GPU=1`; the CPU hosts
the forward/optimizer and the implemented device primitives assist the
backward. It must preserve CPU adapter bytes with proof the device hooks
engaged. This is not a fully GPU-hosted trainer.

## Reproducibility, gradients, and failure behavior

The backward recomputes with the inference primitives and preserves fmaf,
rounding, and accumulation order. The gradient suite covers dense and
quantized bases, Gemma/E-series properties, and strict MoE routing cases.
Strict MoE finite differences hold resolved coordinates to 0.5%; ordinary
and stiff fixtures retain their established, separately justified modes.

The same executable, data, configuration, and seed reproduce the same
adapter bytes and deterministic loss trajectory. Different builds/ISAs
need not write the same bytes. Prompt/completion JSONL masks prompt loss;
weighted examples, training context, and opt-in end-of-turn tokens retain
their specified behavior. DPO evaluates the frozen reference through an
adapter bypass rather than a second base-weight copy.

Checkpoint/output installation preserves the previous artifact on failure.
Allocation, optimizer, save, hashing, record, and signing failures remain
non-successful even if some output bytes were already written. `.train.json`
records base/data/adapter identities, configuration, loss, and build identity;
`--sign-key` signs it. [Lineage](lineage-walkthrough.md) connects train,
merge, quantize, and serving receipts.

## Merge survival

`--merge-lora OUT` folds the adapter delta into each adapted projection and
writes a standalone GGUF at the source tensor type or `--quant` target.
Untouched tensors and zero-delta pairs preserve their specified copy behavior.
The quantization grid can round away a learned delta, so the exact serving
form remains base plus `--lora`.

Every nonzero rewritten delta is measured against the base written alone
at the same output type. The default retained-delta floor is 0.5. A merge
below it refuses before replacing the destination and removes partial
output. `--merge-allow-erased` is the explicit override; `.merge.json`
records `survival` with adapted bytes, changed bytes, retained delta, and
the floor (`null` under the override). Write/install failures leave the
destination untouched. Record/signing errors remain command failures.

Choose a type wider than the base when needed and evaluate the merged
artifact. Q8_0 over a narrower base can keep a delta that its original grid
erases; Q8_0 over a Q8_0 base is not a universal fidelity guarantee.

## Public measurements and their limits

Measured 2026-08-22 on
[Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M),
one base/task with a held-out tool-call evaluation:

| Serving form | Exact held-out tool calls |
| --- | --- |
| Frozen Q4_K_M base | 0.69 |
| Base with the trained adapter | 1.00 |
| Adapter merged into F16 or Q8_0 | 1.00 |
| Adapter merged back into the 4-bit grid | 0.69 |

Two identical training runs wrote identical adapter hashes. Stock llama.cpp
served the adapter at the same 1.00 score. BF16/Q8_0-trained adapters had
cosine 0.9998; the Q4_K_M-trained adapter had cosine 0.9926. The neutral
corpus's NLL moved from 4.063 to 4.026. These are measurements for that
artifact/task, not general quality promises. The reproducibility scope is
[documented separately](determinism-scope.md).

The merge-survival measurement on this base/adapter recorded 1.8% retained
delta onto Q4_K and 99.5% onto Q8_0. The independent 36-prompt unlabeled
boundary bank found three disagreements across the precision-trained
adapters; that is a property of the constructed bank, not a traffic rate.
See the [decision-boundary lane](tool-choice-boundary-lane.md) and its frozen
inputs for that measurement. CPU/device serving comparisons and training
assistance are scoped by the backend/model gates in the manual; no new
hardware result is implied by this reference.

The regression entry points are `tests/test_lora.py`, `tests/test_train.py`,
`tests/test_lora_grad.c`, `tests/test_moe_lora.py`, `tests/test_dpo_grad.c`,
`tests/test_merge.py`, `tests/test_quantize.c`, adapter routing/signature
checks, and lineage/evaluation/receipt checks. Their fixture/hash/tolerance
contracts remain part of the aggregate test gate.
