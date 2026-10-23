# Model admission evidence

Use the [support matrix](../MANUAL.md#support-matrix), `--caps`, and
`--doctor` for current architecture/layout/backend admission. A model-family
label does not certify every quantization, context/cache layout, adapter,
or device. The [coverage inventory](../tests/compatibility/coverage.json)
separates executed evidence from missing checks.

The primary numerical anchor is the publisher's float32 implementation.
The [golden reference comparison](golden-pass-2026-10-22.md) (first pass
[2026-09-07](golden-pass-2026-09-07.md)) and its
[evidence](golden-pass-evidence/) record the measured model/configuration
rows. The [Granite 4.2/Qwen 3.8 admission](granite-42-qwen38-cert-2026-09-06.md)
records the corresponding tokenizer/template/reference cases.
[Qwen4 experimental admission](qwen4exp-admission-evidence/) records CPU and
CUDA checks; its publisher float32 comparison is still unmeasured. No unmeasured
configuration inherits a PASS from a neighboring row.

## AFMoE

Trinity/AFMoE uses gated attention and routed experts with per-branch
normalization. CPU execution is admitted; CUDA/Metal refuse unsupported
layout properties explicitly. The publisher-reference rows are in the
golden comparison. Strict greedy disagreement between quantized engines
can amplify a near-tie; it is not by itself a diagnosis of an incorrect
operation. Report logits, tie classification, and the exact reference
alongside the strict token count.

## Granite

Dense Granite and the admitted hybrid variants have separate topology and
backend rules. The four muP scalars, tokenizer/template choice, recurrent
state, and dense versus sparse/shared-expert FFNs are interpreted from the
actual file. Use the publisher-reference rows and current support table;
a quantized artifact failing a fidelity bar does not imply its higher
precision sibling failed or that the architecture is unsupported.

## gpt-oss

The text path preserves attention sinks, router/expert biases, MXFP4 scale
semantics, YaRN, and Harmony framing. Scalar/eager identity and default
fused/matrix routes carry different numerical obligations. Discrete MoE
routing can amplify a near-tie; inspect reference logits and the tested
route rather than calling every greedy mismatch a wrong operation.

## Gemma sparse and shared-KV layouts

Gemma 4 dense, sparse dual-branch, and E-series shared-KV/embedding layouts
have different geometry and numerical sensitivity. Preserve the artifact
hash and layout, the backend, and the scalar/tolerance route when checking
them. A QAT artifact is not interchangeable with a separately published
PTQ file. Current executable fixtures and the pinned manifest carry those
shape/refusal obligations; an old derivative/pruning campaign is not a live
support matrix.

## Muse Glimmer

The text backbone and native atem tool protocol are implemented; the vision
encoder remains outside the admitted text path. Current native declarations,
results, recipient grammar, bounded parallel calls, reasoning transitions,
truncation closure, and buffered/streamed mapping are covered by template,
tool/attribution, and API conformance gates. The publisher-reference rows
and explicit backend restrictions remain separate from tool-task quality.

## Quantization and backends

NVFP4 is decoded on CPU and supported on CUDA with its scale companions.
Metal admits the codebook i-quants through its current dense kernels and
retains separate NVFP4/MoE/layout refusals. There is no blanket IQ1/IQ2/IQ3
CPU-only rule on Metal. The codebook grids in `src/quants_iq_grids.h` are frozen reference data,
transcribed from the pinned ggml definitions; never regenerate them from
the decoder under test. Independent quantizer-boundary and model-reference
gates protect their use. Use the manual's backend table and actual capability
response rather than a historical artifact-status note.

Model files, reference revisions, corpus/tokenizer hashes, command/settings,
executed checks, and limitations belong in evidence records. Keep frozen
reference captures and deterministic numerical fixtures. Update a missing
or failed disposition only from a relevant verified run; documentation
maintenance does not certify new models or hardware.
