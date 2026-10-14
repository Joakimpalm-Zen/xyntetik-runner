# Runner documentation

Start with the [README](../README.md) for a quick start and
[MANUAL.md](../MANUAL.md) for every CLI option, endpoint, limit, and backend
support rule. This index links current tasks and the evidence for retained
product claims.

## Run and connect

- [OpenCode setup](workflow-opencode.md) and [coding-agent/API compatibility](agent-compatibility.md)
- [Tool-call truncation recovery](truncation-safe-tool-calling.md) and [the deterministic agent matrix](agent-torture.md)
- [Context-grounded drafts](context-drafts.md) and [session images](session-images.md)
- [Tray controller](tray-controller.md) and [Shadow execution/learning boundary](shadow-boundaries.md)
- [Performance routes and idle cost](performance.md), [kernel routes and their obligations](kernel-routes.md), and [buffer lifetimes](buffer-lifetimes.md)

## Work with model files

- [Model support scope](model-scope.md), [admission evidence](model-evidence.md), and [sparse-MoE support](moe-support.md)
- [Train on a quantized GGUF](train-lora-on-quantized-gguf.md) and [adapter/training/merge contracts](adaptation-engine.md)
- [Lineage walkthrough](lineage-walkthrough.md), [sublayer removal](sublayer-removal.md), and [quantization fidelity](quant-fidelity.md)
- [Tool-choice decision boundary](tool-choice-boundary-lane.md)

## Record and verify

- [Record/sign/verify demo](audit-demo.md) and [post-quantum receipts](postquantum-receipts-2026-09-05.md)
- [Determinism scope](determinism-scope.md), [training receipts](reproducible-lora-training-receipts.md), and [portable arithmetic measurement](portable-bitexact-2026-09-05.md)
- [Decision records](decision-record.md), [agent-profile metadata](agent-profile.md), [schemas](schemas/), and [measured-envelope format](envelope-manifests/)
- [Versioning and supported release policy](versioning.md)

## Evidence and contribution

- [Publisher-reference comparison](golden-pass-2026-09-07.md) and [reference data](golden-pass-evidence/)
- [Granite 4.2/Qwen 3.8 reference admission](granite-42-qwen38-cert-2026-09-06.md) and [experimental Qwen4 evidence](qwen4exp-admission-evidence/)
- [Truncation comparison](truncation-benchmark.md), [GPU benchmarks](benchmarks.md), and [idle-memory measurement](idle-coexistence-120b-m5max-2026-09-01.md)
- [Current release/device records](compat-reports/) and [coverage inventory](../tests/compatibility/coverage.json)
- [Compatibility method](compatibility-program.md), [CI/build verification](ci.md), and [Windows remote checks](windows-remote-checks.md)
- [Contributing](../CONTRIBUTING.md), [security](../SECURITY.md), and [community results](community-results.md)

A measurement states its original model, configuration, build, hardware,
and date. Documentation maintenance does not re-certify it for a new build.
