# Documentation index

The user-facing reference is [`MANUAL.md`](../MANUAL.md); the front page is
the [README](../README.md). This folder holds the detail behind them: how a
feature works, the reports and certifications that back a claim, and the
negative results. Dated files are records of the day they name and are not
updated afterwards; the newest record of a topic wins.

## Using Runner

- [One tested workflow: OpenCode against a local model](workflow-opencode.md)
- [Coding-agent compatibility evidence](agent-compatibility.md) and the [agent torture suite](agent-torture.md)
- [Tool calls that survive the token limit](truncation-safe-tool-calling.md) and its [benchmark](truncation-benchmark.md)
- [Context-grounded drafts, `--draft-lookup`](context-drafts.md)
- [Session images: a generation as a file](session-images.md)
- [Tray and menu-bar controller](tray-controller.md)
- [Shadow execution and optional learning](shadow-boundaries.md)
- [Versioning: what 1.x keeps](versioning.md), [determinism scope](determinism-scope.md), [model scope](model-scope.md)

## Proof and provenance

- [The audit demo: record, sign, hand over, verify](audit-demo.md)
- [Post-quantum receipts (ML-DSA-44 beside Ed25519)](postquantum-receipts-2026-09-05.md)
- [Portable bit-exactness across three ISAs](portable-bitexact-2026-09-05.md)
- [The decision record](decision-record.md), [agent profile metadata](agent-profile.md), [specs](specs/), [schemas](schemas/), [envelope manifests](envelope-manifests/)

## Adapters and training

- [The adaptation engine](adaptation-engine.md)
- [Train a LoRA on the quantized GGUF you serve](train-lora-on-quantized-gguf.md)
- [Reproducible LoRA training with receipts](reproducible-lora-training-receipts.md)
- [The local training floor (M5 Max)](training-floor-m5max-2026-09-01.md)
- [Sublayer removal, `--remove-sublayer`](sublayer-removal.md)

## Model support, admissions and certifications

- [Compatibility program](compatibility-program.md), [community results](community-results.md), [sparse-MoE support](moe-support.md)
- [Golden pass over every admitted family (2026-09-07)](golden-pass-2026-09-07.md), [evidence](golden-pass-evidence/)
- [Granite 4.2 and Qwen 3.8 admission](granite-42-qwen38-cert-2026-09-06.md), [evidence](granite-42-qwen38-evidence/); [qwen4exp evidence](qwen4exp-admission-evidence/)
- [Cross-family agent reliability: the remedy record](cross-family-remedy-2026-09-14.md), [evidence](cross-family-remedy-evidence/)
- [SentencePiece divergences read against the publishers' models](tokenizer-spm-2026-10-03.md)
- [Ornith 1.0 9B reference gate](ornith-reference.md)
- Granite 4.1: [certification](granite-cert-2026-08-11.md), [evidence](granite-evidence/)
- Muse Glimmer: [certification](muse-glimmer-cert-2026-08-11.md), [native atem](muse-atem-cert-2026-08-11.md), [evidence](muse-glimmer-evidence/)
- gpt-oss: [Harmony chat](gpt-oss-harmony-2026-08-14.md), [CUDA router bias](cuda-gptoss-router-bias-2026-08-18.md), [CPU/CUDA divergence](cuda-gptoss-divergence-2026-08-19.md)
- Trinity-Nano (afmoe): [goal](afmoe-cert-goal-2026-08-05.md), [report](afmoe-cert-report-2026-08-05.md), [divergence triage](afmoe-divergence-triage-2026-08-05.md), [sensitivity floor](afmoe-sensitivity-floor-2026-08-05.md), [evidence](afmoe-cert-evidence/)
- GPT-OSS x Gemma 4 cert matrix: [goal](cert-matrix-goal-2026-08-05.md), [report](cert-matrix-2026-08-05.md), [status](cert-matrix-status.md), [evidence](cert-matrix-evidence/)
- Release compatibility reports, one set per release: [compat-reports](compat-reports/)

## Fidelity and quantization

- [Quant-vs-tool-call fidelity harness](quant-fidelity.md)
- [Tool-choice decision-boundary lane](tool-choice-boundary-lane.md)
- [Self-sensitivity floors (M5 Max)](sensitivity-floors-m5max-2026-09-01.md)

## Performance and backends

- [GPU benchmarks: Runner and llama.cpp on CUDA](benchmarks.md), [raw](benchmarks-raw/), [RTX 3070 cross-engine run (2026-08-01)](bench-2026-08-01-3070.md)
- [Performance: closing the CPU/GPU gap](performance.md)
- CUDA: [codebook i-quant kernels](cuda-iquants-2026-09-14.md) ([evidence](cuda-iquants-evidence/)), [tensor-core i-quant prefill](cuda-iq-tensorcore-2026-09-14.md) ([evidence](cuda-iq-tensorcore-evidence/)), [decode microbatch identity](cuda-microbatch-identity-2026-08-18.md)
- Metal: [runtime fallback ownership](metal-fallback.md), [dispatch census](metal-dispatch-census-2026-08-13.md), [decode dispatch budget](metal-decode-dispatch-budget-2026-09-01.md), [long-context decode](metal-long-context-decode-2026-08-14.md), [KV traffic](metal-decode-kv-traffic-2026-08-15.md), [grouped-MMA MoE prefill](metal-moe-grouped-mma-2026-09-01.md), [gemma-4 MoE GELU fix](metal-gemma4-moe-divergence-2026-08-31.md)
- M5 Max validations: [gpt-oss-120B](gpt-oss-120b-metal-m5max-2026-08-31.md), [idle coexistence at 120B](idle-coexistence-120b-m5max-2026-09-01.md), [Llama 3.3 70B](llama33-70b-metal-m5max-2026-08-31.md), [Qwen3 30B-A3B](qwen3-30b-a3b-metal-m5max-2026-08-31.md), [Qwen3 235B-A22B](qwen3-235b-metal-m5max-2026-09-01.md)
- Device runs and baselines: [device runs (2026-10-02)](device-runs-2026-10-02/), [artifact baseline (2026-10-02)](artifact-baseline-2026-10-02/), [agent turns (2026-10-02)](agent-turns-2026-10-02/)
- [Windows remote-check protocol](windows-remote-checks.md)

## Negative results

What was tried, measured and not shipped:
[expert-residency cache](negative-result-expert-cache.md) ·
[Harmony analysis bound](negative-result-harmony-analysis-bound.md) ·
[Metal GEMM occupancy](negative-result-metal-gemm-occupancy.md) ·
[expert-major MoE prefill](negative-result-metal-moe-expert-major.md) ·
[multi-row Metal matvec](negative-result-metal-multirow-matvec.md) ·
[Muse Glimmer selective quants](negative-result-muse-glimmer-selective-2026-08-20.md) ·
[Nemotron Lightning artifacts](negative-result-nemotron-lightning-artifacts-2026-08-20.md)
