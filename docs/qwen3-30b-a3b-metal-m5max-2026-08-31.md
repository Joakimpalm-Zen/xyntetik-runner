# Qwen3 30B-A3B Q8_0 Metal validation — M5 Max, 2026-08-31

Validation ran on Apple M5 Max (18 GPU cores), 128 GB unified RAM, macOS 26.5.
The artifact was `models/Qwen3-30B-A3B-Q8_0.gguf`, 32,483,932,064 bytes,
SHA-256 `f804d1c2d37e5c7929bd3541428a471d69a84aa5023d1a7a47fd5941c34475df`.
`scripts/verify-gguf.py` found a complete GGUF v3 with qwen3moe architecture,
48 layers, 579 tensors, and no missing or trailing bytes.

An instrumented short run fully offloaded the model and exercised both Q8_0
dense matmul and native Metal MoE execution. Its eight-token prefill recorded
192 matmul, 385 matvec, and 288 MoE dispatches and produced coherent text.

`test-gpu-identity` compared scalar and Metal inference over 24 teacher-forced
positions. Across 3,646,464 logits, mean absolute delta was 0.0409, or 0.00109
of the mean 37.6 logit range, inside the 0.002 limit. The worst individual
delta was 0.812. The gate passed.

A sustained run used a 732-token engineering document at batch 512 and 128
greedy decode tokens:

- prefill: 183.39 tok/s;
- decode: 65.45 tok/s;
- output: coherent and accurately grounded in the document;
- swap after completion: 0.25 MB.

This closes local Metal coverage for Runner's qwen3moe architecture and its
128-expert Q8_0 path. No code defect was exposed by this rung.
