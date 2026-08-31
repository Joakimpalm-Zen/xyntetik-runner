# gpt-oss-120B fully resident on Metal — M5 Max, 2026-08-31

Machine-locked validation on Apple M5 Max (18 GPU cores), 128 GB unified RAM,
macOS 26.5. Artifact:
`models/gpt-oss-120b-MXFP4.gguf`, 63,387,346,208 bytes, SHA-256
`582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`
(matches the recorded Hugging Face blob hash).

Runner `39b06d4` wrapped the single-file model zero-copy and fully offloaded all
36 layers. A short instrumented run engaged the native Metal MoE route
(`moe=216` dispatches for the nine-token prefill) and generated coherent text.
Swap was zero before and after.

`test-gpu-identity` then compared the established scalar and Metal routes over
24 teacher-forced positions (4,826,112 logits): mean absolute delta 0.00451,
0.000215 of the mean logit range, inside the 0.002 bound. The gate passed.

A realistic sustained run used a 715-token prompt, batch 512, and 128 greedy
decode tokens:

- prefill: 54.65 tok/s
- decode: 64.59 tok/s
- swap after completion: exactly 0

This closes the previously 24-GB-MIG-blocked full-residency validation. It does
not claim CPU/Metal byte identity: gpt-oss routing is sensitivity-gated, and
the measured logit envelope is the existing correctness contract.
