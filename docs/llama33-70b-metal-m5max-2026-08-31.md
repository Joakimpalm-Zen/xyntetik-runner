# Llama 3.3 70B dense Metal validation — M5 Max, 2026-08-31

Machine-locked validation on Apple M5 Max (18 GPU cores), 128 GB unified RAM,
macOS 26.5. The artifact was
`models/Llama-3.3-70B-Instruct-Q4_0.gguf`, 40,116,537,952 bytes, SHA-256
`4c90dbf12b76271bcf2f9eaccf38e3fc64b022b0e58075aa8f62c9c4c921dfd5`.
`scripts/verify-gguf.py` found a complete GGUF v3: llama architecture, 80
layers, 724 tensors, and no trailing or missing bytes. The source blob was not
identified confidently, so the digest is recorded as a local artifact digest
rather than claimed as a Hugging Face match.

An instrumented short run fully offloaded all 80 layers and exercised the
dense Q4_0 Metal batch path (`mm=550` for the nine-token prefill). It produced
coherent text at 2.98 tok/s prefill and 13.32 tok/s decode.

`test-gpu-identity` compared the scalar and Metal routes over 24 teacher-forced
positions. Of 3,078,144 logits, the mean absolute delta was 0.000441, or
0.0000139 of the mean 31.6 logit range, comfortably inside the 0.002 limit;
the worst individual delta was 0.008. The gate passed.

A sustained run read the 468-word determinism note as a 731-token prompt at
batch 512 and generated 128 greedy tokens:

- prefill: 42.25 tok/s;
- decode: 10.57 tok/s;
- output: coherent and responsive to the document;
- swap after completion: 0.25 MB (from a zero baseline).

This closes the dense 70B single-file Metal validation. The 0.25 MB of swap is
negligible bookkeeping-level activity, not model pressure: the host retained
tens of GiB of free/reclaimable memory throughout.
