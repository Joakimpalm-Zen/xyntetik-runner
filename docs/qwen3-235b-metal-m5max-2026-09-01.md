# Qwen3 235B-A22B Q2_K-mix Metal validation — M5 Max, 2026-09-01

Validation ran on Apple M5 Max (18 GPU cores), 128 GB unified RAM, macOS 26.5.
The upstream two-part artifact is from
`unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF`:

- part 1: 49,906,751,904 bytes, SHA-256
  `7cb5dfd9d11f6297477f359439ad906eb92d62fc96b1783871159e7204d427ff`;
- part 2: 35,784,249,568 bytes, SHA-256
  `73801de2867057497a7fb6d0f0350ef7832b98f25c264f3a9909313af785672d`.

Both digests match their Hugging Face LFS identities. The second part uses the
standard compact split layout: only the three `split.*` keys are repeated.
Runner previously required `general.architecture` in every part and rejected
this valid set. Runner `cf50417` now inherits omitted model metadata from part
one while continuing to reject explicit contradictions.

Runner performed the production merge itself:

```text
runner -m ...-00001-of-00002.gguf \
  --quantize ...-merged.gguf --quant keep
```

The output is one complete GGUF v3, 85,691,001,248 bytes, 1,131 tensors,
SHA-256 `1a8bee8fbf8bec585c53a97d8f29e26d50672c898b1cc8c425227501d7dea3da`.
No tensor was converted. Its histogram is F32 471, Q2_K 377, Q3_K 188,
Q4_K 94, and Q6_K 1.

## Metal gaps found and fixed

The first merged load correctly declined Metal because dense Q2_K coverage did
not include an indirect Q2_K expert kernel. Adding it exposed Q3_K expert
tensors in the nominal Q2_K mix, which needed the corresponding expert kernel.
Both kernels use the established quarter-superblock layout. The direct Metal
gate compares their results against Runner's scalar dequantizer, selects expert
1 rather than expert 0 to exercise the expert stride, and passes at the tight
matvec tolerance. The shader roster and embedded-source gates also pass.

With both kernels present, all 94 layers offload. An eight-token instrumented
run recorded 376 matmul, 753 matvec, and 564 native MoE dispatches for prefill,
then produced coherent text at 1.30 tok/s prefill and 25.96 tok/s decode.

## Correctness result and caveat

`test-gpu-identity` compared 3,646,464 logits over 24 teacher-forced positions.
Mean absolute delta was 0.130, or 0.00332 of the mean 39.2 logit range, above
the 0.002 contract; worst delta was 3.17. This is recorded as a failure, not
relabelled as a pass.

The direct Q2_K/Q3_K expert kernel oracle is green. Full router traces agree
until token 4, layer 12, where the eighth selected expert changes from 20 to
120 after accumulated rounding; the seven higher-ranked experts agree. A
short free-running CPU/Metal generation remained byte-identical. These facts
locate the full-model excess in discrete top-8 routing sensitivity, but no
self-sensitivity-floor claim is made without a separate measurement.
*(Addendum 2026-09-01: that measurement exists now — the 235B is
self-identical on only 11/16 prompts within 16 greedy tokens under an
f16-vs-q8 KV perturbation, placing it in the measured chaotic class. The
identity row above stays FAIL against its stated bound; see
`docs/sensitivity-floors-m5max-2026-09-01.md` for what the floor licenses
and what it does not.)*

A sustained batch-512 run used a 732-token engineering note and generated 128
greedy tokens:

- prefill: 24.31 tok/s;
- decode: 21.07 tok/s;
- output: coherent and accurately grounded in the document;
- swap after completion: 1.12 MB.

This closes the machine-locked execution item as validated-with-caveat: the
85.69 GB artifact is usable and fully resident on Metal, the newly required
quant kernels have an absolute scalar anchor, and the model-level identity miss
remains explicit.
