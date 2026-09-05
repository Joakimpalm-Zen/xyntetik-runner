# What the determinism claim covers, exactly

Runner's determinism is a contract with named edges. This page is the
authoritative scope: if a promise is not on the "claimed" list, Runner
does not make it, and the absence is deliberate.

## Claimed, and gated in CI

- **Same executable, same inputs, same sampled tokens.** One binary,
  one model file, one config and seed: the emitted token ids are
  identical across runs and across `-t` thread counts. This is the
  core contract and it is exercised by the test suite on every change.
- **Training byte-determinism, same build.** `--train` with identical
  inputs reproduces the same adapter file, byte for byte, verified by
  sha256 in CI. The `.train.json` sidecar records the binary sha,
  compiler, OS, and arch that produced it.
- **Transcript replay tiers** (`--transcript` / `--verify`):
  - **T1, same binary**: bit-exact replay of a recorded run; verified
    token ids and raw streamed output bytes equal the record exactly. Raw
    bytes are authoritative because an individual byte-token piece need not
    be valid UTF-8; the transcript also keeps a human-readable JSON string.
  - **T2, cross-ISA**: token-level replay on a different machine or
    architecture. Token ids must match; floating-point internals may
    not, and the boundary is libm (`expf` differs between platforms),
    compiler reassociation under fast-math, and SIMD reduction order.
- **Quantization determinism.** `--quantize` and `--merge-lora` write
  the same output bytes for the same inputs on the same build.

## Measured, not yet claimed: T3, any machine, same bytes

An opt-in build (`make T3=1`) compiles the engine with strict floating
point (no fast-math, no fused-multiply-add contraction), portable
transcendental functions (`src/pmath.h`, IEEE add, multiply and divide
only) and canonical-order dot kernels that fix one reduction tree on every
target. In that configuration the decode path produced bit-identical
log-probabilities on arm64, x86-64 and riscv64 for every position of a
165-token measurement (docs/portable-bitexact-2026-09-05.md), at a 7 to
17% decode cost. It is not on the claimed list because the batched prefill
tile, the k-quant formats, MoE experts and the GPU backends are not
canonical yet, and no CI job runs the cross-ISA comparison. A T3 receipt
records `"flavor":"t3"` in its build object so a verifier can tell which
binary wrote it. When the coverage and the gate exist, T3 joins T1 and T2
above as "another machine, same bytes".

## Explicitly not claimed

- **Independent rebuilds are not byte-identical.** A different
  compiler, compiler version, or ISA produces a binary whose
  floating-point sequence can differ legitimately. Cross-build
  verification is what the T2 token tier is for; bit-identity across
  rebuilds is not promised and will not be.
- **Cross-engine identity is not promised.** The same GGUF in another
  runtime samples its own tokens. Adapter files interoperate (measured:
  a Runner-trained adapter scores identically served by llama.cpp);
  token streams do not.
- **Numerical identity of hidden states is not promised, ever.** The
  contract is about emitted tokens and written artifacts, not
  intermediate activations.
- **Wall-clock timing, memory footprint, and throughput are not part
  of the determinism contract.** They vary with the machine and its
  load; the contract is what bytes come out, not when.

## Why the edges are where they are

The determinism boundary follows what a single binary controls. Inside
one executable, the instruction sequence is fixed, so accumulation
order and math-library behavior are fixed, and byte-identity is
enforceable. Across builds, libm and codegen legitimately differ, so
the honest cross-build claim is token identity under replay, which is
what `--verify` implements. Claims stronger than the mechanism can
enforce would eventually be false somewhere; this project would rather
have a narrower promise that is always true.

If you find a reproducible violation of anything on the "claimed"
list, that is a bug and we want the report, with the transcript if you
have one.
