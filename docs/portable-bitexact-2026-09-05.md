# Portable bit-exactness: same model, same bytes, three ISAs (2026-09-05)

**Status:** experiment on branch `r12-frontier`; nothing here is in a
release. Frontier item R12.1 asked whether integer-only inference is the
route to a receipt that verifies bit for bit on any machine (R1.11). This
note answers with a measurement: it is not needed. Float arithmetic gives
bit-identical logits on arm64, x86-64 and riscv64 once three things are
fixed, and the cost is a 7 to 18% decode slowdown, not the 4x of scalar
code.

## Question

Today's determinism contract (docs/determinism-scope.md) is T1 "same
binary, same bytes" and T2 "another machine, same tokens". The T2 edge is
libm. Frontier item R12.1 proposed closing it with an integer-only pipeline
(int8 activations, fixed-point norms, softmax by shifts). Before building
that, measure what actually differs between two machines and what the
smallest change is that removes it.

## Setup

- Model: SmolLM2-135M-Instruct Q8_0, sha256 `5a1395716f79...bba83`
  (bartowski/SmolLM2-135M-Instruct-GGUF), identical file on every host.
- Prompt: one 165-token English paragraph (`--score -f`, which runs one
  decode-shaped forward per position and prints each log-probability with
  nine significant digits, enough to round-trip a float32).
- Hosts: Apple M1 (arm64, NEON, clang), ZEN-GAMING (i7-7700K, x86-64 AVX2,
  mingw gcc), and riscv64 (zig cc, musl static, run under qemu-riscv64
  9.2.2 user mode on a Linux host; no RISC-V board was available).
- Compare: count of positions whose logprob is bit-identical, and the
  maximum absolute difference.

## What differs, isolated one variable at a time

| build, both hosts M1 vs x86 | identical positions | max abs diff |
|---|---:|---:|
| default (`-O3 -ffast-math -march=native`) | 0 / 165 | 4.2e-3 |
| strict FP (`-fno-fast-math -ffp-contract=off`) | 2 / 165 | 2.9e-3 |
| strict + scalar reference kernels (`RUNNER_NO_SIMD`) | 2 / 165 | 2.0e-3 |
| strict + scalar + portable math (`RUNNER_PORTABLE_MATH`) | **165 / 165** | **0** |
| strict + portable math + canonical SIMD kernels (`RUNNER_CANON_KERNELS`) | **165 / 165** | **0** |

Top-1 was identical in every row: T2 already holds. Three independent
causes had to be removed for bit identity:

1. **Compiler freedom.** `-ffast-math` lets each compiler reassociate
   reductions its own way and `-ffp-contract` fuses multiply-adds where it
   likes; the two compilers made different choices. Strict flags remove
   this. Cost on the M1: 113 to 93 tok/s decode (18%); on x86: part of the
   7% below.
2. **Reduction order of the SIMD kernels.** The NEON kernels accumulate in
   four lanes and reduce with `vaddvq`; the AVX2 kernels in eight lanes and
   reduce with a shuffle tree. Same math, different association, different
   last bits. Two fixes were measured: the scalar reference kernels (4x
   slower) and canonical-order kernels (below).
3. **libm.** `expf`, `logf`, `tanhf`, `powf`, `sinf`, `cosf` differ in the
   last bit between Apple libm, glibc and mingw. `src/pmath.h` replaces
   them with double-precision reductions and polynomials over IEEE add,
   mul and div only, rounded to float once. Measured cost: none visible
   (scalar 28.6 vs portable 30.9 tok/s is noise).

## Canonical-order kernels

`RUNNER_CANON_KERNELS` fixes one reduction tree per format and implements
it on every target: eight virtual lanes, lane i accumulating the elements
congruent to i mod 8 within a 32-wide step with fused multiply-adds, then
the tree `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`. The AVX2 kernels already
had that shape (their `hsum8` is that tree), so on x86 the canonical kernel
is the existing kernel. NEON mirrors it with two 4-lane registers per
virtual vector; the no-SIMD fallback mirrors it with `fmaf`, which is what
riscv64 runs. Covered: Q8_0, F16 and F32 row dots, the decode path. Not
covered yet: the batched prefill tile (`vec_dot_f32_multi`), the other
quant formats, the GPU backends.

Results with strict FP + portable math + canonical kernels:

| pair | identical | max abs diff |
|---|---:|---:|
| M1 NEON vs M1 scalar mirror | 165 / 165 | 0 |
| M1 NEON vs x86 AVX2 | 165 / 165 | 0 |
| M1 NEON vs riscv64 scalar (qemu) | 165 / 165 | 0 |
| x86 AVX2 vs riscv64 scalar (qemu) | 165 / 165 | 0 |

## Cost

SmolLM2-135M Q8_0, 166-token prompt, 48 greedy tokens, 4 threads.

| host | build | prefill tok/s | decode tok/s |
|---|---|---:|---:|
| M1 | default | 308 | 113 |
| M1 | strict FP, native kernels | 421 | 93 |
| M1 | strict FP, scalar kernels | 26 | 29 |
| M1 | strict FP, portable math, canonical kernels | 309 | 94 |
| x86 | default | 492 | 145 |
| x86 | strict FP, portable math, canonical kernels | 392 | 135 |

Decode: 7% on x86, 17% on the M1, all of it the strict-FP flags. Prefill
loses 20% on x86 from strict FP alone; its tile kernel is not canonical
yet, so cross-ISA prefill identity is untested and probably false today.
The riscv64 numbers are emulated and say nothing about speed.

## What this means for the theses

- **R12.1 (integer-only for portability): killed as a necessity.** The
  portability claim is reachable in float at a fraction of the cost and
  without touching model fidelity. An integer path would still be a
  legitimate speed or power project, but it is no longer the route to
  "same model, same bytes, any machine".
- **R1.11 (portable bit-exactness): validated, with a concrete route.**
  Strict FP flags for the release binaries, `pmath.h`, canonical kernels
  for every format and the prefill tile, then a T3 replay tier ("any
  machine, same bytes") beside T1 and T2. The remaining work is
  engineering with a clear gate: the score vector above, pinned per model.
- **R12.5 (RISC-V): the engine builds and runs on riscv64.** Cross-compiled
  with `zig cc -target riscv64-linux-musl` (no toolchain installation),
  30 translation units, one warning fixed, statically linked, and it is
  bit-identical to the other two ISAs. Performance on real silicon is
  unknown; that needs a board.

## Gates

- `tests/test_canon_kernels.c` pins each canonical kernel against an
  independent scalar implementation of the same tree, bit for bit, on
  random inputs of every length class (`make test-canon-kernels && ./test-canon-kernels`;
  2176 checks pass on the M1's NEON, the x86 box's AVX2 and the scalar mirror).
- The three score files behind the tables are in the experiment record of
  the suite plan (R12), not in this repository: they depend on a 135 MB
  model that is not pinned here.

## Not done, and why

- Prefill tile, k-quants, Q4_0, MoE experts, GPU backends: the same
  treatment applies but is engineering, not experiment.
- A real RISC-V board: no hardware. The qemu result proves correctness,
  not speed.
- Deciding whether release binaries should ship strict FP: an owner call
  on the 7 to 18% decode cost against a receipt that verifies anywhere.
