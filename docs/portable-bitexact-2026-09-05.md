# Portable arithmetic measurement and T3 scope

The opt-in `make T3=1` build uses strict floating-point compilation,
portable transcendental functions, and canonical-order kernels. A T3
receipt records `flavor: "t3"`; exact continuation/replay still requires
the identity and inputs named by its record.

Measured 2026-09-05 on SmolLM2-135M-Instruct Q8_0, a fixed 165-token
teacher-forced paragraph: the measured canonical decode configuration
produced bit-identical log probabilities on arm64, x86-64, and emulated
riscv64. Decode cost was about 17% on the M1 and 7% on x86. The RISC-V run
used user-mode emulation and establishes no real-board speed result.

This was a scoped decode measurement, not a universal cross-ISA promise for
batch prefill, every quantization, MoE, GPU execution, or independent
rebuilds. The original score vectors were not committed as public fixtures;
the numerical/tree/portable-math tests remain the executable anchors for
the covered primitives. Native T3 CI also tests recorded generation,
training, and session continuation, but is not a cross-ISA comparison.

Use [the determinism scope](determinism-scope.md) for the actual T1/T2/T3
claims and limitations. Do not promote a broader claim from this single
model/prompt/configuration or relabel this measurement for a newer build.
