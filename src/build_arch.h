#ifndef RUNNER_BUILD_ARCH_H
#define RUNNER_BUILD_ARCH_H

// One spelling for every public build/provenance surface. Keep this header
// free of system includes so cross-target preprocess gates can exercise it.
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define RUNNER_BUILD_ARCH "arm64"
#elif defined(__x86_64__) || defined(_M_X64)
#define RUNNER_BUILD_ARCH "x86_64"
#elif defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64
#define RUNNER_BUILD_ARCH "riscv64"
#else
#define RUNNER_BUILD_ARCH "other"
#endif

#endif
