// Non-finite detection for the tolerance gates, in its own translation unit.
//
// The gates are compiled with the engine's CFLAGS, which carry -ffast-math on
// every platform, and under -ffinite-math-only the compiler is licensed to
// assume no NaN or Inf exists. Apple clang 21 takes that licence all the way:
// not only isnan()/isfinite() but a test of the copied IEEE-754 bits
// ((bits & 0x7f800000) == 0x7f800000) folds to false in a fast-math TU, so a
// check written in the gate itself compiles, runs, and detects nothing (the
// v0.5.3 review's NaN case, reproduced 2026-09-14 with a three-line probe).
// The Makefile builds this file with QUANTS_CFLAGS, the same fast-math-free
// flag set that protects the quant arithmetic, so the detection cannot be
// optimised away; tests/test_gpu_identity.c --self-test proves it end to end
// under the release flags.
#include "finite_check.h"

#include <string.h>

static int f32_nonfinite(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return (b & 0x7f800000u) == 0x7f800000u;
}

size_t count_nonfinite_f32(const float *v, size_t n, size_t *first) {
    size_t bad = 0;
    if (first) *first = n;
    for (size_t i = 0; i < n; i++) {
        if (!f32_nonfinite(v[i])) continue;
        if (first && bad == 0) *first = i;
        bad++;
    }
    return bad;
}
