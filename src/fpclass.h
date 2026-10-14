// Floating-point class tests that hold under -ffast-math.
//
// The engine's translation units are built with -ffast-math, and its
// -ffinite-math-only lets the compiler assume that no value is a NaN or an
// infinity. clang 22 acts on that: it folds isfinite() to 1 and isnan() to 0,
// so a check written with them is deleted. Apple clang 21 and gcc 16 do not
// fold the library macros today, but LLVM (Apple clang 21 included) already
// turns an exponent-bits test back into the same class test and folds that.
// The empty asm below hides the bits from the optimizer, so the test runs on
// the value the program actually holds; it costs one register move.
//
// Every class test in a -ffast-math unit goes through here;
// scripts/check-fpclass.py refuses a raw isnan/isfinite/isinf there, and
// tests/test_fpclass.c holds these under the build's own flags.
#ifndef RUNNER_FPCLASS_H
#define RUNNER_FPCLASS_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t fp_bits_f64(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    __asm__("" : "+r"(u));
    return u;
}

static inline uint32_t fp_bits_f32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    __asm__("" : "+r"(u));
    return u;
}

// all-ones exponent: an infinity (zero mantissa) or a NaN (non-zero)
static inline bool fp_finite_f64(double x) {
    return (fp_bits_f64(x) & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
}
static inline bool fp_nan_f64(double x) {
    return (fp_bits_f64(x) & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}
static inline bool fp_finite_f32(float x) {
    return (fp_bits_f32(x) & 0x7f800000u) != 0x7f800000u;
}
static inline bool fp_nan_f32(float x) {
    return (fp_bits_f32(x) & 0x7fffffffu) > 0x7f800000u;
}

#define fp_isfinite(x) _Generic((x), float: fp_finite_f32, default: fp_finite_f64)(x)
#define fp_isnan(x)    _Generic((x), float: fp_nan_f32, default: fp_nan_f64)(x)

#endif
