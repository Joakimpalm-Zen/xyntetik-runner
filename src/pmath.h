// Portable transcendental functions for the bit-exact cross-ISA experiment
// (R12.1 / R1.11). libm's expf, logf, tanhf, powf, sinf and cosf are
// platform-specific: glibc, Apple libm and mingw each return different
// last-bit results, so two machines running the same binary-equivalent
// arithmetic still disagree in the logits. These implementations use only
// IEEE-754 double add/sub/mul/div and integer bit operations, which every
// conforming target performs identically, and round to float once at the
// end. Compile with -fno-fast-math -ffp-contract=off so the compiler neither
// reassociates nor fuses them; under those flags the same source gives the
// same bits on arm64, x86-64 and riscv64.
//
// Accuracy, measured against libm over two million random inputs per range
// (tests/test_pmath.c pins the bounds): expf, logf, tanhf and powf within
// 1 ulp; sinf and cosf within 2 ulp for |x| <= 1000 and within 1.2e-7
// absolute (2 ulp of a value near 1) for |x| <= 1e5: rope angles are
// position times an inverse frequency, so a long context reaches the tens
// of thousands, and near a zero crossing the relative figure is the wrong
// measure (41 ulp of a value of 1e-6 is 2e-12).
// Special values follow libm: NaN in, NaN out; log of +inf is +inf; sin
// and cos of an infinity are NaN; tanh of a subnormal is the subnormal.
// They are NOT libm-compatible (a result can differ from libm in the last
// bit); the property they buy is sameness across machines, not agreement
// with any one platform's libm.
//
// Enabled by -DRUNNER_PORTABLE_MATH; without it this header is inert and
// the engine keeps calling libm.
#ifndef RUNNER_PMATH_H
#define RUNNER_PMATH_H

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline double pm_ldexp2(double x, int k) {
    // x * 2^k by exponent arithmetic, exact
    union { double d; uint64_t u; } v;
    while (k > 1000) { v.d = 0x1p1000; x *= v.d; k -= 1000; }
    while (k < -1000) { v.d = 0x1p-1000; x *= v.d; k += 1000; }
    v.u = (uint64_t)(k + 1023) << 52;
    return x * v.d;
}

static inline double pm_exp_d(double x) {
    // exp(x) = 2^k * exp(r), r = x - k*ln2 in [-ln2/2, ln2/2]
    if (x != x) return x;                 // NaN: the int conversion below is undefined on it
    if (x > 709.0) return 1.0 / 0.0;
    if (x < -745.0) return 0.0;
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    const double inv_ln2 = 1.44269504088896338700e+00;
    double kd = x * inv_ln2;
    int k = (int)(kd >= 0 ? kd + 0.5 : kd - 0.5);
    double r = (x - (double)k * ln2_hi) - (double)k * ln2_lo;
    // Taylor series to degree 13: |r| <= 0.347, term_13/13! ~ 1e-15
    double p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    return pm_ldexp2(p, k);
}

static inline double pm_log_d(double x) {
    // log(x) = k*ln2 + 2*atanh(s), s = (m-1)/(m+1), m in [sqrt(1/2), sqrt(2))
    if (x != x) return x;                       // NaN in, NaN out
    if (x <= 0.0) return x == 0.0 ? -1.0 / 0.0 : 0.0 / 0.0;
    if (x > 1.7976931348623157e308) return x;   // +inf: the exponent field is not a number
    union { double d; uint64_t u; } v = { x };
    int k = (int)((v.u >> 52) & 0x7ff) - 1023;
    v.u = (v.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL; // m in [1,2)
    double m = v.d;
    if (m > 1.41421356237309504880) { m *= 0.5; k += 1; }
    double s = (m - 1.0) / (m + 1.0), s2 = s * s;
    // atanh series: s + s^3/3 + s^5/5 + ... , |s| <= 0.1716 -> 12 terms ~ 1e-19
    double p = 1.0 / 25.0;
    p = p * s2 + 1.0 / 23.0;
    p = p * s2 + 1.0 / 21.0;
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    return (double)k * 6.93147180559945309417e-01 + 2.0 * s * p;
}

static inline void pm_sincos_d(double x, double *sn, double *cs) {
    // reduce by pi/2: x = k*(pi/2) + r, |r| <= pi/4, then Taylor on r
    const double pio2_hi = 1.57079632679489655800e+00;
    const double pio2_lo = 6.12323399573676603587e-17;
    if (!(x < 1e15 && x > -1e15)) {
        // NaN or infinity: no angle. A finite value this large has no
        // fractional radian left in a float anyway (the engine's angles are
        // position times an inverse frequency, at most a few hundred
        // thousand); reduce it with fmod, which is exact in IEEE 754 and so
        // identical on every target, before the two-part reduction below.
        if (!(x == x) || x > 1.7976931348623157e308 || x < -1.7976931348623157e308) {
            *sn = *cs = 0.0 / 0.0;
            return;
        }
        x = fmod(x, 6.28318530717958647693);
    }
    double kd = x * 6.36619772367581382433e-01; // 2/pi
    long long k = (long long)(kd >= 0 ? kd + 0.5 : kd - 0.5);
    double r = (x - (double)k * pio2_hi) - (double)k * pio2_lo;
    double r2 = r * r;
    // sin: r - r^3/3! + ... - r^15/15!   cos: 1 - r^2/2! + ... + r^16/16!
    double s = -1.0 / 1307674368000.0;
    s = s * r2 + 1.0 / 6227020800.0;
    s = s * r2 - 1.0 / 39916800.0;
    s = s * r2 + 1.0 / 362880.0;
    s = s * r2 - 1.0 / 5040.0;
    s = s * r2 + 1.0 / 120.0;
    s = s * r2 - 1.0 / 6.0;
    s = s * r2 + 1.0;
    s = s * r;
    double c = 1.0 / 20922789888000.0;
    c = c * r2 - 1.0 / 87178291200.0;
    c = c * r2 + 1.0 / 479001600.0;
    c = c * r2 - 1.0 / 3628800.0;
    c = c * r2 + 1.0 / 40320.0;
    c = c * r2 - 1.0 / 720.0;
    c = c * r2 + 1.0 / 24.0;
    c = c * r2 - 0.5;
    c = c * r2 + 1.0;
    switch (k & 3) {
        case 0:  *sn = s;  *cs = c;  break;
        case 1:  *sn = c;  *cs = -s; break;
        case 2:  *sn = -s; *cs = -c; break;
        default: *sn = -c; *cs = s;  break;
    }
}

static inline float p_expf(float x)  { return (float)pm_exp_d((double)x); }
static inline float p_logf(float x)  { return (float)pm_log_d((double)x); }
static inline float p_sinf(float x)  { double s, c; pm_sincos_d((double)x, &s, &c); return (float)s; }
static inline float p_cosf(float x)  { double s, c; pm_sincos_d((double)x, &s, &c); return (float)c; }
static inline float p_tanhf(float x) {
    double xd = (double)x;
    if (xd > 20.0) return 1.0f;
    if (xd < -20.0) return -1.0f;
    // near zero exp(2x) rounds to 1 and (e-1)/(e+1) collapses to 0 where
    // tanh x is x; the odd series is exact to double there
    // written as x*(1 - x^2/3), not x - x^3/3: the latter turns -0 into +0
    if (xd < 1e-4 && xd > -1e-4) return (float)(xd * (1.0 - xd * xd * (1.0 / 3.0)));
    double e = pm_exp_d(2.0 * xd);
    return (float)((e - 1.0) / (e + 1.0));
}
static inline float p_powf(float x, float y) {
    // the engine only raises positive bases (rope base, Adam betas)
    if (x <= 0.0f) return x == 0.0f ? 0.0f : 0.0f / 0.0f;
    return (float)pm_exp_d((double)y * pm_log_d((double)x));
}

#ifdef RUNNER_PORTABLE_MATH
#define expf  p_expf
#define logf  p_logf
#define sinf  p_sinf
#define cosf  p_cosf
#define tanhf p_tanhf
#define powf  p_powf
#endif

#endif // RUNNER_PMATH_H
