// Portable math (src/pmath.h) against libm: accuracy bounds over random
// inputs in the engine's ranges, and the special values (NaN, infinities,
// zero, subnormals, overflow edges). The T3 build replaces libm with these,
// so a regression here is a silent change in T3 logits; the bounds pinned
// are the ones measured when the header was written.
//
// Compile WITHOUT -DRUNNER_PORTABLE_MATH: this file needs both the p_*
// functions and the libm originals.
#include "pmath.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static int64_t ulp_diff(float a, float b) {
    if (isnan(a) || isnan(b)) return (isnan(a) && isnan(b)) ? 0 : INT64_C(1) << 40;
    if (isinf(a) || isinf(b)) return (a == b) ? 0 : INT64_C(1) << 40;
    int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = INT32_MIN - ia;
    if (ib < 0) ib = INT32_MIN - ib;
    int64_t d = (int64_t)ia - (int64_t)ib; return d < 0 ? -d : d;
}
static uint64_t rng = 88172645463325252ull;
static float rnd(float lo, float hi) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return lo + (hi - lo) * (float)((rng >> 11) & 0xffffff) / 16777216.0f;
}
typedef float (*f1)(float);
static void sweep(const char *name, f1 pf, f1 lf, float lo, float hi, int64_t max_ulp, double max_abs) {
    int64_t mx = 0; double mxa = 0; float at = 0;
    for (int i = 0; i < 2000000; i++) {
        float x = rnd(lo, hi), p = pf(x), l = lf(x);
        int64_t d = ulp_diff(p, l); double a = fabs((double)p - (double)l);
        // A NaN absolute error compares false below. Absolute-only sweeps
        // must reject a nonfinite mismatch even when their ULP bound is off.
        if (!isfinite(p) || !isfinite(l)) {
            CHECK(d == 0, "%s: nonfinite mismatch at %.9g", name, (double)x);
            if (d != 0) break;
            continue;
        }
        if (d > mx) { mx = d; at = x; }
        if (a > mxa) mxa = a;
    }
    if (max_ulp >= 0) CHECK(mx <= max_ulp, "%s on [%g,%g]: max %" PRId64 " ulp at %.9g (bound %" PRId64 ")", name, (double)lo, (double)hi, mx, (double)at, max_ulp);
    if (max_abs > 0) CHECK(mxa <= max_abs, "%s on [%g,%g]: max abs err %.3g (bound %.3g)", name, (double)lo, (double)hi, mxa, max_abs);
    printf("%-6s [%9g,%9g] max %" PRId64 " ulp, %.2e abs\n", name, (double)lo, (double)hi, mx, mxa);
}
static int same(float a, float b) { return (isnan(a) && isnan(b)) || (a == b && signbit(a) == signbit(b)); }

int main(void) {
    // IEEE binary32 anchors for the oracle itself, including a distance
    // larger than a 32-bit long can hold (Windows uses LLP64).
    CHECK(ulp_diff(1.0f, nextafterf(1.0f, INFINITY)) == 1, "adjacent positive floats differ by one ulp");
    CHECK(ulp_diff(-1.0f, nextafterf(-1.0f, -INFINITY)) == 1, "adjacent negative floats differ by one ulp");
    CHECK(ulp_diff(-0.0f, 0.0f) == 0, "signed zero has zero ulp distance");
    CHECK(ulp_diff(-FLT_MAX, FLT_MAX) == INT64_C(4278190078), "full finite span: twice 0x7f7fffff steps");
    CHECK(ulp_diff(INFINITY, 1.0f) > INT32_MAX && ulp_diff(NAN, 1.0f) > INT32_MAX,
          "nonfinite mismatches cannot pass a finite accuracy bound");
    sweep("expf",  p_expf,  expf,  -87.0f, 88.0f, 1, 0);
    sweep("expf",  p_expf,  expf,  -20.0f, 0.0f, 1, 0);
    sweep("logf",  p_logf,  logf,  1e-30f, 1e30f, 1, 0);
    sweep("logf",  p_logf,  logf,  0.5f, 2.0f, 1, 0);
    sweep("sinf",  p_sinf,  sinf,  -1000.0f, 1000.0f, 2, 0);
    sweep("cosf",  p_cosf,  cosf,  -1000.0f, 1000.0f, 2, 0);
    sweep("sinf",  p_sinf,  sinf,  -1e5f, 1e5f, -1, 1.2e-7);
    sweep("cosf",  p_cosf,  cosf,  -1e5f, 1e5f, -1, 1.2e-7);
    // 2, not 1: glibc's tanhf is itself up to 1 ulp off the correctly
    // rounded value (Apple's is not), and the two errors add
    sweep("tanhf", p_tanhf, tanhf, -30.0f, 30.0f, 2, 0);
    sweep("tanhf", p_tanhf, tanhf, -1e-3f, 1e-3f, 1, 0);
    {
        int64_t mx = 0;
        for (int i = 0; i < 2000000; i++) {
            float x = rnd(1e-3f, 1e6f), y = rnd(-4.0f, 4.0f);
            int64_t d = ulp_diff(p_powf(x, y), powf(x, y)); if (d > mx) mx = d;
        }
        CHECK(mx <= 1, "powf: max %" PRId64 " ulp", mx);
        printf("powf   positive base, |y|<=4    max %" PRId64 " ulp\n", mx);
    }
    // special values, exactly libm's answers
    const float inf = INFINITY, nan = NAN, tiny = 1.401298464e-45f;
    CHECK(same(p_expf(nan), nan) && same(p_logf(nan), nan) && same(p_sinf(nan), nan) &&
          same(p_cosf(nan), nan) && same(p_tanhf(nan), nan), "NaN in, NaN out");
    CHECK(same(p_expf(inf), inf) && same(p_expf(-inf), 0.0f), "exp of infinities");
    CHECK(same(p_logf(inf), inf) && same(p_logf(0.0f), -inf) && same(p_logf(-0.0f), -inf) &&
          isnan(p_logf(-1.0f)) && isnan(p_logf(-inf)), "log of infinities, zero, negatives");
    CHECK(isnan(p_sinf(inf)) && isnan(p_cosf(-inf)), "sin/cos of infinity is NaN");
    CHECK(isfinite(p_sinf(3.4e38f)) && isfinite(p_cosf(-3.4e38f)), "sin/cos of a huge finite angle is finite");
    CHECK(same(p_tanhf(inf), 1.0f) && same(p_tanhf(-inf), -1.0f), "tanh of infinities");
    CHECK(same(p_tanhf(tiny), tiny) && same(p_tanhf(-tiny), -tiny) && same(p_tanhf(-0.0f), -0.0f),
          "tanh of a subnormal is the subnormal, tanh(-0) is -0");
    CHECK(same(p_sinf(-0.0f), -0.0f) && same(p_cosf(0.0f), 1.0f), "sin(-0) is -0, cos(0) is 1");
    CHECK(same(p_expf(88.72f), expf(88.72f)) && same(p_expf(88.73f), inf) &&
          same(p_expf(-103.9f), expf(-103.9f)) && same(p_expf(-104.0f), 0.0f),
          "exp at the float overflow and underflow edges");
    CHECK(same(p_logf(tiny), logf(tiny)) && same(p_logf(3.4e38f), logf(3.4e38f)), "log at the float edges");
    if (g_fail) { fprintf(stderr, "pmath: FAILED\n"); return 1; }
    printf("pmath: OK (10 accuracy sweeps, special values match libm)\n");
    return 0;
}
