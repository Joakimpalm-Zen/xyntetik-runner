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
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static long ulp_diff(float a, float b) {
    if (isnan(a) || isnan(b)) return (isnan(a) && isnan(b)) ? 0 : 1L << 40;
    if (isinf(a) || isinf(b)) return (a == b) ? 0 : 1L << 40;
    int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = INT32_MIN - ia;
    if (ib < 0) ib = INT32_MIN - ib;
    long d = (long)ia - (long)ib; return d < 0 ? -d : d;
}
static uint64_t rng = 88172645463325252ull;
static float rnd(float lo, float hi) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return lo + (hi - lo) * (float)((rng >> 11) & 0xffffff) / 16777216.0f;
}
typedef float (*f1)(float);
static void sweep(const char *name, f1 pf, f1 lf, float lo, float hi, long max_ulp, double max_abs) {
    long mx = 0; double mxa = 0; float at = 0;
    for (int i = 0; i < 2000000; i++) {
        float x = rnd(lo, hi), p = pf(x), l = lf(x);
        long d = ulp_diff(p, l); double a = fabs((double)p - (double)l);
        if (d > mx) { mx = d; at = x; }
        if (a > mxa) mxa = a;
    }
    if (max_ulp >= 0) CHECK(mx <= max_ulp, "%s on [%g,%g]: max %ld ulp at %.9g (bound %ld)", name, (double)lo, (double)hi, mx, (double)at, max_ulp);
    if (max_abs > 0) CHECK(mxa <= max_abs, "%s on [%g,%g]: max abs err %.3g (bound %.3g)", name, (double)lo, (double)hi, mxa, max_abs);
    printf("%-6s [%9g,%9g] max %ld ulp, %.2e abs\n", name, (double)lo, (double)hi, mx, mxa);
}
static int same(float a, float b) { return (isnan(a) && isnan(b)) || (a == b && signbit(a) == signbit(b)); }

int main(void) {
    sweep("expf",  p_expf,  expf,  -87.0f, 88.0f, 1, 0);
    sweep("expf",  p_expf,  expf,  -20.0f, 0.0f, 1, 0);
    sweep("logf",  p_logf,  logf,  1e-30f, 1e30f, 1, 0);
    sweep("logf",  p_logf,  logf,  0.5f, 2.0f, 1, 0);
    sweep("sinf",  p_sinf,  sinf,  -1000.0f, 1000.0f, 2, 0);
    sweep("cosf",  p_cosf,  cosf,  -1000.0f, 1000.0f, 2, 0);
    sweep("sinf",  p_sinf,  sinf,  -1e5f, 1e5f, -1, 1.2e-7);
    sweep("cosf",  p_cosf,  cosf,  -1e5f, 1e5f, -1, 1.2e-7);
    sweep("tanhf", p_tanhf, tanhf, -30.0f, 30.0f, 1, 0);
    sweep("tanhf", p_tanhf, tanhf, -1e-3f, 1e-3f, 1, 0);
    {
        long mx = 0;
        for (int i = 0; i < 2000000; i++) {
            float x = rnd(1e-3f, 1e6f), y = rnd(-4.0f, 4.0f);
            long d = ulp_diff(p_powf(x, y), powf(x, y)); if (d > mx) mx = d;
        }
        CHECK(mx <= 1, "powf: max %ld ulp", mx);
        printf("powf   positive base, |y|<=4    max %ld ulp\n", mx);
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
