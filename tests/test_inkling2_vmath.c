// R4.48.25: the operation-level gate for Inkling 2 Mini's fast CPU route: its own vector kernels in src/inkling2.c,
// each against pmath's double forms (src/pmath.h) over the inputs the route feeds it.
//
// Why: test-inkling2-fast holds the fast route to 1e-4 of the reference route on a tiny file's logits. A kernel 125
// ulps off passed that bound. That was R4.48.20's vector tanh: (1 - e^-2|x|) / (1 + e^-2|x|) in float32 cancels near
// |x| = 1e-3. On the real file, it moved 18 argmaxes on R4.48.24's read, through the routers' near-ties. So each fast
// kernel is held here on its own, to a bound a few roundings wide, and the end-to-end bound stays where it is.
//
// Errors are relative and counted in u = 2^-24, float32's unit roundoff: a correctly rounded operation is within
// 1 u, and 1 ulp is between 1 u and 2 u. The reference is the double value itself, never its float rounding.
//
// The bounds and their rules:
//   EXP and TANH (ink2_vexp8 / ink2_vexp4, ink2_vtanh8 / ink2_vtanh4, or pmath's scalar forms in a build with neither):
//     4 u. Rule: a float32 transcendental that does not cancel loses its last rounding (1 u), its polynomial's
//     roundings (Horner's, damped by the reduced argument: under 2 u) and its range reduction's (1 u).
//     - Every 256th float bit pattern (2^24 inputs: every sign, exponent and high mantissa).
//     - The exp's two edges are its callers' contract, not an error:
//       - at or above 2^127.5 (x >= 88.3762626647949) it saturates to +inf; the callers only take 1 / (1 + e)
//         there, or never reach it (the softmax's arguments are at most 0);
//       - a result under FLT_MIN may flush to 0, an absolute FLT_MIN.
//     - The tanh's subnormal results are held to an absolute 2^-148.
//   SWIGLU (ink2_swiglu_range, the experts' activation: softcap(g, 4) * sigmoid(g) * softcap(u, 25) at Inkling's
//     caps, and at 1 and 2 for the tiny files): 20 u. Rule: the softcap is a division (1 u, which tanh's condition
//     number, at most 1, does not grow), the tanh (4 u) and a multiply (1 u), so 6 u, twice. The sigmoid is the exp
//     (4 u, damped by e / (1 + e) <= 1), an add and a division (2 u), so 6 u. Then two multiplies, 2 u.
//     - The inputs: g and u at every sign and 41 magnitudes from 1e-6 to 1e4, and 0.
//     - F 64 takes the vector blocks, and F 67 the scalar tail.
//     - An absolute floor of gate_cap * up_cap * FLT_MIN: the exp's flush, carried through the two softcaps.
//   ATTENTION (ink2_attn_core, the fast core and the reference core, one query of one head, hd 128 as Inkling's):
//     the first-order worst case of its operations, with gamma(n) = n u / (1 - n u) for an n-term sum in any order:
//     - a logit's error D_j = gamma(hd) * scale * sum |q k_j| + u |l_j|; the max's, Dm = max D_j;
//     - an exp's relative error e_j = D_j + Dm + u |l_j - m| + 4 u;
//     - a probability's 2 max e_j + gamma(nk + 1) + 2 u;
//     - the output's |o - o_ref| <= (2 max e_j + gamma(nk + 1) + gamma(nk) + 2 u) sum p |v| (1 + 1%: the second-order
//       terms, under 1% of it while n u <= 2e-4).
//     - Keys at 1, 2, 7, 8, 9, 15, 16, 17, 33, 500, 1024, 1025 and 2000: the vector blocks' tails, and the fast core's
//       own row past 1,024 keys. Logits spread 1 and 9.
// The linear kernels (vec_dot, vec_dot_multi, vec_dot_exact_tile) are quants.c's. test-quants-simd and
// test-nvfp4-dot hold them against a double dot, and test-dot-route holds the tile to the dot's bits.
//
// LIVE CONTROLS: each sweep must fail for a kernel known to be off.
//   - The cancelling tanh (R4.48.20's formula, in scalar float32) must exceed 4 u on the tanh sweep and 20 u through
//     the SwiGLU.
//   - The exp with the Cody-Waite split's low half dropped (ln2_lo = 0) must exceed 4 u.
//   - The fast core with its last key dropped must exceed the attention bound.
//
// usage: test_inkling2_vmath [--speed]
//   --speed also times the vector tanh against the scalar p_tanhf (information, never a pass or fail)
#include "inkling2.h"
#include "pmath.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static const double U = 5.9604644775390625e-08;   // 2^-24

static float from_bits(uint32_t b) { float f; memcpy(&f, &b, sizeof f); return f; }

// pmath's double forms (pmath.h): p_expf is (float)pm_exp_d(x); p_tanhf's body, before its float rounding
static double ref_exp(float x) { return pm_exp_d((double)x); }
static double ref_tanh(float x) {
    double xd = (double)x;
    if (xd > 20.0) return 1.0;
    if (xd < -20.0) return -1.0;
    if (xd < 1e-4 && xd > -1e-4) return xd * (1.0 - xd * xd * (1.0 / 3.0));
    double e = pm_exp_d(2.0 * xd);
    return (e - 1.0) / (e + 1.0);
}
static double ref_softcap(float v, float cap) { return ref_tanh((float)((double)v / cap)) * cap; }
static double ref_swiglu(float g, float u, float gc, float uc) {
    return ref_softcap(g, gc) * (1.0 / (1.0 + pm_exp_d(-(double)g))) * ref_softcap(u, uc);
}

// THE CONTROLS (scalar float32, uncontracted: the build has -ffp-contract=off)
static float ctl_tanh_cancel(float x) {   // R4.48.20's vector tanh, one lane
    float ax = fabsf(x);
    if (ax < 1e-3f) return x * (1.0f - x * x * (1.0f / 3.0f));
    if (ax >= 10.0f) return copysignf(1.0f, x);
    float e = p_expf(-2.0f * ax);
    return copysignf((1.0f - e) / (1.0f + e), x);
}
static float ctl_exp_no_lo(float x) {      // ink2_vexp8's steps, one lane, with ln2_lo = 0
    if (!(x > -87.3365447504f)) return x != x ? x : 0.0f;
    if (x >= 88.3762626647949f) return INFINITY;
    float kf = rintf(x * 1.44269504088896341f);
    float r = x - kf * 0.693145751953125f;
    float p = 1.98412698e-4f;
    p = p * r + 1.38888889e-3f; p = p * r + 8.33333333e-3f; p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f; p = p * r + 0.5f; p = p * r + 1.0f; p = p * r + 1.0f;
    return ldexpf(p, (int)kf);
}

// ---- EXP and TANH: every 256th bit pattern -------------------------------------------------------------------
enum { NSWEEP = 1 << 24 };
// worst relative error in u over the sweep, and the input where it occurs; a NaN in must give a NaN; fn_ctl, if
// set, is measured instead of the route's kernel
static double sweep(int fn, float (*fn_ctl)(float), float *worst_x, int *nan_bad) {
    float *x = malloc(sizeof(float) * NSWEEP), *y = malloc(sizeof(float) * NSWEEP);
    if (!x || !y) { printf("FAIL: cannot allocate the sweep\n"); exit(1); }
    for (uint32_t i = 0; i < NSWEEP; i++) x[i] = from_bits((i << 8) | (i & 0xffu));
    if (fn_ctl) for (uint32_t i = 0; i < NSWEEP; i++) y[i] = fn_ctl(x[i]);
    else if (ink2_test_vmath(fn, x, y, NSWEEP) != 0) { printf("FAIL: ink2_test_vmath refused fn %d\n", fn); exit(1); }
    double worst = 0.0;
    *worst_x = 0.0f;
    *nan_bad = 0;
    for (uint32_t i = 0; i < NSWEEP; i++) {
        const float xi = x[i], yi = y[i];
        if (xi != xi) { if (yi == yi) (*nan_bad)++; continue; }
        double r, e;
        if (fn == INK2_VM_EXP) {
            if (xi >= 88.3762626647949f) { if (!(yi == INFINITY || fabs(yi - ref_exp(xi)) <= 4 * U * ref_exp(xi))) e = INFINITY; else continue; }
            else {
                r = ref_exp(xi);
                if (r < FLT_MIN) { e = fabs(yi - r) <= FLT_MIN ? 0.0 : INFINITY; }
                else e = fabs(yi - r) / r / U;
            }
        } else {
            r = ref_tanh(xi);
            if (fabs(r) < FLT_MIN) e = fabs(yi - r) <= 0x1p-148 ? 0.0 : INFINITY;
            else e = fabs(yi - r) / fabs(r) / U;
        }
        if (e != e) e = INFINITY;
        if (e > worst) { worst = e; *worst_x = xi; }
    }
    free(x);
    free(y);
    return worst;
}

// ---- SWIGLU ----------------------------------------------------------------------------------------------------
enum { NMAG = 41 };
static int swiglu_inputs(float *v) {   // 0, and +-1e-6 .. 1e4 at 41 log-spaced magnitudes
    int n = 0;
    v[n++] = 0.0f;
    for (int m = 0; m < NMAG; m++) {
        const float a = (float)pow(10.0, -6.0 + 10.0 * m / (NMAG - 1));
        v[n++] = a;
        v[n++] = -a;
    }
    return n;
}
// worst relative error in u of the route's SwiGLU (or, with ctl, the control's) over every (g, u) pair
static double swiglu_worst(float gc, float uc, int F, int ctl, float *wg, float *wu) {
    float vals[1 + 2 * NMAG];
    const int nv = swiglu_inputs(vals);
    const int npairs = nv * nv, rows = (npairs + F - 1) / F;
    float *act = calloc((size_t)rows * 2 * F, sizeof(float));
    float *g_in = calloc((size_t)rows * F, sizeof(float)), *u_in = calloc((size_t)rows * F, sizeof(float));
    if (!act || !g_in || !u_in) { printf("FAIL: cannot allocate the SwiGLU sweep\n"); exit(1); }
    for (int p = 0; p < rows * F; p++) {
        const int q = p % npairs;   // the last row repeats the first pairs
        const int r = p / F, j = p % F;
        g_in[p] = vals[q / nv];
        u_in[p] = vals[q % nv];
        act[(size_t)r * 2 * F + j] = g_in[p];
        act[(size_t)r * 2 * F + F + j] = u_in[p];
    }
    if (ctl) {
        for (int p = 0; p < rows * F; p++) {
            const float g = g_in[p], u = u_in[p];
            const float sg = ctl_tanh_cancel(g / gc) * gc, su = ctl_tanh_cancel(u / uc) * uc;
            act[(size_t)(p / F) * 2 * F + p % F] = sg * (1.0f / (1.0f + p_expf(-g))) * su;
        }
    } else {
        ink2_test_swiglu(gc, uc, act, F, rows);
    }
    const double floor_abs = (double)gc * uc * FLT_MIN;
    double worst = 0.0;
    for (int p = 0; p < rows * F; p++) {
        const double r = ref_swiglu(g_in[p], u_in[p], gc, uc);
        const float y = act[(size_t)(p / F) * 2 * F + p % F];
        double e;
        if (r == 0.0) e = y == 0.0f ? 0.0 : INFINITY;
        else if (fabs(y - r) <= floor_abs) e = 0.0;
        else e = fabs(y - r) / fabs(r) / U;
        if (e != e) e = INFINITY;
        if (e > worst) { worst = e; *wg = g_in[p]; *wu = u_in[p]; }
    }
    free(act);
    free(g_in);
    free(u_in);
    return worst;
}

// R4.48.26: the outputs of the route's SwiGLU (ink2_swiglu_range, as the fast route's expert body calls it) that differ
// in their bits from the reference route's scalar ink2_swiglu, over every (g, u) pair of swiglu_inputs; the first
// differing pair in *fg, *fu
static long swiglu_bit_diffs(float gc, float uc, int F, float *fg, float *fu) {
    float vals[1 + 2 * NMAG];
    const int nv = swiglu_inputs(vals);
    const int npairs = nv * nv, rows = (npairs + F - 1) / F;
    float *act = calloc((size_t)rows * 2 * F, sizeof(float));
    if (!act) { printf("FAIL: cannot allocate the SwiGLU identity sweep\n"); exit(1); }
    for (int p = 0; p < rows * F; p++) {
        const int q = p % npairs;
        act[(size_t)(p / F) * 2 * F + p % F] = vals[q / nv];
        act[(size_t)(p / F) * 2 * F + F + p % F] = vals[q % nv];
    }
    ink2_test_swiglu(gc, uc, act, F, rows);
    long nd = 0;
    for (int p = 0; p < rows * F; p++) {
        const int q = p % npairs;
        const float want = ink2_test_swiglu_ref(gc, uc, vals[q / nv], vals[q % nv]);
        if (memcmp(&want, &act[(size_t)(p / F) * 2 * F + p % F], sizeof want)) {
            if (!nd) { *fg = vals[q / nv]; *fu = vals[q % nv]; }
            nd++;
        }
    }
    free(act);
    return nd;
}

// ---- ATTENTION -------------------------------------------------------------------------------------------------
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static double gauss(void) {   // Box-Muller on a 64-bit LCG: reproducible, no libc rand
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    double a = ((rng >> 11) + 0.5) * 0x1p-53;
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    double b = ((rng >> 11) + 0.5) * 0x1p-53;
    return sqrt(-2.0 * log(a)) * cos(6.283185307179586 * b);
}
static double gam(double n) { return n * U / (1.0 - n * U); }
// the worst ratio |o - o_ref| / bound over the hd outputs, for the core (fast or not) on nk_run of the nk keys
static double attn_ratio(int fast, int hd, int nk, int nk_run, double spread) {
    float *q = malloc(sizeof(float) * hd), *k = malloc(sizeof(float) * (size_t)nk * hd);
    float *v = malloc(sizeof(float) * (size_t)nk * hd), *o = malloc(sizeof(float) * hd);
    double *l = malloc(sizeof(double) * nk), *D = malloc(sizeof(double) * nk);
    if (!q || !k || !v || !o || !l || !D) { printf("FAIL: cannot allocate the attention case\n"); exit(1); }
    const double sd = sqrt(spread);   // q.k / sqrt(hd) then has standard deviation `spread`
    for (int e = 0; e < hd; e++) q[e] = (float)(gauss() * sd);
    for (size_t i = 0; i < (size_t)nk * hd; i++) { k[i] = (float)(gauss() * sd); v[i] = (float)gauss(); }
    const float sink = (float)gauss();
    if (ink2_test_attn_core(fast, hd, nk_run, q, k, v, sink, o) != INK2_OK) {
        printf("FAIL: ink2_test_attn_core refused hd %d nk %d\n", hd, nk_run);
        exit(1);
    }
    const float scale = ink2_inv_sqrt(hd);
    double m = sink, Dm = 0.0;
    for (int j = 0; j < nk; j++) {
        double a = 0.0, aa = 0.0;
        for (int e = 0; e < hd; e++) { a += (double)q[e] * k[(size_t)j * hd + e]; aa += fabs((double)q[e] * k[(size_t)j * hd + e]); }
        l[j] = a * scale;
        D[j] = gam(hd) * scale * aa + U * fabs(l[j]);
        if (l[j] > m) m = l[j];
        if (D[j] > Dm) Dm = D[j];
    }
    double S = exp((double)sink - m), emax = Dm + U * fabs((double)sink - m) + 4 * U;
    for (int j = 0; j < nk; j++) {
        S += exp(l[j] - m);
        const double ej = D[j] + Dm + U * fabs(l[j] - m) + 4 * U;
        if (ej > emax) emax = ej;
    }
    double worst = 0.0;
    for (int e = 0; e < hd; e++) {
        double o_ref = 0.0, mass = 0.0;
        for (int j = 0; j < nk; j++) {
            const double p = exp(l[j] - m) / S;
            o_ref += p * v[(size_t)j * hd + e];
            mass += p * fabs((double)v[(size_t)j * hd + e]);
        }
        const double bound = (2 * emax + gam(nk + 1) + gam(nk) + 2 * U) * mass * 1.01;
        const double ratio = fabs(o[e] - o_ref) / bound;
        if (!(ratio == ratio)) return INFINITY;
        if (ratio > worst) worst = ratio;
    }
    free(q); free(k); free(v); free(o); free(l); free(D);
    return worst;
}

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    const int speed = argc > 1 && !strcmp(argv[1], "--speed");
    printf("form: %s\n", ink2_test_vmath_form());
    float wx;
    int nan_bad;

    double w = sweep(INK2_VM_EXP, NULL, &wx, &nan_bad);
    CHECK(w <= 4.0, "exp: worst %.2f u (at %.9g) <= 4 u over 2^24 inputs", w, wx);
    CHECK(nan_bad == 0, "exp: a NaN in gives a NaN (%d did not)", nan_bad);
    w = sweep(INK2_VM_TANH, NULL, &wx, &nan_bad);
    CHECK(w <= 4.0, "tanh: worst %.2f u (at %.9g) <= 4 u over 2^24 inputs", w, wx);
    CHECK(nan_bad == 0, "tanh: a NaN in gives a NaN (%d did not)", nan_bad);
    w = sweep(INK2_VM_TANH, ctl_tanh_cancel, &wx, &nan_bad);
    CHECK(w > 4.0, "CONTROL: the cancelling tanh fails the tanh sweep (worst %.2f u at %.9g)", w, wx);
    w = sweep(INK2_VM_EXP, ctl_exp_no_lo, &wx, &nan_bad);
    CHECK(w > 4.0, "CONTROL: the exp without ln2_lo fails the exp sweep (worst %.2f u at %.9g)", w, wx);

    const float caps[][2] = { { 4.0f, 25.0f }, { 1.0f, 2.0f } };   // Inkling 2 Mini's, the tiny files'
    for (int c = 0; c < 2; c++)
        for (int F = 64; F <= 67; F += 3) {
            float wg = 0, wu = 0;
            w = swiglu_worst(caps[c][0], caps[c][1], F, 0, &wg, &wu);
            CHECK(w <= 20.0, "swiglu caps %g/%g F %d: worst %.2f u (g %.3g u %.3g) <= 20 u", caps[c][0], caps[c][1], F,
                  w, wg, wu);
        }
    {
        float wg = 0, wu = 0;
        w = swiglu_worst(4.0f, 25.0f, 64, 1, &wg, &wu);
        CHECK(w > 20.0, "CONTROL: the cancelling tanh fails the SwiGLU sweep (worst %.2f u, g %.3g u %.3g)", w, wg, wu);
    }
    // R4.48.26: the fast route's SwiGLU IS the reference route's scalar one, bit for bit
    for (int c = 0; c < 2; c++)
        for (int F = 64; F <= 67; F += 3) {
            float fg = 0, fu = 0;
            const long nd = swiglu_bit_diffs(caps[c][0], caps[c][1], F, &fg, &fu);
            CHECK(nd == 0, "swiglu caps %g/%g F %d: the fast route's outputs equal the reference route's scalar "
                  "ink2_swiglu bit for bit (%ld differ; the first at g %.9g u %.9g)", caps[c][0], caps[c][1], F, nd,
                  fg, fu);
        }

    const int nks[] = { 1, 2, 7, 8, 9, 15, 16, 17, 33, 500, 1024, 1025, 2000 };
    const double spreads[] = { 1.0, 9.0 };
    for (int f = 1; f >= 0; f--) {
        double worst = 0.0;
        int wnk = 0;
        double wsp = 0;
        for (size_t a = 0; a < sizeof nks / sizeof nks[0]; a++)
            for (int b = 0; b < 2; b++) {
                const double r = attn_ratio(f, 128, nks[a], nks[a], spreads[b]);
                if (!(r <= worst)) { worst = r; wnk = nks[a]; wsp = spreads[b]; }
            }
        CHECK(worst <= 1.0, "attention, %s core, hd 128: worst %.3g of the bound (nk %d, spread %g)",
              f ? "fast" : "reference", worst, wnk, wsp);
    }
    {
        const double r = attn_ratio(1, 128, 33, 32, 1.0);
        CHECK(r > 1.0, "CONTROL: the fast core with its last key dropped fails the bound (%.3g of it)", r);
    }

    if (speed) {   // information only
        enum { NS = 1 << 16, REPS = 400 };
        float *x = malloc(sizeof(float) * NS), *y = malloc(sizeof(float) * NS);
        if (!x || !y) return 1;
        for (int i = 0; i < NS; i++) x[i] = (float)((i - NS / 2) * (8.0 / NS));
        double best_v = 1e30, best_s = 1e30, sink = 0;
        for (int round = 0; round < 5; round++) {
            double t0 = now_s();
            for (int r = 0; r < REPS; r++) { ink2_test_vmath(INK2_VM_TANH, x, y, NS); sink += y[r]; }
            double t1 = now_s();
            for (int r = 0; r < REPS; r++) { for (int i = 0; i < NS; i++) y[i] = p_tanhf(x[i]); sink += y[r]; }
            double t2 = now_s();
            if (t1 - t0 < best_v) best_v = t1 - t0;
            if (t2 - t1 < best_s) best_s = t2 - t1;
        }
        printf("speed: vector tanh %.2f ns, scalar p_tanhf %.2f ns an element (%.1fx) [%g]\n",
               best_v * 1e9 / ((double)NS * REPS), best_s * 1e9 / ((double)NS * REPS), best_s / best_v, sink * 0);
        free(x);
        free(y);
    }
    printf(fails ? "FAILED: %d\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
