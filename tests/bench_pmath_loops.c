// R6.86: the attention softmax's loops and SwiGLU's loop, timed with the
// p_expf this change replaced, (float)pm_exp_d, and with the new one,
// interleaved in one binary. A measurement for the record, not a gate:
// make bench-pmath-loops. It is built with test-pmath-vec's flags, the
// engine's default flags in a portable build.
//
// The loops are src/model.c's softmax() and the SiLU branch of its
// gated_act(), with the exp as a macro parameter. The sizes:
//   - a softmax row of 512 keys, scores in [-12, 4]: an attention row of a 512-token prompt;
//   - a SwiGLU row of 9,728: Qwen3-4B's FFN width.
// Each round times both arms on the same inputs, a block of about 50 ms each, the order flipping each round. The
// result is the median over the rounds of the ratio (old time over new time), with the lowest and highest round.
#include "pmath.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline float old_expf(float x) { return (float)pm_exp_d((double)x); }

#define SOFTMAX(NAME, EXP)                                                        \
    static __attribute__((noinline)) void NAME(float *x, int n) {                 \
        float mx = x[0];                                                          \
        for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];                     \
        float s = 0;                                                              \
        for (int i = 0; i < n; i++) { x[i] = EXP(x[i] - mx); s += x[i]; }         \
        for (int i = 0; i < n; i++) x[i] /= s;                                    \
    }
SOFTMAX(softmax_old, old_expf)
SOFTMAX(softmax_new, p_expf)

#define SWIGLU(NAME, EXP)                                                         \
    static __attribute__((noinline)) void NAME(float *h, const float *g, const float *u, int n) { \
        for (int i = 0; i < n; i++) {                                             \
            float gi = g[i];                                                      \
            h[i] = gi < -80.0f ? 0.0f : (gi / (1.0f + EXP(-gi))) * u[i];          \
        }                                                                         \
    }
SWIGLU(swiglu_old, old_expf)
SWIGLU(swiglu_new, p_expf)

enum { NS = 512, NF = 9728, ROUNDS = 15 };
static float g_row[NS], g_work[NS], g_g[NF], g_u[NF], g_h[NF];
static volatile float g_sink;

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static float uni(float lo, float hi) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return lo + (hi - lo) * (float)((g_rng >> 40) & 0xffffff) / 16777216.0f;
}

// one block: reps calls of the arm; returns ns per element
static double block_softmax(int new_arm, long reps) {
    double t0 = now();
    for (long r = 0; r < reps; r++) {
        memcpy(g_work, g_row, sizeof g_work);
        if (new_arm) softmax_new(g_work, NS); else softmax_old(g_work, NS);
        g_sink += g_work[r % NS];
    }
    return (now() - t0) * 1e9 / ((double)reps * NS);
}
static double block_swiglu(int new_arm, long reps) {
    double t0 = now();
    for (long r = 0; r < reps; r++) {
        if (new_arm) swiglu_new(g_h, g_g, g_u, NF); else swiglu_old(g_h, g_g, g_u, NF);
        g_sink += g_h[r % NF];
    }
    return (now() - t0) * 1e9 / ((double)reps * NF);
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void bench(const char *name, double (*block)(int, long)) {
    long reps = 1;
    while (block(0, reps) * (double)reps * (block == block_softmax ? NS : NF) < 5e7) reps *= 2;   // ~50 ms on the old arm
    double t_old[ROUNDS], t_new[ROUNDS], ratio[ROUNDS];
    for (int k = 0; k < ROUNDS; k++) {
        if (k % 2 == 0) { t_old[k] = block(0, reps); t_new[k] = block(1, reps); }
        else            { t_new[k] = block(1, reps); t_old[k] = block(0, reps); }
        ratio[k] = t_old[k] / t_new[k];
    }
    double lo = ratio[0], hi = ratio[0];
    for (int k = 1; k < ROUNDS; k++) { if (ratio[k] < lo) lo = ratio[k]; if (ratio[k] > hi) hi = ratio[k]; }
    double mr = median(ratio, ROUNDS), mo = median(t_old, ROUNDS), mn = median(t_new, ROUNDS);
    printf("%-9s old %.2f ns/element, new %.2f ns/element: new is %.2fx as fast (median of %d interleaved rounds; "
           "lowest %.2fx, highest %.2fx)\n", name, mo, mn, mr, ROUNDS, lo, hi);
}

int main(void) {
    for (int i = 0; i < NS; i++) g_row[i] = uni(-12.0f, 4.0f);
    for (int i = 0; i < NF; i++) { g_g[i] = uni(-6.0f, 6.0f); g_u[i] = uni(-1.0f, 1.0f); }
#ifdef __FAST_MATH__
    printf("bench-pmath-loops: -ffast-math (the engine's default flags), %s\n", __VERSION__);
#else
    printf("bench-pmath-loops: strict float, %s\n", __VERSION__);
#endif
    // the same answers? Each exp is the same bits (test-pmath-expf); a vectorized sum adds in another order
    // under -ffast-math, so the softmax's quotients may move in their last bits. Reported, not judged.
    float a[NS], b[NS], ha[NF], hb[NF];
    memcpy(a, g_row, sizeof a); memcpy(b, g_row, sizeof b);
    softmax_old(a, NS); softmax_new(b, NS);
    swiglu_old(ha, g_g, g_u, NF); swiglu_new(hb, g_g, g_u, NF);
    printf("softmax outputs bit-identical: %s; SwiGLU outputs bit-identical: %s\n",
           memcmp(a, b, sizeof a) ? "no (the sum's order)" : "yes", memcmp(ha, hb, sizeof ha) ? "no" : "yes");
    bench("softmax", block_softmax);
    bench("SwiGLU", block_swiglu);
    return 0;
}
