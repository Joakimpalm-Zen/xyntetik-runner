// R6.86: the attention softmax's OUTPUT in a portable build, bit for bit,
// against the softmax a portable build had before R6.86.
//
// make test-pmath-softmax builds this file with the engine's default flags
// and -DRUNNER_PORTABLE_MATH (a portable build of the default flags, what a
// Windows default would be), and make test runs it. Compared, element for
// element:
//   - new: the engine's softmax and softmax_sink (src/softmax.h): the exp in a
//     loop of its own, which vectorizes, then the sum in its written order;
//   - old: the same two functions as they stood before R6.86, word for word,
//     with that build's expf, (float)pm_exp_d((double)x). That exp had loops of
//     its own, so the compiler kept the fused loop scalar and the sum went one
//     add after another.
// Equal exp bits are not enough: under -ffast-math a fused loop that
// vectorizes adds its sum as a vector of partial sums, and the output moves in
// its last bits. That moved two Windows CPU replies off their pins (the first
// version of R6.86). So the control, which must be live: the fused loop with
// the vector-friendly exp, as that first version had it, must differ from the
// old output on some row under -ffast-math. Under strict float (T3) no loop
// reorders a sum, so there the control is not run and says so.
//
// The rows (deterministic): every length from 1 to 300, then lengths around
// the SIMD steps up to 4096, each with several value shapes: attention-like
// scores, near-equal scores (a large sum), a wide range down to -100 (tiny and
// subnormal exps), one spike over far smaller values, all equal, and the clamp
// edges of the exp. softmax_sink also with a sink above every score.
// The absolute anchor: every row's probabilities are in [0, 1] and sum to 1
// within 1e-5 (n times the float's epsilon would be the bound for an exact
// sum; this is looser, to hold the reassociated control too).
#include "softmax.h"
#include "pmath.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// before R6.86 (src/model.c at 03e98205), word for word, with its expf
static inline float old_expf(float x) { return (float)pm_exp_d((double)x); }
static __attribute__((noinline)) void old_softmax_sink(float *x, int n, float sink) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    if (sink > mx) mx = sink;
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = old_expf(x[i] - mx); s += x[i]; }
    s += old_expf(sink - mx);
    for (int i = 0; i < n; i++) x[i] /= s;
}
static __attribute__((noinline)) void old_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = old_expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
// the control: the fused loop with the vector-friendly exp (the first version of R6.86)
static __attribute__((noinline)) void fused_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = p_expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
static __attribute__((noinline)) void new_softmax(float *x, int n) { softmax(x, n); }
static __attribute__((noinline)) void new_softmax_sink(float *x, int n, float sink) { softmax_sink(x, n, sink); }

static uint64_t g_rng = 0x2545F4914F6CDD1Dull;
static float uni(float lo, float hi) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return lo + (hi - lo) * (float)((g_rng >> 40) & 0xffffff) / 16777216.0f;
}
enum { SHAPES = 6, NMAX = 4096 };
static void fill(float *x, int n, int shape) {
    for (int i = 0; i < n; i++) {
        switch (shape) {
            case 0: x[i] = uni(-12.0f, 4.0f); break;                        // attention-like
            case 1: x[i] = uni(-0.5f, 0.5f); break;                         // near-equal: a large sum
            case 2: x[i] = uni(-100.0f, 0.0f); break;                       // tiny and subnormal exps
            case 3: x[i] = i == n / 2 ? 30.0f : uni(-20.0f, 0.0f); break;   // one spike
            case 4: x[i] = 1.25f; break;                                    // all equal
            default: {                                                      // the exp's clamp edges
                static const float edge[] = { 0.0f, -103.97f, -104.0f, -104.5f, -88.5f, -1e4f };
                x[i] = edge[i % 6];
            }
        }
    }
}

static int g_fail = 0;
static uint64_t g_rows = 0, g_elems = 0, g_control_rows = 0;

// row by row: old and new equal bit for bit; the probabilities are a distribution
static void check(const char *what, const float *a, const float *b, int n) {
    g_rows++;
    g_elems += (uint64_t)n;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        uint32_t ua, ub;
        memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua != ub) {
            if (!g_fail)
                fprintf(stderr, "FAIL: %s, n %d, element %d: new %08" PRIx32 ", before R6.86 %08" PRIx32 "\n",
                        what, n, i, ub, ua);
            g_fail = 1;
            return;
        }
        if (!(b[i] >= 0.0f && b[i] <= 1.0f)) {
            if (!g_fail) fprintf(stderr, "FAIL: %s, n %d: probability %g out of [0, 1]\n", what, n, (double)b[i]);
            g_fail = 1;
            return;
        }
        sum += b[i];
    }
    // softmax_sink's rows carry less than one (the sink's share has no row): only softmax sums to 1
    if (strcmp(what, "softmax") == 0 && fabs(sum - 1.0) > 1e-5) {
        if (!g_fail) fprintf(stderr, "FAIL: softmax, n %d: the probabilities sum to %.9f\n", n, sum);
        g_fail = 1;
    }
}

static void row(int n, int shape) {
    static float src[NMAX], a[NMAX], b[NMAX], c[NMAX];
    fill(src, n, shape);
    memcpy(a, src, sizeof(float) * (size_t)n); old_softmax(a, n);
    memcpy(b, src, sizeof(float) * (size_t)n); new_softmax(b, n);
    check("softmax", a, b, n);
    memcpy(c, src, sizeof(float) * (size_t)n); fused_softmax(c, n);
    if (memcmp(a, c, sizeof(float) * (size_t)n) != 0) g_control_rows++;
    float sink = shape == 3 ? 40.0f : uni(-4.0f, 6.0f);
    memcpy(a, src, sizeof(float) * (size_t)n); old_softmax_sink(a, n, sink);
    memcpy(b, src, sizeof(float) * (size_t)n); new_softmax_sink(b, n, sink);
    check("softmax_sink", a, b, n);
}

int main(void) {
    static const int more[] = { 383, 384, 385, 511, 512, 513, 767, 768, 769, 1000, 1023, 1024, 1025,
                                2047, 2048, 2049, 3000, 4095, 4096 };
    for (int n = 1; n <= 300; n++)
        for (int sh = 0; sh < SHAPES; sh++) row(n, sh);
    for (size_t k = 0; k < sizeof more / sizeof more[0]; k++)
        for (int sh = 0; sh < SHAPES; sh++) row(more[k], sh);
    printf("pmath-softmax: %" PRIu64 " rows, %" PRIu64 " probabilities: %s against the softmax before R6.86\n",
           g_rows, g_elems, g_fail ? "NOT bit-identical" : "bit-identical");
#ifdef __FAST_MATH__
    printf("pmath-softmax: control: the fused loop with the vector-friendly exp differs on %" PRIu64 " of %" PRIu64
           " softmax rows\n", g_control_rows, g_rows / 2);
    if (!g_control_rows) {
        fprintf(stderr, "FAIL: the control is not live: the fused loop gives the old output on every row, so this "
                "test would not see a reordered sum\n");
        g_fail = 1;
    }
#else
    printf("pmath-softmax: control not run: strict float reorders no sum\n");
#endif
    if (g_fail) { fprintf(stderr, "pmath-softmax: FAILED\n"); return 1; }
    printf("pmath-softmax: OK\n");
    return 0;
}
