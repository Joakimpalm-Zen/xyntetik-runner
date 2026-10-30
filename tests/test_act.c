// sigmoid_f and silu_f (src/act.h) under the engine's own CFLAGS
// (-ffast-math): every input from -300 to 300 gives a finite value within
// float rounding of the double reference, in loops the compiler vectorizes
// as it does the forward's. The plain 1/(1+expf(-x)) in the same loop is
// reported, not asserted: x86-64 builds return NaN below about -88 where
// arm64 returns 0, and this test must hold on both.
#include "act.h"
#include "fpclass.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2401

static float xs[N], sg[N], si[N], naive[N];

__attribute__((noinline)) static void run_helpers(void) {
    for (int i = 0; i < N; i++) sg[i] = sigmoid_f(xs[i]);
    for (int i = 0; i < N; i++) si[i] = silu_f(xs[i]);
}

__attribute__((noinline)) static void run_plain(void) {
    for (int i = 0; i < N; i++) naive[i] = 1.0f / (1.0f + expf(-xs[i]));
}

int main(void) {
    for (int i = 0; i < N; i++) xs[i] = -300.0f + 0.25f * (float)i;
    xs[0] = -269.7f; xs[1] = -166.3f; xs[2] = -89.0f; xs[3] = -88.5f;
    run_helpers();
    run_plain();
    int bad = 0, plain_nan = 0;
    for (int i = 0; i < N; i++) {
        double x = xs[i];
        double rs = x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
        double rl = x * rs;
        if (!fp_isfinite(sg[i]) || fabs(sg[i] - rs) > 1e-6 * fmax(1.0, fabs(rs)) + 1e-30) {
            fprintf(stderr, "sigmoid_f(%g) = %g, want %g\n", x, (double)sg[i], rs);
            bad++;
        }
        if (!fp_isfinite(si[i]) || fabs(si[i] - rl) > 1e-6 * fmax(1.0, fabs(rl)) + 1e-30) {
            fprintf(stderr, "silu_f(%g) = %g, want %g\n", x, (double)si[i], rl);
            bad++;
        }
        if (!fp_isfinite(naive[i])) plain_nan++;
    }
    if (bad) {
        fprintf(stderr, "act: %d wrong values\n", bad);
        return 1;
    }
    printf("act: sigmoid_f and silu_f finite and exact to float over [-300, 300] (%d inputs); "
           "the plain sigmoid gives %d non-finite here\n", N, plain_nan);
    return 0;
}
