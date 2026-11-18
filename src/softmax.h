// The attention softmax, in a header of its own so that tests can run the
// engine's own code (R6.86): src/model.c includes it, and so do
// tests/test_pmath_callers.c (what a NaN does here) and
// tests/test_pmath_softmax.c (its output against the softmax before R6.86).
//
// In a portable build (-DRUNNER_PORTABLE_MATH) the exp runs in a loop of its
// own, which vectorizes, and the sum after it in the order it always had:
// one add after another (src/pm_order.h). Fused, the loop vectorized as a
// whole under -ffast-math and added the sum as a vector of partial sums,
// which moved its last bits and two Windows CPU replies with them. The values
// are the same as before: each exp and each add. The default build keeps the
// fused loop, compiled as before.
#ifndef RUNNER_SOFTMAX_H
#define RUNNER_SOFTMAX_H

#include <math.h>

#include "pm_order.h"

// Attention softmax with a learned sink logit (gpt-oss). Transcribed from
// llama.cpp's ggml_compute_forward_soft_max: the sink participates in the max
// and in the denominator, and has NO output row — so the probabilities over
// real positions sum to less than one and the head's output shrinks. Note the
// sink is compared against ALREADY-SCALED scores; it is not itself scaled.
static void softmax_sink(float *x, int n, float sink) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    if (sink > mx) mx = sink;
    float s = 0;
#ifdef RUNNER_PORTABLE_MATH
    for (int i = 0; i < n; i++) x[i] = expf(x[i] - mx);
    for (int i = 0; i < n; i++) { s += x[i]; PM_IN_ORDER(s); }
#else
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
#endif
    s += expf(sink - mx);
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
#ifdef RUNNER_PORTABLE_MATH
    for (int i = 0; i < n; i++) x[i] = expf(x[i] - mx);
    for (int i = 0; i < n; i++) { s += x[i]; PM_IN_ORDER(s); }
#else
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
#endif
    for (int i = 0; i < n; i++) x[i] /= s;
}

#endif
