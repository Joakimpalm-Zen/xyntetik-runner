// The attention softmax, in a header of its own so that a test can run the
// engine's own code (R6.86): src/model.c includes it, and so does
// tests/test_pmath_callers.c, which checks what a NaN does here in a portable
// build against the default build. Moved from src/model.c unchanged.
#ifndef RUNNER_SOFTMAX_H
#define RUNNER_SOFTMAX_H

#include <math.h>

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
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    s += expf(sink - mx);
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

#endif
