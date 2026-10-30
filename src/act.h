// The sigmoid and SiLU, safe under -ffast-math.
//
// The engine builds with -ffast-math, and its finite-math-only lets the
// compiler assume no infinity exists. On x86-64 (clang 22, -march=native) the
// plain forms 1/(1+expf(-x)) and x/(1+expf(-x)) then return NaN for every x
// below about -88, where expf(-x) overflows: the vectorized reciprocal's
// refinement multiplies that infinity by zero. arm64 builds happen to return
// 0. An input that negative is rare but real: RecursiveSI flint's layer-46
// router logits reach -270 on the release file and qwen4exp's
// hyper-connection gate -95 (the lab CPU lane, 2026-10-30, and the Linux
// CUDA host, 2026-10-06), and either one NaN poisons the whole block.
//
// Below -80 both are 0 to within float32 (sigmoid(-80) is 1.8e-35, silu(-80)
// is -1.4e-33) and above 80 the sigmoid is 1, so the cut points move no value
// a float32 forward can use. silu_f is the SiLU the forward's gated_act
// computes, the same -80 cutoff.
//
// Every CPU sigmoid and SiLU goes through these, or carries the same guard on
// its own line; scripts/check-fpclass.py refuses an unguarded form anywhere
// else in src/, and tests/test_act.c holds both under the build's own flags.
#ifndef RUNNER_ACT_H
#define RUNNER_ACT_H

#include <math.h>

static inline float sigmoid_f(float x) {
    return x < -80.0f ? 0.0f : x > 80.0f ? 1.0f : 1.0f / (1.0f + expf(-x));
}

static inline float silu_f(float x) {
    return x < -80.0f ? 0.0f : x / (1.0f + expf(-x));
}

#endif
