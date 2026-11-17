// R6.86: what a NaN does at the portable exp's callers, in a portable build
// of the default flags, against the default build.
//
// make test-pmath-callers builds this file twice, each with the engine's own
// sampler (src/sample.c) and softmax (src/softmax.h):
//   - with the default flags, where expf is the C library's;
//   - with the same flags and -DRUNNER_PORTABLE_MATH, where it is src/pmath.h's.
// Both print the same lines, and the two outputs must be identical. A NaN
// reaches these callers only from a poisoned upstream value. The portable
// math must then do what the default build does, so that switching a build to
// it (a Windows default, R6.85) changes nothing there.
//
// The cases (from the second read of PR 926):
//   - sample_pick with logits [0, NaN], top_k 2, temperature 1, RNG states 1 to 4: a NaN inside top-k. The p_expf
//     before the NaN fix gave +inf for a quiet NaN, which poisoned the normalization, and the NaN token was picked
//     every time where the default build picks the finite one.
//   - softmax_sink with scores [0, -1] and a NaN sink: it gave finite zeros where the default build gives NaNs, so
//     a poisoned sink erased the real positions without a trace.
//   - softmax with a NaN among the scores.
// Each with quiet NaNs of both signs and with payloads, and a signaling one.
// Only NaN cases: for a finite input the C library's expf and pmath's may
// differ in the last bit, which is the point of the portable math.
#include "sample.h"
#include "softmax.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float flt(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static const uint32_t NANS[] = { 0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0xffd2468au, 0x7f800001u };
enum { NNAN = (int)(sizeof(NANS) / sizeof(NANS[0])) };

int main(void) {
    for (int k = 0; k < NNAN; k++) {
        float q = flt(NANS[k]);
        printf("NaN %08" PRIx32 ": sample_pick [0, NaN] top_k 2:", NANS[k]);
        for (uint64_t seed = 1; seed <= 4; seed++) {
            float logits[2] = { 0.0f, q };
            sampler s = { .temp = 1.0f, .top_p = 1.0f, .top_k = 2, .repeat_penalty = 1.0f, .rng = seed };
            printf(" %d", sample_pick(&s, logits, 2, NULL, NULL));
        }
        float x[2] = { 0.0f, -1.0f };
        softmax_sink(x, 2, q);
        printf("; softmax_sink [0, -1] sink NaN: %08" PRIx32 " %08" PRIx32, bits(x[0]), bits(x[1]));
        float y[3] = { 0.0f, q, -1.0f };
        softmax(y, 3);
        printf("; softmax [0, NaN, -1]: %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n", bits(y[0]), bits(y[1]), bits(y[2]));
    }
    return 0;
}
