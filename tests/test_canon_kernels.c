// Canonical-order kernels (RUNNER_CANON_KERNELS) against an independent
// reference of the SAME tree, bit for bit.
//
// The property under test is not "the dot product is accurate" (that is
// tests/test_quants_simd.c) but "every target computes this exact
// association": eight virtual lanes, lane i accumulating the elements
// congruent to i mod 8 in each 32-wide step with fused multiply-adds, the
// block scale fused in per lane, then ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)).
// The reference here is that definition written in plain C with fmaf, which
// is what the no-SIMD fallback is; on an AVX2 or NEON host the kernel under
// test is the intrinsics version, so the comparison is SIMD-vs-scalar on one
// machine. The cross-machine half (arm64 == x86-64 == riscv64 on a real model)
// is the measurement in docs/portable-bitexact-2026-09-05.md; this gate is
// what makes that result stable under kernel edits.
//
// Compile with the engine's strict float regime (-fno-fast-math
// -ffp-contract=off) and -DRUNNER_CANON_KERNELS; without the define the
// engine runs its ISA-native kernels and this gate must not be built.
#include "quants.h"
#include "fp16.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define QK 32
typedef struct { f16_t d; int8_t qs[QK]; } block_q8_0;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) & 0xffffff) / 8388608.0f * 2.0f - 1.0f;
}

static float tree8(const float l[8]) {
    float c0 = l[0] + l[4], c1 = l[1] + l[5], c2 = l[2] + l[6], c3 = l[3] + l[7];
    float d0 = c0 + c2, d1 = c1 + c3;
    return d0 + d1;
}

// the definition, independently
static float ref_q8_0(const block_q8_0 *b, const float *x, int n) {
    float acc[8] = {0};
    for (int i = 0; i < n / QK; i++) {
        float d = f16_to_f32(b[i].d);
        for (int l = 0; l < 8; l++) {
            float t = (float)b[i].qs[l] * x[i * QK + l];
            t = fmaf((float)b[i].qs[8 + l],  x[i * QK + 8 + l],  t);
            t = fmaf((float)b[i].qs[16 + l], x[i * QK + 16 + l], t);
            t = fmaf((float)b[i].qs[24 + l], x[i * QK + 24 + l], t);
            acc[l] = fmaf(d, t, acc[l]);
        }
    }
    return tree8(acc);
}
static float ref_f16(const f16_t *w, const float *x, int n) {
    float a[8] = {0}, b[8] = {0};
    int i = 0;
    for (; i + 16 <= n; i += 16)
        for (int l = 0; l < 8; l++) {
            a[l] = fmaf(f16_to_f32(w[i + l]),     x[i + l],     a[l]);
            b[l] = fmaf(f16_to_f32(w[i + 8 + l]), x[i + 8 + l], b[l]);
        }
    float v[8];
    for (int l = 0; l < 8; l++) v[l] = a[l] + b[l];
    float s = tree8(v);
    for (; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
}
static float ref_f32(const float *w, const float *x, int n) {
    float a[8] = {0}, b[8] = {0}, c[8] = {0}, d[8] = {0};
    int i = 0;
    for (; i + 32 <= n; i += 32)
        for (int l = 0; l < 8; l++) {
            a[l] = fmaf(w[i + l],      x[i + l],      a[l]);
            b[l] = fmaf(w[i + 8 + l],  x[i + 8 + l],  b[l]);
            c[l] = fmaf(w[i + 16 + l], x[i + 16 + l], c[l]);
            d[l] = fmaf(w[i + 24 + l], x[i + 24 + l], d[l]);
        }
    float v[8];
    for (int l = 0; l < 8; l++) { float ab = a[l] + b[l], cd = c[l] + d[l]; v[l] = ab + cd; }
    float s = tree8(v);
    for (; i < n; i++) s += w[i] * x[i];
    return s;
}

static int bits_equal(float a, float b) {
    uint32_t ua, ub; memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
    return ua == ub || (isnan(a) && isnan(b));
}

int main(void) {
    f16_init();
    // lengths: multiples of 32, the f16 16-step, and ragged tails
    const int lens[] = { 32, 64, 96, 128, 256, 1024, 2048, 2064, 2071, 48, 17, 1, 1536 };
    int checks = 0;
    for (size_t li = 0; li < sizeof(lens) / sizeof(*lens); li++) {
        int n = lens[li];
        float *x = malloc(sizeof(float) * (size_t)n);
        float *w = malloc(sizeof(float) * (size_t)n);
        f16_t *h = malloc(sizeof(f16_t) * (size_t)n);
        block_q8_0 *q = malloc(sizeof(block_q8_0) * (size_t)(n / QK + 1));
        for (int trial = 0; trial < 64; trial++) {
            for (int i = 0; i < n; i++) {
                x[i] = frand() * 4.0f;
                w[i] = frand();
                h[i] = f32_to_f16_soft(frand());
            }
            for (int i = 0; i < n / QK; i++) {
                q[i].d = f32_to_f16_soft(frand() * 0.05f);
                for (int j = 0; j < QK; j++) q[i].qs[j] = (int8_t)((rng = rng * 6364136223846793005ull + 1) >> 56);
            }
            float k = vec_dot(T_F32, w, x, n), r = ref_f32(w, x, n);
            CHECK(bits_equal(k, r), "F32 n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
            k = vec_dot(T_F16, h, x, n); r = ref_f16(h, x, n);
            CHECK(bits_equal(k, r), "F16 n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
            if (n % QK == 0) {
                k = vec_dot(T_Q8_0, q, x, n); r = ref_q8_0(q, x, n);
                CHECK(bits_equal(k, r), "Q8_0 n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
                checks++;
            }
            checks += 2;
        }
        free(x); free(w); free(h); free(q);
    }
    if (g_fail) { fprintf(stderr, "canon kernels: FAILED\n"); return 1; }
    printf("canon kernels: %d bit-identical checks against the tree definition\n", checks);
    return 0;
}
