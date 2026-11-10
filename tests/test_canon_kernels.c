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
#include <stddef.h>
#include <stdbool.h>

// setenv is POSIX; MinGW has _putenv_s (main's windows-suite-gates-core
// builds this test, a PR's CI does not)
#ifdef _WIN32
#define setenv_compat(k, v) _putenv_s(k, v)
#else
#define setenv_compat(k, v) setenv(k, v, 1)
#endif

#define QK 32
#define QK_K 256
typedef struct { f16_t d; int8_t qs[QK]; } block_q8_0;
// R12.1b: the k-quant and Q4_0 layouts, from the GGUF format (as quants.c has them)
typedef struct { f16_t d; uint8_t qs[QK / 2]; } block_q4_0;
typedef struct { f16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } block_q4_K;
typedef struct { f16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } block_q5_K;
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; f16_t d; } block_q6_K;

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

// R12.1b, the definitions: see quants.c's canonical section. Each element's
// value is decoded here from the format, not through the engine's helpers.
static int q4_0_val(const block_q4_0 *b, int j) {   // element j of 32
    return (j < 16 ? (b->qs[j] & 0xF) : (b->qs[j - 16] >> 4)) - 8;
}
static void kscale(const uint8_t *s, int j, float *sc, float *mn) {   // the 6-bit pair j of 8
    int a, m;
    if (j < 4) { a = s[j] & 63; m = s[j + 4] & 63; }
    else { a = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
    *sc = (float)a; *mn = (float)m;
}
static float ref_q4_0(const block_q4_0 *b, const float *x, int n) {
    float acc[8] = {0};
    for (int i = 0; i < n / QK; i++) {
        float d = f16_to_f32(b[i].d);
        const float *xp = x + i * QK;
        for (int l = 0; l < 8; l++) {
            float t = (float)q4_0_val(&b[i], l) * xp[l];
            for (int k = 1; k < 4; k++) t = fmaf((float)q4_0_val(&b[i], 8 * k + l), xp[8 * k + l], t);
            acc[l] = fmaf(d, t, acc[l]);
        }
    }
    return tree8(acc);
}
// Q4_K (qh NULL) and Q5_K: element e (0..255) of a super-block is in
// sub-block e / 32; sub-blocks 2g and 2g+1 share qs[32g .. 32g+31] (low, high
// nibbles) and Q5_K's fifth bit is qh[e % 32] bit (e / 32)
static int k4_val(const uint8_t *qs, const uint8_t *qh, int e) {
    int sub = e / 32, p = e % 32, g = sub / 2;
    int v = (sub % 2) ? (qs[32 * g + p] >> 4) : (qs[32 * g + p] & 0xF);
    if (qh && (qh[p] >> sub) & 1) v += 16;
    return v;
}
static float ref_k4(const uint8_t *blk0, size_t sblk, size_t qs_off, const uint8_t *qh_base, size_t qh_off,
                    const float *x, int n) {
    float acc[8] = {0}, msum = 0;
    for (int i = 0; i < n / QK_K; i++) {
        const uint8_t *blk = blk0 + (size_t)i * sblk;
        f16_t dh, mh;
        memcpy(&dh, blk, 2); memcpy(&mh, blk + 2, 2);
        float d = f16_to_f32(dh), dmin = f16_to_f32(mh);
        const uint8_t *qs = blk + qs_off, *qh = qh_base ? blk + qh_off : NULL;
        const float *xp = x + i * QK_K;
        for (int g = 0; g < 4; g++) {
            float sc1, mn1, sc2, mn2;
            kscale(blk + 4, 2 * g, &sc1, &mn1);
            kscale(blk + 4, 2 * g + 1, &sc2, &mn2);
            float d1 = d * sc1, m1 = dmin * mn1, d2 = d * sc2, m2 = dmin * mn2;
            float t1[8], t2[8], s1[8], s2[8];
            for (int l = 0; l < 8; l++) {
                t1[l] = 0.0f; t2[l] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    t1[l] = fmaf((float)k4_val(qs, qh, 64 * g + 8 * k + l),      xp[64 * g + 8 * k + l],      t1[l]);
                    t2[l] = fmaf((float)k4_val(qs, qh, 64 * g + 32 + 8 * k + l), xp[64 * g + 32 + 8 * k + l], t2[l]);
                }
                const float *a = xp + 64 * g, *c = xp + 64 * g + 32;
                s1[l] = (0.0f + (a[l] + a[8 + l])) + (a[16 + l] + a[24 + l]);
                s2[l] = (0.0f + (c[l] + c[8 + l])) + (c[16 + l] + c[24 + l]);
            }
            for (int l = 0; l < 8; l++) acc[l] = fmaf(d1, t1[l], acc[l]);
            for (int l = 0; l < 8; l++) acc[l] = fmaf(d2, t2[l], acc[l]);
            float sx1 = tree8(s1), sx2 = tree8(s2);
            float mm1 = m1 * sx1, mm2 = m2 * sx2;
            msum = msum + (mm1 + mm2);
        }
    }
    return tree8(acc) - msum;
}
static float ref_q4_K(const block_q4_K *b, const float *x, int n) {
    return ref_k4((const uint8_t *)b, sizeof *b, offsetof(block_q4_K, qs), NULL, 0, x, n);
}
static float ref_q5_K(const block_q5_K *b, const float *x, int n) {
    return ref_k4((const uint8_t *)b, sizeof *b, offsetof(block_q5_K, qs), (const uint8_t *)b,
                  offsetof(block_q5_K, qh), x, n);
}
// Q6_K: element e (0..255) of a super-block is in half h = e / 128; within the
// half, quarter k = (e % 128) / 32 and position p = e % 32 take ql[64h + p]
// (k 0) or ql[64h + p + 32] (k 1) low nibbles, the same bytes' high nibbles
// for k 2 and 3, and two bits of qh[32h + p] at bit 2k; less 32
static int q6_val(const block_q6_K *b, int e) {
    int h = e / 128, k = (e % 128) / 32, p = e % 32;
    int lq = b->ql[64 * h + p + ((k & 1) ? 32 : 0)];
    int nib = k < 2 ? (lq & 0xF) : (lq >> 4);
    int hb = (b->qh[32 * h + p] >> (2 * k)) & 3;
    return (nib | (hb << 4)) - 32;
}
static float ref_q6_K(const block_q6_K *b, const float *x, int n) {
    float acc[8] = {0};
    for (int i = 0; i < n / QK_K; i++) {
        float d = f16_to_f32(b[i].d);
        const float *xp = x + i * QK_K;
        for (int h = 0; h < 2; h++)
            for (int base = 0; base < 32; base += 16)
                for (int k = 0; k < 4; k++) {
                    int e0 = 128 * h + 32 * k + base;
                    float ds = d * (float)b[i].scales[8 * h + 2 * k + base / 16];
                    for (int l = 0; l < 8; l++) {
                        float a = (float)q6_val(&b[i], e0 + l) * xp[e0 + l];
                        a = fmaf((float)q6_val(&b[i], e0 + 8 + l), xp[e0 + 8 + l], a);
                        acc[l] = fmaf(ds, a, acc[l]);
                    }
                }
    }
    return tree8(acc);
}

static int bits_equal(float a, float b) {
    uint32_t ua, ub; memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
    return ua == ub || (isnan(a) && isnan(b));
}

// RUNNER_CPU_I8 is honoured only where an int8 kernel exists: the route line
// and receipts read i8_dot_enabled() to name the route, and a build without
// the kernel (this canonical build on arm64, or the 1.11.0 macOS binary built
// for generic armv8-a) runs every matvec f32. Enabled implies a kernel, on
// every build, from the environment and when forced.
static int check_i8_flag_needs_a_kernel(void) {
    bool kernel = i8_dot_ok(T_Q8_0, 32);
    setenv_compat("RUNNER_CPU_I8", "1");
    bool env_on = i8_dot_enabled();
    i8_dot_force(1);
    bool forced_on = i8_dot_enabled();
    i8_dot_force(-1);
    if (env_on != kernel || forced_on != kernel) {
        fprintf(stderr, "FAIL: RUNNER_CPU_I8 enabled=%d forced=%d on a build where an "
                "int8 kernel %s\n", env_on, forced_on, kernel ? "exists" : "does not exist");
        return 1;
    }
    printf("int8 flag: %s (kernel %s)\n", env_on ? "on" : "off, with a notice",
           kernel ? "present" : "absent");
    return 0;
}

int main(void) {
    f16_init();
    if (check_i8_flag_needs_a_kernel()) return 1;
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
    // R12.1b: Q4_0 and the k-quants, at the block lengths they take
    {
        const int klens[] = { 256, 512, 2048, 2560, 9728 };
        for (size_t li = 0; li < sizeof(klens) / sizeof(*klens); li++) {
            int n = klens[li], nb = n / QK_K, n0 = n / QK;
            float *x = malloc(sizeof(float) * (size_t)n);
            block_q4_0 *b0 = malloc(sizeof(block_q4_0) * (size_t)n0);
            block_q4_K *b4 = malloc(sizeof(block_q4_K) * (size_t)nb);
            block_q5_K *b5 = malloc(sizeof(block_q5_K) * (size_t)nb);
            block_q6_K *b6 = malloc(sizeof(block_q6_K) * (size_t)nb);
            for (int trial = 0; trial < 48; trial++) {
                // zeros on some trials, so a zero sum's sign is checked too
                for (int i = 0; i < n; i++) x[i] = (trial % 7 == 3 && i % 5 == 0) ? 0.0f : frand() * 4.0f;
                uint8_t *raw[4] = { (uint8_t *)b0, (uint8_t *)b4, (uint8_t *)b5, (uint8_t *)b6 };
                size_t sz[4] = { sizeof(block_q4_0) * (size_t)n0, sizeof(block_q4_K) * (size_t)nb,
                                 sizeof(block_q5_K) * (size_t)nb, sizeof(block_q6_K) * (size_t)nb };
                for (int t = 0; t < 4; t++)
                    for (size_t k = 0; k < sz[t]; k++) raw[t][k] = (uint8_t)((rng = rng * 6364136223846793005ull + 1) >> 56);
                // scales: ordinary, tiny (subnormal f16 on some trials) and wide
                float span = trial % 11 == 5 ? 1e-6f : trial % 13 == 7 ? 8.0f : 0.05f;
                for (int i = 0; i < n0; i++) b0[i].d = f32_to_f16_soft(frand() * span);
                for (int i = 0; i < nb; i++) {
                    b4[i].d = f32_to_f16_soft(frand() * span); b4[i].dmin = f32_to_f16_soft(frand() * span);
                    b5[i].d = f32_to_f16_soft(frand() * span); b5[i].dmin = f32_to_f16_soft(frand() * span);
                    b6[i].d = f32_to_f16_soft(frand() * span);
                }
                float k = vec_dot(T_Q4_0, b0, x, n), r = ref_q4_0(b0, x, n);
                CHECK(bits_equal(k, r), "Q4_0 n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
                k = vec_dot(T_Q4_K, b4, x, n); r = ref_q4_K(b4, x, n);
                CHECK(bits_equal(k, r), "Q4_K n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
                k = vec_dot(T_Q5_K, b5, x, n); r = ref_q5_K(b5, x, n);
                CHECK(bits_equal(k, r), "Q5_K n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
                k = vec_dot(T_Q6_K, b6, x, n); r = ref_q6_K(b6, x, n);
                CHECK(bits_equal(k, r), "Q6_K n=%d trial %d: kernel %.9g ref %.9g", n, trial, (double)k, (double)r);
                checks += 4;
            }
            free(x); free(b0); free(b4); free(b5); free(b6);
        }
        // the canonical build says which types it runs canonically (R12.1b's canon_gaps)
        const int canon[] = { T_F32, T_F16, T_Q8_0, T_Q4_0, T_Q4_K, T_Q5_K, T_Q6_K };
        for (size_t c = 0; c < sizeof canon / sizeof *canon; c++)
            CHECK(vec_dot_canonical(canon[c]), "type %d is dispatched canonically but not reported so", canon[c]);
        CHECK(!vec_dot_canonical(T_IQ4_XS), "IQ4_XS has no canonical kernel but is reported canonical");
        CHECK(vec_dot_canon_build(), "a canonical build does not say so");
    }
    if (g_fail) { fprintf(stderr, "canon kernels: FAILED\n"); return 1; }
    printf("canon kernels: %d bit-identical checks against the tree definition\n", checks);
    return 0;
}
