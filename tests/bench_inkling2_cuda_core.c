// R4.48.9 (the decode fix): a BENCH, not a pass/fail test. The device attention core at T = 1 (a decode step) on
// Inkling 2 Mini's attention shape (24 query heads on 8 K/V heads, head_dim 128, the relative bias over 1024
// distances), after a 639-position prefill: the production pair (k_ink2_attn_logits, then k_ink2_attn) against the
// TEST-ONLY reference kernel (k_ink2_attn_ref, the core before the fix), each timed over the same single steps, on a
// full layer and on a sliding one (window 1024). gpu_ink2_attn is timed whole (the step's uploads, the ring put, the
// core and the rows back): the two kernels' difference is the core's. The decode diagnosis measured the reference at
// 0.554 ms an application inside the block (its synced phase).
#include "gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }

int main(void) {
    enum { NH = 24, NKV = 8, HD = 128, EXT = 1024, P = 639, STEPS = 64, R = 1100 };
    const int C = NKV * HD, Q = NH * HD, NP = P + STEPS;
    const float scale = 1.0f / sqrtf((float)HD);
    uint32_t seed = 7u;
#define RND() ((float)((int)((seed = seed * 1664525u + 1013904223u) >> 9) % 2001 - 1000) / 1000.0f)
    float *q = malloc(sizeof(float) * (size_t)NP * Q), *k = malloc(sizeof(float) * (size_t)NP * C);
    float *v = malloc(sizeof(float) * (size_t)NP * C), *rl = malloc(sizeof(float) * (size_t)NP * NH * EXT);
    float *sk = malloc(sizeof(float) * (size_t)NP * NH), *att = malloc(sizeof(float) * (size_t)NP * Q);
    int *pos = malloc(sizeof(int) * (size_t)NP);
    if (!q || !k || !v || !rl || !sk || !att || !pos) { fprintf(stderr, "out of memory\n"); return 1; }
    for (size_t i = 0; i < (size_t)NP * Q; i++) q[i] = RND();
    for (size_t i = 0; i < (size_t)NP * C; i++) { k[i] = RND(); v[i] = RND(); }
    for (size_t i = 0; i < (size_t)NP * NH * EXT; i++) rl[i] = RND();
    for (size_t i = 0; i < (size_t)NP * NH; i++) sk[i] = RND();
    for (int i = 0; i < NP; i++) pos[i] = i;
#undef RND
    gpu_ink2 *h = gpu_ink2_bind();
    if (!h || !gpu_ink2_attn_ok(h)) { fprintf(stderr, "bench-inkling2-cuda-core: SKIP (no CUDA device or no core kernels)\n"); return 77; }
    if (!gpu_ink2_reserve_attn(h, P, Q, NH, EXT, NP)) { fprintf(stderr, "no room for the core's buffers\n"); return 1; }
    const int rows[1] = { R }, cw[1] = { C };
    for (int global = 1; global >= 0; global--)
        for (int ref = 1; ref >= 0; ref--) {
            gpu_ink2_kv *kv = gpu_ink2_kv_new(h, 1, rows, cw, 0, 0, NULL);
            gpu_ink2_test_attn_ref(h, ref);
            bool ok = kv && gpu_ink2_attn(h, kv, 0, P, pos, q, k, v, rl, sk, att, NH, NKV, HD, 1024, global, EXT, scale);
            double best = 1e30, sum = 0;
            for (int s = 0; ok && s < STEPS; s++) {
                const int t = P + s;
                const double t0 = now_ms();
                ok = gpu_ink2_attn(h, kv, 0, 1, pos + t, q + (size_t)t * Q, k + (size_t)t * C, v + (size_t)t * C,
                                   rl + (size_t)t * NH * EXT, sk + (size_t)t * NH, att, NH, NKV, HD, 1024, global, EXT, scale);
                const double dt = now_ms() - t0;
                sum += dt;
                if (dt < best) best = dt;
            }
            gpu_ink2_test_attn_ref(h, 0);
            printf("%s layer, %s: %d single steps at %d..%d keys: mean %.4f ms, best %.4f ms a call%s\n",
                   global ? "full" : "sliding", ref ? "reference kernel (before the fix)" : "production pair       ",
                   STEPS, P + 1, P + STEPS, sum / STEPS, best, ok ? "" : " (A CALL FAILED)");
            gpu_ink2_kv_free(kv);
        }
    gpu_ink2_free(h);
    free(q); free(k); free(v); free(rl); free(sk); free(att); free(pos);
    return 0;
}
