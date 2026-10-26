// The attention backward of a training step, in double precision: the
// reference tests/test_attn_bw.c holds the CPU trainer to (model.c
// attn_bw_run) and tests/test_train_gpu.c the device's (k_tab_*). Per
// position t and head h, over the window [t0, t]: scores q.k_s times the
// scale, softmax p, dp_s = dao.v_s, ds_s = p_s (dp_s - sum_s p_s dp_s) scale;
// dq_t += sum_s ds_s k_s, dk_s += ds_s q_t, dv_s += p_s dao_t, with K and V
// f16 cache rows. The includer provides f32_to_f16 / f16_to_f32 (runner.h,
// with f16_init() called).
#ifndef RUNNER_TEST_ATTN_REF_H
#define RUNNER_TEST_ATTN_REF_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int T, H, kv_mul, hd, swa;   // swa: sliding window rows, 0 none
    int n_kv, q_dim, kv_dim;
} ab_shape;

static ab_shape ab_shape_of(int T, int H, int kv_mul, int hd, int swa) {
    ab_shape s = { T, H, kv_mul, hd, swa, H / kv_mul, H * hd, H / kv_mul * hd };
    return s;
}

static float ab_ref_rnd(uint64_t *r) {
    *r ^= *r << 13; *r ^= *r >> 7; *r ^= *r << 17;
    return (float)(int32_t)(uint32_t)(*r >> 32) / 2147483648.0f;
}

// K and V cache rows of kv_dim f16 values, q and dao rows of q_dim floats
static void ab_ref_fill(const ab_shape *s, uint64_t seed, uint8_t *kc, uint8_t *vc,
                        float *q, float *dao) {
    uint64_t r = seed | 1;
    for (size_t i = 0; i < (size_t)s->T * s->kv_dim; i++) {
        f16_t kh = f32_to_f16(ab_ref_rnd(&r)), vh = f32_to_f16(ab_ref_rnd(&r));
        memcpy(kc + 2 * i, &kh, 2);
        memcpy(vc + 2 * i, &vh, 2);
    }
    for (size_t i = 0; i < (size_t)s->T * s->q_dim; i++) {
        q[i] = ab_ref_rnd(&r);
        dao[i] = ab_ref_rnd(&r);
    }
}

// dq [T][q_dim], dk and dv [T][kv_dim], zeroed by the caller
static void ab_ref_run(const ab_shape *s, float scale, const uint8_t *kc,
                       const uint8_t *vc, const float *q, const float *dao,
                       double *dq, double *dk, double *dv) {
    int T = s->T, hd = s->hd, kvd = s->kv_dim;
    double *p = malloc(sizeof(double) * T), *dp = malloc(sizeof(double) * T);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < s->H; h++) {
            int kvh = h / s->kv_mul;
            int t0 = s->swa > 0 && t - s->swa + 1 > 0 ? t - s->swa + 1 : 0;
            const float *qh = q + (size_t)t * s->q_dim + h * hd;
            const float *dh = dao + (size_t)t * s->q_dim + h * hd;
            double mx = -1e300, sum = 0, spd = 0;
            for (int u = t0; u <= t; u++) {
                double sc = 0;
                for (int i = 0; i < hd; i++) {
                    f16_t k;
                    memcpy(&k, kc + 2 * ((size_t)u * kvd + kvh * hd + i), 2);
                    sc += qh[i] * (double)f16_to_f32(k);
                }
                p[u] = sc * scale;
                if (p[u] > mx) mx = p[u];
            }
            for (int u = t0; u <= t; u++) { p[u] = exp(p[u] - mx); sum += p[u]; }
            for (int u = t0; u <= t; u++) {
                p[u] /= sum;
                dp[u] = 0;
                for (int i = 0; i < hd; i++) {
                    f16_t v;
                    memcpy(&v, vc + 2 * ((size_t)u * kvd + kvh * hd + i), 2);
                    dp[u] += dh[i] * (double)f16_to_f32(v);
                }
                spd += p[u] * dp[u];
            }
            for (int u = t0; u <= t; u++) {
                double ds = p[u] * (dp[u] - spd) * scale;
                for (int i = 0; i < hd; i++) {
                    f16_t k;
                    memcpy(&k, kc + 2 * ((size_t)u * kvd + kvh * hd + i), 2);
                    dq[(size_t)t * s->q_dim + h * hd + i] += ds * f16_to_f32(k);
                    dk[(size_t)u * kvd + kvh * hd + i] += ds * qh[i];
                    dv[(size_t)u * kvd + kvh * hd + i] += p[u] * dh[i];
                }
            }
        }
    free(p);
    free(dp);
}

// the largest |got - ref| over n values relative to the largest |ref|
// (exact zeros wanted when the reference is all zero)
static double ab_ref_rel(const float *got, const double *ref, size_t n) {
    double worst = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - ref[i]);
        if (e > worst) worst = e;
        if (fabs(ref[i]) > mx) mx = fabs(ref[i]);
    }
    return mx > 0 ? worst / mx : worst;
}

#endif // RUNNER_TEST_ATTN_REF_H
