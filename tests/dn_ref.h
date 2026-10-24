// The Gated DeltaNet recurrence of a training step, in double precision:
// the reference tests/test_dn_recurrence.c holds the CPU trainer to
// (model.c dn_fw_worker, dn_bw_worker) and tests/test_train_gpu.c the
// device's (k_tdn_fw, k_tdn_bw). Per head h and position t, on the state S
// [hv][hv] (row j a value dim, column i a key dim), with q and k L2-normed:
//   S <- decay S;  pred_j = S_j . k;  delta_j = beta (v_j - pred_j);
//   S_j <- S_j + delta_j k;  y_j = S_j . q / sqrt(sk)
// and its reverse sweep from dy into dq, dk (per head), dv, dbeta, ddecay.
#ifndef RUNNER_TEST_DN_REF_H
#define RUNNER_TEST_DN_REF_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int nh, ng, hv, T;      // hv == sk
    int sk, keydim, inner, convdim;
} dn_shape;

static dn_shape dn_shape_of(int nh, int ng, int hv, int T) {
    dn_shape s = { nh, ng, hv, T, hv, hv * ng, nh * hv, 2 * hv * ng + nh * hv };
    return s;
}

static float dn_ref_rnd(uint64_t *r) {
    *r ^= *r << 13; *r ^= *r >> 7; *r ^= *r << 17;
    return (float)(int32_t)(uint32_t)(*r >> 32) / 2147483648.0f;
}

// inputs as the trainer leaves them: cv rows [T][convdim] with q and k
// L2-normed per group, beta in (0.2, 0.9), decay in (0.8, 1), dy [T][inner]
static void dn_ref_fill(const dn_shape *s, uint64_t seed, float *cv, float *beta,
                        float *decay, float *dy) {
    uint64_t r = seed | 1;
    for (size_t i = 0; i < (size_t)s->T * s->convdim; i++) cv[i] = dn_ref_rnd(&r);
    for (int t = 0; t < s->T; t++)
        for (int g = 0; g < 2 * s->ng; g++) {
            float *x = cv + (size_t)t * s->convdim + (size_t)g * s->sk, ss = 1e-6f;
            for (int i = 0; i < s->sk; i++) ss += x[i] * x[i];
            for (int i = 0; i < s->sk; i++) x[i] /= sqrtf(ss);
        }
    for (size_t i = 0; i < (size_t)s->T * s->nh; i++) {
        beta[i] = 0.55f + 0.35f * dn_ref_rnd(&r);
        decay[i] = 0.9f + 0.099f * dn_ref_rnd(&r);
    }
    for (size_t i = 0; i < (size_t)s->T * s->inner; i++) dy[i] = dn_ref_rnd(&r);
}

// The reference's outputs; S keeps every state, slot t + 1 a head's state
// after position t (slot 0 is zero), so slot T is the final one.
typedef struct {
    double *S, *y, *dq, *dk, *dv, *dbeta, *ddecay;
} dn_ref_out;

static void dn_ref_free(dn_ref_out *o) {
    free(o->S); free(o->y); free(o->dq); free(o->dk); free(o->dv);
    free(o->dbeta); free(o->ddecay);
}

static void dn_ref_run(const dn_shape *s, const float *cv, const float *beta,
                       const float *decay, const float *dy, dn_ref_out *o) {
    const int nh = s->nh, hv = s->hv, sk = s->sk, T = s->T;
    size_t hh = (size_t)hv * hv;
    o->S = calloc((size_t)nh * (T + 1) * hh, sizeof(double));
    o->y = calloc((size_t)T * s->inner, sizeof(double));
    o->dq = calloc((size_t)nh * T * sk, sizeof(double));
    o->dk = calloc((size_t)nh * T * sk, sizeof(double));
    o->dv = calloc((size_t)T * s->inner, sizeof(double));
    o->dbeta = calloc((size_t)T * nh, sizeof(double));
    o->ddecay = calloc((size_t)T * nh, sizeof(double));
    double *dS = malloc(sizeof(double) * hh), sq = sqrt((double)sk);
    for (int h = 0; h < nh; h++) {
        int g = h % s->ng;
        double *Sh = o->S + (size_t)h * (T + 1) * hh;
        for (int t = 0; t < T; t++) {
            const float *c = cv + (size_t)t * s->convdim;
            const float *q = c + g * sk, *k = c + s->keydim + g * sk;
            const float *v = c + 2 * s->keydim + h * hv;
            double dec = decay[(size_t)t * nh + h], bet = beta[(size_t)t * nh + h];
            double *Sp = Sh + (size_t)t * hh, *Sn = Sp + hh;
            for (int j = 0; j < hv; j++) {
                double pred = 0, acc = 0;
                for (int i = 0; i < hv; i++) pred += Sp[j * hv + i] * dec * k[i];
                double delta = (v[j] - pred) * bet;
                for (int i = 0; i < hv; i++) {
                    Sn[j * hv + i] = Sp[j * hv + i] * dec + delta * k[i];
                    acc += Sn[j * hv + i] * q[i];
                }
                o->y[(size_t)t * s->inner + h * hv + j] = acc / sq;
            }
        }
        memset(dS, 0, sizeof(double) * hh);
        for (int t = T - 1; t >= 0; t--) {
            const float *c = cv + (size_t)t * s->convdim;
            const float *q = c + g * sk, *k = c + s->keydim + g * sk;
            const float *v = c + 2 * s->keydim + h * hv;
            double dec = decay[(size_t)t * nh + h], bet = beta[(size_t)t * nh + h];
            const double *Sp = Sh + (size_t)t * hh, *St = Sp + hh;
            double *dq = o->dq + ((size_t)h * T + t) * sk;
            double *dk = o->dk + ((size_t)h * T + t) * sk;
            double db = 0, ddec = 0;
            for (int j = 0; j < hv; j++) {
                double gj = dy[(size_t)t * s->inner + h * hv + j] / sq;
                for (int i = 0; i < hv; i++) {
                    dS[j * hv + i] += gj * q[i];
                    dq[i] += gj * St[j * hv + i];
                }
            }
            for (int j = 0; j < hv; j++) {
                double pred = 0, dd = 0;
                for (int i = 0; i < hv; i++) {
                    pred += Sp[j * hv + i] * dec * k[i];
                    dd += dS[j * hv + i] * k[i];
                }
                double delta = (v[j] - pred) * bet;
                o->dv[(size_t)t * s->inner + h * hv + j] = bet * dd;
                db += dd * (v[j] - pred);
                double dpred = -bet * dd;
                for (int i = 0; i < hv; i++) {
                    double dsd = dS[j * hv + i] + dpred * k[i];
                    dk[i] += dS[j * hv + i] * delta + Sp[j * hv + i] * dec * dpred;
                    ddec += dsd * Sp[j * hv + i];
                    dS[j * hv + i] = dec * dsd;
                }
            }
            o->dbeta[(size_t)t * nh + h] = db;
            o->ddecay[(size_t)t * nh + h] = ddec;
        }
    }
    free(dS);
}

// the largest |got - ref| over n values, where, and the largest |ref|
static double dn_ref_err(const float *got, const double *ref, size_t n,
                         size_t *at, double *refmax) {
    double worst = 0, mx = 0;
    *at = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - ref[i]);
        if (e > worst) { worst = e; *at = i; }
        if (fabs(ref[i]) > mx) mx = fabs(ref[i]);
    }
    *refmax = mx;
    return worst;
}

#endif // RUNNER_TEST_DN_REF_H
