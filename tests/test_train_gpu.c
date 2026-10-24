// R8.19: the training products on the device, every weight type the CPU
// trainer decodes, on random blocks (scale fields made finite).
//
//   transposed  gpu_train_mvt == the CPU trainer's chain (dequant_row, then
//               an ascending-j fmaf per element, a zero dy skipped), memcmp
//   forward     gpu_train_mm is batch-invariant: every column of a 77-column
//               call (the 32-column kernel, three tiles) equals the same
//               column computed alone and in a 5-column call (the 16-column
//               kernel),
//               memcmp; and it is the product (each column within 2^-16 of
//               sum|w x| of a double-precision reference)
//
// R8.21 adds the DeltaNet recurrence's device calls (dn_check below).
//
// Skips (not passes) without a CUDA device. Prints each type's throughput.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float frnd(void) { return (float)(int32_t)rnd32() / 2147483648.0f; }

static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static uint16_t get16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
// a finite block scale, either sign, 2^-10 .. 2^-4
static uint16_t rnd_scale(void) {
    float v = ldexpf(1.0f + (float)(rnd32() % 1024) / 1024.0f, -10 + (int)(rnd32() % 7));
    return f32_to_f16(rnd32() & 1 ? -v : v);
}

static const struct { int type; const char *name; int f16[2]; } TYPES[] = {
    { T_F32, "f32", {-1, -1} },        { T_F16, "f16", {-1, -1} },
    { T_BF16, "bf16", {-1, -1} },      { T_Q8_0, "q8_0", {0, -1} },
    { T_Q4_0, "q4_0", {0, -1} },       { T_Q4_1, "q4_1", {0, 2} },
    { T_Q5_0, "q5_0", {0, -1} },       { T_Q5_1, "q5_1", {0, 2} },
    { T_Q2_K, "q2_K", {80, 82} },      { T_Q3_K, "q3_K", {108, -1} },
    { T_Q4_K, "q4_K", {0, 2} },        { T_Q5_K, "q5_K", {0, 2} },
    { T_Q6_K, "q6_K", {208, -1} },     { T_IQ4_NL, "iq4_nl", {0, -1} },
    { T_IQ4_XS, "iq4_xs", {0, -1} },   { T_IQ2_XXS, "iq2_xxs", {0, -1} },
    { T_IQ2_XS, "iq2_xs", {0, -1} },   { T_IQ2_S, "iq2_s", {0, -1} },
    { T_IQ3_XXS, "iq3_xxs", {0, -1} }, { T_IQ3_S, "iq3_s", {0, -1} },
    { T_IQ1_S, "iq1_s", {0, -1} },     { T_IQ1_M, "iq1_m", {-1, -1} },
    { T_MXFP4, "mxfp4", {-1, -1} },    { T_NVFP4, "nvfp4", {-1, -1} },
    { T_Q2_0, "q2_0", {0, -1} },
};

// n_out rows of n_in random weights of the type
static uint8_t *make_rows(int type, const int *f16, int n_in, int n_out, uint64_t *nbytes) {
    size_t rs = ggml_row_size(type, n_in);
    *nbytes = rs * (size_t)n_out;
    uint8_t *w = malloc(*nbytes);
    if (type == T_F32 || type == T_F16 || type == T_BF16) {
        for (size_t e = 0; e < (size_t)n_in * n_out; e++) {
            float v = frnd();
            if (type == T_F32) memcpy(w + 4 * e, &v, 4);
            else if (type == T_F16) put16(w + 2 * e, f32_to_f16(v));
            else { uint32_t u; memcpy(&u, &v, 4); put16(w + 2 * e, (uint16_t)(u >> 16)); }
        }
        return w;
    }
    for (size_t b = 0; b < *nbytes; b++) w[b] = (uint8_t)rnd32();
    size_t ts = ggml_type_size(type), nb = *nbytes / ts;
    for (size_t k = 0; k < nb; k++) {
        uint8_t *blk = w + k * ts;
        for (int f = 0; f < 2; f++)
            if (f16[f] >= 0) put16(blk + f16[f], rnd_scale());
        if (type == T_IQ1_M) {
            // the block's fp16 scale lives in the top nibbles of the four
            // 16-bit scale words at byte 48
            uint16_t h = rnd_scale();
            for (int s = 0; s < 4; s++) {
                uint16_t sw = get16(blk + 48 + 2 * s);
                sw = (uint16_t)((sw & 0x0FFF) | (((h >> (4 * s)) & 0xF) << 12));
                put16(blk + 48 + 2 * s, sw);
            }
        }
        if (type == T_MXFP4) blk[0] = (uint8_t)(118 + rnd32() % 16);   // 2^-9 .. 2^6
    }
    return w;
}

// the CPU trainer's chain (model.c mvt_worker)
static void cpu_mvt(const gguf_tensor *w, const float *dy, float *dx,
                    int n_in, int n_out, int batch) {
    float *row = malloc(sizeof(float) * (size_t)n_in);
    size_t rs = ggml_row_size(w->type, n_in);
    for (int t = 0; t < batch; t++)
        for (int j = 0; j < n_out; j++) {
            float d = dy[(size_t)t * n_out + j];
            if (d == 0.0f) continue;
            dequant_row(w->type, (const uint8_t *)w->data + (size_t)j * rs, row, n_in);
            for (int i = 0; i < n_in; i++)
                dx[(size_t)t * n_in + i] = fmaf(row[i], d, dx[(size_t)t * n_in + i]);
        }
    free(row);
}

static double clk(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// R8.21: the DeltaNet recurrence on the device (gpu_train_dn_fw / _bw)
// against a double-precision reference of the same equations (model.c
// dn_step and dn_bw_worker), at five head widths (the fixtures' 8, a width
// that is no whole number of warps' columns, and the real models' 128):
//   forward   y within 2^-14 of the reference, relative to the largest |y|;
//             the forward's rounds (the state kept on the device between
//             calls of 16 and 5 positions, from a reset) give the
//             recompute's y bit for bit, and the state brought back to the
//             host after them is the reference's final one within 2^-14
//   backward  dq, dk, dv, dbeta and ddecay within 2^-12 of the reference's,
//             relative to each one's largest magnitude, and bit for bit the
//             same on a second run
//   refusals  a sweep without its recompute, a head width that is no
//             multiple of TDN_ROWS or wider than TDN_MAX_HV, and hv != sk
//             are declined
static double dn_maxabs(const double *a, size_t n) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) mx = fmax(mx, fabs(a[i]));
    return mx;
}

static void dn_close(const char *what, int hv, const float *got,
                     const double *ref, size_t n, double tol) {
    double mx = dn_maxabs(ref, n), worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - ref[i]);
        if (e > worst) { worst = e; at = i; }
    }
    CHECK(mx > 0 && worst <= tol * mx, "dn hv %d: %s off by %.3g of max %.3g at "
          "[%zu] (device %.9g reference %.9g)", hv, what, worst, mx, at,
          (double)got[at], ref[at]);
}

static void dn_check(model_t *m, int nh, int ng, int hv, int T) {
    int sk = hv, keydim = sk * ng, inner = nh * hv, convdim = 2 * keydim + inner;
    int C = 1;
    while (C * C < T) C++;
    tdn_args a = { T, C, nh, ng, sk, hv, convdim, keydim, inner };
    size_t hh = (size_t)hv * hv, ncv = (size_t)T * convdim, ng_ = (size_t)T * nh;
    size_t ny = (size_t)T * inner, nqk = (size_t)nh * T * sk;
    float *cv = malloc(sizeof(float) * ncv), *beta = malloc(sizeof(float) * ng_);
    float *decay = malloc(sizeof(float) * ng_), *dy = malloc(sizeof(float) * ny);
    float *y = malloc(sizeof(float) * ny), *yr = malloc(sizeof(float) * ny);
    float *st = calloc((size_t)nh * hh, sizeof(float));
    float *out[5], *out2[5];
    size_t on[5] = { nqk, nqk, ny, ng_, ng_ };
    for (int i = 0; i < 5; i++) {
        out[i] = malloc(sizeof(float) * on[i]);
        out2[i] = malloc(sizeof(float) * on[i]);
    }
    for (size_t i = 0; i < ncv; i++) cv[i] = frnd();
    for (int t = 0; t < T; t++)   // q and k L2-normed per group, as the CPU leaves them
        for (int g = 0; g < 2 * ng; g++) {
            float *x = cv + (size_t)t * convdim + (size_t)g * sk, ss = 1e-6f;
            for (int i = 0; i < sk; i++) ss += x[i] * x[i];
            for (int i = 0; i < sk; i++) x[i] /= sqrtf(ss);
        }
    for (size_t i = 0; i < ng_; i++) {
        beta[i] = 0.55f + 0.35f * frnd();
        decay[i] = 0.9f + 0.099f * frnd();
    }
    for (size_t i = 0; i < ny; i++) dy[i] = frnd();

    // the reference: every state kept (T is small)
    double *S = calloc((size_t)nh * (T + 1) * hh, sizeof(double));
    double *ry = calloc(ny, sizeof(double)), *rq = calloc(nqk, sizeof(double));
    double *rk = calloc(nqk, sizeof(double)), *rv = calloc(ny, sizeof(double));
    double *rb = calloc(ng_, sizeof(double)), *rd = calloc(ng_, sizeof(double));
    double *dS = malloc(sizeof(double) * hh);
    double sq = sqrt((double)sk);
    for (int h = 0; h < nh; h++) {
        int g = h % ng;
        double *Sh = S + (size_t)h * (T + 1) * hh;   // slot t + 1: after step t
        for (int t = 0; t < T; t++) {
            const float *c = cv + (size_t)t * convdim;
            const float *q = c + g * sk, *k = c + keydim + g * sk, *v = c + 2 * keydim + h * hv;
            double dec = decay[(size_t)t * nh + h], bet = beta[(size_t)t * nh + h];
            double *Sp = Sh + (size_t)t * hh, *Sn = Sp + hh;
            for (int j = 0; j < hv; j++) {
                double pred = 0;
                for (int i = 0; i < hv; i++) pred += Sp[j * hv + i] * dec * k[i];
                double delta = (v[j] - pred) * bet, acc = 0;
                for (int i = 0; i < hv; i++) {
                    Sn[j * hv + i] = Sp[j * hv + i] * dec + delta * k[i];
                    acc += Sn[j * hv + i] * q[i];
                }
                ry[(size_t)t * inner + h * hv + j] = acc / sq;
            }
        }
        memset(dS, 0, sizeof(double) * hh);
        for (int t = T - 1; t >= 0; t--) {
            const float *c = cv + (size_t)t * convdim;
            const float *q = c + g * sk, *k = c + keydim + g * sk, *v = c + 2 * keydim + h * hv;
            double dec = decay[(size_t)t * nh + h], bet = beta[(size_t)t * nh + h];
            const double *Sp = Sh + (size_t)t * hh, *St = Sp + hh;
            double *dq = rq + ((size_t)h * T + t) * sk, *dk = rk + ((size_t)h * T + t) * sk;
            double db = 0, dd_ = 0;
            for (int j = 0; j < hv; j++) {
                double gj = dy[(size_t)t * inner + h * hv + j] / sq;
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
                rv[(size_t)t * inner + h * hv + j] = bet * dd;
                db += dd * (v[j] - pred);
                double dpred = -bet * dd;
                for (int i = 0; i < hv; i++) {
                    double dsd = dS[j * hv + i] + dpred * k[i];
                    dk[i] += dS[j * hv + i] * delta + Sp[j * hv + i] * dec * dpred;
                    dd_ += dsd * Sp[j * hv + i];
                    dS[j * hv + i] = dec * dsd;
                }
            }
            rb[(size_t)t * nh + h] = db;
            rd[(size_t)t * nh + h] = dd_;
        }
    }

    // the forward's rounds: 16 positions a call, the last call 5 or fewer,
    // from a reset (st is zero, as model_recurrent_reset leaves the host's)
    bool ok = true;
    gpu_train_dn_reset(m);
    for (int t0 = 0; ok && t0 < T; ) {
        int n = T - t0 > 16 ? 16 : T - t0 > 5 ? T - t0 - 5 : T - t0;
        tdn_args r = a;
        r.T = n;
        ok = gpu_train_dn_fw(m, &r, cv + (size_t)t0 * convdim, beta + (size_t)t0 * nh,
                             decay + (size_t)t0 * nh, st, yr + (size_t)t0 * inner);
        t0 += n;
    }
    CHECK(ok, "dn hv %d: a forward round declined", hv);
    CHECK(ok && gpu_train_dn_pull(m), "dn hv %d: the state did not come back", hv);
    if (ok) {
        double *fin = malloc(sizeof(double) * (size_t)nh * hh);
        for (int h = 0; h < nh; h++)
            memcpy(fin + (size_t)h * hh, S + ((size_t)h * (T + 1) + T) * hh,
                   sizeof(double) * hh);
        dn_close("final state", hv, st, fin, (size_t)nh * hh, 1.0 / 16384);
        free(fin);
    }
    CHECK(!gpu_train_dn_bw(m, &a, cv, dy, out[0], out[1], out[2], out[3], out[4]),
          "dn hv %d: a sweep ran without its recompute", hv);
    ok = ok && gpu_train_dn_fw(m, &a, cv, beta, decay, NULL, y);
    CHECK(ok, "dn hv %d: the recompute declined", hv);
    if (ok) {
        dn_close("y", hv, y, ry, ny, 1.0 / 16384);
        CHECK(!memcmp(y, yr, sizeof(float) * ny), "dn hv %d: the forward's rounds "
              "and the recompute differ", hv);
        float *junk = malloc(sizeof(float) * ncv);
        memcpy(junk, cv, sizeof(float) * ncv);
        CHECK(!gpu_train_dn_bw(m, &a, junk, dy, out[0], out[1], out[2], out[3], out[4]),
              "dn hv %d: a sweep ran on another recompute's states", hv);
        free(junk);
    }
    for (int run = 0; ok && run < 2; run++) {
        float **o = run ? out2 : out;
        ok = (!run || gpu_train_dn_fw(m, &a, cv, beta, decay, NULL, y)) &&
             gpu_train_dn_bw(m, &a, cv, dy, o[0], o[1], o[2], o[3], o[4]);
        CHECK(ok, "dn hv %d: the sweep declined (run %d)", hv, run);
    }
    if (ok) {
        static const char *const NAMES[5] = { "dq", "dk", "dv", "dbeta", "ddecay" };
        const double *refs[5] = { rq, rk, rv, rb, rd };
        for (int i = 0; i < 5; i++) {
            dn_close(NAMES[i], hv, out[i], refs[i], on[i], 1.0 / 4096);
            CHECK(!memcmp(out[i], out2[i], sizeof(float) * on[i]),
                  "dn hv %d: %s differs between two runs", hv, NAMES[i]);
        }
    }
    tdn_args bad = a;
    bad.hv = bad.sk = 12;
    CHECK(!gpu_train_dn_ok(m, &bad), "dn: hv 12 accepted");
    bad.hv = bad.sk = TDN_MAX_HV + TDN_ROWS;
    CHECK(!gpu_train_dn_ok(m, &bad), "dn: hv %d accepted", bad.hv);
    bad = a;
    bad.sk = hv / 2;
    CHECK(!gpu_train_dn_ok(m, &bad), "dn: hv != sk accepted");
    printf("dn hv %3d nh %d ng %d T %d: %s\n", hv, nh, ng, T, ok ? "checked" : "declined");
    free(cv); free(beta); free(decay); free(dy); free(y); free(yr); free(st);
    for (int i = 0; i < 5; i++) { free(out[i]); free(out2[i]); }
    free(S); free(ry); free(rq); free(rk); free(rv); free(rb); free(rd); free(dS);
}

int main(void) {
    f16_init();
    model_t m;
    memset(&m, 0, sizeof(m));
    m.train_gpu_budget = (uint64_t)1 << 30;   // no model file: the arena is this
    if (!gpu_train_init(&m)) {
        printf("train-gpu: no CUDA device, skipping (not passing)\n");
        return 0;
    }
    // 77 positions: two of k_tmvt's 64-position tiles and three of k_tmmw's
    // 32-column tiles, each with a remainder
    enum { N_IN = 512, N_OUT = 100, B = 77 };   // 100 rows: a partial row block
    float *dy = malloc(sizeof(float) * B * N_OUT);
    float *dx0 = malloc(sizeof(float) * B * N_IN);
    float *dxc = malloc(sizeof(float) * B * N_IN), *dxg = malloc(sizeof(float) * B * N_IN);
    float *x = malloc(sizeof(float) * B * N_IN);
    float *y = malloc(sizeof(float) * B * N_OUT), *y5 = malloc(sizeof(float) * 5 * N_OUT);
    float *y1 = malloc(sizeof(float) * N_OUT), *row = malloc(sizeof(float) * N_IN);
    for (int i = 0; i < B * N_OUT; i++) dy[i] = i % 7 == 0 ? 0.0f : frnd();  // the skip
    for (int i = 0; i < B * N_IN; i++) { dx0[i] = frnd(); x[i] = frnd(); }
    int tested = 0;
    // every type's rows stay allocated until the context is freed: the
    // context knows a tensor by its host address and size (weights live for
    // the run in a mapped file), so a freed buffer the next type's malloc
    // got back would be served the previous type's device copy
    void *keep[sizeof(TYPES) / sizeof(*TYPES) + 1] = {0};
    for (size_t k = 0; k < sizeof(TYPES) / sizeof(*TYPES); k++) {
        gguf_tensor w;
        memset(&w, 0, sizeof(w));
        snprintf(w.name, sizeof w.name, "%s", TYPES[k].name);
        w.type = (uint32_t)TYPES[k].type;
        w.ne[0] = N_IN; w.ne[1] = N_OUT;
        w.scale = 1.0f;
        w.data = make_rows(TYPES[k].type, TYPES[k].f16, N_IN, N_OUT, &w.nbytes);
        // transposed: the CPU chain's bytes
        memcpy(dxc, dx0, sizeof(float) * B * N_IN);
        memcpy(dxg, dx0, sizeof(float) * B * N_IN);
        cpu_mvt(&w, dy, dxc, N_IN, N_OUT, B);
        double t0 = clk();
        bool ok = gpu_train_mvt(&m, &w, dy, dxg, N_IN, N_OUT, B);
        double t1 = clk();
        CHECK(ok, "%s: gpu_train_mvt declined", TYPES[k].name);
        if (ok) {
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < (size_t)B * N_IN; i++)
                if (memcmp(&dxc[i], &dxg[i], 4) && !bad++) first = i;
            CHECK(!bad, "%s: transposed differs from the CPU chain at %zu of %d "
                  "elements (first [%zu]: cpu %.9g gpu %.9g)", TYPES[k].name, bad,
                  B * N_IN, first, dxc[first], dxg[first]);
        }
        // forward: batch-invariant columns, and the product
        ok = gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, B);
        double t2 = clk();
        CHECK(ok, "%s: gpu_train_mm declined", TYPES[k].name);
        bool ok5 = ok && gpu_train_mm(&m, &w, x + 3 * N_IN, N_IN, y5, N_OUT, N_IN,
                                      N_OUT, NULL, 5);
        CHECK(!ok || ok5, "%s: gpu_train_mm declined 5 columns", TYPES[k].name);
        if (ok5) {
            CHECK(!memcmp(y5, y + 3 * N_OUT, sizeof(float) * 5 * N_OUT),
                  "%s: columns 3..7 differ between a %d- and a 5-column call",
                  TYPES[k].name, B);
            double worst = 0;
            for (int c = 0; c < B; c++) {
                bool one = gpu_train_mm(&m, &w, x + (size_t)c * N_IN, N_IN, y1, N_OUT,
                                        N_IN, N_OUT, NULL, 1);
                CHECK(one && !memcmp(y1, y + (size_t)c * N_OUT, sizeof(float) * N_OUT),
                      "%s: column %d alone differs from the %d-column call",
                      TYPES[k].name, c, B);
                for (int j = 0; j < N_OUT; j++) {
                    dequant_row(w.type, (const uint8_t *)w.data +
                                (size_t)j * ggml_row_size(w.type, N_IN), row, N_IN);
                    double s = 0, a = 0;
                    for (int i = 0; i < N_IN; i++) {
                        s += (double)row[i] * x[(size_t)c * N_IN + i];
                        a += fabs((double)row[i] * x[(size_t)c * N_IN + i]);
                    }
                    double e = a > 0 ? fabs(y[(size_t)c * N_OUT + j] - s) / a : 0;
                    if (e > worst) worst = e;
                }
            }
            CHECK(worst < 1.0 / 65536, "%s: forward is not the product (relative "
                  "error %.3g of sum|w x|)", TYPES[k].name, worst);
            printf("train-gpu %-8s transposed bytes == CPU, forward invariant, "
                   "err %.2g; %.2f / %.2f ms\n", TYPES[k].name, worst,
                   (t1 - t0) * 1e3, (t2 - t1) * 1e3);
        }
        keep[k] = w.data;
        tested++;
    }
    // a bias rides on the host after the product; a strided x and y move
    // only their own rows
    {
        gguf_tensor w;
        memset(&w, 0, sizeof(w));
        w.type = T_Q8_0; w.scale = 1.0f; w.ne[0] = N_IN; w.ne[1] = N_OUT;
        int f16[2] = {0, -1};
        w.data = make_rows(T_Q8_0, f16, N_IN, N_OUT, &w.nbytes);
        float bias[N_OUT];
        for (int j = 0; j < N_OUT; j++) bias[j] = frnd();
        enum { XS = N_IN + 8, YS = N_OUT + 5 };
        float *xs = calloc((size_t)3 * XS, sizeof(float)), *ys = malloc(sizeof(float) * 3 * YS);
        for (int c = 0; c < 3; c++) memcpy(xs + (size_t)c * XS, x + (size_t)c * N_IN, sizeof(float) * N_IN);
        for (int i = 0; i < 3 * YS; i++) ys[i] = -7.0f;
        bool ok = gpu_train_mm(&m, &w, xs, XS, ys, YS, N_IN, N_OUT, bias, 3) &&
                  gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, 3);
        CHECK(ok, "strided call declined");
        for (int c = 0; ok && c < 3; c++) {
            for (int j = 0; j < N_OUT; j++)
                CHECK(ys[(size_t)c * YS + j] == y[(size_t)c * N_OUT + j] + bias[j],
                      "strided column %d row %d", c, j);
            for (int j = N_OUT; j < YS; j++)
                CHECK(ys[(size_t)c * YS + j] == -7.0f, "stride gap %d,%d written", c, j);
        }
        // a scaled tensor is the CPU's
        w.scale = 2.0f;
        CHECK(!gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, 1),
              "a scaled tensor went to the device");
        free(xs); free(ys);
        keep[sizeof(TYPES) / sizeof(*TYPES)] = w.data;
    }
    dn_check(&m, 4, 2, 128, 77);
    dn_check(&m, 3, 3, 64, 40);
    dn_check(&m, 4, 2, 40, 33);
    dn_check(&m, 2, 1, 32, 9);
    dn_check(&m, 4, 2, 8, 26);
    gpu_train_free(&m);
    for (size_t k = 0; k < sizeof(keep) / sizeof(*keep); k++) free(keep[k]);
    printf("%s: %d types\n", g_fail ? "FAIL" : "PASS", tested);
    return g_fail;
}
