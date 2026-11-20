// R8.19: the training products on the device, every weight type the CPU
// trainer decodes, on random blocks (scale fields made finite).
//
//   transposed  gpu_train_mvt == the CPU trainer's chain (dequant_row, then
//               an ascending-j fmaf per element, a zero dy skipped), memcmp
//   forward     gpu_train_mm is batch-invariant: every column of a 77-column
//               call (k_tmmr, a 64-column tile and a partial one) equals the
//               same column computed alone and in a 5-column call (its
//               16-column twin k_tmmrn), memcmp; and it is the product (each
//               column within 2^-16 of sum|w x| of a double-precision
//               reference)
//
// RUNNER_TRAIN_GPU_BENCH=1 adds a kernel benchmark: four types at 4096 x
// 5120, 16, 64 and 512 columns, the register-blocked kernels against the
// per-warp ones (RUNNER_TRAIN_MM_KERNEL=warp), the kernels' time alone, best
// of five.
//
// R8.21 adds the DeltaNet recurrence's device calls (dn_check below).
//
// Skips (not passes) without a CUDA device. Prints each type's throughput.
#include "runner.h"
#include "dn_ref.h"
#include "attn_ref.h"
#include "gate_gpu.h"

#include <float.h>
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
// R8.24.1 B (the second reader's F2): a NaN or an infinity, by bits. A
// tolerance written e = |device - reference|, e > bound passes a NaN e (every
// compare with a NaN is false), and -ffast-math may assume it away besides,
// so every default-build bound and every double anchor checks its operands'
// bits first: an output that is not finite where the reference is fails.
static int b_nonfin(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (u & 0x7f800000u) == 0x7f800000u;
}
static int d_nonfin(double d) {
    uint64_t u;
    memcpy(&u, &d, 8);
    return (u & 0x7ff0000000000000ull) == 0x7ff0000000000000ull;
}

// RUNNER_TRAIN_MM_KERNEL for the next products ("" unsets it); the engine
// reads it at every call. MinGW's headers declare no setenv or unsetenv.
static void set_mm_kernel(const char *v) {
#ifdef _WIN32
    _putenv_s("RUNNER_TRAIN_MM_KERNEL", v);   // "" removes it
#else
    if (*v) setenv("RUNNER_TRAIN_MM_KERNEL", v, 1);
    else unsetenv("RUNNER_TRAIN_MM_KERNEL");
#endif
}

static void set_mvt_kernel(const char *v) {
#ifdef _WIN32
    _putenv_s("RUNNER_TRAIN_MVT_KERNEL", v);
#else
    if (*v) setenv("RUNNER_TRAIN_MVT_KERNEL", v, 1);
    else unsetenv("RUNNER_TRAIN_MVT_KERNEL");
#endif
}

// --train-gpu's transposed product (k_tmvtr / k_tmvtrn): each column of a
// B-column call equals it alone and in a 5-column call, memcmp, and it is
// dx0 + W^T dy within 2^-16 of |dx0| + sum|w dy| of a double reference.
static double mvt_dev_check(model_t *m, const gguf_tensor *w, const float *dy,
                            const float *dx0, int n_in, int n_out, int B,
                            const char *name) {
    size_t nx = (size_t)B * n_in;
    float *dxa = malloc(sizeof(float) * nx), *dx5 = malloc(sizeof(float) * 5 * n_in);
    float *dx1 = malloc(sizeof(float) * n_in), *row = malloc(sizeof(float) * n_in);
    double *ref = calloc(nx, sizeof(double)), *mag = calloc(nx, sizeof(double));
    double worst = 1;
    m->train_dev = true;
    memcpy(dxa, dx0, sizeof(float) * nx);
    memcpy(dx5, dx0 + (size_t)3 * n_in, sizeof(float) * 5 * n_in);
    bool ok = gpu_train_mvt(m, w, dy, dxa, n_in, n_out, B) &&
              gpu_train_mvt(m, w, dy + (size_t)3 * n_out, dx5, n_in, n_out, 5);
    CHECK(ok, "%s: device-mode transposed declined", name);
    if (ok) {
        CHECK(!memcmp(dx5, dxa + (size_t)3 * n_in, sizeof(float) * 5 * n_in),
              "%s: device transposed columns 3..7 differ between a %d- and a "
              "5-column call", name, B);
        for (int c = 0; c < B; c++) {
            memcpy(dx1, dx0 + (size_t)c * n_in, sizeof(float) * n_in);
            bool one = gpu_train_mvt(m, w, dy + (size_t)c * n_out, dx1, n_in, n_out, 1);
            CHECK(one && !memcmp(dx1, dxa + (size_t)c * n_in, sizeof(float) * n_in),
                  "%s: device transposed column %d alone differs", name, c);
        }
        for (int j = 0; j < n_out; j++) {
            dequant_row(w->type, (const uint8_t *)w->data +
                        (size_t)j * ggml_row_size(w->type, n_in), row, n_in);
            for (int c = 0; c < B; c++) {
                double d = dy[(size_t)c * n_out + j];
                for (int i = 0; i < n_in; i++) {
                    ref[(size_t)c * n_in + i] += (double)row[i] * d;
                    mag[(size_t)c * n_in + i] += fabs((double)row[i] * d);
                }
            }
        }
        worst = 0;
        for (size_t i = 0; i < nx; i++) {
            double r = ref[i] + dx0[i], a = mag[i] + fabs(dx0[i]);
            double e = a > 0 ? fabs(dxa[i] - r) / a : 0;
            if (e > worst) worst = e;
        }
        CHECK(worst < 1.0 / 65536, "%s: device transposed is not dx0 + W^T dy "
              "(relative error %.3g)", name, worst);
    }
    m->train_dev = false;
    free(dxa); free(dx5); free(dx1); free(row); free(ref); free(mag);
    return worst;
}

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
// against the double-precision reference of the same equations
// (tests/dn_ref.h, which tests/test_dn_recurrence.c holds the CPU's
// recurrence to as well), at five head widths (the fixtures' 8, a width
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
static void dn_close(const char *what, int hv, const float *got,
                     const double *ref, size_t n, double tol) {
    size_t at;
    double mx, worst = dn_ref_err(got, ref, n, &at, &mx);
    CHECK(worst <= tol * mx,   // an all-zero reference wants exact zeros
          "dn hv %d: %s off by %.3g of max %.3g at "
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
    dn_shape sh = dn_shape_of(nh, ng, hv, T);
    dn_ref_fill(&sh, 0x2545F4914F6CDD1Dull ^ (uint64_t)hv, cv, beta, decay, dy);
    dn_ref_out r;
    dn_ref_run(&sh, cv, beta, decay, dy, &r);

    // the forward's rounds: 16 positions a call, the last call 5 or fewer,
    // from a reset (st is zero, as model_recurrent_reset leaves the host's)
    bool ok = true;
    gpu_train_dn_reset(m);
    for (int t0 = 0; ok && t0 < T; ) {
        int n = T - t0 > 16 ? 16 : T - t0 > 5 ? T - t0 - 5 : T - t0;
        tdn_args chunk = a;
        chunk.T = n;
        ok = gpu_train_dn_fw(m, &chunk, cv + (size_t)t0 * convdim, beta + (size_t)t0 * nh,
                             decay + (size_t)t0 * nh, st, yr + (size_t)t0 * inner);
        t0 += n;
    }
    CHECK(ok, "dn hv %d: a forward round declined", hv);
    CHECK(ok && gpu_train_dn_pull(m), "dn hv %d: the state did not come back", hv);
    if (ok) {
        double *fin = malloc(sizeof(double) * (size_t)nh * hh);
        for (int h = 0; h < nh; h++)
            memcpy(fin + (size_t)h * hh, r.S + ((size_t)h * (T + 1) + T) * hh,
                   sizeof(double) * hh);
        dn_close("final state", hv, st, fin, (size_t)nh * hh, 1.0 / 16384);
        free(fin);
    }
    CHECK(!gpu_train_dn_bw(m, &a, cv, dy, out[0], out[1], out[2], out[3], out[4]),
          "dn hv %d: a sweep ran without its recompute", hv);
    ok = ok && gpu_train_dn_fw(m, &a, cv, beta, decay, NULL, y);
    CHECK(ok, "dn hv %d: the recompute declined", hv);
    if (ok) {
        dn_close("y", hv, y, r.y, ny, 1.0 / 16384);
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
        const double *refs[5] = { r.dq, r.dk, r.dv, r.dbeta, r.ddecay };
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
    dn_ref_free(&r);
}

// The attention backward on the device (gpu_train_attn_bw) against the
// double-precision reference (tests/attn_ref.h, the CPU test's too): dq, dk,
// dv within 2^-12 of each one's largest magnitude and bit for bit the same
// on a second run, at GQA, a sliding window, a window past one block of
// TAB_TB positions and head widths 64 and 256.
static void ab_check(model_t *m, int T, int H, int kv_mul, int hd, int swa) {
    ab_shape s = ab_shape_of(T, H, kv_mul, hd, swa);
    size_t row = (size_t)s.kv_dim * 2, nq = (size_t)T * s.q_dim, nk = (size_t)T * s.kv_dim;
    uint8_t *kc = malloc(row * T), *vc = malloc(row * T);
    float *q = malloc(sizeof(float) * nq), *dao = malloc(sizeof(float) * nq);
    float *o[2][3];
    size_t on[3] = { nq, nk, nk };
    for (int r = 0; r < 2; r++)
        for (int i = 0; i < 3; i++) o[r][i] = calloc(on[i], sizeof(float));
    double *ref[3];
    for (int i = 0; i < 3; i++) ref[i] = calloc(on[i], sizeof(double));
    ab_ref_fill(&s, 0x2545F4914F6CDD1Dull ^ (uint64_t)(T * 31 + hd), kc, vc, q, dao);
    float scale = 1.0f / sqrtf((float)hd);
    ab_ref_run(&s, scale, kc, vc, q, dao, ref[0], ref[1], ref[2]);
    tab_args a = { T, H, kv_mul, hd, s.q_dim, s.kv_dim, swa, row, row, scale };
    bool ok = true;
    for (int r = 0; ok && r < 2; r++)
        ok = gpu_train_attn_bw(m, &a, kc, vc, q, dao, o[r][0], o[r][1], o[r][2]);
    CHECK(ok, "attn hd %d: the device declined", hd);
    static const char *const NAMES[3] = { "dq", "dk", "dv" };
    for (int i = 0; ok && i < 3; i++) {
        double rel = ab_ref_rel(o[0][i], ref[i], on[i]);
        CHECK(rel <= 1.0 / 4096, "attn T %d hd %d swa %d: %s off by %.3g of its "
              "largest magnitude", T, hd, swa, NAMES[i], rel);
        CHECK(!memcmp(o[0][i], o[1][i], sizeof(float) * on[i]),
              "attn hd %d: %s differs between two runs", hd, NAMES[i]);
    }
    printf("attn T %d H %d kv_mul %d hd %d swa %d: %s\n", T, H, kv_mul, hd, swa,
           ok ? "checked" : "declined");
    for (int r = 0; r < 2; r++) for (int i = 0; i < 3; i++) free(o[r][i]);
    for (int i = 0; i < 3; i++) free(ref[i]);
    free(kc); free(vc); free(q); free(dao);
}

// R8.24.1 B1 (G0): the device twins of a gated SiLU and its adjoint
// (gpu_train_ffn_check: h = act(g)*u, dg = dh*u*act'(g), du = dh*act(g))
// against the host's own expressions (model_ffn_act, model_ffn_dg,
// model_ffn_du). A T3 build's host does exactly the written operations with
// pmath's exp, which the twins reproduce, so memcmp. The default build's host
// takes the C library's expf, may contract, and runs with subnormals flushed
// (-ffast-math links crtfastmath: FTZ and DAZ for the process), so it is held
// to a stated bound with the device's subnormal results read as zero. In both,
// the anchor: h and du within 4 ulps of the formula in double (the C
// library's double exp, the same 0 below -80); dg within the float formula's
// own conditioning, |dh u| (|g| + 2) 2^-22 (its 1 - sg cancels as g grows).
static int64_t f_ord(float f) {
    int32_t i;
    memcpy(&i, &f, 4);
    return i < 0 ? (int64_t)INT32_MIN - i : i;
}
static int64_t f_ulps(float a, float b) {
    int64_t d = f_ord(a) - f_ord(b);
    return d < 0 ? -d : d;
}
static void ffn_check(model_t *m) {
    // boundary gates: the -80 cutoff and its neighbours, the float exp's
    // overflow edge, subnormals, both zeros, the largest magnitudes kept
    // finite through the product
    const float GB[] = {
        -80.0f, nextafterf(-80.0f, -100.0f), nextafterf(-80.0f, 0.0f), -79.5f,
        -87.33f, -88.73f, -100.0f, -1e30f, -3.0e38f, -50.0f, -20.0f, -10.0f,
        -1.0f, -1e-3f, -1e-30f, -1e-40f, -0.0f, 0.0f, 1e-40f, 1e-30f, 1e-3f,
        0.5f, 1.0f, 2.0f, 10.0f, 15.0f, 16.6f, 20.0f, 30.0f, 50.0f, 80.0f,
        88.0f, 88.73f, 89.0f, 100.0f, 1e10f, 1e30f };
    const float UB[] = { 1.0f, -1.0f, 0.5f, 0.0f, -0.0f, 3.0f };
    const float DB[] = { 1.0f, -2.5f, 0.75f, 0.0f };
    int nb = (int)(sizeof(GB) / sizeof(*GB)), nu = (int)(sizeof(UB) / sizeof(*UB));
    enum { NR = 77777 };   // random elements: no multiple of a block
    int n = nb * nu + NR;
    float *g = malloc(sizeof(float) * n), *u = malloc(sizeof(float) * n);
    float *dh = malloc(sizeof(float) * n), *h = malloc(sizeof(float) * n);
    float *dg = malloc(sizeof(float) * n), *du = malloc(sizeof(float) * n);
    int k = 0;
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < nu; j++, k++) {
            g[k] = GB[i];
            u[k] = UB[j];
            dh[k] = DB[k % 4];
        }
    for (; k < n; k++) {
        // mostly the activations' working range, some far outside it
        g[k] = k % 17 == 0 ? frnd() * 1e6f : frnd() * 90.0f;
        u[k] = frnd() * 4.0f;
        dh[k] = frnd() * 4.0f;
    }
    bool ok = gpu_train_ffn_check(m, ACT_SILU, g, u, dh, h, dg, du, n);
    CHECK(ok, "ffn twin: declined (no kernel in this PTX?)");
    if (!ok) goto done;
    int64_t wh = 0, wdu = 0, ah = 0, adu = 0;
    double wdg = 0, adg = 0;   // dg against its conditioning bound, worst ratio
    size_t bad = 0, abad = 0, hbad = 0;
    for (int i = 0; i < n; i++) {
        float rh = model_ffn_act(ACT_SILU, g[i], u[i]);
        float rdg = model_ffn_dg(ACT_SILU, dh[i], u[i], g[i]);
        float rdu = model_ffn_du(ACT_SILU, dh[i], g[i]);
        float vh = h[i], vdg = dg[i], vdu = du[i];
#ifdef RUNNER_T3_BUILD
        if (memcmp(&rh, &vh, 4) || memcmp(&rdg, &vdg, 4) || memcmp(&rdu, &vdu, 4)) {
            if (!bad++)
                fprintf(stderr, "FAIL: ffn twin differs from the T3 host at [%d] g %.9g u %.9g dh "
                        "%.9g: h %.9g/%.9g dg %.9g/%.9g du %.9g/%.9g\n", i, g[i], u[i], dh[i],
                        vh, rh, vdg, rdg, vdu, rdu);
        }
#else
        // this process flushes subnormals, as the default build's runner does
        if (fabsf(vh) < FLT_MIN) vh = 0.0f;
        if (fabsf(vdg) < FLT_MIN) vdg = 0.0f;
        if (fabsf(vdu) < FLT_MIN) vdu = 0.0f;
#endif
        double gd = g[i], cond = fabs((double)dh[i] * u[i]) * (fabs(gd) + 2.0) * 0x1p-22 + 0x1p-126;
        int64_t e;
        if ((e = f_ulps(vh, rh)) > wh) wh = e;
        if ((e = f_ulps(vdu, rdu)) > wdu) wdu = e;
        double edg = fabs((double)vdg - (double)rdg) / cond;
        // non-finite by bits (a NaN edg passes every compare): equal bits pass, else it fails
        if (b_nonfin(vdg) || b_nonfin(rdg) || d_nonfin(edg)) edg = memcmp(&vdg, &rdg, 4) ? 1e300 : 0;
        if (edg > wdg) wdg = edg;
#ifndef RUNNER_T3_BUILD
        if ((f_ulps(vh, rh) > 4 || f_ulps(vdu, rdu) > 4 || edg > 1.0) && !hbad++)
            fprintf(stderr, "FAIL: ffn twin is off the default build's host at [%d] g %.9g u %.9g "
                    "dh %.9g: h %.9g/%.9g dg %.9g/%.9g du %.9g/%.9g\n", i, g[i], u[i], dh[i],
                    vh, rh, vdg, rdg, vdu, rdu);
#endif
        // the anchor: the formula in double, the C library's double exp
        double sgd = 1.0 / (1.0 + exp(-gd));
        float xh = g[i] < -80.0f ? 0.0f : (float)(gd / (1.0 + exp(-gd)) * u[i]);
        double xdg = g[i] < -80.0f ? 0.0 : (double)dh[i] * u[i] * (sgd * (1.0 + gd * (1.0 - sgd)));
        float xdu = g[i] < -80.0f ? 0.0f : (float)((double)dh[i] * (gd / (1.0 + exp(-gd))));
#ifndef RUNNER_T3_BUILD
        if (fabsf(xh) < FLT_MIN) xh = 0.0f;
        if (fabsf(xdu) < FLT_MIN) xdu = 0.0f;
#endif
        int64_t a1 = f_ulps(vh, xh), a3 = f_ulps(vdu, xdu);
        double a2 = fabs((double)vdg - xdg) / cond;
        if (b_nonfin(vdg) || d_nonfin(xdg) || d_nonfin(a2)) {
            float xf = (float)xdg;
            a2 = memcmp(&vdg, &xf, 4) ? 1e300 : 0;
        }
        if (a1 > ah) ah = a1;
        if (a2 > adg) adg = a2;
        if (a3 > adu) adu = a3;
        if ((a1 > 4 || a2 > 1.0 || a3 > 4) && !abad++)
            fprintf(stderr, "FAIL: ffn twin is not the formula at [%d] g %.9g u %.9g dh %.9g: "
                    "h %.9g (double %.9g) dg %.9g (%.9g) du %.9g (%.9g)\n", i, g[i], u[i], dh[i],
                    vh, xh, vdg, xdg, vdu, xdu);
    }
    CHECK(!bad, "ffn twin: %zu of %d elements differ from the T3 host's bits", bad, n);
    CHECK(!abad, "ffn twin: %zu of %d elements off the double formula (h %lld, du %lld ulps "
          "worst; dg %.3g of its bound)", abad, n, (long long)ah, (long long)adu, adg);
    CHECK(!hbad, "ffn twin: %zu of %d elements off the default build's host (h %lld, du %lld "
          "ulps worst, bound 4; dg %.3g of its bound)", hbad, n, (long long)wh, (long long)wdu, wdg);
    printf("ffn twin (SiLU) %d elements: host %s; worst against the host h %lld du %lld ulps, "
           "dg %.3g of its bound; against the double formula h %lld du %lld ulps, dg %.3g\n", n,
#ifdef RUNNER_T3_BUILD
           "T3 (memcmp)",
#else
           "default (bound, subnormals flushed)",
#endif
           (long long)wh, (long long)wdu, wdg, (long long)ah, (long long)adu, adg);
done:
    free(g); free(u); free(dh); free(h); free(dg); free(du);
}

// R8.24.1 B6 (G0): the device twins of one site's adapter (gpu_train_lora_check)
// against the host's own code (model_lora_site_ref: lora_apply, then the four
// lsb passes of lora_site_bw). Explicit fmaf chains and two products. In a T3
// build that is exactly what the host runs, so memcmp on all four outputs,
// subnormals included. In the default build -ffast-math lets gcc vectorize
// lora_apply's chain over i (a reduction it may reassociate), so the forward
// delta y is the written chain only in T3 and is held to the rounding bound
// there; the backward's passes are laid out as independent chains the
// compiler cannot reassociate, so ga, gb and dx stay memcmp (normal numbers:
// the default build's process flushes subnormals). y, ga, gb and dx start
// from random values (each chain runs from its stored value). Anchor: y
// against a double reference within its chains' rounding, (n_in + r + 2)
// 2^-23 of the sum of the terms' magnitudes.
// smode, the adapter's scale: 0 16/r (the recipe's alpha 16), 1 zero, 2 -16/r,
// 3 the cancellation case: s 3.2 (no power of two, so s acc rounds) and y0 the
// negated s acc of a first host pass, so y is the fused step's rounding residual
// (an unfused y += s acc gives 0 there: RUNNER_TLORA_MUTANT's catch; every other
// case's s is a power of two, or its s acc far below y0's ulp, and cannot see it)
static void lora_case(model_t *m, int r, int n_in, int n_out, int nb, int edge, int smode) {
    size_t na = (size_t)r * n_in, nbb = (size_t)n_out * r;
    size_t nx = (size_t)nb * n_in, ny = (size_t)nb * n_out;
    float *A = malloc(sizeof(float) * na), *B = malloc(sizeof(float) * nbb);
    float *x = malloc(sizeof(float) * nx), *dy = malloc(sizeof(float) * ny);
    float *y0 = malloc(sizeof(float) * ny), *ga0 = malloc(sizeof(float) * na);
    float *gb0 = malloc(sizeof(float) * nbb), *dx0 = malloc(sizeof(float) * nx);
    float *yh = malloc(sizeof(float) * ny), *gah = malloc(sizeof(float) * na);
    float *gbh = malloc(sizeof(float) * nbb), *dxh = malloc(sizeof(float) * nx);
    float *yd = malloc(sizeof(float) * ny), *gad = malloc(sizeof(float) * na);
    float *gbd = malloc(sizeof(float) * nbb), *dxd = malloc(sizeof(float) * nx);
    float sa = 1.0f / sqrtf((float)n_in);
    for (size_t i = 0; i < na; i++) { A[i] = frnd() * sa; ga0[i] = frnd() * 0.01f; }
    for (size_t i = 0; i < nbb; i++) { B[i] = frnd() * 0.1f; gb0[i] = frnd() * 0.01f; }
    for (size_t i = 0; i < nx; i++) { x[i] = frnd(); dx0[i] = frnd(); }
    for (size_t i = 0; i < ny; i++) { dy[i] = frnd(); y0[i] = frnd(); }
    if (edge) {
        // a zero row of x and of dy, both zeros, magnitudes far from 1
        // (products stay normal), and in T3 subnormals
        for (int i = 0; i < n_in; i++) x[i] = 0.0f;
        for (int j = 0; j < n_out; j++) dy[(size_t)(nb - 1) * n_out + j] = 0.0f;
        for (size_t i = 0; i < nx; i += 7) x[i] = i % 2 ? -0.0f : 0.0f;
        for (size_t i = 3; i < nx; i += 11) x[i] *= i % 3 ? 1e15f : 1e-15f;
        for (size_t i = 5; i < ny; i += 13) dy[i] *= i % 3 ? 1e12f : 1e-12f;
        for (size_t i = 1; i < na; i += 17) A[i] = i % 2 ? -0.0f : 0.0f;
#ifdef RUNNER_T3_BUILD
        for (size_t i = 2; i < nx; i += 19) x[i] = i % 2 ? 1e-40f : -1e-41f;
        for (size_t i = 4; i < ny; i += 23) dy[i] = 3e-39f;
#endif
    }
    memcpy(yh, y0, sizeof(float) * ny); memcpy(yd, y0, sizeof(float) * ny);
    memcpy(gah, ga0, sizeof(float) * na); memcpy(gad, ga0, sizeof(float) * na);
    memcpy(gbh, gb0, sizeof(float) * nbb); memcpy(gbd, gb0, sizeof(float) * nbb);
    memcpy(dxh, dx0, sizeof(float) * nx); memcpy(dxd, dx0, sizeof(float) * nx);
    float s = smode == 1 ? 0.0f : smode == 2 ? -16.0f / (float)r : smode == 3 ? 3.2f : 16.0f / (float)r;
    if (smode == 3) {   // y0 = -(s acc) of a first host pass: y is then the rounding residual
        float *y1 = calloc(ny, sizeof(float)), *ga1 = malloc(sizeof(float) * na);
        float *gb1 = malloc(sizeof(float) * nbb), *dx1 = malloc(sizeof(float) * nx);
        memcpy(ga1, ga0, sizeof(float) * na); memcpy(gb1, gb0, sizeof(float) * nbb);
        memcpy(dx1, dx0, sizeof(float) * nx);
        model_lora_site_ref(r, n_in, n_out, nb, s, A, B, x, dy, y1, ga1, gb1, dx1);
        for (size_t i = 0; i < ny; i++) y0[i] = -y1[i];
        memcpy(yh, y0, sizeof(float) * ny); memcpy(yd, y0, sizeof(float) * ny);
        free(y1); free(ga1); free(gb1); free(dx1);
    }
    bool okh = model_lora_site_ref(r, n_in, n_out, nb, s, A, B, x, dy, yh, gah, gbh, dxh);
    bool okd = gpu_train_lora_check(m, r, n_in, n_out, nb, s, A, B, x, dy, yd, gad, gbd, dxd);
    CHECK(okh, "lora twin r %d %dx%d nb %d: the host reference failed", r, n_in, n_out, nb);
    CHECK(okd, "lora twin r %d %dx%d nb %d: declined (no kernel in this PTX?)", r, n_in, n_out, nb);
    if (okh && okd) {
        const struct { const char *nm; const float *h, *d; size_t n; } C[4] = {
            { "y", yh, yd, ny }, { "ga", gah, gad, na }, { "gb", gbh, gbd, nbb }, { "dx", dxh, dxd, nx } };
#ifdef RUNNER_T3_BUILD
        int c0 = 0;
#else
        int c0 = 1;   // y: the bound below, against the host and the double
#endif
        for (int c = c0; c < 4; c++) {
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < C[c].n; i++)
                if (memcmp(&C[c].h[i], &C[c].d[i], 4) && !bad++) first = i;
            CHECK(!bad, "lora twin r %d %dx%d nb %d%s: %s differs from the host at %zu of %zu "
                  "(first [%zu]: host %.9g device %.9g)", r, n_in, n_out, nb, edge ? " edge" : "",
                  C[c].nm, bad, C[c].n, first, C[c].h[first], C[c].d[first]);
        }
        // the anchor: y in double, and (default build) the host's y under
        // the same bound
        double worst = 0, hworst = 0;
        double *tq = malloc(sizeof(double) * r), *ta = malloc(sizeof(double) * r);
        for (int b = 0; b < nb; b++) {
            for (int k = 0; k < r; k++) {
                tq[k] = ta[k] = 0;
                for (int i = 0; i < n_in; i++) {
                    tq[k] += (double)A[(size_t)k * n_in + i] * x[(size_t)b * n_in + i];
                    ta[k] += fabs((double)A[(size_t)k * n_in + i] * x[(size_t)b * n_in + i]);
                }
            }
            for (int j = 0; j < n_out; j++) {
                double v = y0[(size_t)b * n_out + j], mag = fabs(v);
                double d = 0, dm = 0;
                for (int k = 0; k < r; k++) {
                    d += (double)B[(size_t)j * r + k] * tq[k];
                    dm += fabs((double)B[(size_t)j * r + k]) * ta[k];
                }
                v += s * d;
                mag += fabs(s) * dm;
                double bound = (n_in + r + 2) * 0x1p-23 * mag + 0x1p-126;
                double e = fabs(yd[(size_t)b * n_out + j] - v) / bound;
                double eh = fabs((double)yd[(size_t)b * n_out + j] - yh[(size_t)b * n_out + j]) /
                            bound;
                // by bits: the inputs are finite, so a non-finite y (or a NaN e) fails
                if (b_nonfin(yd[(size_t)b * n_out + j]) || d_nonfin(e)) e = 1e300;
                if (b_nonfin(yd[(size_t)b * n_out + j]) || d_nonfin(eh)) eh = 1e300;
                if (e > worst) worst = e;
                if (eh > hworst) hworst = eh;
            }
        }
        CHECK(worst <= 1.0, "lora twin r %d %dx%d nb %d: y is not the delta (%.3g of the "
              "rounding bound)", r, n_in, n_out, nb, worst);
        CHECK(hworst <= 1.0, "lora twin r %d %dx%d nb %d: y is off the host's beyond the "
              "rounding bound (%.3g of it)", r, n_in, n_out, nb, hworst);
        printf("lora twin r %3d %5dx%-5d nb %3d s %-7.4g%s: %s == host; y %.3g of its bound from the "
               "double, %.3g from the host\n", r, n_in, n_out, nb, s, edge ? " edge" : "     ",
               c0 ? "ga gb dx" : "y ga gb dx", worst, hworst);
        free(tq); free(ta);
    }
    free(A); free(B); free(x); free(dy); free(y0); free(ga0); free(gb0); free(dx0);
    free(yh); free(gah); free(gbh); free(dxh); free(yd); free(gad); free(gbd); free(dxd);
}

// R8.24.1 B2 (G0): the device twins of rmsnorm and its adjoint
// (gpu_train_rmsnorm_check: k_trn_fw, k_trn_bw) against the host's own
// functions (model_rmsnorm, model_rmsnorm_bw: model.c rmsnorm, rmsnorm_bw).
// Written before the kernels. A T3 build's host does exactly the written
// operations (the forward's ss a multiply and an add, the adjoint's ss and dot
// fmaf chains, sqrtf correctly rounded), which the twins reproduce, so memcmp,
// except that a NaN compares as a NaN: the device returns the canonical NaN
// where x86 keeps the input's payload, and the host's answer there is "NaN".
// The default build's host runs under -ffast-math: its ss loop may be
// vectorized (reassociated) and the process runs with FTZ and DAZ
// (crtfastmath). So there the bound is stated: every element within 2^-12 of
// its row's largest magnitude, the device's subnormal results read as zero,
// and the subnormal-input row is left to T3 (DAZ reads its inputs as zero on
// the host). In both builds the anchor: the forward and the adjoint within
// (n 2^-24 + 2^-20) of their row's magnitude of the row's formula in double:
// the serial float chain's own error bound (gamma_n), which the host shares
// (rows whose float ss overflows, or that hold the NaN, excluded).
enum { B2_KINDS = 8, B2_SUBNORMAL = 3, B2_NAN = 6, B2_HUGE = 5 };
static void b2_fill_row(float *x, int n, int kind) {
    for (int i = 0; i < n; i++) {
        float v = frnd();
        switch (kind) {
        case 1: v = 0.0f; break;                                        // a zero row
        case 2: v = 0.37109375f; break;                                 // equal values
        case B2_SUBNORMAL: v = v * 1.0e-39f; break;
        case 4: v = (i & 1) ? -0.0f : 0.0f; break;                      // both zeros
        case B2_HUGE: v = v * 1.8e19f; break;                           // x*x near FLT_MAX
        case B2_NAN: if (i == n / 2) { uint32_t q = 0x7fc00001u; memcpy(&v, &q, 4); } break;
        case 7: v = v * ((i % 5 == 0) ? 3.0e4f : 1.0e-3f); break;       // mixed scale
        default: break;
        }
        x[i] = v;
    }
}
static int b2_nan(float f) {   // by bits: -ffast-math may fold isnan away
    uint32_t u;
    memcpy(&u, &f, 4);
    return (u & 0x7fffffffu) > 0x7f800000u;
}
// a subnormal read as zero, as an FTZ process reads it; by bits, so a NaN
// stays a NaN under -ffast-math (a float compare there need not)
#ifndef RUNNER_T3_BUILD
static float b2_ftz(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0) u &= 0x80000000u;
    memcpy(&f, &u, 4);
    return f;
}
#endif
// R8.24.1 B1 (the second reader's gap): non-finite inputs. Every combination
// of g, u and dh over the infinities, NaNs of three payloads and a few finite
// values goes through the twins and the host's own functions. A NaN result
// must meet a NaN (the CPU and the GPU propagate payloads by their own rules);
// any other result must be the host's bits. T3 is the gate: the default
// build's host is compiled with -ffast-math, which assumes no infinity or NaN,
// so its answers there are undefined (measured: dh u act_d(g) at g -inf, u and
// dh +inf is IEEE's inf 0 = NaN on the device and in T3, and 0 in the default
// host, whose compiler folded the product); that build runs the twins over
// the same inputs and reports how many outputs differ, without a verdict.
static void ffn_nonfinite_check(model_t *m) {
    const uint32_t NB[] = { 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7fa00010u };
    const float FV[] = { 1.0f, -2.0f, 0.0f, 90.0f };
    float S[8];
    int ns = 0;
    for (int i = 0; i < 5; i++) memcpy(&S[ns++], &NB[i], 4);
    for (int i = 0; i < 3; i++) S[ns++] = FV[i];
    int n = ns * ns * ns;
    float *g = malloc(sizeof(float) * n), *u = malloc(sizeof(float) * n), *dh = malloc(sizeof(float) * n);
    float *h = malloc(sizeof(float) * n), *dg = malloc(sizeof(float) * n), *du = malloc(sizeof(float) * n);
    int k = 0;
    for (int a = 0; a < ns; a++)
        for (int b = 0; b < ns; b++)
            for (int c = 0; c < ns; c++, k++) { g[k] = S[a]; u[k] = S[b]; dh[k] = S[c]; }
    bool ok = gpu_train_ffn_check(m, ACT_SILU, g, u, dh, h, dg, du, n);
    CHECK(ok, "ffn twin, non-finite inputs: declined");
    size_t bad = 0, nonfin_in = 0;
    for (int i = 0; ok && i < n; i++) {
        float r[3] = { model_ffn_act(ACT_SILU, g[i], u[i]), model_ffn_dg(ACT_SILU, dh[i], u[i], g[i]),
                       model_ffn_du(ACT_SILU, dh[i], g[i]) };
        float d[3] = { h[i], dg[i], du[i] };
        if (b_nonfin(g[i]) || b_nonfin(u[i]) || b_nonfin(dh[i])) nonfin_in++;
        for (int j = 0; j < 3; j++) {
            bool same;
            if (b2_nan(r[j]) || b2_nan(d[j])) same = b2_nan(r[j]) && b2_nan(d[j]);
            else same = !memcmp(&r[j], &d[j], 4);
#ifndef RUNNER_T3_BUILD
            if (!same) bad++;   // reported, not gated: see above
            continue;
#endif
            if (!same && !bad++) {
                uint32_t bg, bu, bd, br, bdv;
                memcpy(&bg, &g[i], 4); memcpy(&bu, &u[i], 4); memcpy(&bd, &dh[i], 4);
                memcpy(&br, &r[j], 4); memcpy(&bdv, &d[j], 4);
                fprintf(stderr, "FAIL: ffn twin, non-finite inputs: g %08x u %08x dh %08x: %s %08x, "
                        "the host %08x\n", (unsigned)bg, (unsigned)bu, (unsigned)bd,
                        j == 0 ? "h" : j == 1 ? "dg" : "du", (unsigned)bdv, (unsigned)br);
            }
        }
    }
#ifdef RUNNER_T3_BUILD
    CHECK(!bad, "ffn twin, non-finite inputs: %zu outputs differ from the host's by class or bits", bad);
    if (ok && !bad)
        printf("ffn twin, non-finite inputs: %d combinations (%zu with an infinity or a NaN in), h dg du "
               "== host by class (NaN meets NaN) and by bits otherwise\n", n, nonfin_in);
#else
    if (ok)
        printf("ffn twin, non-finite inputs: %d combinations (%zu with an infinity or a NaN in) ran; %zu "
               "of %d outputs differ from this -ffast-math host, whose answers there are undefined (T3 "
               "gates them)\n", n, nonfin_in, bad, 3 * n);
#endif
    free(g); free(u); free(dh); free(h); free(dg); free(du);
}

static void b2_check(model_t *m, int n, int rows, int stride, bool weighted) {
    float eps = 1e-6f;
    size_t span = (size_t)(rows - 1) * stride + n;
    float *x = calloc(span, sizeof(float)), *dy = calloc(span, sizeof(float));
    float *w = weighted ? malloc(sizeof(float) * n) : NULL;
    float *dx0 = calloc(span, sizeof(float));
    float *yh = calloc(span, sizeof(float)), *yd = calloc(span, sizeof(float));
    float *dxh = calloc(span, sizeof(float)), *dxd = calloc(span, sizeof(float));
    if (w) for (int i = 0; i < n; i++) w[i] = 1.0f + 0.5f * frnd();
    for (int r = 0; r < rows; r++) {
        b2_fill_row(x + (size_t)r * stride, n, r % B2_KINDS);
        b2_fill_row(dy + (size_t)r * stride, n, r % 3 == 0 ? 0 : 7);
        b2_fill_row(dx0 + (size_t)r * stride, n, 0);
    }
    memcpy(dxh, dx0, sizeof(float) * span);
    memcpy(dxd, dx0, sizeof(float) * span);
    for (int r = 0; r < rows; r++) {
        size_t o = (size_t)r * stride;
        model_rmsnorm(yh + o, x + o, w, n, eps);
        model_rmsnorm_bw(x + o, w, dy + o, dxh + o, n, eps);
    }
    trn_args a = { n, rows, stride, stride, stride, stride, eps };
    bool ok = gpu_train_rmsnorm_check(m, &a, x, w, dy, yd, dxd);
    CHECK(ok, "rmsnorm twin n %d: declined (no kernel in this PTX?)", n);
    size_t bad = 0, hbad = 0, abad = 0;
    double whf = 0, whb = 0, waf = 0, wab = 0;   // worst, as a fraction of the row's magnitude
    for (int r = 0; ok && r < rows; r++) {
        size_t o = (size_t)r * stride;
        int kind = r % B2_KINDS;
        double ss = 0, dot = 0;
        for (int i = 0; i < n; i++) ss += (double)x[o + i] * x[o + i];
        double rr = 1.0 / sqrt(ss / n + eps);
        for (int i = 0; i < n; i++) dot += (w ? w[i] : 1.0) * dy[o + i] * x[o + i];
        double c = rr * rr * rr / n * dot, mf = 0, mb = 0;
        for (int i = 0; i < n; i++) {
            double wi = w ? w[i] : 1.0;
            double f = fabs(x[o + i] * rr * wi);
            double b = fabs(dx0[o + i]) + fabs(rr * wi * dy[o + i]) + fabs(c * x[o + i]);
            if (f > mf) mf = f;
            if (b > mb) mb = b;
        }
        bool anchor = kind != B2_NAN && ss < 3.0e38;
        for (int i = 0; i < n; i++) {
            float hf = yh[o + i], df = yd[o + i], hb = dxh[o + i], db = dxd[o + i];
#ifdef RUNNER_T3_BUILD
            bool same_f = b2_nan(hf) || b2_nan(df) ? b2_nan(hf) && b2_nan(df) : !memcmp(&hf, &df, 4);
            bool same_b = b2_nan(hb) || b2_nan(db) ? b2_nan(hb) && b2_nan(db) : !memcmp(&hb, &db, 4);
            if ((!same_f || !same_b) && !bad++)
                fprintf(stderr, "FAIL: rmsnorm twin differs from the T3 host at n %d row %d [%d] "
                        "x %.9g: y %.9g/%.9g dx %.9g/%.9g\n", n, r, i, x[o + i], df, hf, db, hb);
#else
            // this process flushes subnormals (FTZ and DAZ), as the default build's runner does
            float vf = b2_ftz(df), vb = b2_ftz(db);
            if (kind != B2_SUBNORMAL) {
                double ef = b2_nan(hf) || b2_nan(vf) ? (b2_nan(hf) && b2_nan(vf) ? 0 : 1e300)
                                                     : fabs((double)hf - vf) / (mf + 1e-300);
                double eb = b2_nan(hb) || b2_nan(vb) ? (b2_nan(hb) && b2_nan(vb) ? 0 : 1e300)
                                                     : fabs((double)hb - vb) / (mb + 1e-300);
                if (ss < 3.0e38) {
                    if (ef > whf) whf = ef;
                    if (eb > whb) whb = eb;
                    if ((ef > 0x1p-12 || eb > 0x1p-12) && !hbad++)
                        fprintf(stderr, "FAIL: rmsnorm twin is off the default build's host at n %d "
                                "row %d [%d]: y %.9g/%.9g dx %.9g/%.9g\n", n, r, i, vf, hf, vb, hb);
                }
            }
#endif
            if (anchor && mf > 0) {
                double wi = w ? w[i] : 1.0;
                double xf = x[o + i] * rr * wi;
                double xb = dx0[o + i] + rr * wi * dy[o + i] - c * x[o + i];
                double af = fabs(df - xf) / mf, ab = fabs(db - xb) / mb;
                // by bits: an anchored row's formula is finite, so a non-finite output fails
                if (b_nonfin(df) || d_nonfin(af)) af = 1e300;
                if (b_nonfin(db) || d_nonfin(ab)) ab = 1e300;
                if (af > waf) waf = af;
                if (ab > wab) wab = ab;
                double lim = n * 0x1p-24 + 0x1p-20;
                if ((af > lim || ab > lim) && !abad++)
                    fprintf(stderr, "FAIL: rmsnorm twin is not the formula at n %d row %d [%d]: "
                            "y %.9g (double %.9g) dx %.9g (%.9g)\n", n, r, i, df, xf, db, xb);
            }
        }
    }
    CHECK(!bad, "rmsnorm twin n %d rows %d stride %d %s: %zu elements differ from the T3 host's bits",
          n, rows, stride, w ? "weighted" : "weightless", bad);
    CHECK(!hbad, "rmsnorm twin n %d rows %d: %zu elements off the default build's host (worst "
          "y %.3g, dx %.3g of the row's magnitude, bound 2^-12)", n, rows, hbad, whf, whb);
    CHECK(!abad, "rmsnorm twin n %d rows %d: %zu elements off the double formula (worst y %.3g, "
          "dx %.3g of the row's magnitude, bound n 2^-24 + 2^-20)", n, rows, abad, waf, wab);
    if (ok)
        printf("rmsnorm twin n %4d rows %4d stride %4d %-10s: host %s; against the double "
               "formula y %.3g dx %.3g of the row's magnitude\n", n, rows, stride,
               w ? "weighted" : "weightless",
#ifdef RUNNER_T3_BUILD
               "T3 (memcmp)",
#else
               "default (bound 2^-12, subnormals flushed)",
#endif
               waf, wab);
    free(x); free(dy); free(w); free(dx0); free(yh); free(yd); free(dxh); free(dxd);
}
static void b2_checks(model_t *m) {
    // the step's n (2560 the 4B's attn, ffn and out norms; 128 a head), the
    // block's and the warp's edges, one element, the widest families
    static const int NS[] = { 2560, 128, 1, 31, 32, 33, 127, 129, 255, 256, 257,
                              1023, 1024, 1025, 5120, 8192 };
    for (size_t k = 0; k < sizeof(NS) / sizeof(*NS); k++)
        b2_check(m, NS[k], 2 * B2_KINDS + 1, NS[k], true);
    b2_check(m, 2560, 77, 2560, true);           // attn and ffn norms, T 77
    b2_check(m, 128, 77 * 32, 128, true);        // q-norm per head, T 77 x 32 heads
    b2_check(m, 128, 77 * 8, 128, false);        // the weightless V-norm form
    b2_check(m, 128, 40, 4096, true);            // heads read out of q rows
    b2_check(m, 100, 33, 103, false);            // strides past n
}

// R8.24.1 B3 (G0): the device twin of rope and its adjoint (gpu_train_rope_check:
// k_trope) against the host's own functions (model_rope_ref: model.c
// rope_apply, rope_unapply) on a model that carries only the rope fields.
// Written before the kernel. A T3 build's host takes pmath's sinf and cosf,
// which the twin ports, so memcmp (a NaN compares as a NaN). The default
// build's host takes the C library's sinf and cosf and may contract, and its
// process flushes subnormals (FTZ and DAZ): there each element within 2^-20 of
// its pair's magnitude (|x0| + |x1|) |mscale|, the device's subnormal results
// read as zero and the subnormal-input row left to T3, and only for angles up
// to 1e5: past pmath's stated range it reduces by fmod against a rounded 2 pi,
// the C library exactly, so there the two are different functions and T3's
// memcmp is the gate. In both, the anchor: the rotation in double (the C
// library's double sin and cos of the host's float angle) within 2^-20 of the
// pair's magnitude, for angles up to 1e5; larger or NaN angles are compared
// only to the host.
enum { B3_KINDS = 7, B3_SUBNORMAL = 3 };
static void b3_fill_row(float *x, int n, int kind) {
    for (int i = 0; i < n; i++) {
        float v = frnd();
        switch (kind) {
        case 1: v = 0.0f; break;
        case 2: v = (i & 1) ? -0.0f : 0.0f; break;
        case B3_SUBNORMAL: v = v * 1.0e-39f; break;
        case 4: v = v * 1.0e30f; break;
        case 5: if (i == n / 3) { uint32_t q = 0x7fc00001u; memcpy(&v, &q, 4); } break;
        case 6: v = 0.6875f; break;
        default: break;
        }
        x[i] = v;
    }
}
static void b3_check(model_t *m, int hd, int rd, int neox, int n_heads, int T, int t0,
                     int pad, float ms, int adjoint, int odd_freqs) {
    int half = rd / 2, rowlen = n_heads * hd, stride = rowlen + pad;
    size_t span = (size_t)(T - 1) * stride + rowlen;
    float *fr = malloc(sizeof(float) * half);
    for (int j = 0; j < half; j++) fr[j] = powf(10000.0f, -2.0f * j / (float)rd);
    if (odd_freqs && half > 3) {      // the reduction's far edges: fmod, and NaN
        fr[half - 1] = 1.0e13f;
        {   // +inf by its bits: fast math assumes no infinity, and Apple clang
            // refuses the INFINITY macro there (-Werror=nan-infinity-disabled)
            const uint32_t inf_bits = 0x7f800000u;
            memcpy(&fr[half - 2], &inf_bits, 4);
        }
        fr[half - 3] = 1.0f;          // angle = position: up to 1.3e5
    }
    float *v0 = calloc(span, sizeof(float)), *vh = calloc(span, sizeof(float));
    float *vd = calloc(span, sizeof(float));
    for (int t = 0; t < T; t++) b3_fill_row(v0 + (size_t)t * stride, rowlen, t % B3_KINDS);
    memcpy(vh, v0, sizeof(float) * span);
    memcpy(vd, v0, sizeof(float) * span);
    model_t rm;
    memset(&rm, 0, sizeof(rm));
    rm.n_layer = 1;
    rm.head_dim = hd;
    rm.rope_dim = rd;
    rm.rope_neox = neox != 0;
    rm.rope_mscale = ms;
    rm.rope_inv_freq = fr;
    for (int t = 0; t < T; t++)
        model_rope_ref(&rm, vh + (size_t)t * stride, n_heads, t0 + t, 0, adjoint);
    trope_args a = { n_heads, hd, half, neox, T, t0, stride, adjoint, ms };
    bool ok = gpu_train_rope_check(m, &a, fr, vd);
    CHECK(ok, "rope twin hd %d: declined (no kernel in this PTX?)", hd);
    size_t bad = 0, hbad = 0, abad = 0;
    double wh = 0, wa = 0;
    for (int t = 0; ok && t < T; t++) {
        int kind = t % B3_KINDS;
        (void)kind;   // the default build's subnormal row
        for (int h = 0; h < n_heads; h++)
            for (int j = 0; j < half; j++) {
                size_t o = (size_t)t * stride + (size_t)h * hd;
                size_t i0 = o + (neox ? j : 2 * j), i1 = neox ? i0 + half : i0 + 1;
                float x0 = v0[i0], x1 = v0[i1];
                float h0 = vh[i0], h1 = vh[i1], d0 = vd[i0], d1 = vd[i1];
                double mag = (fabs((double)x0) + fabs((double)x1)) * fabs((double)ms) + 1e-37;
#ifdef RUNNER_T3_BUILD
                bool s0 = b2_nan(h0) || b2_nan(d0) ? b2_nan(h0) && b2_nan(d0) : !memcmp(&h0, &d0, 4);
                bool s1 = b2_nan(h1) || b2_nan(d1) ? b2_nan(h1) && b2_nan(d1) : !memcmp(&h1, &d1, 4);
                if ((!s0 || !s1) && !bad++)
                    fprintf(stderr, "FAIL: rope twin differs from the T3 host at hd %d t %d h %d j %d: "
                            "%.9g/%.9g %.9g/%.9g\n", hd, t, h, j, d0, h0, d1, h1);
#else
                float e0 = b2_ftz(d0), e1 = b2_ftz(d1);
                float ang_h = (float)(t0 + t) * fr[j];
                if (kind != B3_SUBNORMAL && !(fabsf(ang_h) > 1e5f)) {
                    double q = b2_nan(h0) || b2_nan(e0) || b2_nan(h1) || b2_nan(e1)
                        ? ((b2_nan(h0) == b2_nan(e0) && b2_nan(h1) == b2_nan(e1)) ? 0 : 1e300)
                        : fmax(fabs((double)h0 - e0), fabs((double)h1 - e1)) / mag;
                    if (q > wh && q < 1e300) wh = q;
                    if (q > 0x1p-20 && !hbad++)
                        fprintf(stderr, "FAIL: rope twin is off the default build's host at hd %d t %d "
                                "h %d j %d: %.9g/%.9g %.9g/%.9g\n", hd, t, h, j, e0, h0, e1, h1);
                }
#endif
                float ang = (float)(t0 + t) * fr[j];
                if (!b2_nan(x0) && !b2_nan(x1) && fabsf(ang) <= 1e5f) {
                    double c = cos((double)ang) * ms, s = sin((double)ang) * ms;
                    double r0 = adjoint ? x0 * c + x1 * s : x0 * c - x1 * s;
                    double r1 = adjoint ? -x0 * s + x1 * c : x0 * s + x1 * c;
                    double q = fmax(fabs(d0 - r0), fabs(d1 - r1)) / mag;
                    // by bits: fmax drops a NaN argument, and the rotation of finite inputs
                    // is finite, so a non-finite output fails
                    if (b_nonfin(d0) || b_nonfin(d1) || d_nonfin(q)) q = 1e300;
                    if (q > wa) wa = q;
                    if (q > 0x1p-20 && !abad++)
                        fprintf(stderr, "FAIL: rope twin is not the rotation at hd %d t %d h %d j %d "
                                "(angle %.9g): %.9g (%.9g) %.9g (%.9g)\n", hd, t, h, j, ang, d0,
                                r0, d1, r1);
                }
            }
        // the elements past the rope dims stay as they were
        for (int h = 0; h < n_heads && rd < hd; h++)
            for (int i = rd; i < hd; i++) {
                size_t o = (size_t)t * stride + (size_t)h * hd + i;
                if (memcmp(&vd[o], &v0[o], 4) && !(b2_nan(vd[o]) && b2_nan(v0[o])) && !bad++)
                    fprintf(stderr, "FAIL: rope twin touched element %d past rope_dim %d\n", i, rd);
            }
    }
    const char *what = adjoint ? "adjoint" : "forward";
    CHECK(!bad, "rope twin %s hd %d rd %d neox %d t0 %d: %zu pairs differ from the T3 host's bits",
          what, hd, rd, neox, t0, bad);
    CHECK(!hbad, "rope twin %s hd %d rd %d: %zu pairs off the default build's host (worst %.3g of "
          "the pair's magnitude, bound 2^-20)", what, hd, rd, hbad, wh);
    CHECK(!abad, "rope twin %s hd %d rd %d: %zu pairs off the double rotation (worst %.3g of the "
          "pair's magnitude, bound 2^-20)", what, hd, rd, abad, wa);
    if (ok)
        printf("rope twin %-7s hd %3d rd %3d neox %d heads %2d T %3d t0 %6d ms %.4g%s: host %s; "
               "against the double rotation %.3g of the pair's magnitude\n", what, hd, rd, neox,
               n_heads, T, t0, ms, odd_freqs ? " odd-freqs" : "",
#ifdef RUNNER_T3_BUILD
               "T3 (memcmp)",
#else
               "default (bound 2^-20, subnormals flushed)",
#endif
               wa);
    free(fr); free(v0); free(vh); free(vd);
}
static void b3_checks(model_t *m) {
    for (int adj = 0; adj < 2; adj++) {
        b3_check(m, 128, 128, 1, 32, 77, 0, 0, 1.0f, adj, 0);        // the recipe's q (Qwen3)
        b3_check(m, 128, 128, 1, 8, 77, 0, 0, 1.0f, adj, 0);         // its k
        b3_check(m, 128, 64, 0, 4, 33, 1000, 5, 1.0f, adj, 0);       // partial rope, adjacent pairs
        b3_check(m, 64, 64, 0, 3, 31, 32700, 0, 1.2345f, adj, 0);    // YaRN mscale, far positions
        b3_check(m, 256, 256, 1, 2, 17, 131000, 3, 1.0f, adj, 1);    // fmod / NaN / 1e5 angles
        b3_check(m, 128, 128, 1, 1, 128, 0, 0, 1.0f, adj, 0);        // T 128, one head
    }
}

// R8.24.1 B8 (G0): the device twin of the forward's K/V store
// (gpu_train_kv_store_check: k_tkv_store) against the host's own conversion,
// fp16.h's f32_to_f16 (what model.c's kind-0 store calls), bit for bit in
// both builds: the conversion is integers only. Every input is made from its
// bits, never by float arithmetic, which the default build flushes. Inputs:
// every half's float (each must come back as itself, a NaN quieted with its
// payload kept); every f32 exponent with each sign and the rounding's edges
// (zero, one, below, at and above the half-way tie with an even and with an
// odd result, the carry into the exponent, all ones) plus fixed-seed
// mantissas; and for every subnormal result's shift the ties and their
// neighbours. Anchors, IEEE 754 binary16 by hand: 1 -> 3C00, -2 -> C000,
// 65504 -> 7BFF, 65519.9921875 -> 7BFF, 65520 -> 7C00 (the tie goes to the
// even inf), 2^-14 -> 0400, 2^-24 -> 0001, 2^-25 -> 0000 (a tie to even),
// 1.5 * 2^-25 -> 0001, 2^-26 -> 0000, inf -> 7C00, the quiet NaN 7FC00000 ->
// 7E00, the signalling NaN 7F800001 -> 7E00, -0 -> 8000. The rows go at a
// source stride and a row width wider than the data: the bytes between rows
// must come back as they went. Kind 3 (f32) stores the floats as they are.
// RUNNER_TEST_KV_ALL=1 adds every one of the 2^32 inputs, 2^24 at a time.
static uint32_t b8_half_bits(uint16_t h) {   // the float of a half, by integers
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, mt = h & 0x3ffu;
    if (e == 31) return s | 0x7F800000u | (mt << 13);
    if (e) return s | ((e + 112u) << 23) | (mt << 13);
    if (!mt) return s;
    uint32_t k = 0;
    while (!(mt & 0x400u)) { mt <<= 1; k++; }
    return s | ((113u - k) << 23) | ((mt & 0x3ffu) << 13);
}
static float b8_f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
// kind 0 over n inputs at row layout (rows, n_row, src_stride, row_bytes);
// returns the inputs whose half differs from the host's
static size_t b8_store(model_t *m, const uint32_t *in, size_t n, int n_row, int pad_src,
                       int pad_row, const char *what, bool *ran) {
    int rows = (int)((n + (size_t)n_row - 1) / (size_t)n_row);
    int stride = n_row + pad_src;
    size_t row_bytes = (size_t)n_row * 2 + (size_t)pad_row;
    float *src = calloc((size_t)rows * stride, sizeof(float));
    size_t nd = (size_t)rows * row_bytes;
    unsigned char *dst = malloc(nd), *want = malloc(nd);
    memset(dst, 0xA5, nd);
    memset(want, 0xA5, nd);
    for (size_t k = 0; k < (size_t)rows * n_row; k++) {
        uint32_t u = in[k < n ? k : n - 1];   // the last row's tail repeats the last input
        size_t r = k / (size_t)n_row, i = k % (size_t)n_row;
        memcpy(&src[r * stride + i], &u, 4);
        f16_t h = f32_to_f16(b8_f(u));
        memcpy(want + r * row_bytes + 2 * i, &h, 2);
    }
    tkv_args a = { rows, n_row, stride, 0, (ka_u64)row_bytes };
    *ran = gpu_train_kv_store_check(m, &a, src, dst);
    size_t bad = 0;
    for (size_t k = 0; *ran && k < (size_t)rows * n_row; k++) {
        size_t r = k / (size_t)n_row, i = k % (size_t)n_row, o = r * row_bytes + 2 * i;
        if (memcmp(dst + o, want + o, 2) && !bad++) {
            uint16_t g, w;
            memcpy(&g, dst + o, 2); memcpy(&w, want + o, 2);
            fprintf(stderr, "FAIL: kv store twin (%s) of %08x is %04x, the host's %04x\n", what,
                    (unsigned)in[k < n ? k : n - 1], g, w);
        }
    }
    if (*ran && memcmp(dst, want, nd) && !bad++)
        fprintf(stderr, "FAIL: kv store twin (%s) wrote a byte between its rows\n", what);
    free(src); free(dst); free(want);
    return bad;
}
static void b8_checks(model_t *m) {
    static const struct { uint32_t f; uint16_t h; } ANCHOR[] = {
        { 0x3F800000u, 0x3C00 }, { 0xC0000000u, 0xC000 }, { 0x477FE000u, 0x7BFF },
        { 0x477FEFFEu, 0x7BFF }, { 0x477FF000u, 0x7C00 }, { 0x38800000u, 0x0400 },
        { 0x33800000u, 0x0001 }, { 0x33000000u, 0x0000 }, { 0x33400000u, 0x0001 },
        { 0x32800000u, 0x0000 }, { 0x7F800000u, 0x7C00 }, { 0x7FC00000u, 0x7E00 },
        { 0x7F800001u, 0x7E00 }, { 0x80000000u, 0x8000 },
    };
    int na = (int)(sizeof(ANCHOR) / sizeof(ANCHOR[0])), abad = 0;
    for (int i = 0; i < na; i++) {
        f16_t h = f32_to_f16(b8_f(ANCHOR[i].f));
        if (h != ANCHOR[i].h && !abad++)
            fprintf(stderr, "FAIL: the host's f32_to_f16(%08x) is %04x, IEEE binary16 says %04x\n",
                    (unsigned)ANCHOR[i].f, h, ANCHOR[i].h);
    }
    CHECK(!abad, "kv store: %d of the %d IEEE anchors differ on the host", abad, na);
    static const uint32_t EDGE[] = { 0, 1, 0x0FFF, 0x1000, 0x1001, 0x1FFF, 0x2FFF, 0x3000,
                                     0x3001, 0x400000, 0x7FEFFF, 0x7FF000, 0x7FF001, 0x7FFFFF };
    int ne = (int)(sizeof(EDGE) / sizeof(EDGE[0]));
    size_t cap = 65536 + 2 * 256 * (size_t)(ne + 16) + 2 * 11 * 9 + (size_t)na, n = 0;
    uint32_t *in = malloc(sizeof(uint32_t) * cap);
    for (uint32_t h = 0; h < 65536; h++) in[n++] = b8_half_bits((uint16_t)h);
    size_t n_half = n;
    for (uint32_t s = 0; s < 2; s++)
        for (uint32_t e = 0; e < 256; e++) {
            for (int k = 0; k < ne; k++) in[n++] = (s << 31) | (e << 23) | EDGE[k];
            for (int k = 0; k < 16; k++) in[n++] = (s << 31) | (e << 23) | (rnd32() & 0x7FFFFFu);
        }
    for (uint32_t s = 0; s < 2; s++)
        for (uint32_t sh = 14; sh <= 24; sh++) {   // a subnormal half: exponent 14 - sh
            uint32_t hmin = sh <= 23 ? 1u << (23 - sh) : 0, hmax = 0xFFFFFFu >> sh;
            uint32_t hs[3] = { hmin, hmin + 1, hmax };
            for (int j = 0; j < 3; j++)
                for (int d = -1; d <= 1; d++) {
                    uint32_t full = (hs[j] << sh) + (1u << (sh - 1)) + (uint32_t)d;
                    if ((full >> 23) != 1) full = 0x800000u | (full & 0x7FFFFFu);
                    in[n++] = (s << 31) | ((126u - sh) << 23) | (full & 0x7FFFFFu);
                }
        }
    for (int i = 0; i < na; i++) in[n++] = ANCHOR[i].f;
    bool ran = false;
    size_t bad = b8_store(m, in, n, 4096, 7, 6, "f16 edges", &ran);
    CHECK(ran, "kv store twin: declined (no kernel in this PTX?)");
    CHECK(!bad, "kv store twin f16: %zu of %zu inputs differ from the host's bits", bad, n);
    // every half's float comes back as that half (a NaN quieted, its payload kept)
    size_t rbad = 0;
    for (size_t k = 0; k < n_half; k++) {
        uint16_t h = (uint16_t)k, want = ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) ? h | 0x200 : h;
        if (f32_to_f16(b8_f(in[k])) != want && !rbad++)
            fprintf(stderr, "FAIL: the half %04x does not come back as itself on the host\n", h);
    }
    CHECK(!rbad, "kv store: %zu of the 65536 halves do not round-trip", rbad);
    bool r1 = false;
    size_t b1 = b8_store(m, in, 37 * 5, 37, 4, 10, "rows", &r1);   // 5 rows, padded both sides
    CHECK(r1 && !b1, "kv store twin rows: %zu differ", b1);
    // kind 3: the floats as they are, NaN payloads and subnormals included
    {
        int rows = 4, nr = 300, stride = 301;
        size_t rb = (size_t)nr * 4 + 8, nd = (size_t)rows * rb;
        float *src = calloc((size_t)rows * stride, sizeof(float));
        unsigned char *dst = malloc(nd), *want = malloc(nd);
        memset(dst, 0x5A, nd); memset(want, 0x5A, nd);
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < nr; i++) {
                memcpy(&src[(size_t)r * stride + i], &in[(size_t)r * 7919 + (size_t)i * 101], 4);
                memcpy(want + (size_t)r * rb + 4 * (size_t)i, &src[(size_t)r * stride + i], 4);
            }
        tkv_args a = { rows, nr, stride, 3, (ka_u64)rb };
        bool ok = gpu_train_kv_store_check(m, &a, src, dst);
        CHECK(ok && !memcmp(dst, want, nd), "kv store twin f32: the rows are not the floats as they are");
        free(src); free(dst); free(want);
    }
    size_t nall = 0, ball = 0;
    const char *all = getenv("RUNNER_TEST_KV_ALL");
    if (all && *all && strcmp(all, "0") != 0) {   // every input, 2^24 at a time
        uint32_t *ch = malloc(sizeof(uint32_t) << 24);
        for (uint64_t base = 0; base < (1ull << 32); base += 1u << 24) {
            for (uint32_t i = 0; i < (1u << 24); i++) ch[i] = (uint32_t)base + i;
            bool r = false;
            ball += b8_store(m, ch, 1u << 24, 1 << 24, 0, 0, "all", &r);
            CHECK(r, "kv store twin: a chunk at %08x declined", (unsigned)base);
            nall += 1u << 24;
        }
        free(ch);
        CHECK(!ball, "kv store twin over all 2^32 inputs: %zu differ from the host's bits", ball);
    }
    if (ran)
        printf("kv store twin f16: %zu inputs (65536 halves, %d exponents x 2 signs, the subnormal "
               "ties) == host; %d IEEE anchors; rows and kind 3 == host%s\n", n, 256, na,
               nall ? "; all 2^32 inputs == host" : "");
    free(in);
}
// R8.24.1 B4 (G0): the device twin of the forward's attention
// (gpu_train_attn_fw_check: k_tattn) against the host's own attn_run as the
// recompute calls it (model_attn_ref), on K / V rows as the training cache
// holds them. Written before the kernel. A T3 build's host does the written
// chains (a score a multiply and an add a channel, softmax's max, pmath's exp
// and its float sum in key order, a value chain a channel in key order),
// which the twin reproduces, so memcmp (a NaN compares as a NaN). The default
// build's host may vectorize those sums under -ffast-math and takes the C
// library's expf; there, and against a double reference in both builds, each
// element within (hd + span + 8) 2^-24 (1 + |largest score|) of the span's
// largest |v| (the float chains' own error, carried through the
// probabilities), the device's subnormal results read as zero in the default
// build. pos0: the device runs only the chunk of positions pos0..T-1 (a later
// chunk of the taped forward: its query rows, over cache rows 0..T-1), held
// to the host's rows for those positions (a position's attention reads only
// its own q row and the rows before it, so the host's whole run is the
// reference for any chunk).
// plant: the control (the second reader's F2): a quiet NaN written into the
// device's output, which every build's gate must catch. kind, the caches'
// kinds: 0 f16 rows, 3 f32 rows of the same (f16-exact) values, 4 K f16 and
// V f32, 5 K f32 and V f16, 6 f32 rows of values no f16 holds.
static double b4_kv(const uint8_t *base, int kind, size_t row_b, int s, size_t idx) {
    if (kind == 3) {
        float f;
        memcpy(&f, base + (size_t)s * row_b + 4 * idx, 4);
        return f;
    }
    f16_t h;
    memcpy(&h, base + (size_t)s * row_b + 2 * idx, 2);
    return f16_to_f32(h);
}
static void b4_check(model_t *m, int T, int H, int kv_mul, int hd, int swa, float scale_mul,
                     int kind, int zero_q, int pos0, int plant) {
    ab_shape sh = ab_shape_of(T, H, kv_mul, hd, swa);
    size_t row16 = (size_t)sh.kv_dim * 2, nq = (size_t)T * sh.q_dim;
    uint8_t *k16 = malloc(row16 * T), *v16 = malloc(row16 * T);
    float *q = malloc(sizeof(float) * nq), *dao = malloc(sizeof(float) * nq);
    ab_ref_fill(&sh, 0x9E3779B97F4A7C15ull ^ (uint64_t)(T * 131 + hd * 7 + swa), k16, v16, q, dao);
    if (zero_q) for (int i = 0; i < sh.q_dim; i++) q[i] = 0.0f;   // position 0: a flat softmax
    int kk = kind == 3 || kind == 5 || kind == 6 ? 3 : 0;   // K's kind
    int kv = kind == 3 || kind == 4 || kind == 6 ? 3 : 0;   // V's kind
    size_t rowkb = kk == 3 ? (size_t)sh.kv_dim * 4 : row16, rowvb = kv == 3 ? (size_t)sh.kv_dim * 4 : row16;
    uint8_t *kc = k16, *vc = v16;
    for (int w = 0; w < 2; w++) {   // the f32 form of K and of V where asked
        if ((w ? kv : kk) != 3) continue;
        const uint8_t *src = w ? v16 : k16;
        uint8_t *dst = malloc((size_t)sh.kv_dim * 4 * T);
        for (size_t i = 0; i < (size_t)T * sh.kv_dim; i++) {
            f16_t hb;
            memcpy(&hb, src + 2 * i, 2);
            const float f = f16_to_f32(hb);
            uint32_t u;   // the value's bits, end to end: no float written back and
            memcpy(&u, &f, 4);   // read again (CI's gcc flagged that round trip)
            // kind 6: moved off the f16 grid by a few f32 ulps a value, by integers
            // (no float arithmetic, which this build may flush or contract); a
            // signed zero stays as it is
            if (kind == 6 && (u & 0x7fffffffu))
                u += 1u + (uint32_t)((i * 2654435761u) >> 27);   // 1..32 ulps up in magnitude
            memcpy(dst + 4 * i, &u, 4);
        }
        if (w) vc = dst; else kc = dst;
    }
    float scale = scale_mul / sqrtf((float)hd);
    int nb = T - pos0;   // the chunk's rows
    size_t qoff = (size_t)pos0 * sh.q_dim;
    float *aoh = calloc(nq, sizeof(float)), *aod = calloc((size_t)nb * sh.q_dim, sizeof(float));
    bool okh = model_attn_ref(T, H, kv_mul, hd, swa, scale, kk, kv, kc, vc, rowkb, rowvb, q, aoh);
    tattn_args a = { nb, H, kv_mul, hd, sh.q_dim, sh.kv_dim, swa, rowkb, rowvb, scale, kk, kv,
                     pos0 };
    bool ok = okh && gpu_train_attn_fw_check(m, &a, kc, vc, q + qoff, aod);
    CHECK(ok, "attn twin T %d hd %d rows %d..%d: declined (no kernel in this PTX, or a span the "
          "launch could not hold)", T, hd, pos0, T - 1);
    if (ok && plant) {
        uint32_t qn = 0x7fc00000u;
        memcpy(&aod[(size_t)(nb / 2) * sh.q_dim], &qn, 4);
    }
    size_t bad = 0, hbad = 0, abad = 0;
    double wh = 0, wa = 0;
    double *sc = malloc(sizeof(double) * T), *pr = malloc(sizeof(double) * T);
    for (int t = pos0; ok && t < T; t++)
        for (int h = 0; h < H; h++) {
            int kvh = h / kv_mul, t0 = swa > 0 && t - swa + 1 > 0 ? t - swa + 1 : 0, span = t - t0 + 1;
            const float *qh = q + (size_t)t * sh.q_dim + (size_t)h * hd;
            double smax = 0, mx = -1e300, sum = 0, vmax = 0;
            for (int s = t0; s <= t; s++) {
                double d = 0;
                for (int i = 0; i < hd; i++)
                    d += (double)qh[i] * b4_kv(kc, kk, rowkb, s, (size_t)kvh * hd + i);
                sc[s] = d * scale;
                if (fabs(sc[s]) > smax) smax = fabs(sc[s]);
                if (sc[s] > mx) mx = sc[s];
                for (int i = 0; i < hd; i++) {
                    double v = fabs(b4_kv(vc, kv, rowvb, s, (size_t)kvh * hd + i));
                    if (v > vmax) vmax = v;
                }
            }
            for (int s = t0; s <= t; s++) { pr[s] = exp(sc[s] - mx); sum += pr[s]; }
            double lim = (hd + span + 8) * 0x1p-24 * (1.0 + smax) * vmax + 1e-37;
            for (int i = 0; i < hd; i++) {
                size_t o = (size_t)t * sh.q_dim + (size_t)h * hd + i;
                float hv = aoh[o], dv = aod[o - qoff];
#ifdef RUNNER_T3_BUILD
                bool same = b2_nan(hv) || b2_nan(dv) ? b2_nan(hv) && b2_nan(dv) : !memcmp(&hv, &dv, 4);
                if (!same && !bad++ && !plant)
                    fprintf(stderr, "FAIL: attn twin differs from the T3 host at T %d t %d h %d i %d: "
                            "%.9g/%.9g\n", T, t, h, i, dv, hv);
#else
                float ev = b2_ftz(dv);
                double e = fabs((double)ev - hv);
                // by bits first: a NaN e passes every compare below
                bool nf = b_nonfin(ev) || b_nonfin(hv) || d_nonfin(e);
                if (!nf && e / lim > wh) wh = e / lim;
                if ((nf || e > 2 * lim) && !hbad++ && !plant)
                    fprintf(stderr, "FAIL: attn twin is off the default build's host at T %d t %d h %d "
                            "i %d: %.9g/%.9g\n", T, t, h, i, ev, hv);
#endif
                double x = 0;
                for (int s = t0; s <= t; s++)
                    x += pr[s] / sum * b4_kv(vc, kv, rowvb, s, (size_t)kvh * hd + i);
                double ea = fabs((double)dv - x);
                bool anf = b_nonfin(dv) || d_nonfin(x) || d_nonfin(ea);
                if (!anf && ea / lim > wa) wa = ea / lim;
                if ((anf || ea > lim) && !abad++ && !plant)
                    fprintf(stderr, "FAIL: attn twin is not the attention at T %d t %d h %d i %d: "
                            "%.9g (double %.9g)\n", T, t, h, i, dv, x);
            }
        }
    if (plant) {   // the control: the planted NaN must fail the host gate and the anchor
        CHECK(!ok || ((bad || hbad) && abad), "attn twin control: a NaN planted in the device's "
              "output passed the gate (host %zu, default %zu, anchor %zu)", bad, hbad, abad);
        if (ok && (bad || hbad) && abad)
            printf("attn twin control: a NaN planted in the device's output is caught (host gate %zu, "
                   "anchor %zu element(s))\n", bad + hbad, abad);
        bad = hbad = abad = 0;
        ok = false;   // no summary line for the control
    }
    CHECK(!bad, "attn twin T %d H %d kv_mul %d hd %d swa %d kind %d: %zu elements differ from "
          "the T3 host's bits", T, H, kv_mul, hd, swa, kind, bad);
    CHECK(!hbad, "attn twin T %d hd %d: %zu elements off the default build's host (worst %.3g of "
          "the bound, allowed 2)", T, hd, hbad, wh);
    CHECK(!abad, "attn twin T %d hd %d: %zu elements off the double attention (worst %.3g of the "
          "bound)", T, hd, abad, wa);
    if (ok)
        printf("attn twin T %3d H %2d kv_mul %d hd %3d swa %2d kind %d scale x%-4g%s rows %d..%d: "
               "host %s; against the double attention %.3g of the bound\n", T, H, kv_mul, hd, swa,
               kind, scale_mul, zero_q ? " zero-q" : "", pos0, T - 1,
#ifdef RUNNER_T3_BUILD
               "T3 (memcmp)",
#else
               "default (bound, subnormals flushed)",
#endif
               wa);
    if (kc != k16) free(kc);
    if (vc != v16) free(vc);
    free(k16); free(v16); free(q); free(dao); free(aoh); free(aod); free(sc); free(pr);
}
static void b4_checks(model_t *m) {
    b4_check(m, 128, 32, 4, 128, 0, 1.0f, 0, 0, 0, 0);    // the recipe's Qwen3-4B shape, T 128
    b4_check(m, 77, 4, 2, 64, 0, 1.0f, 0, 1, 0, 0);       // grouped heads, a zero q row
    b4_check(m, 90, 4, 1, 256, 24, 1.0f, 0, 0, 0, 0);     // a sliding window, head 256
    b4_check(m, 33, 2, 2, 128, 1, 1.0f, 0, 0, 0, 0);      // a window of one key
    b4_check(m, 64, 8, 4, 128, 0, 16.0f, 0, 0, 0, 0);     // peaked probabilities
    b4_check(m, 50, 4, 2, 96, 7, 1.0f, 3, 0, 0, 0);       // f32 rows, a short window
    b4_check(m, 300, 2, 1, 64, 0, 1.0f, 0, 0, 0, 0);      // spans past a block of threads
    // later chunks of the taped forward (a start that is not 0)
    b4_check(m, 300, 32, 4, 128, 0, 1.0f, 0, 0, 128, 0);  // the recipe's shape, its second chunk
    b4_check(m, 150, 4, 2, 64, 40, 1.0f, 0, 0, 100, 0);   // a window across the chunk's first row
    b4_check(m, 205, 2, 1, 64, 0, 1.0f, 3, 0, 192, 0);    // the last short chunk (13 rows), f32 rows
    b4_check(m, 129, 4, 2, 128, 0, 1.0f, 0, 0, 128, 0);   // a one-row chunk
    // the largest span the scope admits (12288 keys: 48 KiB of scores beside the
    // kernel's static shared memory, past the default 48 KiB a block; the second
    // reader's F1), whole and as the last chunk; and a window that bounds it
    b4_check(m, 12288, 1, 1, 1, 0, 1.0f, 0, 0, 0, 0);
    b4_check(m, 12288, 2, 1, 4, 0, 1.0f, 3, 0, 12000, 0);
    b4_check(m, 12288, 1, 1, 8, 128, 1.0f, 0, 0, 11000, 0);
    b4_check(m, 40, 2, 1, 512, 0, 1.0f, 0, 0, 0, 0);      // the widest head the twin takes
    b4_check(m, 60, 4, 2, 128, 0, 1.0f, 4, 0, 0, 0);      // K f16, V f32
    b4_check(m, 60, 4, 2, 128, 9, 1.0f, 5, 0, 20, 0);     // K f32, V f16, a window, a chunk
    b4_check(m, 70, 4, 2, 96, 0, 1.0f, 6, 0, 0, 0);       // f32 rows no f16 holds
    b4_check(m, 64, 4, 2, 64, 0, 1.0f, 0, 0, 0, 1);       // the control: a planted NaN is caught
}

// A scope case holds when the reason names want, or, with want NULL, when there
// is no reason. strstr never sees a NULL: as a macro arm, strstr(got, NULL)
// stayed visible to gcc 13 at -O3 in the branch that never runs, and
// -Werror=nonnull refused the STRICT build.
static bool scope_reason_holds(const char *got, const char *want) {
    if (!want) return got == NULL;
    return got != NULL && strstr(got, want) != NULL;
}

// R8.24.1 B, the attention slots' scope (model_attn_dev_why), by shape and
// with no device: the recipe's shape (and an f32 cache) runs on the twin, and
// every condition outside the scope declines with its own reason, so a model
// outside it takes today's path and says once why. Nonzero: a case that does
// not hold.
static int b_scope_checks(void) {
    static layer_t ly[1];
    static bool swa1[1] = { true };
    static float knw[1], sink[1];
    model_t s;
    int bad = 0, cases = 0;
#define SCOPE_RESET() do { memset(&s, 0, sizeof s); memset(ly, 0, sizeof ly); \
        s.n_layer = 1; s.layers = ly; s.n_head = 32; s.n_ctx = 512; } while (0)
#define SCOPE_CASE(label, want, pos, n, hd, nkv, qd, os) do { \
        cases++; \
        const char *got = model_attn_dev_why(&s, 0, pos, n, hd, nkv, qd, os); \
        bool okc = scope_reason_holds(got, (want)); \
        if (!okc && !bad++) \
            fprintf(stderr, "FAIL: attention scope (%s): %s, want %s\n", label, \
                    got ? got : "runs on the device", (want) ? (want) : "runs on the device"); \
    } while (0)
    SCOPE_RESET(); SCOPE_CASE("the recipe's shape", NULL, 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.kv_f32 = true; SCOPE_CASE("an f32 cache", NULL, 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); SCOPE_CASE("a later chunk", NULL, 64, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); SCOPE_CASE("12289 rows", "shared memory", 0, 12289, 128, 8, 4096, 4096);
    SCOPE_RESET(); SCOPE_CASE("a chunk ending past 12288", "shared memory", 12000, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.mla_kv_lora = 512; SCOPE_CASE("MLA", "MLA", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.kv_ring = 128; s.l_is_swa = swa1;
    SCOPE_CASE("a ring layer", "ring", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.l_is_swa = swa1;   // a sliding layer without a ring: the twin's window
    SCOPE_CASE("a flat sliding layer", NULL, 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); ly[0].attn_sinks = sink; SCOPE_CASE("sinks", "sinks", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.tied_v = true; s.v_rmsnorm = true; ly[0].knorm_w = knw;
    SCOPE_CASE("tied V", "tied-V", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.kv_q8 = true; SCOPE_CASE("a q8 cache", "q8 or fp4", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.kv_fp4 = true; SCOPE_CASE("an fp4 cache", "q8 or fp4", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); s.kv_split = true; SCOPE_CASE("k8v4", "q8 or fp4", 0, 300, 128, 8, 4096, 4096);
    SCOPE_RESET(); SCOPE_CASE("hd 576", "wider than 512", 0, 300, 576, 8, 32 * 576, 32 * 576);
    SCOPE_RESET(); SCOPE_CASE("q rows", "n_head x head_dim", 0, 300, 128, 8, 4000, 4000);
    SCOPE_RESET(); SCOPE_CASE("output rows", "wider than q", 0, 300, 128, 8, 4096, 5120);
    SCOPE_RESET(); SCOPE_CASE("6 KV heads", "whole groups", 0, 300, 128, 6, 4096, 4096);
    SCOPE_RESET(); SCOPE_CASE("no KV heads", "whole groups", 0, 300, 128, 0, 4096, 4096);
#undef SCOPE_CASE
#undef SCOPE_RESET
    if (!bad) printf("attention scope: %d cases hold (4 run on the twin, %d decline with "
                     "their reason)\n", cases, cases - 4);
    return bad;
}
// R8.24.1 B, R0 of the resident step: its scope by shape, with no device. A
// layer in scope (the recipe's shape) and each condition outside it on its
// own; the step's T, rank and planes against a budget (the planes at the
// recipe's shape, and the largest T and rank a 12 GB and an 8 GB card hold,
// taking the weights as this host measured them).
static size_t b_res_need(int T, int r) {   // Qwen3-4B, f16 K/V, every site at rank r
    const long long per_r = 6656 + 3584 + 3584 + 6656 + 12288 + 12288 + 12288;   // r (n_in + n_out), a layer
    resident_shape rs = { 36, 2560, 4096, 1024, 9728, 32, T, r, 2048, 2048, 36 * per_r * r, 0 };
    return model_resident_need(&rs);
}
static int b_resident_scope_checks(void) {
    static layer_t ly[1];
    static gguf_tensor wt;
    static float bias[1], pnorm[1];
    static int kvsrc[1] = { 1 };
    model_t s;
    int bad = 0, cases = 0;
#define RES_RESET() do { memset(&s, 0, sizeof s); memset(ly, 0, sizeof ly); \
        s.n_layer = 1; s.layers = ly; s.n_head = 32; s.head_dim = 128; s.n_head_kv = 8; \
        s.n_embd = 2560; s.n_ctx = 512; s.resid_scale = 1.0f; s.ffn_act = ACT_SILU; \
        ly[0].w_gate = ly[0].w_up = ly[0].w_down = &wt; ly[0].n_ff = 9728; } while (0)
#define RES_CASE(label, want) do { \
        cases++; \
        const char *got = model_resident_layer_why(&s, 0, 300); \
        bool okc = scope_reason_holds(got, (want)); \
        if (!okc && !bad++) \
            fprintf(stderr, "FAIL: resident scope (%s): %s, want %s\n", label, \
                    got ? got : "resident", (want) ? (want) : "resident"); \
    } while (0)
    RES_RESET(); RES_CASE("the recipe's layer", NULL);
    RES_RESET(); s.full = (struct full_w *)(void *)&wt; RES_CASE("full-weight training", "full-weight");   // a non-NULL mark, never read
    RES_RESET(); s.hyper_conn = true; RES_CASE("hyper-connections", "hyper-connections");
    RES_RESET(); s.norm_layernorm = true; RES_CASE("layer norms", "layer norms");
    RES_RESET(); s.norm_groups = 2; RES_CASE("grouped norms", "grouped");
    RES_RESET(); s.resid_scale = 0.5f; RES_CASE("a residual scale", "residual scale");
    RES_RESET(); s.ffn_act = ACT_GELU; RES_CASE("GELU", "not SiLU");
    RES_RESET(); ly[0].recurrent = true; RES_CASE("a recurrent layer", "recurrent");
    RES_RESET(); ly[0].skip_mixer = true; RES_CASE("no mixer", "without attention");
    RES_RESET(); ly[0].is_moe = true; RES_CASE("MoE", "experts");
    RES_RESET(); ly[0].w_gate = NULL; RES_CASE("an ungated FFN", "not gated");
    RES_RESET(); ly[0].post_attn_norm_w = pnorm; RES_CASE("a post-attention norm", "post-attention");
    RES_RESET(); ly[0].ple_gate = &wt; RES_CASE("PLE", "per-layer embeddings");
    RES_RESET(); ly[0].bq = bias; RES_CASE("a Q bias", "biases");
    RES_RESET(); s.qwen35 = true; RES_CASE("qwen3.5", "qwen3.5");
    RES_RESET(); s.attn_out_gate = true; ly[0].wq_gate = &wt; RES_CASE("an output gate", "output gate");
    RES_RESET(); s.v_rmsnorm = true; RES_CASE("a V norm", "V norm");
    RES_RESET(); s.attn_temp_scale = 0.1f; s.attn_temp_all = true; RES_CASE("temperature", "temperature");
    RES_RESET(); s.n_layer = 2; s.kv_src = kvsrc; RES_CASE("shared K/V", "another layer");
    RES_RESET(); s.kv_q8 = true; RES_CASE("a q8 cache (the attention scope)", "q8 or fp4");
#undef RES_CASE
#undef RES_RESET
    // the step's shape and planes
    const long long per_r = 6656 + 3584 + 3584 + 6656 + 12288 + 12288 + 12288;
    resident_shape rs = { 36, 2560, 4096, 1024, 9728, 32, 300, 8, 2048, 2048, 36 * per_r * 8,
                          (size_t)8e9 };
    const struct { const char *label; int T, r; size_t avail; const char *want; } SC[] = {
        { "the recipe, 8 GB for the planes", 300, 8, (size_t)8e9, NULL },
        { "the recipe, 0.3 GB", 300, 8, (size_t)3e8, "planes need" },
        { "T 12289", 12289, 8, (size_t)8e12, "12288 rows" },
        { "T 1", 1, 8, (size_t)8e9, "12288 rows" },
        { "rank 513", 300, 513, (size_t)8e12, "rank past 512" },
    };
    for (size_t i = 0; i < sizeof SC / sizeof SC[0]; i++) {
        cases++;
        rs.T = SC[i].T; rs.rank_max = SC[i].r; rs.adapter_floats = 36 * per_r * SC[i].r;
        rs.avail = SC[i].avail;
        const char *got = model_resident_shape_why(&rs);
        bool okc = SC[i].want ? got != NULL && strstr(got, SC[i].want) != NULL : got == NULL;
        if (!okc && !bad++)
            fprintf(stderr, "FAIL: resident shape (%s): %s\n", SC[i].label, got ? got : "resident");
    }
    // an absolute anchor for the planes: the tape alone is (L+1) T E floats
    cases++;
    rs.T = 300; rs.adapter_floats = 0; rs.n_ff = 0; rs.q_dim = 0; rs.kv_dim = 0; rs.n_head = 0;
    rs.k_row_b = rs.v_row_b = 0;
    size_t tape_resid = model_resident_need(&rs);
    // (L+1) tape planes, the residual and dx, and a layer's three E-wide work rows (x, y, dx)
    if (tape_resid != (size_t)(37 + 2 + 3) * 300 * 2560 * 4 && !bad++)
        fprintf(stderr, "FAIL: resident planes: the tape, residual and E-wide work planes are %zu "
                "bytes, not (37 + 2 + 3) x 300 x 2560 floats\n", tape_resid);
    if (!bad) {
        // the budgets: this host's 12 GB card offers 11.0 GB free to the run
        // (its desktop holds 0.65 GB) and the recipe's weights take 2.4 GB on
        // it (3.36 GB peak against 0.65 idle, at T 300); a CUDA context is
        // about 0.3 GB. An 8 GB card with the same desktop: 7.2 GB free.
        const struct { const char *card; double avail; } C[] = { { "12 GB", 11.0e9 - 2.4e9 - 0.3e9 },
                                                                 { "8 GB", 7.2e9 - 2.4e9 - 0.3e9 } };
        printf("resident scope: %d cases hold; the recipe's planes at T 300 rank 8: %.0f MB\n", cases,
               b_res_need(300, 8) / 1e6);
        for (int c = 0; c < 2; c++) {
            int tmax = 2, rmax = 1;
            while (tmax < 12288 && b_res_need(tmax + 1, 8) <= C[c].avail) tmax++;
            while (rmax < 512 && b_res_need(300, rmax + 1) <= C[c].avail) rmax++;
            printf("resident scope, a %s card (%.1f GB for the planes beside Qwen3-4B Q4_K_M): the "
                   "largest T at rank 8 %d, the largest rank at T 300 %d\n", C[c].card,
                   C[c].avail / 1e9, tmax, rmax);
        }
    }
    return bad;
}

// R8.24.1 B, R0: the residual add's twin (gpu_train_add_check: k_tadd)
// against the host's own add (layer_branch_out's x += y, one float add).
// Anchors, IEEE 754 binary32 by hand: 1 + 2^-24 = 1 (a tie to even),
// 1 + 3 2^-24 = 1 + 2^-22 (a tie to even, upward), FLT_MAX + FLT_MAX = inf,
// inf + -inf = NaN, -0 + -0 = -0, 0 + -0 = 0. Rows at strides wider than the
// data; boundary values (zeros, the largest finite, the smallest normal,
// subnormals, infinities, NaNs) against every other, then random values. T3:
// every element by bits (NaN meets NaN). The default build's host flushes
// subnormal operands and results (FTZ and DAZ) where the device keeps them:
// there the elements with a subnormal in them are counted, not compared.
static void b_add_checks(model_t *m) {
    static const uint32_t AN[][3] = {   // x, y, x + y
        { 0x3f800000u, 0x33800000u, 0x3f800000u }, { 0x3f800000u, 0x34400000u, 0x3f800002u },
        { 0x7f7fffffu, 0x7f7fffffu, 0x7f800000u }, { 0x80000000u, 0x80000000u, 0x80000000u },
        { 0x00000000u, 0x80000000u, 0x00000000u },
    };
    int na = (int)(sizeof AN / sizeof AN[0]), abad = 0;
    for (int i = 0; i < na; i++) {
        float x, y, z;
        memcpy(&x, &AN[i][0], 4); memcpy(&y, &AN[i][1], 4);
        z = x + y;
        uint32_t u;
        memcpy(&u, &z, 4);
        if (u != AN[i][2] && !abad++)
            fprintf(stderr, "FAIL: the host's %08x + %08x is %08x, IEEE says %08x\n", (unsigned)AN[i][0],
                    (unsigned)AN[i][1], (unsigned)u, (unsigned)AN[i][2]);
    }
    CHECK(!abad, "residual add: %d of the %d IEEE anchors differ on the host", abad, na);
    static const uint32_t BV[] = { 0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f7fffffu,
                                   0xff7fffffu, 0x00800000u, 0x80800000u, 0x00000001u, 0x807fffffu,
                                   0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7fa12345u, 0x33800000u,
                                   0x34400000u };
    int nbv = (int)(sizeof BV / sizeof BV[0]);
    int rows = 9, n = nbv * nbv + 211, xs = n + 5, ys = n + 3;
    float *x = calloc((size_t)rows * xs, sizeof(float)), *y = calloc((size_t)rows * ys, sizeof(float));
    float *xh = malloc(sizeof(float) * (size_t)rows * xs);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < n; i++) {
            float a, b;
            if (i < nbv * nbv) { memcpy(&a, &BV[i / nbv], 4); memcpy(&b, &BV[i % nbv], 4); }
            else { a = frnd() * (r % 2 ? 1e30f : 3.0f); b = frnd() * (r % 3 ? 3.0f : 1e-30f); }
            x[(size_t)r * xs + i] = a;
            y[(size_t)r * ys + i] = b;
        }
    for (size_t k = 0; k < (size_t)rows * xs; k++) xh[k] = x[k];
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < n; i++) xh[(size_t)r * xs + i] += y[(size_t)r * ys + i];
    float *xd = malloc(sizeof(float) * (size_t)rows * xs);
    memcpy(xd, x, sizeof(float) * (size_t)rows * xs);
    tadd_args a = { rows, n, xs, ys };
    bool ok = gpu_train_add_check(m, &a, xd, y);
    CHECK(ok, "residual add twin: declined (no kernel in this PTX?)");
    size_t bad = 0, skipped = 0;
    (void)skipped;   // counted in the default build only
    for (int r = 0; ok && r < rows; r++)
        for (int i = 0; i < xs; i++) {
            size_t o = (size_t)r * xs + i;
            float h = xh[o], d = xd[o];
            if (i >= n) {   // the bytes between rows come back as they went
                if (memcmp(&d, &x[o], 4) && !bad++)
                    fprintf(stderr, "FAIL: residual add twin wrote between rows at row %d [%d]\n", r, i);
                continue;
            }
#ifndef RUNNER_T3_BUILD
            uint32_t ua, ub, ud;
            float fa = x[o], fb = y[(size_t)r * ys + i];
            memcpy(&ua, &fa, 4); memcpy(&ub, &fb, 4); memcpy(&ud, &d, 4);
            bool sub = ((ua & 0x7f800000u) == 0 && (ua & 0x7fffffu)) ||
                       ((ub & 0x7f800000u) == 0 && (ub & 0x7fffffu)) ||
                       ((ud & 0x7f800000u) == 0 && (ud & 0x7fffffu));
            if (sub) { skipped++; continue; }
#endif
            bool same = b2_nan(h) || b2_nan(d) ? b2_nan(h) && b2_nan(d) : !memcmp(&h, &d, 4);
            if (!same && !bad++) {
                uint32_t uh, ud2;
                memcpy(&uh, &h, 4); memcpy(&ud2, &d, 4);
                fprintf(stderr, "FAIL: residual add twin at row %d [%d]: %08x, the host %08x\n", r, i,
                        (unsigned)ud2, (unsigned)uh);
            }
        }
    CHECK(!bad, "residual add twin: %zu elements differ from the host's bits", bad);
    if (ok && !bad)
        printf("residual add twin: %d rows of %d (%d boundary pairs, %d IEEE anchors) == host by bits%s\n",
               rows, n, nbv * nbv, na,
#ifdef RUNNER_T3_BUILD
               "");
#else
               skipped ? "; the subnormal ones counted, not compared (this host flushes them)" : "");
#endif
    free(x); free(y); free(xh); free(xd);
}

// R8.24.1 B5 (G0): the device twin of the head (gpu_train_head_check:
// k_thead) against the host's own code (model_head_fw_ref: apply_head_transforms
// then train_row_lse; model_head_bw_ref: head_rows_bw's per-row adjoint).
// Written before the kernel. A T3 build's host does exactly the written
// operations (pmath's tanh, exp and log; float sums in index order, the LSE's
// in double), which the twin reproduces, so memcmp on the adjoint rows, the
// normalizers and the target logits (a NaN compares as a NaN), and an
// unscored row must come back untouched. The default build's host takes the
// C library's tanhf, expf and logf and may vectorize under -ffast-math; there,
// and against the head in double in both builds, each adjoint element within
// ((V + 16) 2^-23 + (1 + |largest z|) 2^-21) (|its value| + 2^-24 w), w the
// row's |pos_w| |scale| (the sums' own error and the transforms' rounding
// carried through exp), plus under a softcap 2^-21 of the value before its
// slope (1 - th*th in float carries about 2^-23 absolute), the normalizer within 2^-20 (1 + |lse|) + (V + 16)
// 2^-24, and the target logit within 2^-20 (1 + |z|); the device's subnormal
// results read as zero in the default build.
// The bound gates of B5 and B7 (the lead's rule): an error as a fraction of
// its bound, 1e300 when a value is non-finite and its partner is not (two
// non-finite values settled by class: NaN against NaN, an infinity by its
// bits, since the device returns the canonical NaN), never a NaN compared.
static int nf_same(float a, float b) {
    if (b2_nan(a) || b2_nan(b)) return b2_nan(a) && b2_nan(b);
    return !memcmp(&a, &b, 4);
}
static double gate_err(float dev, float host, double lim) {
    if (b_nonfin(dev) || b_nonfin(host)) return nf_same(dev, host) ? 0.0 : 1e300;
    double e = fabs((double)dev - (double)host) / lim;
    return d_nonfin(e) ? 1e300 : e;
}
static double anchor_err(float dev, double ref, double lim) {
    if (b_nonfin(dev) || d_nonfin(ref)) return (b_nonfin(dev) && d_nonfin(ref)) ? 0.0 : 1e300;
    double e = fabs((double)dev - ref) / lim;
    return d_nonfin(e) ? 1e300 : e;
}
// the planted-NaN control: both bound gates must fail a NaN in place of a value
static void nan_control(const char *who, float host, double ref, double lim) {
    float nan; uint32_t q = 0x7fc00000u; memcpy(&nan, &q, 4);
    CHECK(gate_err(nan, host, lim) >= 1e300 && anchor_err(nan, ref, lim) >= 1e300,
          "%s: the bound gates let a planted NaN pass", who);
}
static void b5_fill_row(float *x, int V, int kind) {
    for (int i = 0; i < V; i++) {
        float v = frnd() * 8.0f;
        switch (kind) {
        case 1: v = 0.25f; break;                                   // a flat row
        case 2: v = (i & 1) ? -0.0f : 0.0f; break;
        case 3: v = frnd() * 80.0f; break;                          // wide logits
        case 4: v = i == V / 3 ? 60.0f : frnd(); break;             // one peak
        case 5: v = frnd() * 1.0e4f; break;                         // far past a softcap
        default: break;
        }
        x[i] = v;
    }
}
static void b5_check(model_t *m, int V, int rows, int pad, float ls, float cap, int n_sup,
                     int weighted) {
    int stride = V + pad;
    size_t span = (size_t)(rows - 1) * stride + V;
    float *z0 = malloc(sizeof(float) * span), *zd = malloc(sizeof(float) * span);
    for (size_t i = 0; i < span; i++) z0[i] = 1234.5f;   // the padding, which must survive
    for (int r = 0; r < rows; r++) b5_fill_row(z0 + (size_t)r * stride, V, r % 6);
    int32_t sup[3] = { 3 % V, V / 2, V - 1 };
    int32_t *tg = malloc(sizeof(int32_t) * rows);
    float *pw = weighted ? malloc(sizeof(float) * rows) : NULL;
    for (int r = 0; r < rows; r++) {
        tg[r] = (int32_t)(rnd32() % (uint32_t)V);
        if (pw) pw[r] = 0.25f + 1.5f * (frnd() * 0.5f + 0.5f);
    }
    if (rows > 2) tg[1] = -1;                         // an unscored row
    if (rows > 3 && n_sup) tg[2] = sup[1];            // a suppressed target
    memcpy(zd, z0, sizeof(float) * span);
    double *lsed = calloc(rows, sizeof(double));
    float *ztd = calloc(rows, sizeof(float));
    thead_args a = { V, rows, stride, n_sup, ls, cap };
    bool ok = gpu_train_head_check(m, &a, zd, tg, pw, sup, lsed, ztd);
    CHECK(ok, "head twin V %d: declined (no kernel in this PTX?)", V);
    size_t bad = 0, hbad = 0, abad = 0;
    double wh = 0, wa = 0;
    float *zh = malloc(sizeof(float) * V), *ph = malloc(sizeof(float) * V);
    double *zx = malloc(sizeof(double) * V), *ex = malloc(sizeof(double) * V);
    for (int r = 0; ok && r < rows; r++) {
        const float *x = z0 + (size_t)r * stride;
        float *d = zd + (size_t)r * stride;
        if (tg[r] < 0) {
            if (memcmp(x, d, sizeof(float) * V) && !bad++)
                fprintf(stderr, "FAIL: head twin touched the unscored row %d\n", r);
            continue;
        }
        for (int i = V; r < rows - 1 && i < stride; i++)   // the last row has no padding after it
            if (d[i] != 1234.5f && !bad++) fprintf(stderr, "FAIL: head twin wrote past V\n");
        memcpy(zh, x, sizeof(float) * V);
        float zth;
        double lseh = model_head_fw_ref(zh, V, ls, cap, sup, n_sup, tg[r], &zth);
        memcpy(ph, x, sizeof(float) * V);
        bool okb = model_head_bw_ref(ph, V, 1, ls, cap, sup, n_sup, &tg[r], pw ? &pw[r] : NULL);
        CHECK(okb, "head twin: the host's head_rows_bw failed");
        // the head in double
        double lsx = ls != 0.0f ? ls : 1.0, mx = -1e300, sum = 0, zmax = 0;
        for (int i = 0; i < V; i++) {
            double v = (double)x[i] * lsx;
            if (cap > 0) v = cap * tanh(v / cap);
            zx[i] = v;
        }
        for (int k = 0; k < n_sup; k++) zx[sup[k]] = -1e30;
        for (int i = 0; i < V; i++) {
            if (zx[i] > mx) mx = zx[i];
            if (fabs(zx[i]) > zmax && zx[i] > -1e29) zmax = fabs(zx[i]);
        }
        for (int i = 0; i < V; i++) { ex[i] = exp(zx[i] - mx); sum += ex[i]; }
        double lsex = mx + log(sum), wrow = (pw ? fabs(pw[r]) : 1.0) * fabs(lsx);
        double k1 = (V + 16) * 0x1p-23 + (1.0 + zmax) * 0x1p-21;
#ifdef RUNNER_T3_BUILD
        if ((memcmp(&lseh, &lsed[r], 8) || memcmp(&zth, &ztd[r], 4)) && !bad++)
            fprintf(stderr, "FAIL: head twin's lse or target logit differs from the T3 host at V %d "
                    "row %d: lse %.17g/%.17g z %.9g/%.9g\n", V, r, lsed[r], lseh, ztd[r], zth);
#else
        double llim = 0x1p-20 * (1 + fabs(lseh)) + (V + 16) * 0x1p-24;
        double el = d_nonfin(lsed[r]) || d_nonfin(lseh) ? (memcmp(&lsed[r], &lseh, 8) ? 1e300 : 0)
                                                          : fabs(lsed[r] - lseh) / llim;
        if ((d_nonfin(el) || el > 1.0 || gate_err(ztd[r], zth, 0x1p-20 * (1 + fabs(zth))) > 1.0) && !hbad++)
            fprintf(stderr, "FAIL: head twin's lse or target logit is off the default host at V %d "
                    "row %d: lse %.9g/%.9g z %.9g/%.9g\n", V, r, lsed[r], lseh, ztd[r], zth);
#endif
        double la = d_nonfin(lsed[r]) || d_nonfin(lsex) ? 1e300
                    : fabs(lsed[r] - lsex) / (0x1p-20 * (1 + fabs(lsex)) + (V + 16) * 0x1p-24);
        if ((d_nonfin(la) || la > 1.0) && !abad++)
            fprintf(stderr, "FAIL: head twin's lse is not the double one at V %d row %d: %.9g (%.9g)\n",
                    V, r, lsed[r], lsex);
        for (int i = 0; i < V; i++) {
            double p = ex[i] / sum - (i == tg[r] ? 1.0 : 0.0);
            p *= pw ? pw[r] : 1.0;
            for (int k = 0; k < n_sup; k++) if (i == sup[k]) p = 0.0;
            double pre = p;   // before the softcap's slope: 1 - th*th in float carries ~2^-23 absolute
            if (cap > 0) { double t = tanh((double)x[i] * lsx / cap); p *= 1.0 - t * t; }
            p *= lsx;
            double lim = k1 * (fabs(p) + 0x1p-24 * wrow) + (cap > 0 ? fabs(pre * lsx) * 0x1p-21 : 0.0) + 1e-37;
            float hv = ph[i], dv = d[i];
#ifdef RUNNER_T3_BUILD
            bool same = b2_nan(hv) || b2_nan(dv) ? b2_nan(hv) && b2_nan(dv) : !memcmp(&hv, &dv, 4);
            if (!same && !bad++)
                fprintf(stderr, "FAIL: head twin differs from the T3 host at V %d row %d [%d]: "
                        "%.9g/%.9g\n", V, r, i, dv, hv);
#else
            float ev = b2_ftz(dv);
            double e = gate_err(ev, hv, lim);
            if (e > wh && e < 1e300) wh = e;
            if (e > 2.0 && !hbad++)
                fprintf(stderr, "FAIL: head twin is off the default build's host at V %d row %d [%d]: "
                        "%.9g/%.9g\n", V, r, i, ev, hv);
#endif
            double q = anchor_err(dv, p, lim);
            if (q > wa && q < 1e300) wa = q;
            if (q > 1.0 && !abad++)
                fprintf(stderr, "FAIL: head twin is not the double head at V %d row %d [%d]: %.9g "
                        "(%.9g)\n", V, r, i, dv, p);
        }
    }
    if (ok) nan_control("head twin", 0.5f, 0.5, 1e-6);
    CHECK(!bad, "head twin V %d rows %d scale %g cap %g sup %d: %zu values differ from the T3 host's "
          "bits", V, rows, ls, cap, n_sup, bad);
    CHECK(!hbad, "head twin V %d: %zu values off the default build's host (worst %.3g of the bound, "
          "allowed 2)", V, hbad, wh);
    CHECK(!abad, "head twin V %d: %zu values off the head in double (worst %.3g of the bound)",
          V, abad, wa);
    if (ok)
        printf("head twin V %6d rows %2d pad %d scale %-4g cap %-3g sup %d%s: host %s; against the "
               "head in double %.3g of the bound\n", V, rows, pad, ls, cap, n_sup,
               pw ? " weighted" : "", 
#ifdef RUNNER_T3_BUILD
               "T3 (memcmp)",
#else
               "default (bound, subnormals flushed)",
#endif
               wa);
    free(z0); free(zd); free(tg); free(pw); free(lsed); free(ztd); free(zh); free(ph); free(zx);
    free(ex);
}
static void b5_checks(model_t *m) {
    b5_check(m, 151936, 6, 0, 1.0f, 0.0f, 0, 1);    // Qwen3's vocabulary, the recipe's head
    b5_check(m, 151936, 128, 0, 1.0f, 0.0f, 0, 1);  // and the recipe's T 128 rows at once
    b5_check(m, 32000, 8, 3, 1.0f, 0.0f, 0, 0);     // no weights, a padded stride
    b5_check(m, 32000, 7, 0, 1.5f, 30.0f, 3, 1);    // scale, softcap and suppressed ids
    b5_check(m, 257, 13, 1, 0.0f, 0.0f, 3, 1);      // a block's edge; scale 0 reads as none
    b5_check(m, 1, 3, 0, 1.0f, 0.0f, 0, 1);         // one token
    b5_check(m, 1025, 6, 0, 0.7f, 15.0f, 0, 0);     // a softcap without suppression
}

// R8.24.1 B7 (G0): the device twin of AdamW (gpu_train_adam_check: k_tadam)
// against the host's own code (model_adam_ref: adam_buf). Written before the
// kernel. A T3 build's host does the written operations, so memcmp on th, m
// and v. The default build's host may contract and vectorize under
// -ffast-math (its sqrtf may be a reciprocal estimate refined); there, and
// against the update in double in both builds, each th within 2^-20 (|th| +
// lr), each m within 2^-21 (|m| + |g|), each v within 2^-21 (|v| + g^2); the
// device's subnormals read as zero in the default build. (B7's twin of the
// gradient norms, k_tgnorm, is retired with its case: the step takes the
// norms from the host's code, R4.)
static void b7_adam_check(model_t *m, size_t n, int edges) {   // edges 2: the cancellation case
    float *th = malloc(sizeof(float) * n), *g = malloc(sizeof(float) * n);
    float *mo = malloc(sizeof(float) * n), *vo = malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) {
        th[i] = frnd(); g[i] = frnd() * 1e-2f; mo[i] = frnd() * 1e-3f;
        vo[i] = (frnd() * 0.5f + 0.5f) * 1e-5f;
    }
    if (edges && n > 8) {
        g[0] = 0.0f; g[1] = -0.0f; g[2] = 1.0e19f; g[3] = 1.0e-30f; mo[4] = 0.0f; vo[4] = 0.0f;
        uint32_t q = 0x7fc00001u; memcpy(&g[5], &q, 4); th[6] = 0.0f; g[7] = -3.0e18f;
    }
    // the cancellation case (edges 2): the moment term made to cancel three
    // quarters of wd*th, so the sum's ulp is under the product's and the
    // weight-decay step's rounding shows; wd not a power of two (else wd*th is
    // exact) and lr large enough that a unit of the sum moves th
    float lr = edges == 2 ? 32.0f : 1e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float wd = edges == 2 ? 0.3f : 0.01f;
    float bc1 = 1.0f - powf(b1, 3.0f), bc2 = 1.0f - powf(b2, 3.0f);
    if (edges == 2)
        for (size_t i = 0; i < n; i++) {
            g[i] = 0.0f;
            vo[i] = bc2 / b2;                       // vh = 1: the denominator is 1 + eps
            mo[i] = -0.75f * wd * th[i] * bc1 / b1; // mh = b1 m / bc1 ~ -0.75 wd th
        }
    float *th_h = malloc(sizeof(float) * n), *mo_h = malloc(sizeof(float) * n), *vo_h = malloc(sizeof(float) * n);
    float *th_d = malloc(sizeof(float) * n), *mo_d = malloc(sizeof(float) * n), *vo_d = malloc(sizeof(float) * n);
    memcpy(th_h, th, 4 * n); memcpy(mo_h, mo, 4 * n); memcpy(vo_h, vo, 4 * n);
    memcpy(th_d, th, 4 * n); memcpy(mo_d, mo, 4 * n); memcpy(vo_d, vo, 4 * n);
    model_adam_ref(th_h, g, mo_h, vo_h, n, lr, b1, b2, eps, wd, bc1, bc2);
    tadam_args a = { (uint64_t)n, lr, b1, b2, eps, wd, bc1, bc2 };
    bool ok = gpu_train_adam_check(m, &a, th_d, g, mo_d, vo_d);
    CHECK(ok, "adam twin n %zu: declined (no kernel in this PTX?)", n);
    size_t bad = 0, hbad = 0, abad = 0, nonfin_diff = 0;
    double wa = 0;
    for (size_t i = 0; ok && i < n; i++) {
        double gd = g[i];
        double md = (double)b1 * mo[i] + (1.0 - (double)b1) * gd;
        double vd = (double)b2 * vo[i] + (1.0 - (double)b2) * gd * gd;
        double td = th[i] - lr * ((md / bc1) / (sqrt(vd / bc2) + eps) + wd * (double)th[i]);
        double lt = 0x1p-20 * (fabs(td) + lr) + 1e-37, lm = 0x1p-21 * (fabs(md) + fabs(gd)) + 1e-37;
        double lv = 0x1p-21 * (fabs(vd) + gd * gd) + 1e-37;
        float D[3] = { th_d[i], mo_d[i], vo_d[i] }, H[3] = { th_h[i], mo_h[i], vo_h[i] };
#ifdef RUNNER_T3_BUILD
        for (int k = 0; k < 3; k++) {
            bool same = b2_nan(H[k]) || b2_nan(D[k]) ? b2_nan(H[k]) && b2_nan(D[k]) : !memcmp(&H[k], &D[k], 4);
            if (!same && !bad++)
                fprintf(stderr, "FAIL: adam twin differs from the T3 host at [%zu] (%d): %.9g/%.9g\n",
                        i, k, D[k], H[k]);
        }
#else
        for (int k = 0; k < 3; k++) D[k] = b2_ftz(D[k]);
        double L3[3] = { lt, lm, lv };
        if (b_nonfin(g[i])) {   // a non-finite input: only T3 gates it (the default host is not IEEE there)
            for (int k = 0; k < 3; k++) if (!nf_same(D[k], H[k])) { nonfin_diff++; break; }
            continue;
        }
        for (int k = 0; k < 3; k++)
            if (gate_err(D[k], H[k], L3[k]) > 1.0 && !hbad++)
                fprintf(stderr, "FAIL: adam twin is off the default build's host at [%zu] (%d): %.9g/%.9g\n",
                        i, k, D[k], H[k]);
#endif
        if (!d_nonfin(td) && !d_nonfin(vd) && !b2_nan(g[i])) {
            double e = anchor_err(th_d[i], td, lt), e2 = anchor_err(mo_d[i], md, lm), e3 = anchor_err(vo_d[i], vd, lv);
            if (e2 > e) e = e2;
            if (e3 > e) e = e3;
            if (e > wa) wa = e;
            if (e > 1.0 && !abad++)
                fprintf(stderr, "FAIL: adam twin is not the update in double at [%zu]: th %.9g (%.9g) m %.9g "
                        "(%.9g) v %.9g (%.9g)\n", i, th_d[i], td, mo_d[i], md, vo_d[i], vd);
        }
    }
    if (ok) nan_control("adam twin", 0.5f, 0.5, 1e-6);
    CHECK(!bad, "adam twin n %zu: %zu values differ from the T3 host's bits", n, bad);
    CHECK(!hbad, "adam twin n %zu: %zu values off the default build's host", n, hbad);
    CHECK(!abad, "adam twin n %zu: %zu values off the update in double (worst %.3g of the bound)", n, abad, wa);
    if (ok)
        printf("adam twin n %7zu%s: host %s; against the update in double %.3g of the bound; "
               "non-finite inputs differing (no verdict outside T3): %zu\n", n,
               edges == 2 ? " cancel" : edges ? " edges" : "",
#ifdef RUNNER_T3_BUILD
               "T3 (memcmp)",
#else
               "default (bound, subnormals flushed)",
#endif
               wa, nonfin_diff);
    free(th); free(g); free(mo); free(vo); free(th_h); free(mo_h); free(vo_h); free(th_d); free(mo_d); free(vo_d);
}
static void b7_checks(model_t *m) {
    b7_adam_check(m, 100003, 1);
    b7_adam_check(m, 1, 0);
    b7_adam_check(m, 257, 1);
    b7_adam_check(m, 4099, 2);       // the weight-decay step's rounding made visible
}

// R8.24.1 B R4 (G0): the mean's scale on the device (gpu_train_scale_check:
// k_tscale) against the host's own loop (model_grad_scale_ref:
// model_lora_grad_scale's g[i] *= f). One rounded product each: memcmp in a
// T3 build; in a default build too, except where the input or the product is
// subnormal (that host flushes them, -ffast-math's FTZ and DAZ: counted, not
// compared). The absolute anchor, in both builds: the exact product in
// double, rounded once to float, which IEEE's single product is (a float
// times a float is exact in double). Written before the kernel.
static void r4_scale_check(model_t *m, size_t n, float f) {
    float *g = malloc(sizeof(float) * n), *gh = malloc(sizeof(float) * n);
    float *gd = malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) g[i] = frnd() * (i % 3 ? 1e-2f : 3.0f);
    if (n > 9) {
        uint32_t inf = 0x7f800000u, nan = 0x7fc00001u, sub = 0x00012345u, fmin = 0x00800000u;
        g[0] = 0.0f; g[1] = -0.0f; g[2] = 3.4e38f; g[4] = -1e-30f;
        memcpy(&g[3], &sub, 4); memcpy(&g[5], &inf, 4); memcpy(&g[6], &nan, 4);
        memcpy(&g[7], &fmin, 4); g[8] = -3.4e38f;
    }
    memcpy(gh, g, sizeof(float) * n);
    memcpy(gd, g, sizeof(float) * n);
    model_grad_scale_ref(gh, n, f);
    bool ok = gpu_train_scale_check(m, gd, n, f);
    CHECK(ok, "scale twin n %zu: declined (no kernel in this PTX?)", n);
    size_t bad = 0, abad = 0, flushed = 0;
    for (size_t i = 0; ok && i < n; i++) {
        double pd = (double)g[i] * (double)f;
        float want = (float)pd;
        // by bits and in double: under DAZ a float compare reads a subnormal as zero
        uint32_t ub;
        memcpy(&ub, &g[i], 4);
        bool sub_in = ((ub & 0x7f800000u) == 0 && (ub & 0x007fffffu) != 0) ||
                      (!d_nonfin(pd) && pd != 0.0 && fabs(pd) < 0x1p-126);
#ifdef RUNNER_T3_BUILD
        (void)sub_in;
        if (!nf_same(gd[i], gh[i]) && !bad++)
            fprintf(stderr, "FAIL: scale twin differs from the T3 host at [%zu]: %.9g/%.9g\n", i, gd[i], gh[i]);
        if (!nf_same(gd[i], want) && !abad++)
            fprintf(stderr, "FAIL: scale twin is not the product rounded once at [%zu]: %.9g/%.9g\n", i,
                    gd[i], want);
#else
        if (sub_in) { flushed++; continue; }
        if (!nf_same(gd[i], gh[i]) && !bad++)
            fprintf(stderr, "FAIL: scale twin differs from the default build's host at [%zu]: %.9g/%.9g\n", i,
                    gd[i], gh[i]);
        if (!nf_same(gd[i], want) && !abad++)
            fprintf(stderr, "FAIL: scale twin is not the product rounded once at [%zu]: %.9g/%.9g\n", i,
                    gd[i], want);
#endif
    }
    CHECK(!bad, "scale twin n %zu f %g: %zu values differ from the host's", n, (double)f, bad);
    CHECK(!abad, "scale twin n %zu f %g: %zu values are not the product rounded once", n, (double)f, abad);
    if (ok)
        printf("scale twin n %7zu f %-9g: host %s, the product rounded once; subnormal inputs or products "
               "not compared: %zu\n", n, (double)f,
#ifdef RUNNER_T3_BUILD
               "T3 (memcmp)",
#else
               "default (memcmp, subnormals flushed)",
#endif
               flushed);
    free(g); free(gh); free(gd);
}
static void r4_checks(model_t *m) {
    r4_scale_check(m, 100003, 1.0f / 37.0f);   // a mean's 1 / wsum, blocks and a tail
    r4_scale_check(m, 256, 0.5f);              // one block exactly; a power of two
    r4_scale_check(m, 257, 3.0f);              // one over
    r4_scale_check(m, 1, 1.0f / 3.0f);         // one element
    r4_scale_check(m, 4099, 1e-37f);           // most products subnormal (T3 compares them all)
}

// R8.24.1 B R4: the host scalars held to a T3 build's bits in every build by
// their precise attribute (model.c, model_lora_train_init's note): the device
// optimizer's bias corrections (model_adam_bias_corr), and below them a fresh
// adapter's init scale (model_lora_init_scale). The bias corrections for steps
// 1 to 100000 at AdamW's two betas: an FNV-1a 64 digest of their bits against
// the constant a T3 build gives; in a T3 build also memcmp against 1.0f -
// powf(beta, step), that build's own powf (pmath.h's p_powf, strict), which is
// what makes the constant the T3 values. The anchor: step 1 is 1 - beta
// exactly (beta^1 is beta, and the subtraction rounds once). No device
// needed, so every CI host runs it: gcc on Linux and Windows, clang on macOS,
// each compiler's reading of the attribute.
#define R4_BC_DIGEST 0x001728232e120610ull   // a T3 build's (gcc 16.2.0, x86-64), 2026-11-20
#define R4_INIT_DIGEST 0x972bb79cdde78270ull // the same build's
static int r4_bias_corr_checks(void) {
    static const float betas[2] = { 0.9f, 0.999f };   // TRAIN_ADAM_BETA1, TRAIN_ADAM_BETA2
    uint64_t h = 0xcbf29ce484222325ull;
    int bad = 0;
    for (int b = 0; b < 2; b++) {
        float beta = betas[b], one = 1.0f - beta, at1 = model_adam_bias_corr(beta, 1);
        if (memcmp(&at1, &one, 4) != 0 && !bad++)
            fprintf(stderr, "FAIL: bias correction at step 1, beta %.9g: %.9g, not 1 - beta %.9g\n",
                    (double)beta, (double)at1, (double)one);
        for (int s = 1; s <= 100000; s++) {
            float v = model_adam_bias_corr(beta, s);
            uint32_t u;
            memcpy(&u, &v, 4);
            for (int k = 0; k < 4; k++) { h ^= (u >> (8 * k)) & 0xffu; h *= 0x100000001b3ull; }
#ifdef RUNNER_T3_BUILD
            float w = 1.0f - powf(beta, (float)s);
            if (memcmp(&v, &w, 4) != 0 && !bad++)
                fprintf(stderr, "FAIL: bias correction differs from the T3 build's powf at step %d, beta "
                        "%.9g: %.9g/%.9g\n", s, (double)beta, (double)v, (double)w);
#endif
        }
    }
    if (h != R4_BC_DIGEST && !bad++)
        fprintf(stderr, "FAIL: the bias corrections' digest is %016llx, a T3 build's %016llx\n",
                (unsigned long long)h, (unsigned long long)R4_BC_DIGEST);
    // a fresh adapter's init scale (model_lora_init_scale, the same attribute):
    // 1 / sqrt(n_in) for every n_in 1 to 65536; the anchors are the powers of 4,
    // whose scale is a power of 2 exactly (a default build's rsqrtss and one
    // Newton step gave one ulp more at 2560)
    uint64_t hi = 0xcbf29ce484222325ull;
    for (int n = 1; n <= 65536; n++) {
        float v = model_lora_init_scale(n);
        uint32_t u;
        memcpy(&u, &v, 4);
        for (int k = 0; k < 4; k++) { hi ^= (u >> (8 * k)) & 0xffu; hi *= 0x100000001b3ull; }
        int p = 0;
        while ((1 << (2 * p)) < n) p++;
        if ((1 << (2 * p)) == n && v != 1.0f / (float)(1 << p) && !bad++)
            fprintf(stderr, "FAIL: the init scale at n_in %d is %.9g, not 2^-%d\n", n, (double)v, p);
#ifdef RUNNER_T3_BUILD
        float w = 1.0f / sqrtf((float)n);
        if (memcmp(&v, &w, 4) != 0 && !bad++)
            fprintf(stderr, "FAIL: the init scale differs from the T3 build's 1.0f / sqrtf at n_in %d: "
                    "%.9g/%.9g\n", n, (double)v, (double)w);
#endif
    }
    if (hi != R4_INIT_DIGEST && !bad++)
        fprintf(stderr, "FAIL: the init scales' digest is %016llx, a T3 build's %016llx\n",
                (unsigned long long)hi, (unsigned long long)R4_INIT_DIGEST);
    // and the adapter's scale alpha / rank (model_lora_alpha_scale) at every
    // rank the CLI admits (--lora-rank 1 to 512), whose alpha is 2 x rank
    // (main.c): 2 exactly at each, the anchor
    for (int r = 1; r <= 512; r++) {
        float a = 2.0f * (float)r, v = model_lora_alpha_scale(a, r);
        if (v != 2.0f && !bad++)
            fprintf(stderr, "FAIL: alpha / rank at rank %d is %.9g, not 2\n", r, (double)v);
#ifdef RUNNER_T3_BUILD
        float w = a / (float)r;
        if (memcmp(&v, &w, 4) != 0 && !bad++)
            fprintf(stderr, "FAIL: alpha / rank differs from the T3 build's at rank %d\n", r);
#endif
    }
    if (!bad)
        printf("bias corrections: steps 1 to 100000 at beta 0.9 and 0.999, the init scale at n_in 1 to "
               "65536 and alpha / rank at ranks 1 to 512 are a T3 build's (digests %016llx, %016llx%s)\n",
               (unsigned long long)h, (unsigned long long)hi,
#ifdef RUNNER_T3_BUILD
               ", and memcmp against this T3 build's powf"
#else
               ""
#endif
               );
    return bad;
}

int main(void) {
    f16_init();
    model_t m;
    memset(&m, 0, sizeof(m));
    m.train_gpu_budget = (uint64_t)1 << 30;   // no model file: the arena is this
    if (gate_cc_pins("train-gpu")) return 1;   // no device needed
    if (b_scope_checks()) return 1;   // no device needed
    if (b_resident_scope_checks()) return 1;
    if (r4_bias_corr_checks()) return 1;   // no device needed
    if (!gpu_train_init(&m)) {
        if (gate_no_device("train-gpu")) return 1;
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
    void *keep_pinned = NULL;   // a buffer registered with the context (R8.24.1)
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
        mvt_dev_check(&m, &w, dy, dx0, N_IN, N_OUT, B, TYPES[k].name);
        // R8.24.1: a zero start cleared on the device (gpu_train_mvt0) is the
        // zeros sent, bit for bit, in both kernel families, and reads nothing
        // of the host dx (all-ones bytes going in); the assist's is the CPU
        // chain from zero
        for (int dev = 0; dev < 2; dev++) {
            float *z0 = calloc((size_t)B * N_IN, sizeof(float));
            float *zc = calloc((size_t)B * N_IN, sizeof(float));
            float *zp = malloc(sizeof(float) * B * N_IN);
            memset(zp, 0xFF, sizeof(float) * B * N_IN);
            m.train_dev = dev;
            bool a = gpu_train_mvt(&m, &w, dy, z0, N_IN, N_OUT, B);
            bool b = gpu_train_mvt0(&m, &w, dy, zp, N_IN, N_OUT, B);
            m.train_dev = false;
            CHECK(a && b && !memcmp(z0, zp, sizeof(float) * B * N_IN),
                  "%s: a zero start cleared on the device differs from the "
                  "zeros sent (%s)", TYPES[k].name, dev ? "device mode" : "the assist");
            if (!dev) {
                cpu_mvt(&w, dy, zc, N_IN, N_OUT, B);
                CHECK(!memcmp(zc, zp, sizeof(float) * B * N_IN),
                      "%s: the assist's zero start is not the CPU chain from zero",
                      TYPES[k].name);
            }
            free(z0); free(zc); free(zp);
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
        // R8.24.1: a packed call lands in place, and its bias is the same one
        // add per element as the staged (strided) call's
        float *yb = malloc(sizeof(float) * 3 * N_OUT);
        bool okb = ok && gpu_train_mm(&m, &w, x, N_IN, yb, N_OUT, N_IN, N_OUT, bias, 3);
        CHECK(!ok || okb, "packed call with a bias declined");
        for (int c = 0; okb && c < 3; c++)
            for (int j = 0; j < N_OUT; j++)
                CHECK(yb[(size_t)c * N_OUT + j] == ys[(size_t)c * YS + j],
                      "packed column %d row %d differs from the staged call", c, j);
        free(yb);
        // R8.24.1: three products of one x in one round trip are their own
        // calls' bits (a strided part, a part with a bias); a scaled part is
        // left to the caller
        {
            enum { YS2 = N_OUT + 3 };
            gguf_tensor pw[3];
            for (int s = 0; s < 3; s++) {
                memset(&pw[s], 0, sizeof pw[s]);
                pw[s].type = (uint32_t)TYPES[s].type;
                pw[s].scale = s == 2 ? 2.0f : 1.0f;
                pw[s].ne[0] = N_IN; pw[s].ne[1] = N_OUT;
                pw[s].data = keep[s];
                pw[s].nbytes = ggml_row_size(pw[s].type, N_IN) * N_OUT;
            }
            float b2[N_OUT];
            for (int j = 0; j < N_OUT; j++) b2[j] = frnd();
            float *pa = malloc(sizeof(float) * 3 * YS2), *ra = malloc(sizeof(float) * 3 * YS2);
            float *pb = malloc(sizeof(float) * 3 * N_OUT), *rb = malloc(sizeof(float) * 3 * N_OUT);
            float *pc = malloc(sizeof(float) * 3 * N_OUT);
            for (int i = 0; i < 3 * YS2; i++) pa[i] = ra[i] = -7.0f;
            gpu_mm_part pp[3] = { { &pw[0], pa, YS2, N_OUT, NULL },
                                  { &pw[1], pb, N_OUT, N_OUT, b2 },
                                  { &pw[2], pc, N_OUT, N_OUT, NULL } };
            unsigned done = gpu_train_mm_parts(&m, pp, 3, x, N_IN, N_IN, 3);
            CHECK(done == 3u, "parts: computed %#x, want parts 0 and 1 (the scaled "
                  "part 2 is the caller's)", done);
            bool okr = gpu_train_mm(&m, &pw[0], x, N_IN, ra, YS2, N_IN, N_OUT, NULL, 3) &&
                       gpu_train_mm(&m, &pw[1], x, N_IN, rb, N_OUT, N_IN, N_OUT, b2, 3);
            CHECK(okr, "parts: a reference call declined");
            CHECK(!okr || !memcmp(pa, ra, sizeof(float) * 3 * YS2),
                  "parts: the strided part differs from its own call");
            CHECK(!okr || !memcmp(pb, rb, sizeof(float) * 3 * N_OUT),
                  "parts: the part with a bias differs from its own call");
            free(pa); free(ra); free(pb); free(rb); free(pc);
        }
        // R8.24.1 A4: a product read from a registered buffer and written into
        // a pinned block is the pageable call's, memcmp; the block is handed
        // out again while it is large enough
        {
            float *xr = malloc(sizeof(float) * 3 * N_IN), *yr = malloc(sizeof(float) * 3 * N_OUT);
            memcpy(xr, x, sizeof(float) * 3 * N_IN);
            bool pinned = gpu_train_pin(&m, xr, sizeof(float) * 3 * N_IN);
            float *yp = gpu_train_host_buf(&m, 0, sizeof(float) * 3 * N_OUT);
            CHECK(pinned && yp, "pinning: registered %d, block %p", (int)pinned, (void *)yp);
            CHECK(!yp || gpu_train_host_buf(&m, 0, sizeof(float) * N_OUT) == yp,
                  "pinning: a smaller request did not get the same block");
            bool okp = yp && gpu_train_mm(&m, &w, xr, N_IN, yp, N_OUT, N_IN, N_OUT, bias, 3) &&
                       gpu_train_mm(&m, &w, x, N_IN, yr, N_OUT, N_IN, N_OUT, bias, 3);
            CHECK(!yp || okp, "pinning: a call declined");
            CHECK(!okp || !memcmp(yp, yr, sizeof(float) * 3 * N_OUT),
                  "pinning: the pinned call differs from the pageable one");
            // xr stays registered until gpu_train_free, so it is freed after it
            keep_pinned = xr;
            free(yr);
        }
        // a scaled tensor is the CPU's
        w.scale = 2.0f;
        CHECK(!gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, 1),
              "a scaled tensor went to the device");
        free(xs); free(ys);
        keep[sizeof(TYPES) / sizeof(*TYPES)] = w.data;
    }
    // RUNNER_TRAIN_GPU_BENCH=1: the forward product's two kernel families at
    // 4096 x 5120, 16, 64 and 512 columns (the narrow twin, a device-mode
    // round, a recompute window), the kernels' time alone, best of five
    const char *bench = getenv("RUNNER_TRAIN_GPU_BENCH");
    void *bkeep[4] = {0};   // freed after the context, like keep[]
    if (bench && strcmp(bench, "0") != 0) {
        static const struct { int type; const char *name; int f16[2]; } BT[] = {
            { T_IQ3_S, "iq3_s", {0, -1} }, { T_Q4_K, "q4_K", {0, 2} },
            { T_Q6_K, "q6_K", {208, -1} }, { T_Q8_0, "q8_0", {0, -1} },
        };
        enum { BN_IN = 5120, BN_OUT = 4096, BB = 512 };
        float *bx = malloc(sizeof(float) * (size_t)BB * BN_IN);
        float *by = malloc(sizeof(float) * (size_t)BB * BN_OUT);
        for (size_t i = 0; i < (size_t)BB * BN_IN; i++) bx[i] = frnd();
        for (size_t k = 0; k < sizeof(BT) / sizeof(*BT); k++) {
            gguf_tensor w;
            memset(&w, 0, sizeof(w));
            w.type = (uint32_t)BT[k].type; w.scale = 1.0f;
            w.ne[0] = BN_IN; w.ne[1] = BN_OUT;
            w.data = make_rows(BT[k].type, BT[k].f16, BN_IN, BN_OUT, &w.nbytes);
            static const int NBS[] = { 16, 64, 512 };
            for (int ni = 0; ni < 3; ni++)
                for (int v = 0; v < 2; v++) {
                    set_mm_kernel(v ? "" : "warp");
                    double best = 1e9;
                    for (int rep = 0; rep < 5; rep++) {
                        double t_mm, t_mvt, k_mm, k_mvt;
                        long n_mm, n_mvt;
                        gpu_train_prof(&m, &t_mm, &n_mm, &t_mvt, &n_mvt, &k_mm, &k_mvt);
                        if (!gpu_train_mm(&m, &w, bx, BN_IN, by, BN_OUT, BN_IN,
                                          BN_OUT, NULL, NBS[ni])) break;
                        gpu_train_prof(&m, &t_mm, &n_mm, &t_mvt, &n_mvt, &k_mm, &k_mvt);
                        if (k_mm < best) best = k_mm;
                    }
                    set_mm_kernel("");
                    printf("bench %-6s %4d columns %-8s %7.3f ms  %6.2f TFLOP/s\n",
                           BT[k].name, NBS[ni], v ? "tile" : "per-warp", best * 1e3,
                           2.0 * BN_IN * BN_OUT * NBS[ni] / best / 1e12);
                }
            // the transposed product in device mode: the CPU chain (k_tmvt)
            // against the register-blocked twins
            m.train_dev = true;
            float *bdx = malloc(sizeof(float) * (size_t)BB * BN_IN);
            for (int ni = 0; ni < 3; ni++)
                for (int v = 0; v < 2; v++) {
                    set_mvt_kernel(v ? "" : "chain");
                    double best = 1e9;
                    for (int rep = 0; rep < 5; rep++) {
                        double t_mm, t_mvt, k_mm, k_mvt;
                        long n_mm, n_mvt;
                        memset(bdx, 0, sizeof(float) * (size_t)NBS[ni] * BN_IN);
                        gpu_train_prof(&m, &t_mm, &n_mm, &t_mvt, &n_mvt, &k_mm, &k_mvt);
                        if (!gpu_train_mvt(&m, &w, by, bdx, BN_IN, BN_OUT, NBS[ni])) break;
                        gpu_train_prof(&m, &t_mm, &n_mm, &t_mvt, &n_mvt, &k_mm, &k_mvt);
                        if (k_mvt < best) best = k_mvt;
                    }
                    set_mvt_kernel("");
                    printf("bench %-6s %4d columns %-8s %7.3f ms  %6.2f TFLOP/s (transposed)\n",
                           BT[k].name, NBS[ni], v ? "tile" : "chain", best * 1e3,
                           2.0 * BN_IN * BN_OUT * NBS[ni] / best / 1e12);
                }
            free(bdx);
            m.train_dev = false;
            bkeep[k] = w.data;
        }
        free(bx); free(by);
    }
    ffn_check(&m);
    ffn_nonfinite_check(&m);
    lora_case(&m, 8, 2560, 1024, 104, 0, 0);
    lora_case(&m, 8, 9728, 2560, 104, 0, 0);
    lora_case(&m, 8, 37, 300, 7, 1, 0);
    lora_case(&m, 1, 64, 5, 1, 1, 0);
    lora_case(&m, 5, 129, 33, 17, 1, 0);
    lora_case(&m, 512, 300, 200, 9, 0, 0);    // LORA_R_MAX, the largest rank --lora-rank takes
    lora_case(&m, 512, 37, 61, 3, 1, 2);      // and with a negative scale, edge rows
    lora_case(&m, 8, 129, 33, 17, 0, 1);      // a zero scale: y and the gradients' s terms vanish
    lora_case(&m, 8, 37, 300, 7, 1, 2);       // a negative scale
    lora_case(&m, 5, 129, 33, 17, 0, 3);      // the cancellation case: y is the fused step's residual
    lora_case(&m, 8, 2560, 1024, 13, 0, 3);   // and at the recipe's width
    ab_check(&m, 77, 4, 2, 64, 0);
    ab_check(&m, 150, 8, 4, 64, 0);
    ab_check(&m, 90, 4, 1, 256, 24);
    dn_check(&m, 4, 2, 128, 77);
    dn_check(&m, 3, 3, 64, 40);
    dn_check(&m, 4, 2, 40, 33);
    dn_check(&m, 2, 1, 32, 9);
    dn_check(&m, 4, 2, 8, 26);
    b2_checks(&m);
    b3_checks(&m);
    b8_checks(&m);
    b_add_checks(&m);
    b4_checks(&m);
    b5_checks(&m);
    b7_checks(&m);
    r4_checks(&m);
    gpu_train_free(&m);
    free(keep_pinned);
    for (size_t k = 0; k < sizeof(keep) / sizeof(*keep); k++) free(keep[k]);
    for (size_t k = 0; k < sizeof(bkeep) / sizeof(*bkeep); k++) free(bkeep[k]);
    printf("%s: %d types\n", g_fail ? "FAIL" : "PASS", tested);
    return g_fail;
}
