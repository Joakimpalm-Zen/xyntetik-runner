// The CPU dot route gate (2026-10-08). Byte-identity tests cannot see a
// route that is correct and slow: 1.1.1 sent the one-row decode through
// vec_dot_multi, every output stayed byte-equal, and dense IQ3_S decode fell
// from 1.45 to 0.83 tok/s until a hotfix. Two checks:
//  1. dispatch: a one-token forward never calls vec_dot_multi; a 3-row batch
//     does (the small-batch route exists and is used);
//  2. a same-process ratio canary: for 2..7 columns vec_dot_multi must not
//     cost more than 1.5x the per-column vec_dot loop it replaces. A ratio
//     on one machine in one process, not an absolute tok/s. Best of 5
//     interleaved rounds removes a single spike but not a stretch of
//     contention on a shared hosted runner (two macos-suite failures, each
//     at ratio ~2 and ~1.0 on re-run), so a case over the limit is measured
//     again with 25 rounds and fails only when that confirms it: a slow
//     route is slow every time, a busy neighbour is not.
//  3. the same kind of canary for a weight decode (R3.73): on an x86 AVX2
//     build, dequant_row's MXFP4 rows (every expert row a batched prefill
//     of gpt-oss decodes) must cost at most 1.3x Q4_K's per weight, a
//     decode with an AVX2 kernel of long standing (a ratio of two library
//     routes on one machine, steadier than a hand-written scalar loop).
//     Measured on hosted x86, 2026-11-16: the SIMD decode 1.05x on a Xeon
//     6975P-C and 0.63x on an EPYC 9V74; the C decode it replaced 1.58x on
//     the Xeon (and about 6x on the EPYC, four workers, the R3.60 bench).
//     Its bytes are checked too, over all
//     256 exponents: this binary is built with the runner's own flags
//     (-ffast-math, so flush-to-zero and denormals-are-zero are on), which
//     test-quants-simd's are not.
#include "model.h"
#include "quants.h"
#include "fp16.h"
#include "runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static int g_fail = 0;
static void ck(int cond, const char *what) {
    fprintf(stderr, "%s: %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) g_fail = 1;
}

static double route_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void dispatch(const char *path) {
    model_t m; memset(&m, 0, sizeof m);
    model_params p; memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF; p.n_threads = 1; p.n_ctx = 64; p.n_batch = 8;
    if (!model_load(&m, path, &p)) { fprintf(stderr, "cannot load %s\n", path); g_fail = 1; return; }
    unsigned long c0 = vec_dot_multi_calls();
    model_forward(&m, 1, 0);
    unsigned long c1 = vec_dot_multi_calls();
    ck(c1 == c0, "a one-row forward never takes the multi-column dot");
    int32_t toks[3] = { 2, 3, 4 };
    model_forward_batch(&m, toks, 3, 1, true);
    ck(vec_dot_multi_calls() > c1, "a 3-row batch takes the multi-column dot");
    model_free(&m);
}

static void ratio(int type, const char *name) {
    const int n = 4096, rows = 256;
    size_t rs = ggml_row_size(type, n);
    unsigned char *w = malloc(rs * rows);
    float *x = malloc(sizeof(float) * n * VEC_DOT_MULTI_MAX);
    unsigned s = 12345;
    for (size_t i = 0; i < rs * rows; i++) { s = s * 1103515245u + 12345u; w[i] = (unsigned char)(s >> 16); }
    // keep fp16 block scales finite and small: every format here leads its
    // block with an fp16 scale, and a random one can be Inf/NaN
    size_t bs = ggml_row_size(type, ggml_block_size(type));
    for (size_t b = 0; b < rs * rows; b += bs) { w[b] = 0x00; w[b + 1] = 0x3c; w[b + 2] = 0x00; w[b + 3] = 0x3c; }
    for (int i = 0; i < n * VEC_DOT_MULTI_MAX; i++) { s = s * 1103515245u + 12345u; x[i] = ((s >> 9) & 1023) / 1024.0f - 0.5f; }
    float out[VEC_DOT_MULTI_MAX], sink = 0;
    for (int nc = 2; nc < VEC_DOT_MULTI_MAX; nc++) {
        const float *xs[VEC_DOT_MULTI_MAX];
        for (int c = 0; c < nc; c++) xs[c] = x + (size_t)c * n;
        double best_m, best_l;
        for (int attempt = 0, rounds = 5; attempt < 2; attempt++, rounds = 25) {
            best_m = 1e9; best_l = 1e9;
            for (int rep = 0; rep < rounds; rep++) {
                double t0 = route_now();
                for (int r = 0; r < rows; r++) { vec_dot_multi(type, w + rs * r, xs, nc, n, out); sink += out[0]; }
                double t1 = route_now();
                for (int r = 0; r < rows; r++) for (int c = 0; c < nc; c++) sink += vec_dot(type, w + rs * r, xs[c], n);
                double t2 = route_now();
                if (t1 - t0 < best_m) best_m = t1 - t0;
                if (t2 - t1 < best_l) best_l = t2 - t1;
            }
            if (best_m <= 1.5 * best_l) break;
            if (attempt == 0)
                fprintf(stderr, "re-measuring: %s, %d columns: ratio %.2f in 5 rounds\n",
                        name, nc, best_m / best_l);
        }
        char msg[160];
        snprintf(msg, sizeof msg, "%s, %d columns: multi %.3f ms, per-column loop %.3f ms (ratio %.2f, limit 1.50)",
                 name, nc, best_m * 1e3, best_l * 1e3, best_m / best_l);
        ck(best_m <= 1.5 * best_l, msg);
    }
    if (sink == 12345.678f) puts("");   // keep the work observable
    free(w); free(x);
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
static const float kv_mxfp4_f[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
static float g_e8m0_f[256];   // 2^(e - 127), the table quants.c's dq_mxfp4 reads
// The C decode: a table value times a table scale, both loaded, as dq_mxfp4
// does. Not `kv * ldexpf(1.0f, e - 127)`: under -ffast-math GCC rewrites that
// to ldexpf(kv, e - 127), which neither flushes a subnormal scale nor makes
// 0 * inf a NaN, so it is not the C decode's arithmetic (seen on hosted x86:
// two exponents of 256 differed until the scale came from a table).
static void scalar_mxfp4_rows(const unsigned char *w, float *y, int nblk) {
    for (int i = 0; i < nblk; i++, w += 17, y += 32) {   // block: e, then 16 bytes
        float d = g_e8m0_f[w[0]];
        for (int j = 0; j < 16; j++) {
            y[j]      = kv_mxfp4_f[w[1 + j] & 0xF] * d;
            y[j + 16] = kv_mxfp4_f[w[1 + j] >> 4]  * d;
        }
    }
}
static void random_rows(int type, unsigned char *w, size_t bytes, unsigned s) {
    for (size_t i = 0; i < bytes; i++) { s = s * 1103515245u + 12345u; w[i] = (unsigned char)(s >> 16); }
    size_t bs = ggml_type_size(type);
    for (size_t b = 0; b + bs <= bytes; b += bs) {
        if (type == T_MXFP4) w[b] = (unsigned char)(117 + (b / bs) % 21);
        else { w[b] = 0x00; w[b + 1] = 0x2c; w[b + 2] = 0x00; w[b + 3] = 0x2c; }   // d, dmin 1/16
    }
}
static double decode_pass(int type, const unsigned char *w, float *y, int n, int rows) {
    size_t rs = ggml_row_size(type, n);
    double t0 = route_now();
    for (int r = 0; r < rows; r++) dequant_row(type, w + rs * r, y + (size_t)n * r, n);
    return route_now() - t0;
}
static void decode_ratio_mxfp4(void) {
    const int n = 4096, rows = 256;
    f16_init();   // quants.c's tables (once; a loaded model built them already)
    for (int e = 0; e < 256; e++) g_e8m0_f[e] = ldexpf(1.0f, e - 127);
    size_t rm = ggml_row_size(T_MXFP4, n) * rows, rk = ggml_row_size(T_Q4_K, n) * rows;
    unsigned char *wm = malloc(rm), *wk = malloc(rk);
    float *y = malloc(sizeof(float) * (size_t)n * rows);
    random_rows(T_MXFP4, wm, rm, 54321);
    random_rows(T_Q4_K, wk, rk, 12345);
    double best_m = 1e9, best_k = 1e9;
    for (int attempt = 0, rounds = 5; attempt < 2; attempt++, rounds = 25) {
        best_m = 1e9; best_k = 1e9;
        for (int rep = 0; rep < rounds; rep++) {
            double tm = decode_pass(T_MXFP4, wm, y, n, rows);
            double tk = decode_pass(T_Q4_K, wk, y, n, rows);
            if (tm < best_m) best_m = tm;
            if (tk < best_k) best_k = tk;
        }
        if (best_m <= 1.3 * best_k) break;
        if (attempt == 0)
            fprintf(stderr, "re-measuring: MXFP4 row decode at %.2fx Q4_K's in 5 rounds\n",
                    best_m / best_k);
    }
    double per = 1e9 / ((double)n * rows);
    char msg[220];
    snprintf(msg, sizeof msg, "MXFP4 row decode %.3f ns a weight, Q4_K's (an AVX2 decode) %.3f "
             "(%.2fx, at most 1.30x)", best_m * per, best_k * per, best_m / best_k);
    ck(best_m <= 1.3 * best_k, msg);
    // the C decode's bytes under this binary's float environment, every
    // exponent (0 is the subnormal scale denormals-are-zero flushes, 255 the
    // infinite one)
    float *z = malloc(sizeof(float) * (size_t)n);
    int bad = 0;
    for (int e = 0; e < 256; e++) {
        for (int k = 0; k < n / 32; k++) wm[(size_t)k * 17] = (unsigned char)e;
        dequant_row(T_MXFP4, wm, y, n);
        scalar_mxfp4_rows(wm, z, n / 32);
        if (memcmp(y, z, sizeof(float) * (size_t)n)) bad++;
    }
    snprintf(msg, sizeof msg, "MXFP4 row decode under the runner's float flags: the C decode's "
             "bytes at %d of 256 exponents", 256 - bad);
    ck(bad == 0, msg);
    free(z); free(wm); free(wk); free(y);
}
#endif

int main(int argc, char **argv) {
    dispatch(argc > 1 ? argv[1] : "test-q8.gguf");
    ratio(T_Q8_0, "Q8_0");
    ratio(T_Q4_K, "Q4_K");
    ratio(T_IQ3_S, "IQ3_S");
    ratio(T_IQ4_XS, "IQ4_XS");
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    decode_ratio_mxfp4();
#endif
    if (!g_fail) puts("dot route gate ok");
    return g_fail;
}
