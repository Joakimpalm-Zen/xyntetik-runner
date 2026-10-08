// The CPU dot route gate (2026-10-08). Byte-identity tests cannot see a
// route that is correct and slow: 1.1.1 sent the one-row decode through
// vec_dot_multi, every output stayed byte-equal, and dense IQ3_S decode fell
// from 1.45 to 0.83 tok/s until a hotfix. Two checks:
//  1. dispatch: a one-token forward never calls vec_dot_multi; a 3-row batch
//     does (the small-batch route exists and is used);
//  2. a same-process ratio canary: for 2..7 columns vec_dot_multi must not
//     cost more than 1.5x the per-column vec_dot loop it replaces. A ratio
//     on one machine in one process, not an absolute tok/s.
#include "model.h"
#include "quants.h"
#include "runner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
        double best_m = 1e9, best_l = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = route_now();
            for (int r = 0; r < rows; r++) { vec_dot_multi(type, w + rs * r, xs, nc, n, out); sink += out[0]; }
            double t1 = route_now();
            for (int r = 0; r < rows; r++) for (int c = 0; c < nc; c++) sink += vec_dot(type, w + rs * r, xs[c], n);
            double t2 = route_now();
            if (t1 - t0 < best_m) best_m = t1 - t0;
            if (t2 - t1 < best_l) best_l = t2 - t1;
        }
        char msg[160];
        snprintf(msg, sizeof msg, "%s, %d columns: multi %.3f ms, per-column loop %.3f ms (ratio %.2f, limit 1.50)",
                 name, nc, best_m * 1e3, best_l * 1e3, best_m / best_l);
        ck(best_m <= 1.5 * best_l, msg);
    }
    if (sink == 12345.678f) puts("");   // keep the work observable
    free(w); free(x);
}

int main(int argc, char **argv) {
    dispatch(argc > 1 ? argv[1] : "test-q8.gguf");
    ratio(T_Q8_0, "Q8_0");
    ratio(T_Q4_K, "Q4_K");
    ratio(T_IQ3_S, "IQ3_S");
    ratio(T_IQ4_XS, "IQ4_XS");
    if (!g_fail) puts("dot route gate ok");
    return g_fail;
}
