// R4.48.15: Inkling 2 Mini's fast CPU route, on the tiny inkling2_loop files (tests/inkling2_tiny.py: the dense
// projections F32, BF16 and Q8_0; the experts NVFP4 with their per-expert .scale).
//
// The fast route keeps every weight in its file format and decodes it at use, keeps the activations float32 and
// sums in the house kernels' vectorized order. It is not the reference order, so its rows are close to the
// reference route's, not equal; the threads split outputs, never a sum, so its rows do not depend on the count.
//
//  (a) the route: fast by default; RUNNER_INKLING2_ROUTE=reference selects the reference
//  (b) on each file, the fast rows within 1e-4 of the reference rows (relative to the largest logit) and not equal
//      to them byte for byte (it is a different route)
//  (c) the fast rows byte-identical at one and four threads, with loops split over four (the count observed)
//  (d) the fast rows byte-identical at -b 64 and -b 8 (the prefill takes one order at every width)
//  (e) no float32 copy of any weight and no expert cache on the fast route (model_inkling2_fit)
//  (f) CONTROL: the route with its activations quantized to 8 bits (RUNNER_INKLING2_ACT8=1, test-only) is at least
//      ten times further from the reference than the fast route
//
// usage: test_inkling2_fast TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf
#include "model.h"
#include "gpu.h"
#include "inkling2.h"
#include "inkling2_model.h"
#include "quants.h"
#include "tpool.h"
#include "compat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOK = 24 };
static const int32_t PROMPT[N_TOK] = { 1, 5, 9, 2, 7, 3, 8, 4, 6, 0, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61 };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

// the route and the control are read at load, so each load sets them first (an empty value is unset)
static model_t *load(const char *path, const char *route, bool act8, int threads, int n_batch) {
    plat_setenv("RUNNER_INKLING2_ROUTE", route ? route : "");
    plat_setenv("RUNNER_INKLING2_ACT8", act8 ? "1" : "");
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = threads;
    p.n_ctx = 64;
    p.n_batch = n_batch;
    model_t *m = calloc(1, sizeof *m);
    if (!m || !model_load(m, path, &p)) { free(m); return NULL; }
    return m;
}
// every row of the prompt, [N_TOK][n_vocab], in calls of at most -b tokens (a call keeps at most -b rows), as the
// engine's prefill makes them
static float *rows(model_t *m) {
    float *r = malloc(sizeof(float) * (size_t)N_TOK * (size_t)m->n_vocab);
    for (int pos = 0; pos < N_TOK;) {
        const int n = N_TOK - pos < m->n_batch ? N_TOK - pos : m->n_batch;
        if (!model_inkling2_forward(m, PROMPT + pos, n, pos, true, true)) { free(r); return NULL; }
        for (int b = 0; b < n; b++)
            memcpy(r + (size_t)(pos + b) * (size_t)m->n_vocab, model_inkling2_row(m, b), sizeof(float) * (size_t)m->n_vocab);
        pos += n;
    }
    return r;
}
static void drop(model_t *m) { if (m) { model_free(m); free(m); } }
// the largest difference relative to the largest reference logit
static double rel_diff(const float *a, const float *ref, size_t n) {
    double d = 0, top = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabs((double)a[i] - ref[i]) > d) d = fabs((double)a[i] - ref[i]);
        if (fabs((double)ref[i]) > top) top = fabs((double)ref[i]);
    }
    return top > 0 ? d / top : d;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf\n", argv[0]); return 2; }
    plat_setenv("INK2_PAR_MIN", "1");

    // (a)
    model_t *d = load(argv[1], NULL, false, 1, 64);
    model_t *r = load(argv[1], "reference", false, 1, 64);
    CHECK(d && r, "(a) the tiny file loads on both routes");
    if (!d || !r) return 1;
    CHECK(model_inkling2_route(d) == 'F', "(a) the default route is %c, want F (fast)", model_inkling2_route(d));
    CHECK(model_inkling2_route(r) == 'R', "(a) RUNNER_INKLING2_ROUTE=reference gives %c, want R", model_inkling2_route(r));
    drop(d); drop(r);

    // (b) on each file
    const char *kind[3] = { "F32", "BF16", "Q8_0" };
    for (int f = 0; f < 3; f++) {
        model_t *fm = load(argv[1 + f], NULL, false, 1, 64), *rm = load(argv[1 + f], "reference", false, 1, 64);
        if (!fm || !rm) { CHECK(0, "(b) %s: the file loads", kind[f]); drop(fm); drop(rm); continue; }
        const size_t n = (size_t)N_TOK * (size_t)fm->n_vocab;
        float *a = rows(fm), *b = rows(rm);
        const double rd = a && b ? rel_diff(a, b, n) : 1;
        CHECK(a && b && rd <= 1e-4, "(b) %s dense: fast rows within %.2e of the reference rows, want <= 1e-4", kind[f], rd);
        CHECK(a && b && memcmp(a, b, sizeof(float) * n), "(b) %s dense: the fast rows are not the reference rows byte for byte",
              kind[f]);
        free(a); free(b); drop(fm); drop(rm);
    }

    // (c) threads, (d) widths, (e) resident, (f) the control, on the BF16 file (the exact file's dense type)
    model_t *t1 = load(argv[2], NULL, false, 1, 64), *t4 = load(argv[2], NULL, false, 1, 64);
    model_t *b8 = load(argv[2], NULL, false, 1, 8), *ref = load(argv[2], "reference", false, 1, 64);
    model_t *q8 = load(argv[2], NULL, true, 1, 64);
    if (!t1 || !t4 || !b8 || !ref || !q8) { printf("FAIL: the BF16 file does not load five times\n"); return 1; }
    tpool_destroy(t4->tp);
    t4->tp = tpool_create(4);   // as a server swaps a slot's pool
    const size_t n = (size_t)N_TOK * (size_t)t1->n_vocab;
    float *x1 = rows(t1);
    long s0, s1;
    int last;
    ink2_par_stats(&s0, NULL);
    float *x4 = rows(t4);
    ink2_par_stats(&s1, &last);
    CHECK(x1 && x4 && !memcmp(x1, x4, sizeof(float) * n), "(c) the fast rows are byte-identical at 1 and 4 threads");
    CHECK(s1 > s0 && last == 4, "(c) at 4 threads %ld loops split, the last over %d threads, want > 0 and 4", s1 - s0, last);
    float *x8 = rows(b8);
    CHECK(x1 && x8 && !memcmp(x1, x8, sizeof(float) * n), "(d) the fast rows are byte-identical at -b 64 and -b 8");
    inkling2_fit fit;
    CHECK(model_inkling2_fit(t1, &fit) && fit.dense_f32 == 0 && fit.expert_cache == 0,
          "(e) the fast route holds %llu bytes of float32 weights and %llu of expert cache, want 0 and 0",
          (unsigned long long)fit.dense_f32, (unsigned long long)fit.expert_cache);
    float *xr = rows(ref), *xq = rows(q8);
    const double fast_err = x1 && xr ? rel_diff(x1, xr, n) : 1, act8_err = xq && xr ? rel_diff(xq, xr, n) : 0;
    CHECK(act8_err >= 10 * fast_err && act8_err > 0,
          "(f) control: 8-bit activations %.2e from the reference, the fast route %.2e, want at least 10x", act8_err, fast_err);
    free(x1); free(x4); free(x8); free(xr); free(xq);
    // (g) last-row-only prefill (keep_all = false) runs the head once (on the final row of the final chunk) while
    // leaving keep_all / row-dump callers bit-identical, and want_logits = false skips the head altogether
    {
        model_t *p64 = load(argv[2], NULL, false, 1, 64), *p8 = load(argv[2], NULL, false, 1, 8),
                *p0 = load(argv[2], NULL, false, 1, 64), *rk = load(argv[2], NULL, false, 1, 64);
        float *krows = rk ? rows(rk) : NULL;
        const size_t nv = (size_t)t1->n_vocab;
        const long h0 = ink2_head_rows();
        float *last64 = p64 ? model_inkling2_forward(p64, PROMPT, N_TOK, 0, true, false) : NULL;
        const long h1 = ink2_head_rows();
        CHECK(last64 && h1 - h0 == 1, "(g) prefill asking for last row only (-b 64) runs the head %ld row(s), want 1", h1 - h0);
        const long h2 = ink2_head_rows();
        float *last8 = p8 ? model_inkling2_forward(p8, PROMPT, N_TOK, 0, true, false) : NULL;
        const long h3 = ink2_head_rows();
        CHECK(last8 && h3 - h2 == 1, "(g) prefill asking for last row only (-b 8) runs the head %ld row(s), want 1", h3 - h2);
        CHECK(last64 && krows && !memcmp(last64, krows + (size_t)(N_TOK - 1) * nv, sizeof(float) * nv),
              "(g) last-row-only prefill (-b 64) is byte-identical to keep_all final row");
        CHECK(last8 && krows && !memcmp(last8, krows + (size_t)(N_TOK - 1) * nv, sizeof(float) * nv),
              "(g) last-row-only prefill (-b 8) is byte-identical to keep_all final row");
        const long h4 = ink2_head_rows();
        float *no_lg = p0 ? model_inkling2_forward(p0, PROMPT, N_TOK, 0, false, false) : (float *)1;
        const long h5 = ink2_head_rows();
        CHECK(no_lg == NULL && h5 - h4 == 0, "(g) prefill with want_logits=false runs the head %ld row(s), want 0", h5 - h4);
        free(krows); drop(p64); drop(p8); drop(p0); drop(rk);
    }
    // (h) Tier C prefill dispatch: a T=64 prefill takes the exact wide-batch tile (>0 columns)
    {
        model_t *p64t = load(argv[2], NULL, false, 1, 64);
        int32_t prompt64[64];
        for (int i = 0; i < 64; i++) prompt64[i] = PROMPT[i % N_TOK];
        const unsigned long tc0 = vec_dot_exact_tile_cols();
        float *out64 = p64t ? model_inkling2_forward(p64t, prompt64, 64, 0, true, false) : NULL;
        const unsigned long tc1 = vec_dot_exact_tile_cols();
        CHECK(out64 && tc1 > tc0, "(h) T=64 BF16 prefill takes exact tile path (%lu cols, want > 0)", tc1 - tc0);
        drop(p64t);
    }
    drop(t1); drop(t4); drop(b8); drop(ref); drop(q8);
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
