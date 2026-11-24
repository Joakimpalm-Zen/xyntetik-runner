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
static model_t *load_ctx(const char *path, const char *route, bool act8, int threads, int n_ctx, int n_batch) {
    plat_setenv("RUNNER_INKLING2_ROUTE", route ? route : "");
    plat_setenv("RUNNER_INKLING2_ACT8", act8 ? "1" : "");
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = threads;
    p.n_ctx = n_ctx;
    p.n_batch = n_batch;
    model_t *m = calloc(1, sizeof *m);
    if (!m || !model_load(m, path, &p)) { free(m); return NULL; }
    return m;
}
static model_t *load(const char *path, const char *route, bool act8, int threads, int n_batch) {
    return load_ctx(path, route, act8, threads, 64, n_batch);
}
// every row of tok[0..n_tok-1], in calls of at most -b tokens (a call keeps at most -b rows), as the engine's
// prefill makes them
static float *rows_n(model_t *m, const int32_t *tok, int n_tok) {
    float *r = malloc(sizeof(float) * (size_t)n_tok * (size_t)m->n_vocab);
    for (int pos = 0; pos < n_tok;) {
        const int n = n_tok - pos < m->n_batch ? n_tok - pos : m->n_batch;
        if (!model_inkling2_forward(m, tok + pos, n, pos, true, true)) { free(r); return NULL; }
        for (int b = 0; b < n; b++)
            memcpy(r + (size_t)(pos + b) * (size_t)m->n_vocab, model_inkling2_row(m, b), sizeof(float) * (size_t)m->n_vocab);
        pos += n;
    }
    return r;
}
static float *rows(model_t *m) { return rows_n(m, PROMPT, N_TOK); }
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
    // (h) Tier C prefill dispatch: a T=64 prefill takes the exact wide-batch tile (>0 columns) wherever the build has
    // one for BF16 at the model's width. A build without one leaves vec_dot_exact_tile_ok false and its prefill takes
    // the multi-column dot, so the check follows that capability and skips, saying why: a T3 build's canonical
    // kernels (RUNNER_CANON_KERNELS), or a build with no SIMD kernels, such as a sanitized build with no -march
    // (R4.48.20.4; both failed here before). A build compiled for the SIMD set quants.c keys its kernels on (AVX2 with
    // FMA and F16C, or aarch64 NEON) and not T3 must report the tile, so the skip cannot hide a tile that went missing.
    {
#if !defined(RUNNER_CANON_KERNELS) && ((defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) || \
                                       (defined(__aarch64__) && defined(__ARM_NEON)))
        const bool simd_build = true;
#else
        const bool simd_build = false;
#endif
        model_t *p64t = load(argv[2], NULL, false, 1, 64);
        const int hn = p64t ? p64t->n_embd : 0;
        if (!p64t) CHECK(0, "(h) the BF16 model did not load at -b 64");
        else if (!vec_dot_exact_tile_ok(T_BF16, hn)) {
            if (simd_build) CHECK(0, "(h) a SIMD build reports no BF16 exact tile at n=%d", hn);
#ifdef RUNNER_CANON_KERNELS
            const char *why = "a T3 build's canonical kernels have none";
#else
            const char *why = "no AVX2/FMA/F16C or NEON kernels in this build";
#endif
            printf("ok: (h) skipped: no BF16 exact tile at n=%d (%s), so the prefill takes vec_dot_multi\n", hn, why);
        } else {
#ifdef RUNNER_CANON_KERNELS
            CHECK(0, "(h) a T3 build reports an exact tile it should not have");
#endif
            int32_t prompt64[64];
            for (int i = 0; i < 64; i++) prompt64[i] = PROMPT[i % N_TOK];
            const unsigned long tc0 = vec_dot_exact_tile_cols();
            float *out64 = model_inkling2_forward(p64t, prompt64, 64, 0, true, false);
            const unsigned long tc1 = vec_dot_exact_tile_cols();
            CHECK(out64 && tc1 > tc0, "(h) T=64 BF16 prefill takes exact tile path (%lu cols, want > 0)", tc1 - tc0);
        }
        drop(p64t);
    }
    // (i) default chunk size 256 (clamped to n_ctx) and lazy r->rows allocation (only allocated on keep_all / dump)
    enum { N_SPLIT = 280 };
    int32_t split_tok[N_SPLIT];
    for (int i = 0; i < N_SPLIT; i++) split_tok[i] = PROMPT[i % N_TOK];
    {
        model_t *m384 = load_ctx(argv[2], NULL, false, 1, 384, 0);
        model_t *m128 = load_ctx(argv[2], NULL, false, 1, 128, 0);
        model_t *m64c = load_ctx(argv[2], NULL, false, 1, 64, 0);
        CHECK(m384 && m384->n_batch == 256, "(i) default n_batch at n_ctx=384 is %d, want 256", m384 ? m384->n_batch : -1);
        CHECK(m128 && m128->n_batch == 128, "(i) default n_batch clamped at n_ctx=128 is %d, want 128", m128 ? m128->n_batch : -1);
        CHECK(m64c && m64c->n_batch == 64, "(i) default n_batch clamped at n_ctx=64 is %d, want 64", m64c ? m64c->n_batch : -1);
        if (m384) {
            const int nv = m384->n_vocab;
            inkling2_fit f0, f1;
            float *last_lg = model_inkling2_forward(m384, split_tok, N_SPLIT, 0, true, false);
            bool ok0 = model_inkling2_fit(m384, &f0);
            CHECK(last_lg && ok0 && f0.logits == sizeof(float) * (uint64_t)nv,
                  "(i) non-keep_all forward keeps fit.logits=%llu (want %llu, no eager rows)",
                  (unsigned long long)f0.logits, (unsigned long long)(sizeof(float) * (uint64_t)nv));
            CHECK(model_inkling2_row(m384, 0) == NULL, "(i) model_inkling2_row returns NULL when rows not allocated");
            float *ka = model_inkling2_forward(m384, PROMPT, N_TOK, 0, true, true);
            bool ok1 = model_inkling2_fit(m384, &f1);
            CHECK(ka && ok1 && f1.logits == sizeof(float) * (uint64_t)nv * (uint64_t)(m384->n_batch + 1),
                  "(i) keep_all forward lazily allocates rows: fit.logits=%llu (want %llu)",
                  (unsigned long long)f1.logits, (unsigned long long)(sizeof(float) * (uint64_t)nv * (uint64_t)(m384->n_batch + 1)));
        }
        drop(m384); drop(m128); drop(m64c);
    }
    // (j) -b invariance across chunk split (N_TOK=280, n_ctx=384) on BOTH fast and reference routes, with control
    for (int ri = 0; ri < 2; ri++) {
        const char *route = ri == 0 ? NULL : "reference";
        const char *rname = ri == 0 ? "fast" : "reference";
        model_t *m256 = load_ctx(argv[2], route, false, ri == 0 ? 2 : 1, 384, 256);
        model_t *m64s = load_ctx(argv[2], route, false, ri == 0 ? 2 : 1, 384, 64);
        model_t *m8s  = load_ctx(argv[2], route, false, ri == 0 ? 2 : 1, 384, 8);
        const size_t nbytes = m256 ? (size_t)N_SPLIT * (size_t)m256->n_vocab * sizeof(float) : 0;
        float *r256 = m256 ? rows_n(m256, split_tok, N_SPLIT) : NULL;
        float *r64s = m64s ? rows_n(m64s, split_tok, N_SPLIT) : NULL;
        float *r8s  = m8s  ? rows_n(m8s, split_tok, N_SPLIT)  : NULL;
        CHECK(r256 && r64s && !memcmp(r256, r64s, nbytes), "(j) %s split -b 256 == -b 64 byte-identical (N=%d)", rname, N_SPLIT);
        CHECK(r256 && r8s  && !memcmp(r256, r8s, nbytes),  "(j) %s split -b 256 == -b 8 byte-identical (N=%d)",  rname, N_SPLIT);
        // non-vacuous control: perturb continuation chunk p0 > 0, so -b 256 (split at 256) != -b 64 (split at 64)
        ink2_set_test_chunk_bump(1e-2f);
        float *c256 = m256 ? rows_n(m256, split_tok, N_SPLIT) : NULL;
        float *c64s = m64s ? rows_n(m64s, split_tok, N_SPLIT) : NULL;
        ink2_set_test_chunk_bump(0.0f);
        CHECK(c256 && c64s && memcmp(c256, c64s, nbytes) != 0,
              "(j) %s split control: chunk-boundary perturbation makes -b 256 != -b 64", rname);
        free(r256); free(r64s); free(r8s); free(c256); free(c64s);
        drop(m256); drop(m64s); drop(m8s);
    }
    drop(t1); drop(t4); drop(b8); drop(ref); drop(q8);
    // (k) decode fusion + batched experts: <= half the decode barriers per T=1 step on 4T while staying byte-identical
    // across F32/BF16/Q8_0, 1T/4T, -b 256/64/8, and RUNNER_INK2_DECODE_FUSE=0
    for (int f = 0; f < 3; f++) {
        model_t *m_unfused = load_ctx(argv[1 + f], NULL, false, 4, 384, 64);
        model_t *m_f4 = load_ctx(argv[1 + f], NULL, false, 4, 384, 64);
        model_t *m_f1 = load_ctx(argv[1 + f], NULL, false, 1, 384, 64);
        if (!m_unfused || !m_f4 || !m_f1) { CHECK(0, "(k) %s load", kind[f]); drop(m_unfused); drop(m_f4); drop(m_f1); continue; }
        const size_t nv = (size_t)m_f4->n_vocab;
        float *u_dec = malloc(sizeof(float) * 8 * nv), *f4_dec = malloc(sizeof(float) * 8 * nv), *f1_dec = malloc(sizeof(float) * 8 * nv);
        plat_setenv("RUNNER_INK2_DECODE_FUSE", "0");
        bool ok_u = model_inkling2_forward(m_unfused, PROMPT, 8, 0, true, false) != NULL;
        long su0 = 0, su1 = 0, sf0 = 0, sf1 = 0;
        int lu = 0, lf = 0;
        for (int t = 0; t < 8 && ok_u; t++) {
            if (t == 0) ink2_par_stats(&su0, &lu);
            float *lg = model_inkling2_forward(m_unfused, &PROMPT[8 + t], 1, 8 + t, true, false);
            if (t == 0) ink2_par_stats(&su1, &lu);
            if (!lg) ok_u = false; else memcpy(u_dec + (size_t)t * nv, lg, sizeof(float) * nv);
        }
        plat_setenv("RUNNER_INK2_DECODE_FUSE", "");
        bool ok_f4 = model_inkling2_forward(m_f4, PROMPT, 8, 0, true, false) != NULL;
        for (int t = 0; t < 8 && ok_f4; t++) {
            if (t == 0) ink2_par_stats(&sf0, &lf);
            float *lg = model_inkling2_forward(m_f4, &PROMPT[8 + t], 1, 8 + t, true, false);
            if (t == 0) ink2_par_stats(&sf1, &lf);
            if (!lg) ok_f4 = false; else memcpy(f4_dec + (size_t)t * nv, lg, sizeof(float) * nv);
        }
        bool ok_f1 = model_inkling2_forward(m_f1, PROMPT, 8, 0, true, false) != NULL;
        for (int t = 0; t < 8 && ok_f1; t++) {
            float *lg = model_inkling2_forward(m_f1, &PROMPT[8 + t], 1, 8 + t, true, false);
            if (!lg) ok_f1 = false; else memcpy(f1_dec + (size_t)t * nv, lg, sizeof(float) * nv);
        }
        const long du = su1 - su0, df = sf1 - sf0;
        CHECK(ok_u && ok_f4 && ok_f1 && df > 0 && lf == 4 && df * 2 <= du,
              "(k) %s decode splits %ld barriers (unfused %ld, want >0, last_pool=4, <=50%% of unfused)", kind[f], df, du);
        CHECK(ok_u && ok_f4 && !memcmp(f4_dec, u_dec, sizeof(float) * 8 * nv),
              "(k) %s decode fused (4T) == RUNNER_INK2_DECODE_FUSE=0 byte-identical", kind[f]);
        CHECK(ok_f4 && ok_f1 && !memcmp(f4_dec, f1_dec, sizeof(float) * 8 * nv),
              "(k) %s decode fused (4T) == fused (1T) byte-identical", kind[f]);
        free(u_dec); free(f4_dec); free(f1_dec);
        drop(m_unfused); drop(m_f4); drop(m_f1);
    }
    // Verify -b 256 / -b 64 / -b 8 prefill (272 tokens) + 8 fused decode steps stay byte-identical on BF16
    {
        const int bs[3] = { 256, 64, 8 };
        float *bdec[3] = { NULL, NULL, NULL };
        size_t nv = 0;
        bool ok_b[3] = { true, true, true };
        for (int bi = 0; bi < 3; bi++) {
            model_t *mb = load_ctx(argv[2], NULL, false, 4, 384, bs[bi]);
            if (!mb) { ok_b[bi] = false; continue; }
            nv = (size_t)mb->n_vocab;
            bdec[bi] = malloc(sizeof(float) * 8 * nv);
            for (int p0 = 0; p0 < 272 && ok_b[bi]; p0 += bs[bi]) {
                int nc = 272 - p0; if (nc > bs[bi]) nc = bs[bi];
                float *ret = model_inkling2_forward(mb, &split_tok[p0], nc, p0, nc == 272 - p0, false);
                if (nc == 272 - p0 && !ret) ok_b[bi] = false;
            }
            for (int t = 0; t < 8 && ok_b[bi]; t++) {
                float *lg = model_inkling2_forward(mb, &split_tok[272 + t], 1, 272 + t, true, false);
                if (!lg) ok_b[bi] = false; else memcpy(bdec[bi] + (size_t)t * nv, lg, sizeof(float) * nv);
            }
            drop(mb);
        }
        CHECK(ok_b[0] && ok_b[1] && !memcmp(bdec[0], bdec[1], sizeof(float) * 8 * nv),
              "(k) BF16 272p+8d fused decode: -b 256 == -b 64 byte-identical");
        CHECK(ok_b[0] && ok_b[2] && !memcmp(bdec[0], bdec[2], sizeof(float) * 8 * nv),
              "(k) BF16 272p+8d fused decode: -b 256 == -b 8 byte-identical");
        for (int bi = 0; bi < 3; bi++) free(bdec[bi]);
    }
    // (l) R4.48.20.4: the MoE layer allocates nothing a call. It allocated its routing tables (T x E bytes and T x E
    // floats: 320 KB a layer at 512 positions and 128 experts), its scores and the expert body's position lists on
    // every call, past glibc's 128 KB mmap threshold and the size Windows' heap returns to the system on free. On each
    // file at 1 and 4 threads and -b 1, 8, 64 and 256, a prefill after the first at the same width allocates nothing:
    // the tables live in the instance's scratch.
    for (int f = 0; f < 3; f++)
        for (int th = 1; th <= 4; th += 3) {
            static const int BS[4] = { 1, 8, 64, 256 };
            long extra[4] = { -1, -1, -1, -1 };
            for (int bi = 0; bi < 4; bi++) {
                model_t *m = load_ctx(argv[1 + f], NULL, false, th, 384, BS[bi]);
                const int nt = BS[bi] == 1 ? N_TOK : N_SPLIT;
                float *r0 = m ? rows_n(m, split_tok, nt) : NULL;
                const long a0 = ink2_moe_allocs();
                float *r1 = r0 ? rows_n(m, split_tok, nt) : NULL, *r2 = r1 ? rows_n(m, split_tok, nt) : NULL;
                if (r2) extra[bi] = ink2_moe_allocs() - a0;
                free(r0); free(r1); free(r2); drop(m);
            }
            CHECK(extra[0] == 0 && extra[1] == 0 && extra[2] == 0 && extra[3] == 0,
                  "(l) %s, %d thread(s): two prefills after the first allocate in the MoE layer %ld / %ld / %ld / %ld "
                  "times at -b 1 / 8 / 64 / 256 (want 0 each)", kind[f], th, extra[0], extra[1], extra[2], extra[3]);
        }
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
