// R8.25.2 gate A1: full-weight training's dense gradient against central
// differences of the engine's own loss, on the F32 fixture.
//
// The read copy of an F32 model is the master itself, so perturbing the
// tensor's bytes perturbs the trained weight. Three directional derivatives:
// a seeded random direction over every trained matrix, a second seed, and one
// matrix alone (the last layer's ffn_down). Each must agree with <dL/dW, V> to
// a relative 1e-3 at the best step of a sweep (see directional() for why one
// fixed step is not a portable grade). The dense gradient's DETERMINISM is pinned too (two
// backwards, byte-identical gradients and a bit-equal loss), that the forward
// reads the master itself after a step (what the variant ships), and that a
// Q8_0 file is refused (no Q8 read copy: R8.25.2.2's Gate B found it no better
// than quantizing the float variant afterwards).
//
//     ./test-full-grad test.gguf test-q8.gguf
#include "runner.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

enum { T = 14 };
static const int SLOTS[] = { LW_Q, LW_K, LW_V, LW_O, LW_GATE, LW_UP, LW_DOWN };
enum { NS = 7 };

static gguf_tensor *site(model_t *m, int l, int s) {
    layer_t *ly = &m->layers[l];
    switch (s) {
        case LW_Q: return ly->wq;
        case LW_K: return ly->wk;
        case LW_V: return ly->wv;
        case LW_O: return ly->wo;
        case LW_GATE: return ly->w_gate;
        case LW_UP: return ly->w_up;
        case LW_DOWN: return ly->w_down;
    }
    return NULL;
}

static double loss_of(model_t *m, const int32_t *toks) {
    double loss = 0;
    model_full_grad_zero(m);
    if (!model_lora_backward(m, toks, T, &loss)) CHECK(0, "the backward failed");
    return loss;
}

static uint64_t g_lcg;
static float urand(void) {   // uniform in [-1, 1)
    g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((g_lcg >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
}

static bool g_kv_f32 = false, g_grade = false;
static model_t load(const char *path) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = 64;
    p.n_batch = 32;
    p.kv_f32 = g_kv_f32;
    model_t m;
    memset(&m, 0, sizeof(m));
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", path);
        exit(1);
    }
    return m;
}

// <G, V> and the central difference of the loss along V, for V over the
// sites `only` selects (-1: all).
//
// A central difference at step eps carries two errors: truncation, which falls
// as eps^2, and the f32 forward's rounding, which is a roughly fixed absolute
// error in the loss divided by eps. Measured at eps 1e-3 over each matrix alone
// (R6.28, 2026-11-09), that rounding term is 3e-4 to 4e-3 in absolute terms
// whatever the matrix, so the RELATIVE error is that floor over |<G,V>|: an
// attention matrix (|<G,V>| near 0.07) reads 5e-3, the last ffn_down (0.75)
// read 3.7e-4 with gcc 13.3 on Linux, 4.6e-4 on macOS and 1.10e-3 with MinGW
// gcc on Windows, where the reduction order and contraction differ. A fixed
// eps with a fixed relative bound therefore graded the platform's rounding,
// not the gradient. The graded directions sweep the step instead, from where
// truncation shows (1e-2) down into the rounding regime (1e-4), and grade at
// the step where the central difference is most STABLE: the smallest change to
// the next step down, where truncation and rounding balance. That choice reads
// the finite differences only, never <G,V>, so a wrong gradient cannot pick the
// step that hides it. (The best agreement over the sweep would: a 0.2% bias on
// the last ffn_down's gradient met the rounding noise at eps 3e-4 and read
// 1.9e-4.) The stability there is the measured noise floor: it must resolve
// half of A1's 1e-3, and the agreement at that step must be within 1e-3.
static void directional(model_t *m, const int32_t *toks, float **g, uint64_t seed,
                        int only_layer, int only_slot, const char *what) {
    float *v[64][NS];
    float *w0[64][NS];
    double d = 0;
    g_lcg = seed;
    for (int l = 0; l < m->n_layer; l++)
        for (int k = 0; k < NS; k++) {
            v[l][k] = NULL;
            w0[l][k] = NULL;
            gguf_tensor *t = site(m, l, SLOTS[k]);
            if (!t || (only_layer >= 0 && (l != only_layer || SLOTS[k] != only_slot))) continue;
            size_t ne = (size_t)t->ne[0] * (size_t)t->ne[1];
            v[l][k] = malloc(sizeof(float) * ne);
            w0[l][k] = malloc(sizeof(float) * ne);
            memcpy(w0[l][k], t->data, sizeof(float) * ne);
            for (size_t i = 0; i < ne; i++) {
                v[l][k][i] = urand();
                d += (double)g[l * NS + k][i] * v[l][k][i];
            }
        }
    const char *scan = getenv("RUNNER_TEST_FD_SCAN");
    bool graded = g_kv_f32 && g_grade;
    // eps_list[0] is the fixed step of the printed reading; a graded direction
    // (or RUNNER_TEST_FD_SCAN) takes the whole sweep
    float eps_list[] = { 1e-3f, 1e-2f, 3e-3f, 3e-4f, 1e-4f, 3e-5f };
    int n_eps = scan ? 6 : graded ? 5 : 1;
    double fd_at[6] = { 0 };
    for (int e = 0; e < n_eps; e++) {
    const float eps = eps_list[e];
    double lp = 0, lm = 0;
    for (int sgn = 1; sgn >= -1; sgn -= 2) {
        for (int l = 0; l < m->n_layer; l++)
            for (int k = 0; k < NS; k++) {
                if (!v[l][k]) continue;
                gguf_tensor *t = site(m, l, SLOTS[k]);
                float *w = t->data;
                size_t ne = (size_t)t->ne[0] * (size_t)t->ne[1];
                for (size_t i = 0; i < ne; i++) w[i] = w0[l][k][i] + (float)sgn * eps * v[l][k][i];
            }
        double L = loss_of(m, toks);
        if (sgn > 0) lp = L; else lm = L;
    }
    fd_at[e] = (lp - lm) / (2.0 * eps);
    if (scan) printf("full-grad: scan %s: eps %.0e, relative %.2e\n", what, eps,
                     fabs(fd_at[e] - d) / fabs(d));
    }
    for (int l = 0; l < m->n_layer; l++)
        for (int k = 0; k < NS; k++) {
            if (!v[l][k]) continue;
            gguf_tensor *t = site(m, l, SLOTS[k]);
            memcpy(t->data, w0[l][k], sizeof(float) * (size_t)t->ne[0] * (size_t)t->ne[1]);
            free(v[l][k]);
            free(w0[l][k]);
        }
    double ad = fabs(d) > 1e-12 ? fabs(d) : 1e-12;
    double fd = fd_at[0];
    double rel = fabs(fd - d) / ad;
    printf("full-grad: reading, %s: <G,V> %.6g, central difference %.6g at eps %.0e, relative %.2e "
           "(absolute %.2e)\n", what, d, fd, (double)eps_list[0], rel, fabs(fd - d));
    // with an f32 KV cache (--train-kv-f32) there is no f16 staircase, and the
    // declared A1 (1e-3 along a random direction) is graded at the best step
    if (graded) {
        // the sweep from the largest step down: 1e-2, 3e-3, 1e-3, 3e-4, 1e-4
        static const int order[5] = { 1, 2, 0, 3, 4 };
        int pick = -1;
        double stab = 1e30;
        for (int j = 0; j + 1 < 5; j++) {
            double a = fd_at[order[j]], b = fd_at[order[j + 1]];
            double s = fabs(a - b) / (fabs(a) > 1e-12 ? fabs(a) : 1e-12);
            if (s < stab) { stab = s; pick = order[j]; }
        }
        double r = fabs(fd_at[pick] - d) / ad;
        printf("full-grad: %s: most stable at step %.0e (the next step down moves it %.2e), "
               "agreement there %.2e\n", what, (double)eps_list[pick], stab, r);
        CHECK(stab <= 5e-4, "%s: no step of the sweep resolves A1's 1e-3 (most stable %.2e at eps %.0e)",
              what, stab, (double)eps_list[pick]);
        CHECK(r <= 1e-3, "%s: the directional derivative misses (relative %.2e at eps %.0e, the sweep's "
              "most stable step)", what, r, (double)eps_list[pick]);
    }
}

// The graded check, the house method of tests/test_lora_grad.c: the f16 KV
// cache staircases a finite difference along an attention weight, so the
// direction is sign(G) over the selected coordinates and the best of six
// steps must sit within 1%; a wrong jacobian term is a bias no step removes.
static void sign_directional(model_t *m, const int32_t *toks, float **g, int only_layer,
                             int only_slot, const char *what) {
    double expect = 0;
    for (int l = 0; l < m->n_layer; l++)
        for (int k = 0; k < NS; k++) {
            if (!g[l * NS + k] || (only_layer >= 0 && (l != only_layer || SLOTS[k] != only_slot))) continue;
            gguf_tensor *t = site(m, l, SLOTS[k]);
            size_t ne = (size_t)t->ne[0] * (size_t)t->ne[1];
            for (size_t i = 0; i < ne; i++) expect += fabs((double)g[l * NS + k][i]);
        }
    const float steps[6] = { 5e-6f, 1e-5f, 2e-5f, 5e-5f, 1e-4f, 2e-4f };
    double best = 1e30;
    float best_e = 0;
    for (int si = 0; si < 6; si++) {
        float e = steps[si];
        double lv[2];
        for (int sg = 0; sg < 2; sg++) {
            float sign = sg ? -1.0f : 1.0f;
            for (int l = 0; l < m->n_layer; l++)
                for (int k = 0; k < NS; k++) {
                    if (!g[l * NS + k] || (only_layer >= 0 && (l != only_layer || SLOTS[k] != only_slot))) continue;
                    gguf_tensor *t = site(m, l, SLOTS[k]);
                    float *w = t->data;
                    size_t ne = (size_t)t->ne[0] * (size_t)t->ne[1];
                    for (size_t i = 0; i < ne; i++)
                        if (g[l * NS + k][i] != 0.0f) w[i] += sign * e * (g[l * NS + k][i] > 0 ? 1.0f : -1.0f);
                }
            lv[sg] = loss_of(m, toks);
            for (int l = 0; l < m->n_layer; l++)
                for (int k = 0; k < NS; k++) {
                    if (!g[l * NS + k] || (only_layer >= 0 && (l != only_layer || SLOTS[k] != only_slot))) continue;
                    gguf_tensor *t = site(m, l, SLOTS[k]);
                    float *w = t->data;
                    size_t ne = (size_t)t->ne[0] * (size_t)t->ne[1];
                    for (size_t i = 0; i < ne; i++)
                        if (g[l * NS + k][i] != 0.0f) w[i] -= sign * e * (g[l * NS + k][i] > 0 ? 1.0f : -1.0f);
                }
        }
        double slope = (lv[0] - lv[1]) / (2.0 * e);
        double rel = expect > 0 ? fabs(slope - expect) / expect : 0;
        if (rel < best) { best = rel; best_e = e; }
    }
    printf("full-grad: %s: along sign(G), sum|G| %.6g, best agreement %.2e at step %.0e\n", what, expect,
           best, (double)best_e);
    CHECK(best <= 0.01, "%s: the directional derivative's best agreement %.3g is outside 1%%", what, best);
}

int main(int argc, char **argv) {
    const char *base = argc > 1 ? argv[1] : "test.gguf";
    const char *q8 = argc > 2 ? argv[2] : "test-q8.gguf";
    f16_init();
    model_t m = load(base);
    char why[256] = "";
    if (!model_full_train_init(&m, why, sizeof why)) {
        fprintf(stderr, "FAIL: model_full_train_init: %s\n", why);
        return 1;
    }
    CHECK(m.n_layer <= 64, "fixture too deep");
    int32_t toks[T];
    for (int i = 0; i < T; i++) toks[i] = 3 + (i * 7) % (m.n_vocab - 4);

    // the analytic gradient, twice: byte-identical
    double l0 = loss_of(&m, toks);
    float *g[64 * NS];
    for (int l = 0; l < m.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            int ni = 0, no = 0;
            const float *gr = model_full_grad(&m, l, SLOTS[k], &ni, &no);
            g[l * NS + k] = NULL;
            if (!gr) continue;
            g[l * NS + k] = malloc(sizeof(float) * (size_t)ni * no);
            memcpy(g[l * NS + k], gr, sizeof(float) * (size_t)ni * no);
        }
    double l1 = loss_of(&m, toks);
    CHECK(l0 == l1, "the loss is not bit-equal across two backwards (%.17g, %.17g)", l0, l1);
    int same = 1, nonzero = 0;
    for (int l = 0; l < m.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            int ni = 0, no = 0;
            const float *gr = model_full_grad(&m, l, SLOTS[k], &ni, &no);
            if (!gr) continue;
            if (memcmp(gr, g[l * NS + k], sizeof(float) * (size_t)ni * no)) same = 0;
            for (size_t i = 0; i < (size_t)ni * no; i++) nonzero += gr[i] != 0.0f;
        }
    CHECK(same, "the dense gradient is not byte-identical across two backwards");
    CHECK(nonzero > 0, "the dense gradient is all zero");
    printf("full-grad: loss %.6f, the gradient byte-identical across two backwards\n", l0);

    directional(&m, toks, g, 0x9E3779B97F4A7C15ull, -1, -1, "every matrix, seed 1");
    directional(&m, toks, g, 0xD1B54A32D192ED03ull, -1, -1, "every matrix, seed 2");
    directional(&m, toks, g, 0x2545F4914F6CDD1Dull, m.n_layer - 1, LW_DOWN, "the last layer's ffn_down alone");
    if (getenv("RUNNER_TEST_FD_EACH"))
        for (int l = 0; l < m.n_layer; l++)
            for (int k = 0; k < NS; k++) {
                char w[64];
                snprintf(w, sizeof w, "layer %d slot %d alone", l, SLOTS[k]);
                directional(&m, toks, g, 0x51ED2701ull + 97u * (uint64_t)(l * NS + k), l, SLOTS[k], w);
            }
    // the graded checks: every matrix together, then each alone
    sign_directional(&m, toks, g, -1, -1, "every matrix");
    for (int l = 0; l < m.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            char w[64];
            snprintf(w, sizeof w, "layer %d slot %d", l, SLOTS[k]);
            sign_directional(&m, toks, g, l, SLOTS[k], w);
        }
    for (int i = 0; i < m.n_layer * NS; i++) free(g[i]);
    model_free(&m);

    // the same gradient with an f32 KV cache (--train-kv-f32): the random
    // directions graded at the declared 1e-3, each matrix alone too
    g_kv_f32 = true;
    model_t mf = load(base);
    CHECK(mf.kv_f32, "the f32 KV cache was not taken");
    if (!model_full_train_init(&mf, why, sizeof why)) {
        fprintf(stderr, "FAIL: model_full_train_init (f32 KV): %s\n", why);
        return 1;
    }
    loss_of(&mf, toks);
    float *gf[64 * NS];
    for (int l = 0; l < mf.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            int ni = 0, no = 0;
            const float *gr = model_full_grad(&mf, l, SLOTS[k], &ni, &no);
            gf[l * NS + k] = NULL;
            if (!gr) continue;
            gf[l * NS + k] = malloc(sizeof(float) * (size_t)ni * no);
            memcpy(gf[l * NS + k], gr, sizeof(float) * (size_t)ni * no);
        }
    // A1: two random directions over every matrix and the last layer's
    // ffn_down alone, within 1e-3 at the best step of the sweep
    g_grade = true;
    directional(&mf, toks, gf, 0x9E3779B97F4A7C15ull, -1, -1, "f32 KV, every matrix, seed 1");
    directional(&mf, toks, gf, 0xD1B54A32D192ED03ull, -1, -1, "f32 KV, every matrix, seed 2");
    directional(&mf, toks, gf, 0x2545F4914F6CDD1Dull, mf.n_layer - 1, LW_DOWN, "f32 KV, the last layer's ffn_down alone");
    g_grade = false;
    // each matrix alone: a reading (an attention matrix's small derivative sits
    // in the f32 forward's rounding noise at one eps; the sign check grades it)
    for (int l = 0; l < mf.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            char w[64];
            snprintf(w, sizeof w, "f32 KV, layer %d slot %d alone", l, SLOTS[k]);
            directional(&mf, toks, gf, 0x51ED2701ull + 97u * (uint64_t)(l * NS + k), l, SLOTS[k], w);
        }
    sign_directional(&mf, toks, gf, -1, -1, "f32 KV, every matrix");
    for (int l = 0; l < mf.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            char w[64];
            snprintf(w, sizeof w, "f32 KV, layer %d slot %d", l, SLOTS[k]);
            sign_directional(&mf, toks, gf, l, SLOTS[k], w);
        }
    for (int i = 0; i < mf.n_layer * NS; i++) free(gf[i]);
    model_free(&mf);
    g_kv_f32 = false;

    // after a step the forward reads the master itself, which the variant ships
    model_t ms = load(base);
    if (!model_full_train_init(&ms, why, sizeof why)) {
        fprintf(stderr, "FAIL: model_full_train_init (step): %s\n", why);
        return 1;
    }
    uint64_t params = 0;
    int nt = model_full_count(&ms, &params);
    loss_of(&ms, toks);
    CHECK(model_full_adam_step(&ms, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 1), "the step failed");
    int reads = 0;
    for (int l = 0; l < ms.n_layer; l++)
        for (int k = 0; k < NS; k++) {
            gguf_tensor *t = site(&ms, l, SLOTS[k]);
            const float *w = model_full_master(&ms, l, SLOTS[k]);
            if (t && w && t->data == (const void *)w) reads++;
        }
    CHECK(reads == nt, "%d of %d trained tensors read their master", reads, nt);
    printf("full-grad: the forward reads the master after a step (%d matrices, %llu parameters)\n",
           nt, (unsigned long long)params);
    model_free(&ms);

    // a Q8_0 file is refused, and the refusal names the float route
    model_t mq = load(q8);
    CHECK(!model_full_train_init(&mq, why, sizeof why), "a Q8_0 file was taken");
    CHECK(strstr(why, "is not f32") && strstr(why, "--rewrite --quant"), "the refusal reads: %s", why);
    CHECK(!model_full_on(&mq), "a refused init left training state");
    printf("full-grad: a q8_0 file is refused (%s)\n", why);
    model_free(&mq);
    if (!g_fail) printf("full-grad: ok\n");
    return g_fail;
}

/* R6.29 CI selection probe: no test behavior change. */
