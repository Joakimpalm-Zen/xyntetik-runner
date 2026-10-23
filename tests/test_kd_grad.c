// R8.15 gate: the distillation objective (model_lora_backward_kd) and its
// gradient, before any training time is spent on it.
//
//  1. ONE-HOT IS CROSS-ENTROPY. A teacher with one entry, log-probability 0,
//     on the next token is the weighted cross-entropy: the same loss and the
//     same gradient buffers, bit for bit, under a prompt mask, a fractional
//     weight and a negative one.
//
//  2. LOSS vs an independent formula. The KL is recomputed here from the
//     model's own logits in double, by a different expression than model.c
//     uses (log-ratios against a double log-softmax, the student's tail as
//     a masked sum of its probabilities), for a top-k teacher with a tail
//     and for a teacher that lists the whole vocabulary (no tail).
//
//  3. DETERMINISM: two backward calls, bit-equal loss and byte-identical
//     gradient buffers.
//
//  4. CENTRAL DIFFERENCES of the actual loss: per coordinate on the largest
//     gradients, and the directional derivative over the whole adapter (the
//     arbiter, as in test_lora_grad.c: the f16 KV staircase averages out
//     along a direction that moves thousands of parameters).
//
//     ./test-kd-grad test.gguf test-lora.full.gguf
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

// a prompt of P tokens, then the completion the teacher scores
enum { P = 4, T = 14, K = 5, MAX_BUF = 64 };

static float *snap[MAX_BUF];
static int snap_n[MAX_BUF], lay[MAX_BUF], slo[MAX_BUF], whi[MAX_BUF], nsnap;

static void snapshot(model_t *m) {
    for (int i = 0; i < nsnap; i++) free(snap[i]);
    nsnap = 0;
    for (int l = 0; l < m->n_layer; l++)
        for (int s = 0; s < LW_SLOTS; s++)
            for (int w = 0; w < 2; w++) {
                int cnt = 0;
                float *g = model_lora_gradbuf(m, l, s, w, &cnt);
                if (!g || nsnap >= MAX_BUF) continue;
                snap[nsnap] = malloc(sizeof(float) * (size_t)cnt);
                memcpy(snap[nsnap], g, sizeof(float) * (size_t)cnt);
                snap_n[nsnap] = cnt;
                lay[nsnap] = l; slo[nsnap] = s; whi[nsnap] = w;
                nsnap++;
            }
}

static bool grads_equal_snapshot(model_t *m) {
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *g = model_lora_gradbuf(m, lay[i], slo[i], whi[i], &cnt);
        if (!g || cnt != snap_n[i] ||
            memcmp(g, snap[i], sizeof(float) * (size_t)cnt) != 0) return false;
    }
    return true;
}

static double kd(model_t *m, const int32_t *toks, const float *w,
                 const lora_teacher *tch) {
    double loss = 0;
    model_lora_grad_zero(m);
    if (!model_lora_backward_kd(m, toks, T, w, tch, &loss)) {
        CHECK(0, "model_lora_backward_kd failed");
        return 0;
    }
    return loss;
}

static uint64_t rs = 0x2545F4914F6CDD1Dull;
static double urand(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double)(rs >> 11) * (1.0 / 9007199254740992.0);
}

// A top-k teacher: the next token first, k-1 other distinct ids, random
// probabilities scaled to `mass` (the tail holds 1 - mass).
static void soft_teacher(const model_t *m, const int32_t *toks, int k,
                         double mass, int32_t *ids, float *lp) {
    for (int t = 0; t + 1 < T; t++) {
        int32_t *row = ids + (size_t)t * k;
        float *lrow = lp + (size_t)t * k;
        row[0] = toks[t + 1];
        for (int i = 1; i < k; i++) {
            bool dup;
            do {
                row[i] = (int32_t)(urand() * m->n_vocab);
                dup = false;
                for (int j = 0; j < i; j++) dup |= row[j] == row[i];
            } while (dup || row[i] >= m->n_vocab);
        }
        double wsum = 0, wts[KD_K_MAX];
        for (int i = 0; i < k; i++) { wts[i] = 0.05 + urand(); wsum += wts[i]; }
        for (int i = 0; i < k; i++) lrow[i] = (float)log(mass * wts[i] / wsum);
    }
}

// A teacher anchored on the student: its top-k tokens at every transition,
// holding `mass` in proportions perturbed from the student's own, so the tail
// holds 1 - mass while the student's tail is small and uneven. On a random
// fixture a teacher over random ids leaves the student's tail nearly flat,
// where the tail term's gradient averages out (a mutation dropping it passed
// the central differences below at 0.26% on that teacher); here the tail
// term carries most of the gradient.
static void anchored_teacher(model_t *m, const int32_t *toks, int k,
                             double mass, int32_t *ids, float *lp) {
    int V = m->n_vocab;
    for (int t = 0; t < T; t++) {
        float *lg = model_forward(m, toks[t], t);
        if (!lg) { CHECK(0, "forward failed"); return; }
        if (t + 1 >= T) break;
        int32_t *row = ids + (size_t)t * k;
        float *lrow = lp + (size_t)t * k;
        for (int i = 0; i < k; i++) {
            int best = -1;
            for (int j = 0; j < V; j++) {
                bool taken = false;
                for (int c = 0; c < i; c++) taken |= row[c] == j;
                if (!taken && (best < 0 || lg[j] > lg[best])) best = j;
            }
            row[i] = best;
        }
        double wts[KD_K_MAX], wsum = 0;
        for (int i = 0; i < k; i++) {
            wts[i] = exp((double)lg[row[i]] - (double)lg[row[0]]) * (0.5 + urand());
            wsum += wts[i];
        }
        for (int i = 0; i < k; i++) lrow[i] = (float)log(mass * wts[i] / wsum);
    }
}

// The bound on the gap to the independent formula: the listed terms take
// log q against the float normalizer the cross-entropy uses (mx + logf of a
// float sum, rounded at about 6e-8 of a log-sum-exp near 5 nats), so the gap
// grows with the weight summed over the window: 1.3e-6 on the full-vocabulary
// teacher on Linux x86-64, 0.96e-6 on arm64. A wrong formula is off by 1e-2
// and more.
static double kd_tol(const float *w) {
    double sw = 0;
    for (int t = 0; t + 1 < T; t++) sw += w ? fabs((double)w[t]) : 1.0;
    return 3e-6 * sw;
}

// The KL written independently: a double log-softmax of the model's logits,
// log-ratios, and the student's tail as the masked sum of its probabilities.
static double reference_kl(model_t *m, const int32_t *toks, const float *w,
                           const lora_teacher *tch) {
    int V = m->n_vocab;
    double total = 0;
    char *listed = malloc((size_t)V);
    double *lq = malloc(sizeof(double) * (size_t)V);
    for (int t = 0; t < T; t++) {
        float *lg = model_forward(m, toks[t], t);
        if (!lg) { CHECK(0, "forward failed"); break; }
        if (t + 1 >= T || (w && w[t] == 0.0f)) continue;
        double mx = lg[0], s = 0;
        for (int j = 1; j < V; j++) if (lg[j] > mx) mx = lg[j];
        for (int j = 0; j < V; j++) s += exp((double)lg[j] - mx);
        double lse = mx + log(s);
        for (int j = 0; j < V; j++) lq[j] = (double)lg[j] - lse;
        memset(listed, 0, (size_t)V);
        const int32_t *ids = tch->ids + (size_t)t * tch->k;
        const float *lp = tch->logp + (size_t)t * tch->k;
        double pin = 0, kl = 0;
        for (int i = 0; i < tch->k; i++) {
            listed[ids[i]] = 1;
            double p = exp((double)lp[i]);
            pin += p;
            kl += p * log(p / exp(lq[ids[i]]));
        }
        double qtail = 0;
        for (int j = 0; j < V; j++) if (!listed[j]) qtail += exp(lq[j]);
        double ptail = 1.0 - pin;
        if (tch->k < V && ptail > 0) kl += ptail * log(ptail / qtail);
        total += (w ? (double)w[t] : 1.0) * kl;
    }
    free(listed);
    free(lq);
    return total;
}

// Central differences: the largest gradients one coordinate at a time, then
// the directional derivative along the gradient's sign over every coordinate
// above the mean magnitude (expected slope: the sum of those magnitudes).
static void fd_checks(model_t *m, const int32_t *toks, const float *w,
                      const lora_teacher *tch, const char *what) {
    double base = kd(m, toks, w, tch);
    snapshot(m);
    double l2 = kd(m, toks, w, tch);
    CHECK(base == l2, "%s: loss not deterministic: %.17g vs %.17g", what, base, l2);
    CHECK(grads_equal_snapshot(m), "%s: gradient buffers not byte-deterministic", what);

    // the eight largest gradient coordinates
    int top_b[8], top_k[8];
    double top_v[8];
    for (int s = 0; s < 8; s++) { top_b[s] = -1; top_k[s] = 0; top_v[s] = 0; }
    for (int i = 0; i < nsnap; i++)
        for (int k = 0; k < snap_n[i]; k++) {
            double a = fabs(snap[i][k]);
            if (a <= top_v[7]) continue;
            int s = 7;
            while (s > 0 && a > top_v[s - 1]) {
                top_v[s] = top_v[s - 1]; top_b[s] = top_b[s - 1]; top_k[s] = top_k[s - 1];
                s--;
            }
            top_v[s] = a; top_b[s] = i; top_k[s] = k;
        }
    double worst = 0;
    for (int s = 0; s < 8 && top_b[s] >= 0; s++) {
        int i = top_b[s], k = top_k[s], cnt = 0;
        float *theta = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
        float save = theta[k];
        double fd = 0, an = (double)snap[i][k];
        const float eps[2] = { 1e-2f, 8e-2f };
        for (int e = 0; e < 2; e++) {
            theta[k] = save + eps[e];
            double lp = kd(m, toks, w, tch);
            theta[k] = save - eps[e];
            double lm = kd(m, toks, w, tch);
            theta[k] = save;
            fd = (lp - lm) / (2.0 * (double)eps[e]);
            double mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
            if (fabs(fd - an) <= 0.05 * mag) break;
        }
        double mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
        double rel = mag > 0 ? fabs(fd - an) / mag : 0;
        if (rel > worst) worst = rel;
        CHECK(rel <= 0.05, "%s: layer %d slot %d %s[%d]: fd %.6g vs analytic %.6g (rel %.3g)",
              what, lay[i], slo[i], whi[i] ? "B" : "A", k, fd, an, rel);
    }

    double sum_abs = 0; long n_all = 0;
    for (int i = 0; i < nsnap; i++)
        for (int k = 0; k < snap_n[i]; k++) { sum_abs += fabs(snap[i][k]); n_all++; }
    double thresh = n_all ? sum_abs / (double)n_all : 0, expect = 0;
    long n_sel = 0;
    for (int i = 0; i < nsnap; i++)
        for (int k = 0; k < snap_n[i]; k++)
            if (fabs(snap[i][k]) > thresh) { expect += fabs(snap[i][k]); n_sel++; }
    const float steps[6] = { 5e-6f, 1e-5f, 2e-5f, 5e-5f, 1e-4f, 2e-4f };
    double best = 1.0;
    for (int si = 0; si < 6; si++) {
        double lv[2];
        for (int sg = 0; sg < 2; sg++) {
            float e = sg ? -steps[si] : steps[si];
            for (int i = 0; i < nsnap; i++) {
                int cnt = 0;
                float *theta = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
                for (int k = 0; k < cnt; k++)
                    if (fabs(snap[i][k]) > thresh)
                        theta[k] += e * (snap[i][k] > 0 ? 1.0f : -1.0f);
            }
            lv[sg] = kd(m, toks, w, tch);
            for (int i = 0; i < nsnap; i++) {
                int cnt = 0;
                float *theta = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
                for (int k = 0; k < cnt; k++)
                    if (fabs(snap[i][k]) > thresh)
                        theta[k] -= e * (snap[i][k] > 0 ? 1.0f : -1.0f);
            }
        }
        double slope = (lv[0] - lv[1]) / (2.0 * (double)steps[si]);
        double rel = expect > 0 ? fabs(slope - expect) / expect : 0;
        if (rel < best) best = rel;
    }
    CHECK(best <= 0.01, "%s: directional derivative: best agreement %.4f is outside 1%%",
          what, best);
    printf("ok: %s: loss %.6f, deterministic; top-8 coordinates worst rel %.4f; "
           "directional best rel %.5f over %ld coords\n", what, base, worst, best, n_sel);
}

int main(int argc, char **argv) {
    const char *base = argc > 1 ? argv[1] : "test.gguf";
    const char *adapter = argc > 2 ? argv[2] : "test-lora.full.gguf";
    f16_init();
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = 64;
    p.n_batch = 32;
    model_t m;
    memset(&m, 0, sizeof(m));
    if (!model_load(&m, base, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", base);
        return 1;
    }
    if (!model_lora_load(&m, adapter, 1.0f)) {
        fprintf(stderr, "FAIL: cannot load adapter %s\n", adapter);
        return 1;
    }
    int V = m.n_vocab;
    int32_t toks[T];
    for (int i = 0; i < T; i++) toks[i] = 3 + (i * 7) % (V - 4);
    // the prompt masked, one fractional weight, one negative one
    float w[T - 1];
    for (int i = 0; i < T - 1; i++) w[i] = i >= P - 1 ? 1.0f : 0.0f;
    w[P + 1] = 0.5f;
    w[P + 3] = -1.0f;

    // ---- 1. a one-hot teacher is the weighted cross-entropy, bit for bit
    {
        int32_t ids[T - 1];
        float lp[T - 1];
        for (int t = 0; t + 1 < T; t++) { ids[t] = toks[t + 1]; lp[t] = 0.0f; }
        lora_teacher one = { 1, ids, lp };
        const float *ws[2] = { NULL, w };
        for (int c = 0; c < 2; c++) {
            double ce = 0, kl = 0;
            model_lora_grad_zero(&m);
            CHECK(model_lora_backward_w(&m, toks, T, ws[c], &ce), "backward_w failed");
            snapshot(&m);
            kl = kd(&m, toks, ws[c], &one);
            CHECK(kl == ce, "one-hot KL %.17g != cross-entropy %.17g (%s)", kl, ce,
                  c ? "weighted" : "unweighted");
            CHECK(grads_equal_snapshot(&m),
                  "one-hot gradient buffers differ from the cross-entropy's (%s)",
                  c ? "weighted" : "unweighted");
        }
        printf("ok: a one-hot teacher is the cross-entropy bit for bit "
               "(%d buffers, unweighted and weighted)\n", nsnap);
    }

    // ---- 2. the loss against an independent formula, with and without a tail
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(T - 1) * (V < KD_K_MAX ? V : KD_K_MAX));
    float *lp = malloc(sizeof(float) * (size_t)(T - 1) * (V < KD_K_MAX ? V : KD_K_MAX));
    lora_teacher top = { K, ids, lp };
    soft_teacher(&m, toks, K, 0.85, ids, lp);
    {
        double got = kd(&m, toks, w, &top), want = reference_kl(&m, toks, w, &top);
        double tol = kd_tol(w);
        CHECK(fabs(got - want) <= tol, "top-%d KL %.17g vs independent %.17g", K, got, want);
        printf("ok: random top-%d teacher, tail 0.15: KL %.9f matches the "
               "independent formula (%.2e, bound %.2e)\n", K, got, fabs(got - want), tol);
    }
    fd_checks(&m, toks, w, &top, "random top-k, tail 0.15");
    anchored_teacher(&m, toks, K, 0.1, ids, lp);
    {
        double got = kd(&m, toks, w, &top), want = reference_kl(&m, toks, w, &top);
        double tol = kd_tol(w);
        CHECK(fabs(got - want) <= tol, "anchored top-%d KL %.17g vs independent %.17g",
              K, got, want);
        printf("ok: student-anchored top-%d teacher, tail 0.9: KL %.9f matches "
               "the independent formula (%.2e, bound %.2e)\n", K, got, fabs(got - want), tol);
    }
    fd_checks(&m, toks, w, &top, "anchored top-k, tail 0.9");

    if (V <= KD_K_MAX) {
        int32_t *fids = malloc(sizeof(int32_t) * (size_t)(T - 1) * V);
        float *flp = malloc(sizeof(float) * (size_t)(T - 1) * V);
        soft_teacher(&m, toks, V, 1.0, fids, flp);
        lora_teacher full = { V, fids, flp };
        double got = kd(&m, toks, w, &full), want = reference_kl(&m, toks, w, &full);
        double tol = kd_tol(w);
        CHECK(fabs(got - want) <= tol, "full-vocabulary KL %.17g vs independent %.17g",
              got, want);
        printf("ok: full-vocabulary teacher (k %d, no tail): KL %.9f matches the "
               "independent formula (%.2e, bound %.2e)\n", V, got, fabs(got - want), tol);
        fd_checks(&m, toks, w, &full, "full vocabulary");
        free(fids); free(flp);
    }

    // ---- refusals: an id outside the vocabulary, k out of range
    {
        int32_t save = ids[(P - 1) * K + 2];
        ids[(P - 1) * K + 2] = V;
        double l = 0;
        CHECK(!model_lora_backward_kd(&m, toks, T, w, &top, &l),
              "a teacher id past the vocabulary was accepted");
        ids[(P - 1) * K + 2] = save;
        lora_teacher bad = { KD_K_MAX + 1, ids, lp };
        CHECK(!model_lora_backward_kd(&m, toks, T, w, &bad, &l),
              "k above KD_K_MAX was accepted");
        // a NaN by its bits: the build's fast math refuses the NAN macro
        float keep = lp[(P - 1) * K + 1], nan;
        uint32_t qnan = 0x7fc00000u;
        memcpy(&nan, &qnan, sizeof nan);
        lp[(P - 1) * K + 1] = nan;
        CHECK(!model_lora_backward_kd(&m, toks, T, w, &top, &l),
              "a NaN teacher log-probability was accepted");
        lp[(P - 1) * K + 1] = keep;
        printf("ok: an out-of-vocabulary id, an oversized k and a NaN "
               "log-probability are refused\n");
    }

    free(ids); free(lp);
    for (int i = 0; i < nsnap; i++) free(snap[i]);
    model_lora_free(&m);
    model_free(&m);
    if (g_fail) { fprintf(stderr, "test-kd-grad: FAILED\n"); return 1; }
    printf("test-kd-grad: ok\n");
    return 0;
}
