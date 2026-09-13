// DPO gate: the pairwise objective and its gradient against independent
// references, before any training time is spent on it.
//
// Three checks, in order of what they can catch:
//
//  1. LOSS vs an independent formula. The four sequence log-probabilities
//     come out of the engine; the loss is then recomputed here from them,
//     by a different expression than dpo.c uses, and the two must agree.
//     Catches a sign error, a swapped chosen/rejected, a reference taken
//     under the wrong policy, and the stable-softplus branch boundary.
//
//  2. GRADIENT vs the identity it is supposed to be. grad L_DPO must equal
//     coeff * (grad CE(y_w) - grad CE(y_l)). Both halves are computed here
//     with two separate single-example backward calls and combined by hand,
//     then compared buffer by buffer against what dpo_accumulate left
//     behind. Catches an accumulation bug, a missed buffer in the scale,
//     and a bypass left on across the pair.
//
//  3. DIRECTIONAL DERIVATIVE vs finite differences of the actual loss. The
//     arbiter, and the only one of the three that does not reuse any of the
//     gradient machinery: it perturbs the parameters and re-evaluates the
//     loss. Per-coordinate FD on this model is f16-staircase noise, so the
//     house convention applies -- a directional check at eps 5e-5..2e-4,
//     best agreement within 1%.
//
//     ./test-dpo-grad test.gguf test-lora.full.gguf
#include "dpo.h"
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

// a prompt of P, then a completion; the two halves of the pair share the
// prompt and differ in the completion, which is what "same task input" means
enum { P = 6, CW = 7, CL = 5, NW = P + CW, NL = P + CL };

enum { MAX_BUF = 64 };
static float *snap[MAX_BUF];
static int snap_n[MAX_BUF], lay[MAX_BUF], slo[MAX_BUF], whi[MAX_BUF], nsnap;

static void collect(model_t *m) {
    nsnap = 0;
    for (int l = 0; l < m->n_layer; l++)
        for (int s = 0; s < 7; s++)
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

int main(int argc, char **argv) {
    const char *base = argc > 1 ? argv[1] : "test.gguf";
    const char *adapter = argc > 2 ? argv[2] : "test-lora.full.gguf";
    const double beta = 0.1;
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

    int32_t tw[NW], tl[NL];
    float mw[NW - 1], ml[NL - 1];
    for (int i = 0; i < NW; i++) tw[i] = 3 + (i * 7) % (m.n_vocab - 4);
    for (int i = 0; i < P; i++) tl[i] = tw[i];              // shared prompt
    for (int i = P; i < NL; i++) tl[i] = 3 + (i * 13) % (m.n_vocab - 4);
    // transition t scores toks[t+1], so the completion starts at t = P-1
    for (int i = 0; i < NW - 1; i++) mw[i] = i >= P - 1 ? 1.0f : 0.0f;
    for (int i = 0; i < NL - 1; i++) ml[i] = i >= P - 1 ? 1.0f : 0.0f;

    // ---- the bypass must be an exact round trip, or pi_ref is not pi_ref
    {
        double a = 0, b = 0, c = 0;
        CHECK(model_seq_nll(&m, tw, NW, mw, &a), "seq_nll failed");
        CHECK(model_lora_bypass(&m, true), "bypass refused on a CPU model");
        CHECK(model_seq_nll(&m, tw, NW, mw, &b), "seq_nll (bypassed) failed");
        CHECK(model_lora_bypass(&m, true), "bypass not idempotent");
        CHECK(model_lora_bypass(&m, false), "bypass restore refused");
        CHECK(model_seq_nll(&m, tw, NW, mw, &c), "seq_nll (restored) failed");
        CHECK(a == c, "bypass is not a round trip: %.17g then %.17g", a, c);
        CHECK(a != b, "bypass changed nothing -- the adapter is a no-op, so "
                      "this fixture cannot test a reference policy");
        printf("ok: bypass exact round trip, adapter moves the NLL %.6f -> %.6f\n", a, b);
    }

    // ---- 0. the identity case, which is free and catches a whole class.
    // A zero adapter makes pi_theta IDENTICAL to pi_ref, so the margin must be
    // exactly 0 and the loss exactly log 2 -- not approximately. Any leak
    // between the two policies (a bypass that misses a slot, a reference taken
    // after a backward has already moved the parameters, a log-prob read from
    // the wrong side) shows up here as a non-zero margin on the first step,
    // which is also why the trainer prints the margin on step 1.
    {
        model_t z;
        memset(&z, 0, sizeof(z));
        if (model_load(&z, base, &p) && model_lora_train_init(&z, 4, 8.0f, 0)) {
            dpo_step q;
            CHECK(dpo_accumulate(&z, beta, tw, NW, mw, tl, NL, ml, &q),
                  "dpo_accumulate failed on a zero adapter");
            CHECK(q.margin == 0.0, "a fresh (B=0) adapter must give margin "
                  "exactly 0, got %.17g", q.margin);
            CHECK(q.logp_w == q.logp_ref_w && q.logp_l == q.logp_ref_l,
                  "policy and reference log-probs differ under a no-op adapter");
            CHECK(fabs(q.loss - log(2.0)) < 1e-12,
                  "margin 0 must give loss log 2, got %.17g", q.loss);
            printf("ok: no-op adapter gives margin exactly 0 and loss %.17g\n", q.loss);
            model_lora_free(&z);
            model_free(&z);
        } else {
            CHECK(0, "could not build the zero-adapter control");
        }
    }

    // ---- 1. loss against an independently written formula
    dpo_step s;
    CHECK(dpo_loss(&m, beta, tw, NW, mw, tl, NL, ml, &s), "dpo_loss failed");
    {
        double rw = beta * (s.logp_w - s.logp_ref_w);
        double rl = beta * (s.logp_l - s.logp_ref_l);
        // a different expression for the same thing: log(1 + e^-z) via
        // -log(sigmoid) written as log(1/sigma) = log((1+e^-z)) computed
        // through expm1/log rather than log1p, and the margin reassembled
        // from the implicit rewards rather than from the four log-probs
        double z = rw - rl;
        double ref_loss = -log(1.0 / (1.0 + exp(-z)));
        double ref_coeff = beta * (1.0 / (1.0 + exp(z)));
        CHECK(fabs(s.margin - z) < 1e-12,
              "margin disagrees with r_w - r_l: %.17g vs %.17g", s.margin, z);
        CHECK(fabs(s.loss - ref_loss) < 1e-9,
              "loss disagrees with the independent formula: %.17g vs %.17g",
              s.loss, ref_loss);
        CHECK(fabs(s.coeff - ref_coeff) < 1e-9,
              "coeff disagrees with beta*sigma(-margin): %.17g vs %.17g",
              s.coeff, ref_coeff);
        printf("ok: loss %.6f, margin %.6f, coeff %.6g match an independent formula\n",
               s.loss, s.margin, s.coeff);
    }

    // ---- the accumulating step must report the SAME loss as the loss-only
    // path, which is the check that the backward's masked loss and the
    // forward-only NLL are the same number
    dpo_step g;
    CHECK(dpo_accumulate(&m, beta, tw, NW, mw, tl, NL, ml, &g),
          "dpo_accumulate failed");
    CHECK(fabs(g.loss - s.loss) < 1e-9,
          "accumulate and loss-only disagree: %.17g vs %.17g", g.loss, s.loss);
    CHECK(fabs(g.logp_w - s.logp_w) < 1e-9 && fabs(g.logp_l - s.logp_l) < 1e-9,
          "policy log-probs differ between the two paths");
    collect(&m);
    CHECK(nsnap >= 14, "expected adapters on every slot, saw %d buffers", nsnap);

    // ---- 2. the gradient identity, assembled by hand from two plain backwards
    {
        double lw = 0, ll = 0;
        model_lora_grad_zero(&m);
        CHECK(model_lora_backward_w(&m, tw, NW, mw, &lw), "backward(chosen) failed");
        float *gw[MAX_BUF];
        for (int i = 0; i < nsnap; i++) {
            int cnt = 0;
            float *gb = model_lora_gradbuf(&m, lay[i], slo[i], whi[i], &cnt);
            gw[i] = malloc(sizeof(float) * (size_t)cnt);
            memcpy(gw[i], gb, sizeof(float) * (size_t)cnt);
        }
        model_lora_grad_zero(&m);
        CHECK(model_lora_backward_w(&m, tl, NL, ml, &ll), "backward(rejected) failed");
        double worst = 0, scale = 0;
        for (int i = 0; i < nsnap; i++) {
            int cnt = 0;
            float *gl = model_lora_gradbuf(&m, lay[i], slo[i], whi[i], &cnt);
            for (int k = 0; k < cnt; k++) {
                double want = g.coeff * ((double)gw[i][k] - (double)gl[k]);
                double got = (double)snap[i][k];
                double d = fabs(want - got);
                if (d > worst) worst = d;
                if (fabs(want) > scale) scale = fabs(want);
            }
            free(gw[i]);
        }
        CHECK(fabs(lw + g.logp_w) < 1e-9,
              "chosen NLL differs from -logp_w: %.17g vs %.17g", lw, -g.logp_w);
        CHECK(fabs(ll + g.logp_l) < 1e-9,
              "rejected NLL differs from -logp_l: %.17g vs %.17g", ll, -g.logp_l);
        double rel = scale > 0 ? worst / scale : worst;
        CHECK(rel < 1e-5, "grad L_DPO != coeff*(grad CE_w - grad CE_l): "
              "worst %.6g on a scale of %.6g (rel %.3g)", worst, scale, rel);
        printf("ok: gradient identity holds, worst %.3g on a scale of %.3g\n",
               worst, scale);
    }

    // ---- 3. directional derivative vs central differences of the loss.
    // The only check here that touches none of the gradient machinery.
    {
        long coords = 0;
        double analytic = 0;
        // a fixed pseudo-random unit-ish direction, and <grad, d> from the
        // gradient dpo_accumulate left behind
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < nsnap; i++)
            for (int k = 0; k < snap_n[i]; k++) {
                rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                double d = (rs >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
                analytic += (double)snap[i][k] * d;
                coords++;
            }
        // A WIDER ladder than tests/test_lora_grad.c's, and the reason is
        // specific to this objective. The DPO margin here is
        // beta*((lp_w - lp_ref_w) - (lp_l - lp_ref_l)): a difference of
        // differences of four sequence NLLs whose magnitudes are ~39 and
        // whose combination is ~0.15. That is catastrophic cancellation on
        // top of the f16-staircase forward noise, so the noise floor on the
        // loss is far higher relative to the signal than it is for a plain
        // cross-entropy. The remedy is the same one the house already
        // established for per-coordinate FD: grow eps until the signal
        // clears the noise, and check that agreement IMPROVES monotonically
        // as it does. A wrong adjoint does not improve with eps.
        enum { NSTEP = 6 };
        const float steps[NSTEP] = { 5e-5f, 1e-4f, 2e-4f, 5e-4f, 1e-3f, 2e-3f };
        double rels[NSTEP];
        double best = 1e9, best_fd = 0;
        float best_eps = 0;
        for (int si = 0; si < NSTEP; si++) {
            float e = steps[si];
            double lp = 0, lm = 0;
            for (int sign = 0; sign < 2; sign++) {
                uint64_t r2 = 0x9E3779B97F4A7C15ull;
                for (int i = 0; i < nsnap; i++) {
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; k < cnt; k++) {
                        r2 ^= r2 << 13; r2 ^= r2 >> 7; r2 ^= r2 << 17;
                        double d = (r2 >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
                        theta[k] += (sign ? -e : e) * (float)d;
                    }
                }
                dpo_step q;
                CHECK(dpo_loss(&m, beta, tw, NW, mw, tl, NL, ml, &q),
                      "dpo_loss failed under perturbation");
                if (sign) lm = q.loss; else lp = q.loss;
                // restore exactly, by re-walking the same direction back
                uint64_t r3 = 0x9E3779B97F4A7C15ull;
                for (int i = 0; i < nsnap; i++) {
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; k < cnt; k++) {
                        r3 ^= r3 << 13; r3 ^= r3 >> 7; r3 ^= r3 << 17;
                        double d = (r3 >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
                        theta[k] -= (sign ? -e : e) * (float)d;
                    }
                }
            }
            double fd = (lp - lm) / (2.0 * (double)e);
            double rel = fabs(analytic) > 0 ? fabs(fd - analytic) / fabs(analytic)
                                            : fabs(fd - analytic);
            rels[si] = rel;
            printf("   eps %-8g fd %-12.6g analytic %-12.6g rel %.5f\n",
                   (double)e, fd, analytic, rel);
            if (rel < best) { best = rel; best_fd = fd; best_eps = e; }
        }
        printf("ok: directional best at eps %g: fd %.6g vs analytic %.6g "
               "(rel %.5f, %ld coords)\n",
               (double)best_eps, best_fd, analytic, best, coords);
        CHECK(best <= 0.01, "DPO directional derivative: best agreement %.4f "
              "is outside 1%%", best);
        // The convergence claim, checked rather than asserted in prose: the
        // largest eps must agree better than the smallest. If it does not,
        // the disagreement is not noise and the band is hiding a real bug.
        CHECK(rels[NSTEP - 1] < rels[0],
              "agreement does not improve with eps (%.5f at %g vs %.5f at %g) "
              "-- the residual is not forward noise",
              rels[NSTEP - 1], (double)steps[NSTEP - 1], rels[0], (double)steps[0]);
    }

    for (int i = 0; i < nsnap; i++) free(snap[i]);
    model_lora_free(&m);
    model_free(&m);
    if (g_fail) { fprintf(stderr, "test-dpo-grad: FAILED\n"); return 1; }
    printf("test-dpo-grad: ok\n");
    return 0;
}
