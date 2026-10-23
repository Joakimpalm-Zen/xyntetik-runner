// Adaptation D3 gate: the backward pass against finite differences.
//
// The RED-first equivalent for gradients: for every adapted projection slot
// (attention q/k/v/output + ffn gate/up/down, both layers of the fixture,
// A and B sides), perturb sampled coordinates and compare the analytic
// gradient against the central difference of the ACTUAL loss the engine
// computes. A backward pass that used the wrong input, the wrong side of a
// residual, a wrong softmax jacobian or a wrong rope adjoint lands orders of
// magnitude outside the float-forward FD noise this tolerates.
//
// Also pins gradient DETERMINISM — two identical backward calls produce
// byte-identical gradient buffers and bit-equal losses — the property the
// whole "same data + same seed -> byte-identical adapter" claim (D5) stands
// on.
//
//     ./test-lora-grad test.gguf test-lora.full.gguf
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

static double run_backward(model_t *m, const int32_t *toks) {
    double loss = 0;
    model_lora_grad_zero(m);
    if (!model_lora_backward(m, toks, T, &loss)) {
        CHECK(0, "model_lora_backward failed");
        return 0;
    }
    return loss;
}

int main(int argc, char **argv) {
    const char *base = argc > 1 ? argv[1] : "test.gguf";
    const char *adapter = argc > 2 ? argv[2] : "test-lora.full.gguf";
    // "stiff" (R8.9.6): a fixture whose loss the f16 KV staircase makes
    // unresolvable per coordinate (per-layer embeddings, shared KV, the
    // weightless V norm: the set of coordinates that miss changes with the
    // FD step, which a jacobian bias never does). Per-coordinate misses are
    // reported, not failed; the whole-adapter directional derivative and
    // the gradient cosine are the checks. Established on the muse fixture
    // in R8.9.4 as a reading, made a mode here.
    bool stiff = argc > 3 && strcmp(argv[3], "stiff") == 0;
    // "strict" (R17.3): the sparse-MoE fixtures. An attention adapter's
    // gradient reaches its layer input through the frozen experts AND the
    // router's softmax weights, and the router's share is small next to the
    // 5% per-coordinate tolerance: a mutation that dropped the router term
    // passed every check at 5% (worst rel err 0.013 to 0.020 on the three
    // MoE fixtures) while the correct backward reads 0. Strict holds the
    // resolved coordinates to 0.5%.
    bool strict = argc > 3 && strcmp(argv[3], "strict") == 0;
    // "slots" (R8.16): also a directional derivative along each projection
    // slot's own coordinates. The whole-adapter direction is dominated by
    // the large slots: on the qwen35 fixture a backward that dropped the
    // DeltaNet decay's gradient (the ssm_alpha slot, gradients near 1e-3)
    // passed it at cosine 0.99983, where its own direction reads an
    // analytic 0 against a nonzero finite difference.
    bool slots = argc > 3 && strcmp(argv[3], "slots") == 0;
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
    int32_t toks[T];
    for (int i = 0; i < T; i++) toks[i] = 3 + (i * 7) % (m.n_vocab - 4);

    // ---- the activation derivative itself, against central differences of
    // the forward's own gated_act (u = 1), both kinds, over the range real
    // pre-activations occupy. Decisive where the fixture-level gate is not:
    // a SiLU derivative substituted for GELU's passes the whole-adapter
    // check at fixture scale but fails here at |g| ~ 1 by more than 5%.
    {
        const int kinds[2] = { ACT_SILU, ACT_GELU };
        double worst_act = 0;
        for (int k = 0; k < 2; k++)
            for (double g = -6.0; g <= 6.0; g += 0.05) {
                const double h = 1e-2;
                double fd = ((double)model_ffn_act(kinds[k], (float)(g + h), 1.0f) -
                             (double)model_ffn_act(kinds[k], (float)(g - h), 1.0f)) / (2 * h);
                double an = model_ffn_act_deriv(kinds[k], (float)g);
                double err = fabs(fd - an);
                if (err > worst_act) worst_act = err;
                CHECK(err < 2e-3, "act_d(%s, %.2f) = %.6f, central difference %.6f",
                      kinds[k] == ACT_GELU ? "gelu" : "silu", g, an, fd);
            }
        printf("ok: activation derivatives match central differences, worst abs err %.2e\n",
               worst_act);
    }

    // ---- determinism: two runs, byte-identical grads and bit-equal loss
    double l1 = run_backward(&m, toks);
    // snapshot every grad buffer
    enum { MAX_BUF = 128 };
    float *snap[MAX_BUF]; int snap_n[MAX_BUF]; int nsnap = 0;
    int lay[MAX_BUF], slo[MAX_BUF], whi[MAX_BUF];
    for (int l = 0; l < m.n_layer; l++)
        for (int s = 0; s < LW_SLOTS; s++)
            for (int w = 0; w < 2; w++) {
                int cnt = 0;
                float *g = model_lora_gradbuf(&m, l, s, w, &cnt);
                if (!g || nsnap >= MAX_BUF) continue;
                snap[nsnap] = malloc(sizeof(float) * (size_t)cnt);
                memcpy(snap[nsnap], g, sizeof(float) * (size_t)cnt);
                snap_n[nsnap] = cnt;
                lay[nsnap] = l; slo[nsnap] = s; whi[nsnap] = w;
                nsnap++;
            }
    CHECK(nsnap >= 14, "expected adapters on every slot, saw %d buffers", nsnap);
    double l2 = run_backward(&m, toks);
    CHECK(l1 == l2, "loss not deterministic: %.17g vs %.17g", l1, l2);
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *g = model_lora_gradbuf(&m, lay[i], slo[i], whi[i], &cnt);
        CHECK(g && cnt == snap_n[i] &&
              memcmp(g, snap[i], sizeof(float) * (size_t)cnt) == 0,
              "grad buffer %d not byte-deterministic", i);
    }
    printf("ok: loss %.6f, %d grad buffers byte-deterministic across runs\n",
           l1, nsnap);
    // RUNNER_GRAD_DUMP=FILE: the loss, then per buffer its layer, slot, side
    // (0 A, 1 B) and count as int32 and its gradients as float32, for
    // tests/test_deltanet_autograd.py
    const char *dump = getenv("RUNNER_GRAD_DUMP");
    if (dump) {
        FILE *f = fopen(dump, "wb");
        CHECK(f != NULL, "cannot write RUNNER_GRAD_DUMP %s", dump);
        if (f) {
            fwrite(&l1, sizeof l1, 1, f);
            for (int i = 0; i < nsnap; i++) {
                int32_t h[4] = { lay[i], slo[i], whi[i], snap_n[i] };
                fwrite(h, sizeof h, 1, f);
                fwrite(snap[i], sizeof(float), (size_t)snap_n[i], f);
            }
            CHECK(fclose(f) == 0, "cannot write RUNNER_GRAD_DUMP %s", dump);
        }
    }

    // ---- finite differences on sampled coordinates of every buffer
    double worst = 0, cos_num = 0, cos_a = 0, cos_f = 0;
    int checked = 0, unresolved = 0;
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
        CHECK(theta != NULL, "param buffer missing for grad buffer %d", i);
        if (!theta) continue;
        int coords[3] = { 0, cnt / 2, cnt - 1 };
        for (int c = 0; c < 3; c++) {
            int k = coords[c];
            float save = theta[k];
            // eps large enough to average across the f16 rounding of the
            // cached K/V rows the loss actually flows through (the forward
            // quantizes v/k to ~5e-4 relative; a step well above that sees
            // the smooth loss, a smaller one sees the staircase)
            // RUNNER_FD_EPS overrides the step (diagnostic only): a wrong
            // adjoint stays wrong as eps grows, FD noise shrinks toward it
            float eps = getenv("RUNNER_FD_EPS") ? (float)atof(getenv("RUNNER_FD_EPS")) : 1e-2f;
            theta[k] = save + eps;
            double lp = run_backward(&m, toks);
            theta[k] = save - eps;
            double lm = run_backward(&m, toks);
            theta[k] = save;
            double fd = (lp - lm) / (2.0 * (double)eps);
            double an = (double)snap[i][k];
            // two-scale escape: the f16-KV rounding staircases the finite
            // difference non-monotonically in eps (probed: fd oscillates
            // AROUND the analytic value and converges to it as eps grows —
            // a real jacobian bias would persist instead). A coordinate that
            // misses at the small step gets one retry at 8x, where the
            // staircase averages out.
            if (fabs(fd - an) > 1e-2 &&
                fabs(fd - an) > 0.05 * (fabs(an) > fabs(fd) ? fabs(an)
                                                            : fabs(fd))) {
                float e2 = 8e-2f;
                theta[k] = save + e2;
                lp = run_backward(&m, toks);
                theta[k] = save - e2;
                lm = run_backward(&m, toks);
                theta[k] = save;
                double fd8 = (lp - lm) / (2.0 * (double)e2);
                // a finite difference that does not agree with itself across
                // the two steps has not converged on this coordinate (the
                // staircase, amplified by sandwich norms on the muse fixture):
                // it is counted as unresolved, never compared. One that agrees
                // with itself and still misses the analytic value is a real
                // failure. The directional check below is the noise-free
                // arbiter for both.
                double m2 = fabs(fd) > fabs(fd8) ? fabs(fd) : fabs(fd8);
                if (fabs(fd - fd8) > 1e-2 && fabs(fd - fd8) > 0.05 * m2) {
                    unresolved++;
                    continue;
                }
                fd = fd8;
            }
            double mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
            double rel = mag > 0.02 ? fabs(fd - an) / mag : fabs(fd - an);
            if (rel > worst && fabs(fd - an) > 1e-2) worst = rel;
            // two-sided bound: 5% relative for well-resolved coordinates,
            // or inside the measured FD noise floor for tiny gradients. The
            // floor is REAL and was probed, not assumed: every layer-0
            // parameter's loss path crosses the f16 rounding of layer-1's
            // cached K/V, which staircases the finite difference by up to
            // ~7e-3 absolute at eps=1e-2 — and doubling eps halves the
            // discrepancy while the analytic value stands still, the noise
            // signature. A wrong jacobian term produces O(1) relative error
            // on the LARGE coordinates, which the relative arm still
            // catches; the cosine below guards the aggregate.
            if (stiff && !(rel <= 0.05 || fabs(fd - an) <= 1e-2))
                printf("warn: stiff fixture, layer %d slot %d %s[%d]: fd %g vs analytic %g (rel %.3g)\n",
                       lay[i], slo[i], whi[i] ? "B" : "A", k, fd, an, rel);
            else
            CHECK(rel <= 0.05 || fabs(fd - an) <= 1e-2,
                  "layer %d slot %d %s[%d]: fd %.6g vs analytic %.6g "
                  "(rel %.3g)", lay[i], slo[i], whi[i] ? "B" : "A", k, fd,
                  an, rel);
            cos_num += fd * an; cos_a += an * an; cos_f += fd * fd;
            checked++;
        }
    }
    double cosine = cos_num / (sqrt(cos_a) * sqrt(cos_f) + 1e-30);
    CHECK(cosine > 0.999, "fd/analytic cosine %.6f over %d coords", cosine,
          checked);
    CHECK(unresolved * 4 <= checked + unresolved,
          "%d of %d coordinates unresolved by the finite difference", unresolved,
          checked + unresolved);
    if (stiff)
        CHECK(cos_num / (sqrt(cos_a) * sqrt(cos_f) + 1e-30) >= 0.999,
              "stiff fixture: gradient cosine below 0.999");
    if (strict)
        CHECK(worst <= 0.005, "strict fixture: worst per-coordinate rel err %.4g above 0.005",
              worst);
    printf("ok: %d FD coordinates across %d buffers (%d unresolved), worst rel err %.4g, "
           "cosine %.6f\n", checked, nsnap, unresolved, worst, cosine);

    // ---- directional derivative over the whole adapter (R8.9.4). The
    // per-coordinate check above rides on the f16 staircase of the cached
    // K/V: its noise is per coordinate and random in sign, so it averages
    // out along a direction that moves thousands of parameters at once,
    // while a wrong jacobian term biases every one of them the same way.
    // Direction: the sign of the analytic gradient on every coordinate above
    // the mean magnitude, so the expected slope is the sum of those
    // magnitudes and the staircase contributes ~sqrt(N) of its per-coordinate
    // noise against a signal of order N. Three steps, all inside the linear
    // regime: thousands of coordinates move at once, so the step per
    // coordinate is tiny (the loss moves by tenths of a percent); each must land
    // within 2% of the analytic slope.
    {
        double sum_abs = 0; long n_all = 0;
        for (int i = 0; i < nsnap; i++)
            for (int k = 0; k < snap_n[i]; k++) { sum_abs += fabs(snap[i][k]); n_all++; }
        double thresh = n_all ? sum_abs / (double)n_all : 0;
        double expect = 0; long n_sel = 0;
        for (int i = 0; i < nsnap; i++)
            for (int k = 0; k < snap_n[i]; k++)
                if (fabs(snap[i][k]) > thresh) { expect += fabs(snap[i][k]); n_sel++; }
        // R8.9.6: two smaller steps for the stiff fixtures (per-layer
        // embeddings and a weightless V norm over near-zero rows curve the
        // loss hard along thousands of coordinates); a wrong term is a bias
        // no step removes, so the best-of-five rule is not loosened
        const float steps[6] = { 5e-6f, 1e-5f, 2e-5f, 5e-5f, 1e-4f, 2e-4f };
        double best = 1.0;
        for (int si = 0; si < 6; si++) {
            float e = steps[si];
            for (int sign = 1; sign >= -1; sign -= 2) {
                for (int i = 0; i < nsnap; i++) {
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; theta && k < cnt; k++)
                        if (fabs(snap[i][k]) > thresh)
                            theta[k] += (float)sign * e * (snap[i][k] > 0 ? 1.0f : -1.0f);
                }
                double lv = run_backward(&m, toks);
                if (sign == 1) cos_num = lv; else cos_f = lv;   // reuse as lp/lm
                for (int i = 0; i < nsnap; i++) {
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; theta && k < cnt; k++)
                        if (fabs(snap[i][k]) > thresh)
                            theta[k] -= (float)sign * e * (snap[i][k] > 0 ? 1.0f : -1.0f);
                }
            }
            double slope = (cos_num - cos_f) / (2.0 * (double)e);
            double rel = expect > 0 ? fabs(slope - expect) / expect : 0;
            if (rel < best) best = rel;
            printf("ok: directional eps %g: fd %.6g vs analytic %.6g (rel %.5f, %ld coords)\n",
                   (double)e, slope, expect, rel, n_sel);
        }
        // the smallest step is inside the linear regime on every fixture
        // measured (the larger ones show the loss's curvature, which grows
        // with the sandwich norms); a wrong jacobian term is a bias that no
        // step removes, so the best of the three must sit within 1%
        CHECK(best <= 0.01, "directional derivative: best agreement %.4f is outside 1%%", best);
    }

    // ---- exact zeros (mode "slots"): a backward that drops a term leaves
    // the coordinates only that term reached at exactly 0 (the qwen35
    // attention gate's rows of the fused Q adapter; the ssm_alpha slot when
    // the decay's gradient is lost), and both directional checks below are
    // blind there: the sign direction skips them and a random one lets them
    // cancel. A correct backward is exactly 0 only where the loss does not
    // depend on the coordinate, and then the loss does not move at all, the
    // f16 staircase included. Up to three zeros per buffer are probed.
    for (int i = 0; slots && i < nsnap; i++) {
        int cnt = 0, probed = 0;
        float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
        for (int k = 0; theta && k < snap_n[i] && probed < 3; k++) {
            if (snap[i][k] != 0.0f) continue;
            if (probed++ && k != snap_n[i] - 1 && k % 7) continue;
            float save = theta[k];
            theta[k] = save + 1e-2f;
            double lp = run_backward(&m, toks);
            theta[k] = save - 1e-2f;
            double lm = run_backward(&m, toks);
            theta[k] = save;
            double fd = (lp - lm) / 2e-2;
            CHECK(fabs(fd) <= 1e-5, "layer %d slot %d %s[%d]: analytic gradient exactly 0 "
                  "where the loss moves (fd %.3g)", lay[i], slo[i], whi[i] ? "B" : "A", k, fd);
        }
    }

    // ---- per-slot directional derivatives (mode "slots")
    for (int sl = 0; slots && sl < LW_SLOTS; sl++) {
        double expect = 0;
        long n_sel = 0;
        for (int i = 0; i < nsnap; i++)
            if (slo[i] == sl)
                for (int k = 0; k < snap_n[i]; k++) { expect += fabs(snap[i][k]); n_sel++; }
        if (!n_sel) continue;
        // up to 1e-2: the small slots (ssm_beta, ssm_alpha: slopes near 0.3)
        // move the loss by too little at 1e-4 to clear its float rounding
        const float steps[6] = { 1e-4f, 2e-4f, 5e-4f, 1e-3f, 2e-3f, 5e-3f };
        double best = 1e9, best_fd = 0;
        for (int si = 0; si < 6; si++) {
            double lv[2];
            for (int sg = 0; sg < 2; sg++) {
                float e = sg ? -steps[si] : steps[si];
                for (int i = 0; i < nsnap; i++) {
                    if (slo[i] != sl) continue;
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; theta && k < cnt; k++)
                        theta[k] += e * (snap[i][k] > 0 ? 1.0f : snap[i][k] < 0 ? -1.0f : 0.0f);
                }
                lv[sg] = run_backward(&m, toks);
                for (int i = 0; i < nsnap; i++) {
                    if (slo[i] != sl) continue;
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; theta && k < cnt; k++)
                        theta[k] -= e * (snap[i][k] > 0 ? 1.0f : snap[i][k] < 0 ? -1.0f : 0.0f);
                }
            }
            double fd = (lv[0] - lv[1]) / (2.0 * (double)steps[si]);
            double mag = fabs(fd) > expect ? fabs(fd) : expect;
            double rel = mag > 0 ? fabs(fd - expect) / mag : 0;
            if (rel < best) { best = rel; best_fd = fd; }
        }
        printf("ok: slot %d direction: fd %.6g vs analytic %.6g (rel %.5f, %ld coords)\n",
               sl, best_fd, expect, best, n_sel);
        CHECK(best <= 0.02, "slot %d: directional derivative off by %.4f (fd %.6g, analytic %.6g)",
              sl, best, best_fd, expect);
        // and along a fixed +-1 direction that does not come from the
        // gradient: the sign direction above is empty where a backward
        // leaves a slot's gradient at zero (a dropped term), and passes it.
        // The bound is the slot's L1 gradient norm, the scale of the slope
        // along the sign direction.
        double best2 = 1e9, fd2b = 0, an2 = 0;
        for (int si = 0; si < 6; si++) {
            double lv[2];
            an2 = 0;
            for (int sg = 0; sg < 2; sg++) {
                uint64_t r = 0x9E3779B97F4A7C15ull ^ (uint64_t)(sl + 1);
                float e = sg ? -steps[si] : steps[si];
                for (int i = 0; i < nsnap; i++) {
                    if (slo[i] != sl) continue;
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; theta && k < cnt; k++) {
                        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                        float dk = (r >> 63) ? 1.0f : -1.0f;
                        theta[k] += e * dk;
                        if (sg == 0) an2 += (double)snap[i][k] * dk;
                    }
                }
                lv[sg] = run_backward(&m, toks);
                r = 0x9E3779B97F4A7C15ull ^ (uint64_t)(sl + 1);
                for (int i = 0; i < nsnap; i++) {
                    if (slo[i] != sl) continue;
                    int cnt = 0;
                    float *theta = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
                    for (int k = 0; theta && k < cnt; k++) {
                        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                        theta[k] -= e * ((r >> 63) ? 1.0f : -1.0f);
                    }
                }
            }
            double fd = (lv[0] - lv[1]) / (2.0 * (double)steps[si]);
            double scale = expect > 0 ? expect : fabs(fd);
            double rel = scale > 0 ? fabs(fd - an2) / scale : 0;
            if (rel < best2) { best2 = rel; fd2b = fd; }
        }
        printf("ok: slot %d random direction: fd %.6g vs analytic %.6g (off %.5f of |g|_1)\n",
               sl, fd2b, an2, best2);
        CHECK(best2 <= 0.02, "slot %d: random-direction derivative off by %.4f of |g|_1 "
              "(fd %.6g, analytic %.6g)", sl, best2, fd2b, an2);
    }

    for (int i = 0; i < nsnap; i++) free(snap[i]);
    model_free(&m);
    printf(g_fail ? "lora-grad: FAILED\n" : "lora-grad: ok\n");
    return g_fail;
}
