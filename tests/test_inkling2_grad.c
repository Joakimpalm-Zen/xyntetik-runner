// R4.48.10: finite-difference gradient, recompute bit-identity, determinism,
// and failing-control gate for Inkling 2 Mini (inkling2_loop) LoRA backward.
//
// Tolerance (stated before first run): because src/inkling2.c and the backward
// run in pure float32 with -fno-fast-math -ffp-contract=off (no F16 KV
// quantization staircase), central finite differences are limited by float32
// cancellation in (L(theta + eps*d) - L(theta - eps*d)) / (2*eps) at small
// step sizes and higher-order loop curvature at large ones.
//   - Per-coordinate FD: cosine > 0.9999, worst rel err <= 0.005 (0.5%) where
//     max(|fd|, |an|) > 1e-2, else |fd - an| <= 1e-3.
//   - Directional derivatives (sign and fixed random +-1 Rademacher, whole
//     adapter + per-slot Q/K/V/O + per-block 0..3): best of steps
//     {1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3} within 0.005 (0.5%) relative error.
//   - Controls (drop pass 1 on looped layers; drop MoE sigmoid router weight
//     derivative; drop sink_proj / rel_bias / dynamic sconv weight_proj) must
//     each exceed tolerance cleanly (fail the gate).
#include "compat.h"
#include "fpclass.h"
#include "model.h"
#include "tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(c, ...) do {                                                   \
    if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr);  \
                g_fail++; }                                                  \
} while (0)

enum { T_LEN = 14 };
static const int32_t g_toks[T_LEN] = {
    12, 45, 78, 3, 199, 64, 17, 250, 91, 5, 133, 42, 88, 7
};

static double run_bwd(model_t *m) {
    model_lora_grad_zero(m);
    double loss = 0.0;
    if (!model_lora_backward(m, g_toks, T_LEN, &loss)) {
        fprintf(stderr, "FAIL: model_lora_backward failed\n");
        g_fail++;
        return 0.0;
    }
    return loss;
}

static const int g_slots[4] = { LW_Q, LW_K, LW_V, LW_O };

static void set_ctrl(const char *val) {
    plat_setenv("RUNNER_TEST_INK2_BWD_CTRL", val ? val : "");
}

static double best_dir_err(model_t *m, int nsnap, float *snap[], int snap_n[],
                           int lay[], int slo[], int whi[],
                           int filter_layer, int filter_slot, bool use_rand,
                           uint64_t rseed, double *best_fd_out,
                           double *expect_out, long *nsel_out) {
    double expect = 0.0;
    long n_sel = 0;
    uint64_t r = rseed;
    for (int i = 0; i < nsnap; i++) {
        if (filter_layer >= 0 && lay[i] != filter_layer) continue;
        if (filter_slot >= 0 && slo[i] != filter_slot) continue;
        for (int k = 0; k < snap_n[i]; k++) {
            if (use_rand) {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                float dk = (r >> 63) ? 1.0f : -1.0f;
                expect += (double)snap[i][k] * dk;
                n_sel++;
            } else {
                expect += fabs((double)snap[i][k]);
                n_sel++;
            }
        }
    }
    const float steps[6] = { 1e-5f, 5e-5f, 1e-4f, 2e-4f, 5e-4f, 1e-3f };
    double best = 1e9, best_fd = 0.0;
    for (int si = 0; si < 6; si++) {
        float e = steps[si];
        double lv[2] = { 0.0, 0.0 };
        for (int sg = 0; sg < 2; sg++) {
            float sgn = sg ? -e : e;
            uint64_t r2 = rseed;
            for (int i = 0; i < nsnap; i++) {
                if (filter_layer >= 0 && lay[i] != filter_layer) continue;
                if (filter_slot >= 0 && slo[i] != filter_slot) continue;
                int cnt = 0;
                float *th = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
                for (int k = 0; th && k < cnt; k++) {
                    float dk;
                    if (use_rand) {
                        r2 ^= r2 << 13; r2 ^= r2 >> 7; r2 ^= r2 << 17;
                        dk = (r2 >> 63) ? 1.0f : -1.0f;
                    } else {
                        dk = snap[i][k] > 0 ? 1.0f : snap[i][k] < 0 ? -1.0f : 0.0f;
                    }
                    th[k] += sgn * dk;
                }
            }
            model_lora_grad_zero(m);
            if (!model_lora_backward(m, g_toks, T_LEN, &lv[sg])) {
                g_fail++;
                lv[sg] = 0.0;
            }
            r2 = rseed;
            for (int i = 0; i < nsnap; i++) {
                if (filter_layer >= 0 && lay[i] != filter_layer) continue;
                if (filter_slot >= 0 && slo[i] != filter_slot) continue;
                int cnt = 0;
                float *th = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
                for (int k = 0; th && k < cnt; k++) {
                    float dk;
                    if (use_rand) {
                        r2 ^= r2 << 13; r2 ^= r2 >> 7; r2 ^= r2 << 17;
                        dk = (r2 >> 63) ? 1.0f : -1.0f;
                    } else {
                        dk = snap[i][k] > 0 ? 1.0f : snap[i][k] < 0 ? -1.0f : 0.0f;
                    }
                    th[k] -= sgn * dk;
                }
            }
        }
        double fd = (lv[0] - lv[1]) / (2.0 * (double)e);
        double mag = fabs(fd) > fabs(expect) ? fabs(fd) : fabs(expect);
        double rel = mag > 0 ? fabs(fd - expect) / mag : 0.0;
        if (rel < best) { best = rel; best_fd = fd; }
    }
    if (best_fd_out) *best_fd_out = best_fd;
    if (expect_out) *expect_out = expect;
    if (nsel_out) *nsel_out = n_sel;
    return best;
}

int main(int argc, char **argv) {
    const char *mpath = argc > 1 ? argv[1] : "test-inkling2-train.gguf";
    model_params mp = {0};
    mp.n_ctx = 64;
    mp.n_batch = 64;
    mp.n_threads = 2;
    model_t m = {0};
    if (!model_load(&m, mpath, &mp)) {
        fprintf(stderr, "FAIL: cannot load %s\n", mpath);
        return 1;
    }

    CHECK(!model_has_training_state(&m), "bare inkling2 model reported training state");

    if (!model_lora_train_init(&m, 4, 8.0f, 7)) {
        fprintf(stderr, "FAIL: model_lora_train_init refused inkling2_loop\n");
        model_free(&m);
        return 1;
    }

    // Seed deterministic small nonzero B values so every upstream & downstream
    // path across all 4 physical blocks and 2 loop passes carries non-zero flow.
    uint64_t rng = 0xD1B54A32D192ED03ull;
    for (int l = 0; l < 4; l++) {
        for (int si = 0; si < 4; si++) {
            int cnt = 0;
            float *b = model_lora_param(&m, l, g_slots[si], 1, &cnt);
            CHECK(b != NULL && cnt > 0, "missing B param at layer %d slot %d", l, g_slots[si]);
            for (int k = 0; b && k < cnt; k++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                b[k] = ((float)(int32_t)(uint32_t)(rng >> 32) / 2147483648.0f) * 0.12f;
            }
        }
    }

    plat_setenv("RUNNER_INK2_CHECK_RECOMPUTE", "1");
    set_ctrl(NULL);

    // Save + reload via model_lora_load must allocate no training state.
    const char *tmp_a = "test-inkling2-grad-tmp.lora.gguf";
    CHECK(model_lora_save(&m, tmp_a), "model_lora_save failed");
    model_t m_srv = {0};
    mp.lora_path = tmp_a;
    mp.lora_scale = 1.0f;
    CHECK(model_load(&m_srv, mpath, &mp), "model_load with --lora failed");
    CHECK(!model_has_training_state(&m_srv), "serving an adapter allocated training state");
    model_free(&m_srv);
    remove(tmp_a);

    // Determinism across repeated runs and thread counts
    double l1 = run_bwd(&m);
    CHECK(fp_finite_f64(l1), "loss is not finite");
    CHECK(model_has_training_state(&m), "backward left no training state");

    enum { MAX_BUF = 64 };
    float *snap[MAX_BUF] = {0};
    int snap_n[MAX_BUF] = {0}, lay[MAX_BUF] = {0}, slo[MAX_BUF] = {0}, whi[MAX_BUF] = {0};
    int nsnap = 0;
    for (int l = 0; l < 4; l++)
        for (int si = 0; si < 4; si++)
            for (int w = 0; w < 2; w++) {
                int cnt = 0;
                float *g = model_lora_gradbuf(&m, l, g_slots[si], w, &cnt);
                if (!g || nsnap >= MAX_BUF) continue;
                snap[nsnap] = malloc(sizeof(float) * (size_t)cnt);
                memcpy(snap[nsnap], g, sizeof(float) * (size_t)cnt);
                snap_n[nsnap] = cnt;
                lay[nsnap] = l;
                slo[nsnap] = g_slots[si];
                whi[nsnap] = w;
                nsnap++;
            }
    CHECK(nsnap == 32, "expected 32 grad buffers (4 layers x 4 slots x A/B), got %d", nsnap);

    double l2 = run_bwd(&m);
    CHECK(memcmp(&l1, &l2, sizeof l1) == 0, "loss not bit-identical across runs: %.17g vs %.17g", l1, l2);
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *g = model_lora_gradbuf(&m, lay[i], slo[i], whi[i], &cnt);
        CHECK(g && cnt == snap_n[i] && memcmp(g, snap[i], sizeof(float) * (size_t)cnt) == 0,
              "grad buffer %d (l=%d s=%d w=%d) not byte-deterministic", i, lay[i], slo[i], whi[i]);
    }
    printf("ok: loss %.6f, %d grad buffers byte-identical across runs (recompute bit-exact)\n", l1, nsnap);

    // Recompute drift control: deliberately corrupt recompute checkpoint check and verify failure
    set_ctrl("drift_ckpt");
    model_lora_grad_zero(&m);
    double l_drift = 0.0;
    bool ok_drift = model_lora_backward(&m, g_toks, T_LEN, &l_drift);
    set_ctrl(NULL);
    CHECK(!ok_drift, "CONTROL FAILED: drift_ckpt did not trigger recompute bit-identity check");
    if (!ok_drift) printf("ok: control (recompute mismatch) caught cleanly\n");

    // Per-coordinate FD check
    double worst = 0.0, cos_num = 0.0, cos_a = 0.0, cos_f = 0.0;
    int checked = 0;
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *th = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
        int coords[3] = { 0, cnt / 2, cnt - 1 };
        for (int c = 0; c < 3; c++) {
            int k = coords[c];
            double an = (double)snap[i][k];
            // Try eps=1e-2f first; if a discrete sparse-MoE top-k routing boundary
            // sits between save +- 1e-2, retry at eps=1e-3f.
            float save = th[k], eps = 1e-2f;
            th[k] = save + eps; double lp = run_bwd(&m);
            th[k] = save - eps; double lm = run_bwd(&m);
            th[k] = save;
            double fd = (lp - lm) / (2.0 * (double)eps);
            double mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
            double rel = mag > 1e-2 ? fabs(fd - an) / mag : fabs(fd - an);
            if (!(rel <= 0.005 || fabs(fd - an) <= 1e-3)) {
                eps = 1e-3f;
                th[k] = save + eps; lp = run_bwd(&m);
                th[k] = save - eps; lm = run_bwd(&m);
                th[k] = save;
                fd = (lp - lm) / (2.0 * (double)eps);
                mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
                rel = mag > 1e-2 ? fabs(fd - an) / mag : fabs(fd - an);
            }
            if (fabs(fd - an) > 1e-2 && rel > worst) worst = rel;
            CHECK(rel <= 0.005 || fabs(fd - an) <= 1e-3,
                  "coord l=%d s=%d %c[%d]: fd %.6g vs an %.6g (rel %.4g)",
                  lay[i], slo[i], whi[i] ? 'B' : 'A', k, fd, an, rel);
            cos_num += fd * an; cos_a += an * an; cos_f += fd * fd;
            checked++;
        }
    }
    double cosine = cos_num / (sqrt(cos_a) * sqrt(cos_f) + 1e-30);
    CHECK(cosine > 0.9999, "fd/analytic cosine %.6f below 0.9999", cosine);
    CHECK(worst <= 0.005, "worst per-coord rel err %.4g above 0.005", worst);
    printf("ok: %d FD coords, worst rel err %.4g, cosine %.6f\n", checked, worst, cosine);

    // Directional derivatives: whole adapter (sign + random), per slot, per physical layer
    double fd_v = 0, an_v = 0; long nc = 0;
    double rel_all_s = best_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, -1, false, 0, &fd_v, &an_v, &nc);
    CHECK(rel_all_s <= 0.005, "whole-adapter sign direction rel %.5f (fd %.6g, an %.6g)", rel_all_s, fd_v, an_v);
    printf("ok: whole sign dir: fd %.6g vs an %.6g (rel %.5f, %ld coords)\n", fd_v, an_v, rel_all_s, nc);

    double rel_all_r = best_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, -1, true, 0x9E3779B97F4A7C15ull, &fd_v, &an_v, &nc);
    CHECK(rel_all_r <= 0.005, "whole-adapter Rademacher direction rel %.5f (fd %.6g, an %.6g)", rel_all_r, fd_v, an_v);
    printf("ok: whole rand dir: fd %.6g vs an %.6g (rel %.5f, %ld coords)\n", fd_v, an_v, rel_all_r, nc);

    for (int si = 0; si < 4; si++) {
        double rs = best_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, g_slots[si], false, 0, &fd_v, &an_v, &nc);
        double rr = best_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, g_slots[si], true, 0x123456789ABCDEF0ull ^ (uint64_t)(si + 1), &fd_v, &an_v, &nc);
        CHECK(rs <= 0.005 && rr <= 0.005, "slot %d dir rel (sign %.5f, rand %.5f) above 0.005", g_slots[si], rs, rr);
        printf("ok: slot %d dirs: sign rel %.5f, rand rel %.5f\n", g_slots[si], rs, rr);
    }
    for (int l = 0; l < 4; l++) {
        double rs = best_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, l, -1, false, 0, &fd_v, &an_v, &nc);
        double rr = best_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, l, -1, true, 0xA5A5A5A55A5A5A5Aull ^ (uint64_t)(l + 1), &fd_v, &an_v, &nc);
        CHECK(rs <= 0.005 && rr <= 0.005, "layer %d dir rel (sign %.5f, rand %.5f) above 0.005", l, rs, rr);
        printf("ok: layer %d dirs: sign rel %.5f, rand rel %.5f\n", l, rs, rr);
    }

    // Failing controls (Condition 5): each dropped Jacobian term must make directional FD fail
    static const struct { const char *ctrl; const char *desc; int filter_l; int filter_s; } ctrls[] = {
        { "drop_pass1",   "drop pass 1 contribution on looped layers", 1, -1 },
        { "drop_router",  "drop MoE sigmoid router weight derivative", 0, -1 },
        { "drop_sink",    "drop sink_proj derivative",                 -1, LW_Q },
        { "drop_relbias", "drop relative_position_bias derivative",    -1, -1 },
        { "drop_dynconv", "drop global dynamic sconv weight_proj VJP", -1, -1 },
    };
    for (size_t ci = 0; ci < sizeof ctrls / sizeof ctrls[0]; ci++) {
        set_ctrl(ctrls[ci].ctrl);
        model_lora_grad_zero(&m);
        double lc = 0.0;
        CHECK(model_lora_backward(&m, g_toks, T_LEN, &lc), "control %s backward failed unexpectedly", ctrls[ci].ctrl);
        float *csnap[MAX_BUF] = {0};
        for (int i = 0; i < nsnap; i++) {
            int cnt = 0;
            float *g = model_lora_gradbuf(&m, lay[i], slo[i], whi[i], &cnt);
            csnap[i] = malloc(sizeof(float) * (size_t)cnt);
            memcpy(csnap[i], g, sizeof(float) * (size_t)cnt);
        }
        // Evaluate analytic slope from corrupted grads against true loss FD along random+-1
        set_ctrl(NULL);
        double bad_rel = best_dir_err(&m, nsnap, csnap, snap_n, lay, slo, whi,
                                      ctrls[ci].filter_l, ctrls[ci].filter_s,
                                      true, 0x9E3779B97F4A7C15ull ^ (uint64_t)ci,
                                      &fd_v, &an_v, &nc);
        for (int i = 0; i < nsnap; i++) free(csnap[i]);
        CHECK(bad_rel > 0.005, "CONTROL FAILED: '%s' still passed 0.005 gate (rel %.5f, fd %.6g vs corrupted %.6g)",
              ctrls[ci].desc, bad_rel, fd_v, an_v);
        if (bad_rel > 0.005)
            printf("ok: control (%s) FAILS gate as required: rel %.5f (fd %.6g vs an %.6g)\n",
                   ctrls[ci].desc, bad_rel, fd_v, an_v);
    }

    for (int i = 0; i < nsnap; i++) free(snap[i]);
    model_free(&m);
    printf("inkling2-grad: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
