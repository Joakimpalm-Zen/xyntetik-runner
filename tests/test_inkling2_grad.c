// R4.48.10: finite-difference gradient, recompute bit-identity, determinism,
// and failing-control gate for Inkling 2 Mini (inkling2_loop) LoRA backward.
//
// Tolerance (stated before first run): because src/inkling2.c and the backward
// run in pure float32 with -fno-fast-math -ffp-contract=off (no F16 KV
// quantization staircase), central finite differences are limited by float32
// cancellation in (L(theta + eps*d) - L(theta - eps*d)) / (2*eps) at small
// step sizes and higher-order loop curvature at large ones.
//   - Thread invariance (G1): model_lora_backward at 1, 2, and 4 threads (with
//     INK2_PAR_MIN=1 so worker pools split chunks) must produce bit-identical
//     loss and all 32 gradient buffers (memcmp == 0). Control `thread_order_dep`
//     (chunked over T across threads) must fail thread invariance at 4 threads.
//   - Top-k stability & non-null draw rule (G3, stated before run): any
//     Rademacher vector with |g . d| < 1.0 (before evaluating FD) or whose
//     router top-k selection vector `app_sel` differs between theta + eps*d
//     and theta - eps*d is redrawn deterministically via LCG
//     (`rseed = rseed * 6364136223846793005ULL + 1442695040888963407ULL`) up
//     to 64 tries, and total redraws are reported.
//   - Per-coordinate FD (G5, stated before run): step eps = 1e-2f (falling back
//     to 1e-3f only if a coordinate straddles a top-k flip at +eps vs -eps),
//     checked over ALL 96 sampled coordinates (none excluded); true worst rel
//     AND true worst abs reported. Pass rule: cosine > 0.9999 and every coord
//     satisfies |fd - an| <= 0.005 * max(|fd|, |an|) + 2e-3 (noise floor from
//     float32 logit/loss ULP ~4e-6 divided by 2*1e-3).
//   - Directional derivatives (G2, stated before run): fixed step eps = 5e-5f
//     for sign directions and eps = 1e-3f for Rademacher directions (no best-of-N
//     selection) on whole adapter, per slot Q/K/V/O, and per block 0..3 within
//     0.005 (0.5%) relative error.
//   - Controls: drop pass 1 & pass 2 on looped layers; scale d_att by 1.02x
//     in attention backward; drop MoE router weight derivative; drop sink_proj
//     / rel_bias / dynamic sconv weight_proj; each must fail 0.005 cleanly.
#include "compat.h"
#include "fpclass.h"
#include "inkling2.h"
#include "inkling2_model.h"
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

static void seed_adapter_b(model_t *m) {
    uint64_t rng = 0xD1B54A32D192ED03ull;
    for (int l = 0; l < 4; l++) {
        for (int si = 0; si < 4; si++) {
            int cnt = 0;
            float *b = model_lora_param(m, l, g_slots[si], 1, &cnt);
            CHECK(b != NULL && cnt > 0, "missing B param at layer %d slot %d", l, g_slots[si]);
            for (int k = 0; b && k < cnt; k++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                b[k] = ((float)(int32_t)(uint32_t)(rng >> 32) / 2147483648.0f) * 0.12f;
            }
        }
    }
}

static bool capture_last_sel(const model_t *m, int **buf_out, int *n_out) {
    const int *sel = NULL;
    int n = 0;
    if (!model_inkling2_last_sel(m, &sel, &n) || !sel || n <= 0) return false;
    int *cp = malloc(sizeof(int) * (size_t)n);
    if (!cp) return false;
    memcpy(cp, sel, sizeof(int) * (size_t)n);
    *buf_out = cp;
    *n_out = n;
    return true;
}

static double fixed_dir_err(model_t *m, int nsnap, float *snap[], int snap_n[],
                            int lay[], int slo[], int whi[],
                            int filter_layer, int filter_slot, bool use_rand,
                            uint64_t rseed, float step, double *fd_out,
                            double *expect_out, long *nsel_out, int *redraws_out) {
    int redraws = 0;
    double fd = 0.0, expect = 0.0, rel = 0.0;
    long n_sel = 0;
    for (int attempt = 0; attempt < 64; attempt++) {
        expect = 0.0;
        n_sel = 0;
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
        if (use_rand && fabs(expect) < 1.0) {
            redraws++;
            rseed = rseed * 6364136223846793005ULL + 1442695040888963407ULL;
            CHECK(attempt < 63, "exceeded 64 redraws looking for non-orthogonal direction");
            continue;
        }
        double lv[2] = { 0.0, 0.0 };
        int *sel[2] = { NULL, NULL };
        int sel_n[2] = { 0, 0 };
        for (int sg = 0; sg < 2; sg++) {
            float sgn = sg ? -step : step;
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
            CHECK(capture_last_sel(m, &sel[sg], &sel_n[sg]), "failed to capture app_sel");
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
        bool sel_eq = sel[0] && sel[1] && sel_n[0] == sel_n[1] &&
                      memcmp(sel[0], sel[1], sizeof(int) * (size_t)sel_n[0]) == 0;
        free(sel[0]); free(sel[1]);
        fd = (lv[0] - lv[1]) / (2.0 * (double)step);
        double mag = fabs(fd) > fabs(expect) ? fabs(fd) : fabs(expect);
        rel = mag > 0 ? fabs(fd - expect) / mag : 0.0;
        if (sel_eq || !use_rand) {
            CHECK(sel_eq, "top-k selection differed across +eps/-eps on fixed direction");
            break;
        }
        redraws++;
        rseed = rseed * 6364136223846793005ULL + 1442695040888963407ULL;
        CHECK(attempt < 63, "exceeded 64 redraws looking for top-k-stable direction");
    }
    if (fd_out) *fd_out = fd;
    if (expect_out) *expect_out = expect;
    if (nsel_out) *nsel_out = n_sel;
    if (redraws_out) *redraws_out = redraws;
    return rel;
}

int main(int argc, char **argv) {
    plat_setenv("INK2_PAR_MIN", "1");
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

    seed_adapter_b(&m);

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
    // Thread-count invariance (G1): 1, 2, and 4 threads must match bit for bit,
    // and ink2_par_stats confirms worker pools actually split at >1 threads.
    static const int ths[2] = { 1, 4 };
    for (int ti = 0; ti < 2; ti++) {
        model_params mp_t = {0};
        mp_t.n_ctx = 64; mp_t.n_batch = 64; mp_t.n_threads = ths[ti];
        model_t mt = {0};
        CHECK(model_load(&mt, mpath, &mp_t), "model_load failed at -t %d", ths[ti]);
        CHECK(model_lora_train_init(&mt, 4, 8.0f, 7), "lora_train_init failed at -t %d", ths[ti]);
        seed_adapter_b(&mt);
        long sp0 = 0, sp1 = 0; int last_p = 1;
        ink2_par_stats(&sp0, NULL);
        double lt = run_bwd(&mt);
        ink2_par_stats(&sp1, &last_p);
        if (ths[ti] == 1) CHECK(sp1 == sp0, "1-thread run split %ld loops (expected 0)", sp1 - sp0);
        else              CHECK(sp1 > sp0 && last_p == ths[ti], "%d-thread run split %ld loops last_pool %d", ths[ti], sp1 - sp0, last_p);
        CHECK(memcmp(&l1, &lt, sizeof l1) == 0, "loss differs at %d threads vs 2: %.17g vs %.17g", ths[ti], l1, lt);
        for (int i = 0; i < nsnap; i++) {
            int cnt = 0;
            float *g = model_lora_gradbuf(&mt, lay[i], slo[i], whi[i], &cnt);
            CHECK(g && cnt == snap_n[i] && memcmp(g, snap[i], sizeof(float) * (size_t)cnt) == 0,
                  "grad buffer %d (l=%d s=%d w=%d) differs at %d threads", i, lay[i], slo[i], whi[i], ths[ti]);
        }
        if (ths[ti] == 4) {
            set_ctrl("thread_order_dep");
            double l_ord = run_bwd(&mt);
            set_ctrl(NULL);
            int order_diff = (memcmp(&l1, &l_ord, sizeof l1) != 0);
            for (int i = 0; i < nsnap && !order_diff; i++) {
                int cnt = 0;
                float *g = model_lora_gradbuf(&mt, lay[i], slo[i], whi[i], &cnt);
                if (!g || cnt != snap_n[i] || memcmp(g, snap[i], sizeof(float) * (size_t)cnt) != 0)
                    order_diff = 1;
            }
            CHECK(order_diff, "CONTROL FAILED: thread_order_dep at 4 threads stayed identical to 2 threads");
            if (order_diff) printf("ok: control (thread_order_dep at 4 threads) broke bit-identity as required\n");
        }
        model_free(&mt);
    }
    printf("ok: loss %.6f, %d grad buffers byte-identical across runs & 1/2/4 threads\n", l1, nsnap);

    // Recompute drift control: deliberately corrupt recompute checkpoint check and verify failure
    set_ctrl("drift_ckpt");
    model_lora_grad_zero(&m);
    double l_drift = 0.0;
    bool ok_drift = model_lora_backward(&m, g_toks, T_LEN, &l_drift);
    set_ctrl(NULL);
    CHECK(!ok_drift, "CONTROL FAILED: drift_ckpt did not trigger recompute bit-identity check");
    if (!ok_drift) printf("ok: control (recompute mismatch) caught cleanly\n");

    // OOM propagation checks (E1): injected allocation failures inside formerly void helpers fail cleanly
    static const char *oom_sites[] = { "lora_proj", "scaled_clip", "depth_attn", "kv_conv", "attn_fwd", "attn_bwd" };
    for (size_t oi = 0; oi < sizeof oom_sites / sizeof oom_sites[0]; oi++) {
        plat_setenv("RUNNER_TEST_INK2_ALLOC_FAIL", oom_sites[oi]);
        model_lora_grad_zero(&m);
        double loom = 0.0;
        bool ok_oom = model_lora_backward(&m, g_toks, T_LEN, &loom);
        plat_setenv("RUNNER_TEST_INK2_ALLOC_FAIL", "");
        CHECK(!ok_oom, "OOM site '%s' succeeded silently instead of failing", oom_sites[oi]);
    }
    printf("ok: %zu OOM injection sites propagated failure cleanly\n", sizeof oom_sites / sizeof oom_sites[0]);

    // Per-coordinate FD check (G5): stated rule step 1e-2f (fallback to 1e-3f only if top-k flips),
    // report true worst rel & abs across ALL 96 coords.
    double worst_rel = 0.0, worst_abs = 0.0, cos_num = 0.0, cos_a = 0.0, cos_f = 0.0;
    const double abs_floor = 2e-3;
    int checked = 0, coord_flips = 0;
    for (int i = 0; i < nsnap; i++) {
        int cnt = 0;
        float *th = model_lora_param(&m, lay[i], slo[i], whi[i], &cnt);
        int coords[3] = { 0, cnt / 2, cnt - 1 };
        for (int c = 0; c < 3; c++) {
            int k = coords[c];
            double an = (double)snap[i][k];
            float save = th[k], eps = 1e-2f;
            th[k] = save + eps; double lp = run_bwd(&m);
            int *sp = NULL, np = 0; CHECK(capture_last_sel(&m, &sp, &np), "coord sel+");
            th[k] = save - eps; double lm = run_bwd(&m);
            int *sm = NULL, nm = 0; CHECK(capture_last_sel(&m, &sm, &nm), "coord sel-");
            th[k] = save;
            bool same_sel = sp && sm && np == nm && memcmp(sp, sm, sizeof(int) * (size_t)np) == 0;
            free(sp); free(sm);
            if (!same_sel) {
                coord_flips++;
                eps = 1e-3f;
                th[k] = save + eps; lp = run_bwd(&m);
                CHECK(capture_last_sel(&m, &sp, &np), "coord retry sel+");
                th[k] = save - eps; lm = run_bwd(&m);
                CHECK(capture_last_sel(&m, &sm, &nm), "coord retry sel-");
                th[k] = save;
                CHECK(sp && sm && np == nm && memcmp(sp, sm, sizeof(int) * (size_t)np) == 0,
                      "coord l=%d s=%d %c[%d] still flipped top-k at 1e-3f", lay[i], slo[i], whi[i] ? 'B' : 'A', k);
                free(sp); free(sm);
            }
            double fd = (lp - lm) / (2.0 * (double)eps);
            double diff = fabs(fd - an);
            double mag = fabs(fd) > fabs(an) ? fabs(fd) : fabs(an);
            double rel = mag > 0.0 ? diff / mag : 0.0;
            if (rel > worst_rel) worst_rel = rel;
            if (diff > worst_abs) worst_abs = diff;
            CHECK(diff <= 0.005 * mag + abs_floor,
                  "coord l=%d s=%d %c[%d]: fd %.6g vs an %.6g (diff %.4g, rel %.4g, tol %.4g)",
                  lay[i], slo[i], whi[i] ? 'B' : 'A', k, fd, an, diff, rel, 0.005 * mag + abs_floor);
            cos_num += fd * an; cos_a += an * an; cos_f += fd * fd;
            checked++;
        }
    }
    double cosine = cos_num / (sqrt(cos_a) * sqrt(cos_f) + 1e-30);
    CHECK(cosine > 0.9999, "fd/analytic cosine %.6f below 0.9999", cosine);
    printf("ok: %d FD coords (%d top-k fallbacks): true worst abs %.4g, true worst rel %.4g, cosine %.6f\n",
           checked, coord_flips, worst_abs, worst_rel, cosine);

    // Directional derivatives (G2+G3): fixed step 5e-5f for sign, 1e-3f for Rademacher
    const float sign_eps = 5e-5f, rand_eps = 1e-3f;
    double fd_v = 0, an_v = 0; long nc = 0;
    int total_redraws = 0, rd = 0;
    double rel_all_s = fixed_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, -1, false, 0, sign_eps, &fd_v, &an_v, &nc, &rd);
    total_redraws += rd;
    CHECK(rel_all_s <= 0.005, "whole-adapter sign direction rel %.5f (fd %.6g, an %.6g)", rel_all_s, fd_v, an_v);
    printf("ok: whole sign dir (eps=5e-5): fd %.6g vs an %.6g (rel %.5f, %ld coords)\n", fd_v, an_v, rel_all_s, nc);

    double rel_all_r = fixed_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, -1, true, 0x9E3779B97F4A7C15ull, rand_eps, &fd_v, &an_v, &nc, &rd);
    total_redraws += rd;
    CHECK(rel_all_r <= 0.005, "whole-adapter Rademacher direction rel %.5f (fd %.6g, an %.6g)", rel_all_r, fd_v, an_v);
    printf("ok: whole rand dir (eps=1e-3, redraws=%d): fd %.6g vs an %.6g (rel %.5f, %ld coords)\n", rd, fd_v, an_v, rel_all_r, nc);

    for (int si = 0; si < 4; si++) {
        double fd_r = 0, an_r = 0;
        double rs = fixed_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, g_slots[si], false, 0, sign_eps, &fd_v, &an_v, &nc, &rd);
        total_redraws += rd;
        double rr = fixed_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, -1, g_slots[si], true, 0x123456789ABCDEF0ull ^ (uint64_t)(si + 1), rand_eps, &fd_r, &an_r, &nc, &rd);
        total_redraws += rd;
        CHECK(rs <= 0.005 && rr <= 0.005, "slot %d dir rel (sign %.5f, rand %.5f [fd %.4g vs an %.4g]) above 0.005", g_slots[si], rs, rr, fd_r, an_r);
        printf("ok: slot %d dirs: sign rel %.5f, rand rel %.5f (fd %.4g vs an %.4g, redraws=%d)\n", g_slots[si], rs, rr, fd_r, an_r, rd);
    }
    for (int l = 0; l < 4; l++) {
        double fd_r = 0, an_r = 0;
        double rs = fixed_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, l, -1, false, 0, sign_eps, &fd_v, &an_v, &nc, &rd);
        total_redraws += rd;
        double rr = fixed_dir_err(&m, nsnap, snap, snap_n, lay, slo, whi, l, -1, true, 0xA5A5A5A55A5A5A5Aull ^ (uint64_t)(l + 1), rand_eps, &fd_r, &an_r, &nc, &rd);
        total_redraws += rd;
        CHECK(rs <= 0.005 && rr <= 0.005, "layer %d dir rel (sign %.5f, rand %.5f [fd %.4g vs an %.4g]) above 0.005", l, rs, rr, fd_r, an_r);
        printf("ok: layer %d dirs: sign rel %.5f, rand rel %.5f (fd %.4g vs an %.4g, redraws=%d)\n", l, rs, rr, fd_r, an_r, rd);
    }
    printf("ok: G3 top-k flip redraw count across all directional probes: %d\n", total_redraws);

    // Failing controls: each dropped or mis-scaled Jacobian term must make directional FD fail at fixed eps=1e-3
    static const struct { const char *ctrl; const char *desc; int filter_l; int filter_s; } ctrls[] = {
        { "drop_pass1",   "drop pass 1 contribution on looped layers",    1, -1 },
        { "drop_pass2",   "drop pass 2 contribution on looped layers",    1, -1 },
        { "scale_102",    "scale d_att inside attention backward by 1.02", -1, LW_Q },
        { "drop_router",  "drop MoE sigmoid router weight derivative",    0, -1 },
        { "drop_sink",    "drop sink_proj derivative",                    -1, LW_Q },
        { "drop_relbias", "drop relative_position_bias derivative",       -1, -1 },
        { "drop_dynconv", "drop global dynamic sconv weight_proj VJP",    -1, -1 },
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
        // Evaluate analytic slope from corrupted grads against true loss FD along top-k-stable random+-1
        set_ctrl(NULL);
        double bad_rel = fixed_dir_err(&m, nsnap, csnap, snap_n, lay, slo, whi,
                                       ctrls[ci].filter_l, ctrls[ci].filter_s,
                                       true, 0x9E3779B97F4A7C15ull ^ (uint64_t)ci,
                                       rand_eps, &fd_v, &an_v, &nc, &rd);
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
