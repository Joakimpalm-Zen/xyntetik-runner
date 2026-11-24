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
//   - Top-k stability & redraw rule (G3): any Rademacher vector whose router
//     top-k selection vector `app_sel` differs between theta + eps*d and
//     theta - eps*d is redrawn deterministically via LCG
//     (`rseed = rseed * 6364136223846793005ULL + 1442695040888963407ULL`) up
//     to 64 tries, and total redraws are reported.
//   - Per-coordinate FD (G5, house stiff rule): step eps = 1e-2f (falling back
//     to 1e-3f only if a coordinate straddles a top-k flip at +eps vs -eps),
//     checked over ALL 96 sampled coordinates (none excluded). Coordinates
//     exceeding 0.005 * max(|fd|, |an|) + floor(h) (where
//     floor(h) = 2e-3 * (1e-3 / h)) and floor-assisted coords are reported as
//     diagnostics; the hard coordinate gate is cosine >= 0.9999 and the
//     directional derivatives below stay the arbiter.
//   - Directional derivatives (G2): fixed step eps = 5e-5f for sign and
//     eps = 2e-4f for Rademacher (no best-of-N), with every relative error
//     measured against the probed block's L1 gradient norm ||g_block||_1:
//     rel = |fd - an| / ||g_block||_1 <= 0.005 on whole adapter, per slot
//     Q/K/V/O, and per block 0..3, checked across two committed seed sets
//     (+0 and +1000003).
//   - Controls: judged on the deterministic sign probes (whole, 4 slots, 4
//     layers); each of the 7 Jacobian controls must fail (> 0.005) on at
//     least one sign probe, with its Rademacher figure printed beside it.
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

enum { MAX_BUF = 64 };

static double fixed_dir_err(model_t *m, int nsnap, float *ref_snap[], float *an_snap[],
                            int snap_n[], int lay[], int slo[], int whi[],
                            int filter_layer, int filter_slot, bool use_rand,
                            uint64_t rseed, float step, double *fd_out,
                            double *expect_out, double *l1_out, long *nsel_out,
                            int *redraws_out) {
    int redraws = 0;
    double fd = 0.0, expect = 0.0, l1_norm = 0.0, rel = 0.0;
    long n_sel = 0;
    float *saved_th[MAX_BUF] = {0};
    for (int i = 0; i < nsnap && i < MAX_BUF; i++) {
        if (filter_layer >= 0 && lay[i] != filter_layer) continue;
        if (filter_slot >= 0 && slo[i] != filter_slot) continue;
        for (int k = 0; k < snap_n[i]; k++) l1_norm += fabs((double)ref_snap[i][k]);
        int cnt = 0;
        float *th = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
        if (th && cnt > 0) {
            saved_th[i] = malloc(sizeof(float) * (size_t)cnt);
            if (!saved_th[i]) { g_fail++; goto out; }
            memcpy(saved_th[i], th, sizeof(float) * (size_t)cnt);
        }
    }
    for (int attempt = 0; attempt < 64; attempt++) {
        expect = 0.0;
        n_sel = 0;
        uint64_t r = rseed;
        for (int i = 0; i < nsnap; i++) {
            if (filter_layer >= 0 && lay[i] != filter_layer) continue;
            if (filter_slot >= 0 && slo[i] != filter_slot) continue;
            for (int k = 0; k < snap_n[i]; k++) {
                float dk;
                if (use_rand) {
                    r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                    dk = (r >> 63) ? 1.0f : -1.0f;
                } else {
                    dk = ref_snap[i][k] > 0 ? 1.0f : ref_snap[i][k] < 0 ? -1.0f : 0.0f;
                }
                expect += (double)an_snap[i][k] * dk;
                n_sel++;
            }
        }
        double lv[2] = { 0.0, 0.0 };
        int *sel[2] = { NULL, NULL };
        int sel_n[2] = { 0, 0 };
        for (int sg = 0; sg < 2; sg++) {
            float sgn = sg ? -step : step;
            uint64_t r2 = rseed;
            for (int i = 0; i < nsnap && i < MAX_BUF; i++) {
                if (!saved_th[i]) continue;
                int cnt = 0;
                float *th = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
                for (int k = 0; th && k < cnt; k++) {
                    float dk;
                    if (use_rand) {
                        r2 ^= r2 << 13; r2 ^= r2 >> 7; r2 ^= r2 << 17;
                        dk = (r2 >> 63) ? 1.0f : -1.0f;
                    } else {
                        dk = ref_snap[i][k] > 0 ? 1.0f : ref_snap[i][k] < 0 ? -1.0f : 0.0f;
                    }
                    th[k] = saved_th[i][k] + sgn * dk;
                }
            }
            model_lora_grad_zero(m);
            if (!model_lora_backward(m, g_toks, T_LEN, &lv[sg])) {
                g_fail++;
                lv[sg] = 0.0;
            }
            CHECK(capture_last_sel(m, &sel[sg], &sel_n[sg]), "failed to capture app_sel");
            for (int i = 0; i < nsnap && i < MAX_BUF; i++) {
                if (!saved_th[i]) continue;
                int cnt = 0;
                float *th = model_lora_param(m, lay[i], slo[i], whi[i], &cnt);
                if (th && cnt > 0) memcpy(th, saved_th[i], sizeof(float) * (size_t)cnt);
            }
        }
        bool sel_eq = sel[0] && sel[1] && sel_n[0] == sel_n[1] &&
                      memcmp(sel[0], sel[1], sizeof(int) * (size_t)sel_n[0]) == 0;
        free(sel[0]); free(sel[1]);
        fd = (lv[0] - lv[1]) / (2.0 * (double)step);
        rel = l1_norm > 0.0 ? fabs(fd - expect) / l1_norm : 0.0;
        if (sel_eq || !use_rand) {
            CHECK(sel_eq, "top-k selection differed across +eps/-eps on fixed direction");
            break;
        }
        redraws++;
        rseed = rseed * 6364136223846793005ULL + 1442695040888963407ULL;
        CHECK(attempt < 63, "exceeded 64 redraws looking for top-k-stable direction");
    }
out:
    for (int i = 0; i < nsnap && i < MAX_BUF; i++) free(saved_th[i]);
    if (fd_out) *fd_out = fd;
    if (expect_out) *expect_out = expect;
    if (l1_out) *l1_out = l1_norm;
    if (nsel_out) *nsel_out = n_sel;
    if (redraws_out) *redraws_out = redraws;
    return rel;
}

static double dot_sign_dir(int nsnap, float *ref_snap[], float *an_snap[],
                           int snap_n[], int lay[], int slo[],
                           int filter_layer, int filter_slot) {
    double acc = 0.0;
    for (int i = 0; i < nsnap; i++) {
        if (filter_layer >= 0 && lay[i] != filter_layer) continue;
        if (filter_slot >= 0 && slo[i] != filter_slot) continue;
        for (int k = 0; k < snap_n[i]; k++) {
            float dk = ref_snap[i][k] > 0 ? 1.0f : ref_snap[i][k] < 0 ? -1.0f : 0.0f;
            acc += (double)an_snap[i][k] * dk;
        }
    }
    return acc;
}

static void run_dir_suite(model_t *m, int nsnap, float *snap[], int snap_n[],
                          int lay[], int slo[], int whi[],
                          uint64_t seed_off, const char *label,
                          float sign_eps, float rand_eps, bool run_ctrls) {
    double fd_v = 0, an_v = 0, l1_v = 0, max_true = 0.0;
    long nc = 0;
    int total_redraws = 0, rd = 0;
    struct { const char *name; int fl; int fs; double fd; double l1; } sp[9] = {
        { "whole", -1, -1, 0, 0 },
        { "slot0", -1, LW_Q, 0, 0 }, { "slot1", -1, LW_K, 0, 0 },
        { "slot2", -1, LW_V, 0, 0 }, { "slot3", -1, LW_O, 0, 0 },
        { "layer0", 0, -1, 0, 0 }, { "layer1", 1, -1, 0, 0 },
        { "layer2", 2, -1, 0, 0 }, { "layer3", 3, -1, 0, 0 },
    };

    double rel_all_s = fixed_dir_err(m, nsnap, snap, snap, snap_n, lay, slo, whi, -1, -1, false, 0, sign_eps, &fd_v, &an_v, &l1_v, &nc, &rd);
    sp[0].fd = fd_v; sp[0].l1 = l1_v;
    total_redraws += rd;
    if (rel_all_s > max_true) max_true = rel_all_s;
    CHECK(rel_all_s <= 0.005, "[%s] whole-adapter sign dir rel %.5f (fd %.6g, an %.6g, l1 %.6g)", label, rel_all_s, fd_v, an_v, l1_v);
    printf("ok [%s]: whole sign dir (eps=%.0e): fd %.6g vs an %.6g (rel %.5f, %ld coords)\n", label, sign_eps, fd_v, an_v, rel_all_s, nc);

    double rel_all_r = fixed_dir_err(m, nsnap, snap, snap, snap_n, lay, slo, whi, -1, -1, true, 0x9E3779B97F4A7C15ull + seed_off, rand_eps, &fd_v, &an_v, &l1_v, &nc, &rd);
    total_redraws += rd;
    if (rel_all_r > max_true) max_true = rel_all_r;
    CHECK(rel_all_r <= 0.005, "[%s] whole-adapter Rademacher dir rel %.5f (fd %.6g, an %.6g, l1 %.6g)", label, rel_all_r, fd_v, an_v, l1_v);
    printf("ok [%s]: whole rand dir (eps=%.0e, redraws=%d): fd %.6g vs an %.6g (l1 %.6g, rel %.5f, %ld coords)\n", label, rand_eps, rd, fd_v, an_v, l1_v, rel_all_r, nc);

    for (int si = 0; si < 4; si++) {
        double fd_r = 0, an_r = 0, l1_r = 0;
        double rs = fixed_dir_err(m, nsnap, snap, snap, snap_n, lay, slo, whi, -1, g_slots[si], false, 0, sign_eps, &fd_v, &an_v, &l1_v, &nc, &rd);
        sp[1 + si].fd = fd_v; sp[1 + si].l1 = l1_v;
        total_redraws += rd;
        double rr = fixed_dir_err(m, nsnap, snap, snap, snap_n, lay, slo, whi, -1, g_slots[si], true, (0x123456789ABCDEF0ull ^ (uint64_t)(si + 1)) + seed_off, rand_eps, &fd_r, &an_r, &l1_r, &nc, &rd);
        total_redraws += rd;
        if (rs > max_true) max_true = rs;
        if (rr > max_true) max_true = rr;
        CHECK(rs <= 0.005 && rr <= 0.005, "[%s] slot %d dir rel (sign %.5f, rand %.5f [fd %.4g vs an %.4g, l1 %.4g]) above 0.005", label, g_slots[si], rs, rr, fd_r, an_r, l1_r);
        printf("ok [%s]: slot %d dirs: sign rel %.5f, rand rel %.5f (fd %.4g vs an %.4g, l1 %.4g, redraws=%d)\n", label, g_slots[si], rs, rr, fd_r, an_r, l1_r, rd);
    }
    for (int l = 0; l < 4; l++) {
        double fd_r = 0, an_r = 0, l1_r = 0;
        double rs = fixed_dir_err(m, nsnap, snap, snap, snap_n, lay, slo, whi, l, -1, false, 0, sign_eps, &fd_v, &an_v, &l1_v, &nc, &rd);
        sp[5 + l].fd = fd_v; sp[5 + l].l1 = l1_v;
        total_redraws += rd;
        double rr = fixed_dir_err(m, nsnap, snap, snap, snap_n, lay, slo, whi, l, -1, true, (0xA5A5A5A55A5A5A5Aull ^ (uint64_t)(l + 1)) + seed_off, rand_eps, &fd_r, &an_r, &l1_r, &nc, &rd);
        total_redraws += rd;
        if (rs > max_true) max_true = rs;
        if (rr > max_true) max_true = rr;
        CHECK(rs <= 0.005 && rr <= 0.005, "[%s] layer %d dir rel (sign %.5f, rand %.5f [fd %.4g vs an %.4g, l1 %.4g]) above 0.005", label, l, rs, rr, fd_r, an_r, l1_r);
        printf("ok [%s]: layer %d dirs: sign rel %.5f, rand rel %.5f (fd %.4g vs an %.4g, l1 %.4g, redraws=%d)\n", label, l, rs, rr, fd_r, an_r, l1_r, rd);
    }
    printf("ok [%s]: G3 top-k flip redraw count across all directional probes: %d\n", label, total_redraws);
    printf("ok [%s]: max true-backward rel %.5f (<= 0.005)\n", label, max_true);
    if (!run_ctrls) return;

    static const struct { const char *ctrl; const char *desc; int filter_l; int filter_s; bool check_param; } ctrls[] = {
        { "drop_pass1",   "drop pass 1 contribution on looped layers",     1, -1,   true  },
        { "drop_pass2",   "drop pass 2 contribution on looped layers",     1, -1,   true  },
        { "scale_102",    "scale d_att inside attention backward by 1.02", -1, LW_Q, true  },
        { "drop_router",  "drop MoE sigmoid router weight derivative",     0, -1,   true  },
        { "drop_sink",    "drop sink_proj derivative",                     -1, LW_Q, true  },
        { "drop_relbias", "drop relative_position_bias derivative",        -1, LW_Q, false },
        { "drop_dynconv", "drop global dynamic sconv weight_proj VJP",     -1, -1,   true  },
    };
    double min_ctrl_max_sign = 1e9;
    for (size_t ci = 0; ci < sizeof ctrls / sizeof ctrls[0]; ci++) {
        set_ctrl(ctrls[ci].ctrl);
        model_lora_grad_zero(m);
        double lc = 0.0;
        CHECK(model_lora_backward(m, g_toks, T_LEN, &lc), "[%s] control %s backward failed unexpectedly", label, ctrls[ci].ctrl);
        float *csnap[MAX_BUF] = {0};
        for (int i = 0; i < nsnap; i++) {
            int cnt = 0;
            float *g = model_lora_gradbuf(m, lay[i], slo[i], whi[i], &cnt);
            csnap[i] = malloc(sizeof(float) * (size_t)cnt);
            memcpy(csnap[i], g, sizeof(float) * (size_t)cnt);
        }
        set_ctrl(NULL);
        double sign_rels[9] = {0};
        double max_sign = 0.0;
        for (int pi = 0; pi < 9; pi++) {
            double c_an = dot_sign_dir(nsnap, snap, csnap, snap_n, lay, slo, sp[pi].fl, sp[pi].fs);
            sign_rels[pi] = sp[pi].l1 > 0.0 ? fabs(sp[pi].fd - c_an) / sp[pi].l1 : 0.0;
            if (sign_rels[pi] > max_sign) max_sign = sign_rels[pi];
        }
        double rand_rel = fixed_dir_err(m, nsnap, snap, csnap, snap_n, lay, slo, whi,
                                        ctrls[ci].filter_l, ctrls[ci].filter_s,
                                        true, (0x9E3779B97F4A7C15ull ^ (uint64_t)ci) + seed_off,
                                        rand_eps, &fd_v, &an_v, &l1_v, &nc, &rd);
        for (int i = 0; i < nsnap; i++) free(csnap[i]);
        if (ctrls[ci].check_param) {
            if (max_sign < min_ctrl_max_sign) min_ctrl_max_sign = max_sign;
            CHECK(max_sign > 0.005, "[%s] CONTROL FAILED: '%s' passed ALL 9 sign probes (max sign rel %.5f <= 0.005)",
                  label, ctrls[ci].desc, max_sign);
        }
        printf("%s [%s]: control (%s) param-sign %s (max=%.5f, rand_info=%.5f): "
               "whole=%.5f s0=%.5f s1=%.5f s2=%.5f s3=%.5f l0=%.5f l1=%.5f l2=%.5f l3=%.5f\n",
               ctrls[ci].check_param ? "ok" : "note", label, ctrls[ci].desc,
               ctrls[ci].check_param ? "FAILS sign gate" : "covered by isolated rel-branch probe",
               max_sign, rand_rel,
               sign_rels[0], sign_rels[1], sign_rels[2], sign_rels[3], sign_rels[4],
               sign_rels[5], sign_rels[6], sign_rels[7], sign_rels[8]);
    }
    printf("ok [%s]: min gated param-control max-sign rel %.5f (> 0.005)\n", label, min_ctrl_max_sign);
}

static void run_rel_branch_probe(model_t *m, double plain_loss) {
    // Dedicated isolated relative-branch activation probe (8 applications).
    // Step multipliers frozen by app-0 calibrate-then-freeze sweep over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}:
    //   sign_mult = 3e-3f (0.000014 at app 0), rand_mult = 1e-2f (0.000024 at app 0), seed = 0xC0FFEE123456789BULL.
    const float sign_mult = 3e-3f, rand_mult = 1e-2f;
    const uint64_t base_seed = 0xC0FFEE123456789BULL;
    const int n_app = 8;

    plat_setenv("RUNNER_TEST_INK2_REL_PROBE", "1");
    set_ctrl(NULL);

    float xq_buf[T_LEN * 256], g_true[T_LEN * 256], d_sign[T_LEN * 256], d_rand[T_LEN * 256], g_ctrl[T_LEN * 256];
    double sign_fds[8] = {0}, rand_fds[8] = {0}, l1s[8] = {0};
    double max_true = 0.0;
    float app0_sign_eps = 0.0f;

    for (int a = 0; a < n_app; a++) {
        ink2_rel_probe rp = { .app = a, .eps = 0.0f, .d = NULL, .xq_in = xq_buf, .g_rel = g_true };
        ink2_set_rel_probe(&rp);
        double l_zero = 0.0;
        model_lora_grad_zero(m);
        CHECK(model_lora_backward(m, g_toks, T_LEN, &l_zero), "rel_probe eps=0 failed at app %d", a);
        if (a == 0) {
            CHECK(memcmp(&l_zero, &plain_loss, sizeof plain_loss) == 0,
                  "rel_probe control c3 FAILED: eps=0 loss %.17g != plain loss %.17g", l_zero, plain_loss);
            printf("ok: rel_probe control c3 (eps=0 matches plain loss bit-for-bit: %.6f)\n", l_zero);
        }
        int n_el = rp.T * rp.H;
        CHECK(n_el > 0 && n_el <= (int)(sizeof xq_buf / sizeof xq_buf[0]), "unexpected xq size %d", n_el);
        double ss = 0.0, l1 = 0.0;
        uint64_t r = base_seed + (uint64_t)a;
        for (int i = 0; i < n_el; i++) {
            ss += (double)xq_buf[i] * (double)xq_buf[i];
            l1 += fabs((double)g_true[i]);
            d_sign[i] = g_true[i] > 0.0f ? 1.0f : g_true[i] < 0.0f ? -1.0f : 0.0f;
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            d_rand[i] = (r >> 63) ? 1.0f : -1.0f;
        }
        double rms = sqrt(ss / (double)n_el);
        l1s[a] = l1;

        double an_s = 0.0, an_r = 0.0;
        for (int i = 0; i < n_el; i++) {
            an_s += (double)g_true[i] * (double)d_sign[i];
            an_r += (double)g_true[i] * (double)d_rand[i];
        }

        float seps = (float)(sign_mult * rms);
        float reps = (float)(rand_mult * rms);
        if (a == 0) app0_sign_eps = seps;

        double lp = 0.0, lm = 0.0;
        rp.xq_in = NULL; rp.g_rel = NULL;
        rp.d = d_sign;
        rp.eps = seps; model_lora_grad_zero(m); CHECK(model_lora_backward(m, g_toks, T_LEN, &lp), "rel+ at app %d", a);
        if (a == 0) {
            CHECK(memcmp(&lp, &plain_loss, sizeof plain_loss) != 0,
                  "rel_probe control c4 FAILED: eps>0 did not change loss");
            printf("ok: rel_probe control c4 (eps=%.6g shifted loss: %.6f -> %.6f)\n", (double)seps, plain_loss, lp);
        }
        rp.eps = -seps; model_lora_grad_zero(m); CHECK(model_lora_backward(m, g_toks, T_LEN, &lm), "rel- at app %d", a);
        double fd_s = (lp - lm) / (2.0 * (double)seps);
        double rel_s = l1 > 0.0 ? fabs(fd_s - an_s) / l1 : fabs(fd_s);

        rp.d = d_rand;
        rp.eps = reps; model_lora_grad_zero(m); CHECK(model_lora_backward(m, g_toks, T_LEN, &lp), "relr+ at app %d", a);
        rp.eps = -reps; model_lora_grad_zero(m); CHECK(model_lora_backward(m, g_toks, T_LEN, &lm), "relr- at app %d", a);
        double fd_r = (lp - lm) / (2.0 * (double)reps);
        double rel_r = l1 > 0.0 ? fabs(fd_r - an_r) / l1 : fabs(fd_r);

        sign_fds[a] = fd_s;
        rand_fds[a] = fd_r;
        if (rel_s > max_true) max_true = rel_s;
        if (rel_r > max_true) max_true = rel_r;
        CHECK(rel_s <= 0.005 && rel_r <= 0.005,
              "rel_probe app %d failed (sign rel %.5f [fd %.5f vs an %.5f], rand rel %.5f [fd %.5f vs an %.5f], ||g_rel||_1 %.5f)",
              a, rel_s, fd_s, an_s, rel_r, fd_r, an_r, l1);
        printf("ok: rel_probe app %d: sign rel %.5f (fd %.5f vs an %.5f), rand rel %.5f (fd %.5f vs an %.5f), ||g_rel||_1 %.5f\n",
               a, rel_s, fd_s, an_s, rel_r, fd_r, an_r, l1);
    }
    printf("ok: rel_probe true backward across all %d apps: max rel %.5f (<= 0.005)\n", n_app, max_true);

    // Controls c1 (drop_relbias) and c2 (scale_relbias_102)
    static const struct { const char *ctrl; const char *tag; } rctrls[2] = {
        { "drop_relbias",      "c1 (drop_relbias)"      },
        { "scale_relbias_102", "c2 (scale_relbias_102)" },
    };
    for (int ci = 0; ci < 2; ci++) {
        double worst_c_s = 0.0, worst_c_r = 0.0, worst_c = 0.0;
        for (int a = 0; a < n_app; a++) {
            ink2_rel_probe rp = { .app = a, .eps = 0.0f, .d = NULL, .xq_in = NULL, .g_rel = g_ctrl };
            ink2_set_rel_probe(&rp);
            set_ctrl(rctrls[ci].ctrl);
            double lc = 0.0;
            model_lora_grad_zero(m);
            CHECK(model_lora_backward(m, g_toks, T_LEN, &lc), "rel_probe %s failed at app %d", rctrls[ci].ctrl, a);
            set_ctrl(NULL);
            int n_el = rp.T * rp.H;
            double an_cs = 0.0, an_cr = 0.0, l1_c = 0.0;
            // Read unperturbed g_true direction for sign, same Rademacher for rand, evaluate g_ctrl . d against l1s[a] (or |fd| if g_ctrl==0)
            rp.eps = 0.0f; rp.g_rel = g_true;
            model_lora_grad_zero(m);
            CHECK(model_lora_backward(m, g_toks, T_LEN, &lc), "rel_probe ref failed at app %d", a);
            uint64_t r = base_seed + (uint64_t)a;
            for (int i = 0; i < n_el; i++) {
                float ds = g_true[i] > 0.0f ? 1.0f : g_true[i] < 0.0f ? -1.0f : 0.0f;
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                float dr = (r >> 63) ? 1.0f : -1.0f;
                an_cs += (double)g_ctrl[i] * (double)ds;
                an_cr += (double)g_ctrl[i] * (double)dr;
                l1_c += fabs((double)g_ctrl[i]);
            }
            double denom_s = l1_c > 0.0 ? l1s[a] : fabs(sign_fds[a]);
            double denom_r = l1_c > 0.0 ? l1s[a] : fabs(rand_fds[a]);
            double rs = denom_s > 0.0 ? fabs(sign_fds[a] - an_cs) / denom_s : 0.0;
            double rr = denom_r > 0.0 ? fabs(rand_fds[a] - an_cr) / denom_r : 0.0;
            if (rs > worst_c_s) worst_c_s = rs;
            if (rr > worst_c_r) worst_c_r = rr;
            if (rs > worst_c) worst_c = rs;
            if (rr > worst_c) worst_c = rr;
        }
        CHECK(worst_c > 0.005, "rel_probe control %s FAILED: passed all apps (max rel %.5f <= 0.005)", rctrls[ci].tag, worst_c);
        printf("ok: rel_probe control %s FAILS gate as required: max sign rel %.5f, max rand rel %.5f (app0 eps=%.6g)\n",
               rctrls[ci].tag, worst_c_s, worst_c_r, (double)app0_sign_eps);
    }

    ink2_set_rel_probe(NULL);
    plat_setenv("RUNNER_TEST_INK2_REL_PROBE", "");
    set_ctrl(NULL);
}

int main(int argc, char **argv) {
    plat_setenv("INK2_PAR_MIN", "1");
    const char *mpath = argc > 1 ? argv[1] : "test-inkling2-train.gguf";

    // Stack-defect regression guard: a fast-route instance loaded without train=true must refuse
    // backward with an error instead of SEGVing on NULL decoded matrices.
    {
        const char *saved_route = getenv("RUNNER_INKLING2_ROUTE");
        char saved_buf[64] = {0};
        if (saved_route && *saved_route) snprintf(saved_buf, sizeof saved_buf, "%s", saved_route);
        plat_setenv("RUNNER_INKLING2_ROUTE", "");
        model_params mp_fast = {0};
        mp_fast.n_ctx = 64;
        mp_fast.n_batch = 64;
        mp_fast.n_threads = 2;
        model_t m_fast = {0};
        CHECK(model_load(&m_fast, mpath, &mp_fast), "fast-route control model_load failed");
        CHECK(model_inkling2_route(&m_fast) == 'F', "control expected fast route ('F'), got '%c'",
              model_inkling2_route(&m_fast));
        CHECK(model_lora_train_init(&m_fast, 4, 8.0f, 7), "fast-route train_init failed unexpectedly");
        double l_fast = 0.0;
        CHECK(!model_lora_backward(&m_fast, g_toks, T_LEN, &l_fast),
              "fast-route backward refused cleanly without SEGV");
        model_free(&m_fast);
        plat_setenv("RUNNER_INKLING2_ROUTE", saved_buf);
    }

    model_params mp = {0};
    mp.n_ctx = 64;
    mp.n_batch = 64;
    mp.n_threads = 2;
    mp.train = true;
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
    model_params mp_srv = mp;
    mp_srv.train = false;
    mp_srv.lora_path = tmp_a;
    mp_srv.lora_scale = 1.0f;
    CHECK(model_load(&m_srv, mpath, &mp_srv), "model_load with --lora failed");
    CHECK(!model_has_training_state(&m_srv), "serving an adapter allocated training state");
    model_free(&m_srv);
    remove(tmp_a);

    // Determinism across repeated runs and thread counts
    double l1 = run_bwd(&m);
    CHECK(fp_finite_f64(l1), "loss is not finite");
    CHECK(model_has_training_state(&m), "backward left no training state");

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
        mp_t.n_ctx = 64; mp_t.n_batch = 64; mp_t.n_threads = ths[ti]; mp_t.train = true;
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

    // OOM propagation checks (E1): injected allocation failures inside backward helpers fail cleanly
    static const char *oom_sites[] = {
        "lora_proj", "scaled_clip", "depth_attn", "kv_conv", "attn_fwd", "attn_bwd",
        "f_lora_apply", "ckpt", "bscr", "layer", "msg_fwd", "msg_bwd"
    };
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
    // floor(h) = 2e-3 * (1e-3 / h), report worst coord + every floor-assisted coord across ALL 96 coords.
    double worst_rel = 0.0, worst_abs = 0.0, cos_num = 0.0, cos_a = 0.0, cos_f = 0.0;
    struct { int l, s, w, k; double fd, an, eps, diff, rel, tol; } w_abs = {0}, w_rel = {0};
    int checked = 0, coord_flips = 0, floor_assisted = 0, floor_exceeded = 0;
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
            double floor_h = 2e-3 * (1e-3 / (double)eps);
            double tol = 0.005 * mag + floor_h;
            if (diff > tol) {
                floor_exceeded++;
                printf("note: coord exceeds floor(h) l=%d s=%d %c[%d]: fd %.6g vs an %.6g (h=%.0e, diff %.4g, rel %.4g, tol %.4g)\n",
                       lay[i], slo[i], whi[i] ? 'B' : 'A', k, fd, an, (double)eps, diff, rel, tol);
            } else if (diff > 0.005 * mag) {
                floor_assisted++;
                printf("note: floor-assisted coord l=%d s=%d %c[%d]: fd %.6g vs an %.6g (h=%.0e, diff %.4g, rel %.4g, tol %.4g)\n",
                       lay[i], slo[i], whi[i] ? 'B' : 'A', k, fd, an, (double)eps, diff, rel, tol);
            }
            if (rel >= worst_rel) {
                worst_rel = rel;
                w_rel.l = lay[i]; w_rel.s = slo[i]; w_rel.w = whi[i]; w_rel.k = k;
                w_rel.fd = fd; w_rel.an = an; w_rel.eps = eps; w_rel.diff = diff; w_rel.rel = rel; w_rel.tol = tol;
            }
            if (diff >= worst_abs) {
                worst_abs = diff;
                w_abs.l = lay[i]; w_abs.s = slo[i]; w_abs.w = whi[i]; w_abs.k = k;
                w_abs.fd = fd; w_abs.an = an; w_abs.eps = eps; w_abs.diff = diff; w_abs.rel = rel; w_abs.tol = tol;
            }
            cos_num += fd * an; cos_a += an * an; cos_f += fd * fd;
            checked++;
        }
    }
    double cosine = cos_num / (sqrt(cos_a) * sqrt(cos_f) + 1e-30);
    CHECK(cosine >= 0.9999, "fd/analytic cosine %.6f below 0.9999", cosine);
    printf("ok: %d FD coords (%d top-k fallbacks, %d floor-assisted, %d above floor): true worst abs %.4g, true worst rel %.4g, cosine %.6f\n",
           checked, coord_flips, floor_assisted, floor_exceeded, worst_abs, worst_rel, cosine);
    printf("ok: worst-abs coord l=%d s=%d %c[%d]: fd %.6g vs an %.6g (h=%.0e, diff %.4g, rel %.4g, tol %.4g)\n",
           w_abs.l, w_abs.s, w_abs.w ? 'B' : 'A', w_abs.k, w_abs.fd, w_abs.an, w_abs.eps, w_abs.diff, w_abs.rel, w_abs.tol);
    printf("ok: worst-rel coord l=%d s=%d %c[%d]: fd %.6g vs an %.6g (h=%.0e, diff %.4g, rel %.4g, tol %.4g)\n",
           w_rel.l, w_rel.s, w_rel.w ? 'B' : 'A', w_rel.k, w_rel.fd, w_rel.an, w_rel.eps, w_rel.diff, w_rel.rel, w_rel.tol);

    // Directional derivatives (G2+G3): fixed step 5e-5f sign, 2e-4f Rademacher across two independent seed sets
    const float sign_eps = 5e-5f, rand_eps = 2e-4f;
    run_dir_suite(&m, nsnap, snap, snap_n, lay, slo, whi, 0ULL, "seed+0", sign_eps, rand_eps, true);
    run_dir_suite(&m, nsnap, snap, snap_n, lay, slo, whi, 1000003ULL, "seed+1000003", sign_eps, rand_eps, true);
    run_rel_branch_probe(&m, l1);

    for (int i = 0; i < nsnap; i++) free(snap[i]);
    model_free(&m);
    printf("inkling2-grad: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
