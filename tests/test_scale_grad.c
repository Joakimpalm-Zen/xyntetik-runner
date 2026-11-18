// R4.41 Phase 0 gates (a) & (b):
//   (a) Per-block fp16 scale decomposition and in-place GGUF writer for
//       K-quants (Q3_K, Q4_K, Q5_K, Q6_K) plus IQ3_S:
//       - delta = 0 reconstructs dequant_row bit-for-bit (0 ULP, memcmp == 0)
//       - non-zero delta patch + dequant matches unit reconstruction
//       - model_scales_write preserves every byte outside trainable block d/dmin
//         fields (checked by full-file comparison)
//   (b) Scale gradient through existing KD backward (model_lora_backward_kd) on
//       dense SiLU (qwen3) and qwen35, at theta == 0 AND non-zero theta:
//       - cross-thread determinism (-t 1 vs -t 4): bit-equal loss & grad buffers
//       - directional derivative finite-difference arbiter (sign sweep on f16 KV,
//         random V sweep on f32 KV)
//       - named refusal when a model has no trainable K-quant/IQ3_S tensors
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

enum { P = 4, T = 14, K = 5 };

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static double urand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) * (1.0 / 9007199254740992.0);
}

static void make_teacher(const model_t *m, const int32_t *toks, int k,
                         int32_t *ids, float *lp) {
    int V = m->n_vocab;
    for (int t = 0; t + 1 < T; t++) {
        int32_t *row = ids + (size_t)t * k;
        float *lrow = lp + (size_t)t * k;
        row[0] = toks[t + 1];
        for (int i = 1; i < k; i++) {
            bool dup;
            do {
                row[i] = (int32_t)(urand() * V);
                dup = false;
                for (int j = 0; j < i; j++) dup |= row[j] == row[i];
            } while (dup || row[i] >= V);
        }
        double w[32], sum = 0.0;
        for (int i = 0; i < k; i++) {
            w[i] = 0.2 + urand() * (i == 0 ? 3.0 : 1.0);
            sum += w[i];
        }
        double scale = 0.85 / sum;
        for (int i = 0; i < k; i++) lrow[i] = (float)log(w[i] * scale);
    }
}

static model_t load_model(const char *path, int threads, bool kv_f32) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = threads;
    p.n_ctx = 64;
    p.n_batch = 32;
    p.kv_f32 = kv_f32;
    model_t m;
    memset(&m, 0, sizeof m);
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", path);
        exit(1);
    }
    return m;
}

static double run_kd(model_t *m, const int32_t *toks, const float *w,
                     const lora_teacher *tch) {
    double loss = 0.0;
    model_scales_grad_zero(m);
    if (!model_lora_backward_kd(m, toks, T, w, tch, &loss)) {
        CHECK(0, "model_lora_backward_kd failed");
        return 0.0;
    }
    return loss;
}

static void verify_block_decomp_and_writer(const char *path, const char *label) {
    model_t m = load_model(path, 2, false);
    char why[256] = "";
    CHECK(model_scales_train_init(&m, why, sizeof why), "%s: train_init: %s", label, why);
    uint64_t params = 0;
    int nt = model_scales_count(&m, &params);
    CHECK(nt > 0 && params > 0, "%s: expected >0 trainable scale tensors/params", label);

    int checked_blocks = 0;
    float orig_w[256], zero_w[256], unit_a[256], unit_b[256], nz_w[256], patch_w[256];
    uint8_t blk_patched[256];
    for (int l = 0; l < m.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            int64_t nb = 0;
            float *th_d = model_scales_param(&m, l, s, 0, &nb);
            if (!th_d) continue;
            float *th_m = model_scales_param(&m, l, s, 1, NULL);
            const gguf_tensor *t = NULL;
            // find matching tensor in gf by layer prefix/dimensions; read via layer struct
            (void)t;
            // Inspect actual blocks using quants_scale_info via model layer tensors
            const layer_t *ly = &m.layers[l];
            const gguf_tensor *cands[LW_SLOTS] = {
                ly->wq, ly->wk, ly->wv, ly->wo, ly->w_gate, ly->w_up, ly->w_down,
                ly->wqkv, ly->wq_gate, ly->ssm_beta, ly->ssm_alpha, ly->ssm_out
            };
            const gguf_tensor *wt = cands[s];
            const scale_qinfo *si = quants_scale_info((int)wt->type);
            CHECK(si != NULL, "%s: missing scale_qinfo for layer %d slot %d", label, l, s);
            int sample = nb < 6 ? (int)nb : 6;
            for (int b = 0; b < sample; b++) {
                const uint8_t *blk = (const uint8_t *)wt->data + (size_t)b * (size_t)si->blk_bytes;
                dequant_row(si->type, blk, orig_w, 256);
                scale_block_reconstruct(si, blk, 0.0f, 0.0f, zero_w);
                CHECK(memcmp(orig_w, zero_w, sizeof orig_w) == 0,
                      "%s: delta=0 reconstruction is not 0 ULP (l=%d s=%d b=%d)", label, l, s, b);

                float d0 = 0.0f, m0 = 0.0f;
                scale_block_read(si, blk, &d0, &m0);
                scale_block_units(si, blk, unit_a, unit_b);
                double max_abs = 0.0, max_ref = 1e-6;
                for (int i = 0; i < 256; i++) {
                    double ref = (double)orig_w[i];
                    double got = (double)d0 * (double)unit_a[i] +
                                 (si->off_dmin >= 0 ? (double)m0 * (double)unit_b[i] : 0.0);
                    double err = fabs(got - ref);
                    if (err > max_abs) max_abs = err;
                    if (fabs(ref) > max_ref) max_ref = fabs(ref);
                }
                CHECK(max_abs <= 1e-5 * max_ref,
                      "%s: unit decomp mismatch l=%d s=%d b=%d: err %.3e ref %.3e",
                      label, l, s, b, max_abs, max_ref);

                // Non-zero perturbation: patch to fp16, then reconstruct with the fp16-quantized log delta.
                float dd = 0.15f, dm = si->off_dmin >= 0 ? -0.10f : 0.0f;
                CHECK(scale_block_patch(si, blk, dd, dm, blk_patched),
                      "%s: scale_block_patch failed", label);
                dequant_row(si->type, blk_patched, patch_w, 256);
                float dq_d = 0.0f, dq_m = 0.0f;
                scale_block_read(si, blk_patched, &dq_d, &dq_m);
                float qdd = (d0 != 0.0f && dq_d != 0.0f) ? logf(dq_d / d0) : 0.0f;
                float qdm = (si->off_dmin >= 0 && m0 != 0.0f && dq_m != 0.0f) ? logf(dq_m / m0) : 0.0f;
                scale_block_reconstruct(si, blk, qdd, qdm, nz_w);
                double nz_err = 0.0;
                for (int i = 0; i < 256; i++) {
                    double err = fabs((double)patch_w[i] - (double)nz_w[i]);
                    if (err > nz_err) nz_err = err;
                }
                CHECK(nz_err <= 2e-5 * max_ref,
                      "%s: patched block dequant vs reconstruct mismatch %.3e", label, nz_err);
                checked_blocks++;
            }
            (void)th_m;
        }
    }
    CHECK(checked_blocks > 0, "%s: checked 0 blocks", label);

    // Whole-file write test: 0-delta must be byte-identical; non-zero must alter ONLY scale bytes.
    char out0[256], out1[256];
    snprintf(out0, sizeof out0, "test-scale-zero-%s.gguf", label);
    snprintf(out1, sizeof out1, "test-scale-nz-%s.gguf", label);
    CHECK(model_scales_write(&m, path, out0, why, sizeof why), "%s: zero write: %s", label, why);

    FILE *fs = fopen(path, "rb"), *f0 = fopen(out0, "rb");
    CHECK(fs && f0, "%s: cannot reopen written files", label);
    fseek(fs, 0, SEEK_END); long szs = ftell(fs); rewind(fs);
    fseek(f0, 0, SEEK_END); long sz0 = ftell(f0); rewind(f0);
    CHECK(szs == sz0, "%s: size mismatch %ld vs %ld", label, szs, sz0);
    uint8_t *bs = malloc((size_t)szs), *b0 = malloc((size_t)sz0);
    CHECK(bs && b0 && fread(bs, 1, (size_t)szs, fs) == (size_t)szs &&
          fread(b0, 1, (size_t)sz0, f0) == (size_t)sz0, "%s: read failed", label);
    fclose(fs); fclose(f0);
    CHECK(memcmp(bs, b0, (size_t)szs) == 0, "%s: zero-delta written file differs from source", label);
    free(b0);

    // Non-zero perturbations across every trainable block
    for (int l = 0; l < m.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            int64_t nb = 0;
            float *th_d = model_scales_param(&m, l, s, 0, &nb);
            if (!th_d) continue;
            float *th_m = model_scales_param(&m, l, s, 1, NULL);
            for (int64_t i = 0; i < nb; i++) {
                th_d[i] = (float)((urand() - 0.5) * 0.25);
                if (th_m) th_m[i] = (float)((urand() - 0.5) * 0.25);
            }
        }
    }
    CHECK(model_scales_write(&m, path, out1, why, sizeof why), "%s: nz write: %s", label, why);
    FILE *f1 = fopen(out1, "rb");
    CHECK(f1 != NULL, "%s: cannot open %s", label, out1);
    fseek(f1, 0, SEEK_END); long sz1 = ftell(f1); rewind(f1);
    CHECK(sz1 == szs, "%s: nz size mismatch", label);
    uint8_t *b1 = malloc((size_t)sz1);
    CHECK(b1 && fread(b1, 1, (size_t)sz1, f1) == (size_t)sz1, "%s: nz read failed", label);
    fclose(f1);

    // Mask out trainable scale offsets in both bs and b1 -> remainder must be byte-identical!
    int n_scale_changed = 0;
    const uint8_t *base_ptr = (const uint8_t *)m.gf.map;
    for (int l = 0; l < m.n_layer; l++) {
        const layer_t *ly = &m.layers[l];
        const gguf_tensor *cands[LW_SLOTS] = {
            ly->wq, ly->wk, ly->wv, ly->wo, ly->w_gate, ly->w_up, ly->w_down,
            ly->wqkv, ly->wq_gate, ly->ssm_beta, ly->ssm_alpha, ly->ssm_out
        };
        for (int s = 0; s < LW_SLOTS; s++) {
            int64_t nb = 0;
            if (!model_scales_param(&m, l, s, 0, &nb)) continue;
            const gguf_tensor *wt = cands[s];
            const scale_qinfo *si = quants_scale_info((int)wt->type);
            uint64_t toff = (uint64_t)((const uint8_t *)wt->data - base_ptr);
            for (int64_t b = 0; b < nb; b++) {
                uint64_t boff = toff + (uint64_t)b * (uint64_t)si->blk_bytes;
                if (memcmp(bs + boff + si->off_d, b1 + boff + si->off_d, 2) != 0) n_scale_changed++;
                bs[boff + si->off_d] = 0; bs[boff + si->off_d + 1] = 0;
                b1[boff + si->off_d] = 0; b1[boff + si->off_d + 1] = 0;
                if (si->off_dmin >= 0) {
                    if (memcmp(bs + boff + si->off_dmin, b1 + boff + si->off_dmin, 2) != 0) n_scale_changed++;
                    bs[boff + si->off_dmin] = 0; bs[boff + si->off_dmin + 1] = 0;
                    b1[boff + si->off_dmin] = 0; b1[boff + si->off_dmin + 1] = 0;
                }
            }
        }
    }
    CHECK(n_scale_changed > 0, "%s: expected scale bytes to change", label);
    CHECK(memcmp(bs, b1, (size_t)szs) == 0,
          "%s: bytes OUTSIDE trainable scale fields changed in written GGUF!", label);

    free(bs); free(b1);
    remove(out0); remove(out1);
    model_free(&m);
    printf("ok: gate(a) %s: %d tensors, %llu params, 0 ULP at delta=0, %d scale fields modified, 0 non-scale bytes altered\n",
           label, nt, (unsigned long long)params, n_scale_changed);
}

static void verify_kd_grad(const char *path, const char *label, bool nonzero_init) {
    // 1. Thread-count determinism (-t 1 vs -t 4)
    model_t m1 = load_model(path, 1, false);
    model_t m4 = load_model(path, 4, true);
    char why[256] = "";
    CHECK(model_scales_train_init(&m1, why, sizeof why), "%s t1 init: %s", label, why);
    CHECK(model_scales_train_init(&m4, why, sizeof why), "%s t4 init: %s", label, why);

    // Seed theta identically if nonzero_init
    if (nonzero_init) {
        g_rng = 0x12345678abcdef01ull;
        for (int l = 0; l < m1.n_layer; l++) {
            for (int s = 0; s < LW_SLOTS; s++) {
                int64_t nb = 0;
                float *p1d = model_scales_param(&m1, l, s, 0, &nb);
                float *p4d = model_scales_param(&m4, l, s, 0, NULL);
                if (!p1d) continue;
                float *p1m = model_scales_param(&m1, l, s, 1, NULL);
                float *p4m = model_scales_param(&m4, l, s, 1, NULL);
                for (int64_t i = 0; i < nb; i++) {
                    float vd = (float)((urand() - 0.5) * 0.25);
                    p1d[i] = p4d[i] = vd;
                    if (p1m) {
                        float vm = (float)((urand() - 0.5) * 0.25);
                        p1m[i] = p4m[i] = vm;
                    }
                }
            }
        }
    }

    int32_t toks[T];
    for (int i = 0; i < T; i++) toks[i] = 3 + (i * 7) % (m1.n_vocab - 4);
    float w[T - 1];
    for (int i = 0; i < T - 1; i++) w[i] = i >= P - 1 ? 1.0f : 0.0f;
    w[P + 1] = 0.5f;

    int32_t tids[(T - 1) * K];
    float tlp[(T - 1) * K];
    g_rng = 0x2545F4914F6CDD1Dull;
    make_teacher(&m1, toks, K, tids, tlp);
    lora_teacher tch = { K, tids, tlp };

    // Thread-count invariance check (m1 vs m4_f16)
    model_t m4_f16 = load_model(path, 4, false);
    CHECK(model_scales_train_init(&m4_f16, why, sizeof why), "%s t4_f16 init: %s", label, why);
    if (nonzero_init) {
        for (int l = 0; l < m1.n_layer; l++) {
            for (int s = 0; s < LW_SLOTS; s++) {
                int64_t nb = 0;
                float *src_d = model_scales_param(&m1, l, s, 0, &nb);
                float *dst_d = model_scales_param(&m4_f16, l, s, 0, NULL);
                if (!src_d) continue;
                memcpy(dst_d, src_d, sizeof(float) * (size_t)nb);
                float *src_m = model_scales_param(&m1, l, s, 1, NULL);
                float *dst_m = model_scales_param(&m4_f16, l, s, 1, NULL);
                if (src_m) memcpy(dst_m, src_m, sizeof(float) * (size_t)nb);
            }
        }
    }
    double l_t1 = run_kd(&m1, toks, w, &tch);
    double l_t4 = run_kd(&m4_f16, toks, w, &tch);
    CHECK(l_t1 == l_t4, "%s: -t 1 loss %.17g != -t 4 loss %.17g", label, l_t1, l_t4);
    for (int l = 0; l < m1.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            for (int wh = 0; wh < 2; wh++) {
                int64_t nb = 0;
                float *g1 = model_scales_grad(&m1, l, s, wh, &nb);
                float *g4 = model_scales_grad(&m4_f16, l, s, wh, NULL);
                if (!g1) continue;
                CHECK(memcmp(g1, g4, sizeof(float) * (size_t)nb) == 0,
                      "%s: -t 1 vs -t 4 grad mismatch at l=%d s=%d wh=%d", label, l, s, wh);
            }
        }
    }
    model_free(&m4_f16);

    // 2. F32 KV sign(G) directional derivative (house arbiter, <= 1e-3) + random-V diagnostic
    double loss_f32 = run_kd(&m4, toks, w, &tch);
    float *snap_d[64][LW_SLOTS] = {{0}}, *snap_m[64][LW_SLOTS] = {{0}};
    float *v_d[64][LW_SLOTS] = {{0}}, *v_m[64][LW_SLOTS] = {{0}};
    float *p0_d[64][LW_SLOTS] = {{0}}, *p0_m[64][LW_SLOTS] = {{0}};
    double gv_expect = 0.0, f32_sign_expect = 0.0;
    double f32_sum_abs = 0.0;
    long f32_n_all = 0;
    for (int l = 0; l < m4.n_layer; l++)
        for (int s = 0; s < LW_SLOTS; s++)
            for (int wh = 0; wh < 2; wh++) {
                int64_t nb = 0;
                float *g = model_scales_grad(&m4, l, s, wh, &nb);
                if (!g) continue;
                for (int64_t i = 0; i < nb; i++) { f32_sum_abs += fabs((double)g[i]); f32_n_all++; }
            }
    double f32_thresh = f32_n_all ? f32_sum_abs / (double)f32_n_all : 0.0;
    g_rng = 0x55aa55aa11223344ull;
    for (int l = 0; l < m4.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            int64_t nb = 0;
            float *gd = model_scales_grad(&m4, l, s, 0, &nb);
            if (!gd) continue;
            float *pd = model_scales_param(&m4, l, s, 0, NULL);
            snap_d[l][s] = malloc(sizeof(float) * (size_t)nb);
            v_d[l][s] = malloc(sizeof(float) * (size_t)nb);
            p0_d[l][s] = malloc(sizeof(float) * (size_t)nb);
            memcpy(snap_d[l][s], gd, sizeof(float) * (size_t)nb);
            memcpy(p0_d[l][s], pd, sizeof(float) * (size_t)nb);
            for (int64_t i = 0; i < nb; i++) {
                v_d[l][s][i] = (float)(urand() * 2.0 - 1.0);
                gv_expect += (double)snap_d[l][s][i] * (double)v_d[l][s][i];
                if (fabs((double)snap_d[l][s][i]) > f32_thresh) f32_sign_expect += fabs((double)snap_d[l][s][i]);
            }
            float *gm = model_scales_grad(&m4, l, s, 1, NULL);
            if (gm) {
                float *pm = model_scales_param(&m4, l, s, 1, NULL);
                snap_m[l][s] = malloc(sizeof(float) * (size_t)nb);
                v_m[l][s] = malloc(sizeof(float) * (size_t)nb);
                p0_m[l][s] = malloc(sizeof(float) * (size_t)nb);
                memcpy(snap_m[l][s], gm, sizeof(float) * (size_t)nb);
                memcpy(p0_m[l][s], pm, sizeof(float) * (size_t)nb);
                for (int64_t i = 0; i < nb; i++) {
                    v_m[l][s][i] = (float)(urand() * 2.0 - 1.0);
                    gv_expect += (double)snap_m[l][s][i] * (double)v_m[l][s][i];
                    if (fabs((double)snap_m[l][s][i]) > f32_thresh) f32_sign_expect += fabs((double)snap_m[l][s][i]);
                }
            }
        }
    }
    const float eps_sweep[5] = { 1e-2f, 3e-3f, 1e-3f, 3e-4f, 1e-4f };
    double f32_sign_best = 1e30;
    for (int ei = 0; ei < 5; ei++) {
        float eps = eps_sweep[ei];
        double lp = 0.0, lm = 0.0;
        for (int sg = 0; sg < 2; sg++) {
            float mul = sg == 0 ? eps : -eps;
            for (int l = 0; l < m4.n_layer; l++) {
                for (int s = 0; s < LW_SLOTS; s++) {
                    if (!snap_d[l][s]) continue;
                    int64_t nb = 0;
                    float *pd = model_scales_param(&m4, l, s, 0, &nb);
                    for (int64_t i = 0; i < nb; i++) {
                        double g = snap_d[l][s][i];
                        pd[i] = fabs(g) > f32_thresh ? p0_d[l][s][i] + mul * (g > 0 ? 1.0f : -1.0f) : p0_d[l][s][i];
                    }
                    float *pm = model_scales_param(&m4, l, s, 1, NULL);
                    if (pm)
                        for (int64_t i = 0; i < nb; i++) {
                            double g = snap_m[l][s][i];
                            pm[i] = fabs(g) > f32_thresh ? p0_m[l][s][i] + mul * (g > 0 ? 1.0f : -1.0f) : p0_m[l][s][i];
                        }
                }
            }
            double L = run_kd(&m4, toks, w, &tch);
            if (sg == 0) lp = L; else lm = L;
        }
        double slope = (lp - lm) / (2.0 * eps);
        double rel = f32_sign_expect > 0 ? fabs(slope - f32_sign_expect) / f32_sign_expect : 0.0;
        if (rel < f32_sign_best) f32_sign_best = rel;
    }
    double best_rel = 1e30;
    for (int ei = 0; ei < 5; ei++) {
        float eps = eps_sweep[ei];
        double lp = 0.0, lm = 0.0;
        for (int sg = 0; sg < 2; sg++) {
            float mul = sg == 0 ? eps : -eps;
            for (int l = 0; l < m4.n_layer; l++) {
                for (int s = 0; s < LW_SLOTS; s++) {
                    int64_t nb = 0;
                    float *pd = model_scales_param(&m4, l, s, 0, &nb);
                    if (!pd) continue;
                    for (int64_t i = 0; i < nb; i++) pd[i] = p0_d[l][s][i] + mul * v_d[l][s][i];
                    float *pm = model_scales_param(&m4, l, s, 1, NULL);
                    if (pm)
                        for (int64_t i = 0; i < nb; i++) pm[i] = p0_m[l][s][i] + mul * v_m[l][s][i];
                }
            }
            double L = run_kd(&m4, toks, w, &tch);
            if (sg == 0) lp = L; else lm = L;
        }
        double fd = (lp - lm) / (2.0 * eps);
        double mag = fabs(gv_expect) > 1e-12 ? fabs(gv_expect) : 1e-12;
        double rel = fabs(fd - gv_expect) / mag;
        if (rel < best_rel) best_rel = rel;
    }
    // Separate dmin (theta_m-only) directional FD check so dropping dmin cannot hide behind theta_d.
    double m_sum = 0.0;
    long m_cnt = 0;
    for (int l = 0; l < m4.n_layer; l++)
        for (int s = 0; s < LW_SLOTS; s++)
            if (snap_m[l][s]) {
                int64_t nb = 0;
                (void)model_scales_param(&m4, l, s, 1, &nb);
                for (int64_t i = 0; i < nb; i++) { m_sum += fabs((double)snap_m[l][s][i]); m_cnt++; }
            }
    if (m_cnt > 0) {
        double m_thresh = m_sum / (double)m_cnt;
        double m_expect = 0.0;
        long m_sel = 0;
        for (int l = 0; l < m4.n_layer; l++)
            for (int s = 0; s < LW_SLOTS; s++)
                if (snap_m[l][s]) {
                    int64_t nb = 0;
                    (void)model_scales_param(&m4, l, s, 1, &nb);
                    for (int64_t i = 0; i < nb; i++) {
                        double g = snap_m[l][s][i];
                        if (fabs(g) > m_thresh) { m_expect += fabs(g); m_sel++; }
                    }
                }
        double m_best = 1e30;
        for (int ei = 0; ei < 5; ei++) {
            float eps = eps_sweep[ei];
            double lp = 0.0, lm = 0.0;
            for (int sg = 0; sg < 2; sg++) {
                float mul = sg == 0 ? eps : -eps;
                for (int l = 0; l < m4.n_layer; l++)
                    for (int s = 0; s < LW_SLOTS; s++)
                        if (snap_m[l][s]) {
                            int64_t nb = 0;
                            float *pm = model_scales_param(&m4, l, s, 1, &nb);
                            for (int64_t i = 0; i < nb; i++) {
                                double g = snap_m[l][s][i];
                                if (m_sel == 0)
                                    pm[i] = p0_m[l][s][i] + mul;
                                else if (fabs(g) > m_thresh)
                                    pm[i] = p0_m[l][s][i] + mul * (g > 0 ? 1.0f : -1.0f);
                                else
                                    pm[i] = p0_m[l][s][i];
                            }
                        }
                double L = run_kd(&m4, toks, w, &tch);
                if (sg == 0) lp = L; else lm = L;
            }
            double slope = (lp - lm) / (2.0 * eps);
            double denom = fmax(fabs(slope), fmax(fabs(m_expect), 1e-12));
            double rel = fabs(slope - m_expect) / denom;
            if (rel < m_best) m_best = rel;
        }
        CHECK(m_sel > 0 && m_best <= 1e-3,
              "%s (%s): f32-KV dmin (theta_m-only) directional FD rel %.3e > 1e-3 (sel %ld)",
              label, nonzero_init ? "nz" : "zero", m_best, m_sel);
    }
    for (int l = 0; l < m4.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            if (!snap_d[l][s]) continue;
            int64_t nb = 0;
            float *pd = model_scales_param(&m4, l, s, 0, &nb);
            memcpy(pd, p0_d[l][s], sizeof(float) * (size_t)nb);
            float *pm = model_scales_param(&m4, l, s, 1, NULL);
            if (pm) memcpy(pm, p0_m[l][s], sizeof(float) * (size_t)nb);
            free(snap_d[l][s]); free(v_d[l][s]); free(p0_d[l][s]);
            free(snap_m[l][s]); free(v_m[l][s]); free(p0_m[l][s]);
        }
    }
    CHECK(f32_sign_best <= 1e-3, "%s (%s): f32-KV sign directional FD rel %.3e > 1e-3",
          label, nonzero_init ? "nz" : "zero", f32_sign_best);

    // 3. F16 KV sign(G) directional sweep (matching test_kd_grad / test_lora_grad house arbiter)
    double sum_abs = 0.0;
    long n_all = 0;
    for (int l = 0; l < m1.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            for (int wh = 0; wh < 2; wh++) {
                int64_t nb = 0;
                float *g = model_scales_grad(&m1, l, s, wh, &nb);
                if (!g) continue;
                for (int64_t i = 0; i < nb; i++) { sum_abs += fabs((double)g[i]); n_all++; }
            }
        }
    }
    double thresh = n_all ? sum_abs / (double)n_all : 0.0;
    double sign_expect = 0.0;
    long n_sel = 0;
    float *g1_snap[64][LW_SLOTS][2] = {{{0}}}, *p1_save[64][LW_SLOTS][2] = {{{0}}};
    for (int l = 0; l < m1.n_layer; l++) {
        for (int s = 0; s < LW_SLOTS; s++) {
            for (int wh = 0; wh < 2; wh++) {
                int64_t nb = 0;
                float *g = model_scales_grad(&m1, l, s, wh, &nb);
                float *p = model_scales_param(&m1, l, s, wh, NULL);
                if (!g) continue;
                g1_snap[l][s][wh] = malloc(sizeof(float) * (size_t)nb);
                p1_save[l][s][wh] = malloc(sizeof(float) * (size_t)nb);
                memcpy(g1_snap[l][s][wh], g, sizeof(float) * (size_t)nb);
                memcpy(p1_save[l][s][wh], p, sizeof(float) * (size_t)nb);
                for (int64_t i = 0; i < nb; i++) {
                    if (fabs((double)g[i]) > thresh) {
                        sign_expect += fabs((double)g[i]);
                        n_sel++;
                    }
                }
            }
        }
    }
    const float steps[6] = { 1e-5f, 3e-5f, 1e-4f, 3e-4f, 1e-3f, 3e-3f };
    double sign_best = 1e30;
    for (int si = 0; si < 6; si++) {
        float e = steps[si];
        double lv[2] = { 0.0, 0.0 };
        for (int sg = 0; sg < 2; sg++) {
            float mul = sg == 0 ? e : -e;
            for (int l = 0; l < m1.n_layer; l++) {
                for (int s = 0; s < LW_SLOTS; s++) {
                    for (int wh = 0; wh < 2; wh++) {
                        if (!g1_snap[l][s][wh]) continue;
                        int64_t nb = 0;
                        float *p = model_scales_param(&m1, l, s, wh, &nb);
                        for (int64_t i = 0; i < nb; i++) {
                            double g = g1_snap[l][s][wh][i];
                            if (fabs(g) > thresh)
                                p[i] = p1_save[l][s][wh][i] + mul * (g > 0 ? 1.0f : -1.0f);
                            else
                                p[i] = p1_save[l][s][wh][i];
                        }
                    }
                }
            }
            lv[sg] = run_kd(&m1, toks, w, &tch);
        }
        double slope = (lv[0] - lv[1]) / (2.0 * (double)e);
        double rel = sign_expect > 0 ? fabs(slope - sign_expect) / sign_expect : 0.0;
        if (rel < sign_best) sign_best = rel;
    }
    for (int l = 0; l < m1.n_layer; l++)
        for (int s = 0; s < LW_SLOTS; s++)
            for (int wh = 0; wh < 2; wh++) {
                free(g1_snap[l][s][wh]);
                free(p1_save[l][s][wh]);
            }
    CHECK(sign_best <= 0.01, "%s (%s): f16-KV sign directional rel %.4f > 0.01",
          label, nonzero_init ? "nz" : "zero", sign_best);

    printf("ok: gate(b) %s (%s): loss %.4f, -t1==-t4 bit-exact, f32-KV sign rel %.2e (rand %.2e), f16-KV sign rel %.4f (%ld coords)\n",
           label, nonzero_init ? "nz" : "zero", loss_f32, f32_sign_best, best_rel, sign_best, n_sel);
    model_free(&m1);
    model_free(&m4);
}

int main(int argc, char **argv) {
    const char *q3_q3k  = argc > 1 ? argv[1] : "test-scale-qwen3-q3k.gguf";
    const char *q3_q4k  = argc > 2 ? argv[2] : "test-scale-qwen3-q4k.gguf";
    const char *q3_q5k  = argc > 3 ? argv[3] : "test-scale-qwen3-q5k.gguf";
    const char *q3_q6k  = argc > 4 ? argv[4] : "test-scale-qwen3-q6k.gguf";
    const char *q3_iq3s = argc > 5 ? argv[5] : "test-scale-iq3s.gguf";
    const char *q35_q4k = argc > 6 ? argv[6] : "test-scale-qwen35-q4k.gguf";
    const char *q35_q6k = argc > 7 ? argv[7] : "test-scale-qwen35-q6k.gguf";
    const char *f32_ref = argc > 8 ? argv[8] : "test.gguf";

    f16_init();

    verify_block_decomp_and_writer(q3_q3k,  "qwen3-q3k");
    verify_block_decomp_and_writer(q3_q4k,  "qwen3-q4k");
    verify_block_decomp_and_writer(q3_q5k,  "qwen3-q5k");
    verify_block_decomp_and_writer(q3_q6k,  "qwen3-q6k");
    verify_block_decomp_and_writer(q3_iq3s, "llama-iq3s");
    verify_block_decomp_and_writer(q35_q4k, "qwen35-q4k");

    // Gate (b): KD directional FD on dense SiLU (qwen3) & qwen35 at zero and non-zero theta
    verify_kd_grad(q3_q3k,  "qwen3-q3k",  false);
    verify_kd_grad(q3_q4k,  "qwen3-q4k",  false);
    verify_kd_grad(q3_q4k,  "qwen3-q4k",  true);
    verify_kd_grad(q3_iq3s, "llama-iq3s", true);
    verify_kd_grad(q35_q4k, "qwen35-q4k", false);
    verify_kd_grad(q35_q4k, "qwen35-q4k", true);
    verify_kd_grad(q35_q6k, "qwen35-q6k", true);

    // Refusal check: F32 model has no K-quant / IQ3_S projections
    {
        model_t mf = load_model(f32_ref, 2, false);
        char why[256] = "";
        CHECK(!model_scales_train_init(&mf, why, sizeof why),
              "F32-only model should be refused by model_scales_train_init");
        CHECK(strstr(why, "no K-quant") != NULL, "unexpected refusal message: %s", why);
        model_free(&mf);
        printf("ok: F32-only model refused cleanly: %s\n", why);
    }

    if (g_fail) {
        fprintf(stderr, "test-scale-grad: FAILED\n");
        return 1;
    }
    printf("test-scale-grad: ok\n");
    return 0;
}
