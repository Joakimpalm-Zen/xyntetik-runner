// R4.48.10: LoRA backward pass for Inkling 2 Mini (inkling2_loop), built STRICT
// (-fno-fast-math -ffp-contract=off) matching src/inkling2.c.
//
// Design:
//   - Attention-only adapters (LW_Q, LW_K, LW_V, LW_O) on all physical layers.
//   - Forward records only per-application input/output checkpoints ckpt[0..n_app]
//     of shape [T][H]; every cur[], prev[] and last_h state references one of
//     these saved slabs, so no redundant cross-pass buffers are copied.
//   - Backward walks applications in reverse order a = n_app - 1 .. 0, recomputing
//     each single application from its checkpoint, asserting recomputed h_out equals
//     saved ckpt[h_out_id] bit for bit (Condition 4), and accumulating adapter
//     gradients ga/gb in serial ascending float32 fmaf order across all passes.
//   - Gradient flows through frozen sparse MoE + SoftcappedSwiGLU + sigmoid
//     router weights, static and dynamic short convolutions, relative position
//     bias + l2_clip + log_tau + sink softmax column, gated o_proj, depth-wise
//     attention residuals, loop message_input / message_step gates, and softcapped
//     lm_head.
#include "inkling2.h"
#include "inkling2_model.h"
#include "fpclass.h"
#include "pmath.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    CTRL_DROP_PASS1   = 1 << 0,
    CTRL_DROP_PASS2   = 1 << 1,
    CTRL_SCALE_102    = 1 << 2,
    CTRL_THREAD_ORDER = 1 << 3,
    CTRL_DROP_ROUTER  = 1 << 4,
    CTRL_DROP_SINK    = 1 << 5,
    CTRL_DROP_RELBIAS     = 1 << 6,
    CTRL_DROP_DYNCONV     = 1 << 7,
    CTRL_DRIFT_CKPT       = 1 << 8,
    CTRL_SCALE_RELBIAS_102 = 1 << 9,
};

static int read_bwd_ctrl(void) {
    const char *e = getenv("RUNNER_TEST_INK2_BWD_CTRL");
    if (!e || !*e) return 0;
    int f = 0;
    if (strstr(e, "drop_pass1"))       f |= CTRL_DROP_PASS1;
    if (strstr(e, "drop_pass2"))       f |= CTRL_DROP_PASS2;
    if (strstr(e, "scale_102"))        f |= CTRL_SCALE_102;
    if (strstr(e, "thread_order_dep")) f |= CTRL_THREAD_ORDER;
    if (strstr(e, "drop_router"))      f |= CTRL_DROP_ROUTER;
    if (strstr(e, "drop_sink"))        f |= CTRL_DROP_SINK;
    if (strstr(e, "drop_relbias"))     f |= CTRL_DROP_RELBIAS;
    if (strstr(e, "scale_relbias_102")) f |= CTRL_SCALE_RELBIAS_102;
    if (strstr(e, "drop_dynconv"))     f |= CTRL_DROP_DYNCONV;
    if (strstr(e, "drift_ckpt"))       f |= CTRL_DRIFT_CKPT;
    return f;
}

static bool bwd_alloc_fail(const char *tag) {
    const char *e = getenv("RUNNER_TEST_INK2_ALLOC_FAIL");
    return e && *e && (!strcmp(e, "all") || !strcmp(e, tag));
}

static inline float ink2b_sigmoid(float x) { return 1.0f / (1.0f + p_expf(-x)); }

// ---- forward primitives duplicated/shared under strict IEEE rules for single-app recompute
static void f_matvec(const float *W, int n_out, int n_in, const float *x, float *y) {
    for (int o = 0; o < n_out; o++) {
        float a = 0.0f;
        const float *row = W + (size_t)o * n_in;
        for (int i = 0; i < n_in; i++) a += row[i] * x[i];
        y[o] = a;
    }
}

typedef struct { const float *W, *x; float *y; int n_out, n_in; } mv_par_ctx;
static void f_mv_range(void *cv, int64_t t0, int64_t t1) {
    const mv_par_ctx *c = cv;
    for (int64_t t = t0; t < t1; t++)
        f_matvec(c->W, c->n_out, c->n_in, c->x + (size_t)t * c->n_in, c->y + (size_t)t * c->n_out);
}
static void f_rows(const float *W, int n_out, int n_in, int T, const float *x, float *y) {
    mv_par_ctx c = { W, x, y, n_out, n_in };
    ink2_par(T, (int64_t)n_out * n_in, f_mv_range, &c);
}

static ink2_status f_lora_apply(const float *a, const float *b, int r, float scale, int n_out, int n_in,
                                int T, const float *x, float *y) {
    if (r <= 0 || !a || !b || scale == 0.0f) return INK2_OK;
    if (r > INK2_MAX_LORA_RANK) {
        fprintf(stderr, "error: inkling2_loop f_lora_apply: rank %d exceeds limit %d\n", r, INK2_MAX_LORA_RANK);
        return INK2_BAD_INPUT;
    }
    float tmp_stack[256];
    float *tmp = bwd_alloc_fail("f_lora_apply") ? NULL : (r <= 256 ? tmp_stack : malloc(sizeof(float) * (size_t)r));
    if (!tmp) {
        fprintf(stderr, "error: inkling2_loop f_lora_apply: out of memory allocating rank buffer\n");
        return INK2_BAD_INPUT;
    }
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * n_in;
        float *yt = y + (size_t)t * n_out;
        for (int k = 0; k < r; k++) {
            const float *ak = a + (size_t)k * n_in;
            float acc = 0.0f;
            for (int i = 0; i < n_in; i++) acc = fmaf(ak[i], xt[i], acc);
            tmp[k] = acc;
        }
        for (int j = 0; j < n_out; j++) {
            const float *bj = b + (size_t)j * r;
            float acc = 0.0f;
            for (int k = 0; k < r; k++) acc = fmaf(bj[k], tmp[k], acc);
            yt[j] = fmaf(scale, acc, yt[j]);
        }
    }
    if (tmp != tmp_stack) free(tmp);
    return INK2_OK;
}

static void f_unweighted_norm(const float *x, int n, float eps, float *y) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r;
}

static void f_rms_norm(const float *x, int n, float eps, const float *wt, float *y) {
    f_unweighted_norm(x, n, eps, y);
    for (int i = 0; i < n; i++) y[i] = wt[i] * y[i];
}

static float f_log_tau(const ink2_shape *s, int pos) {
    float a = (float)(pos + 1) / (float)s->log_floor;
    if (a < 1.0f) a = 1.0f;
    return 1.0f + s->log_alpha * p_logf(a);
}

static void f_l2_clip(float *y, int n, float thr) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += y[i] * y[i];
    float f = (1.0f / (sqrtf(ss) + 1e-6f)) * thr;
    if (f > 1.0f) f = 1.0f;
    for (int i = 0; i < n; i++) y[i] = y[i] * f;
}

static void f_scaled_clip(const ink2_shape *s, int g, float *y, const float *scale, int n, int d, float thr, int pos) {
    float tau = g ? f_log_tau(s, pos) : 1.0f;
    for (int h = 0; h < n; h++) {
        float *yh = y + (size_t)h * d;
        for (int i = 0; i < d; i++) yh[i] = yh[i] * scale[(size_t)h * d + i];
        f_l2_clip(yh, d, thr);
        if (g) for (int i = 0; i < d; i++) yh[i] = yh[i] * tau;
    }
}

static void f_short_conv(const float *cw, int C, int K, int T, const float *lin, float *out) {
    const float div = (float)sqrt((double)(K + 1));
    for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
        float a = 0.0f;
        for (int j = 0; j < K; j++) {
            int u = t - (K - 1) + j;
            if (u >= 0) a += cw[(size_t)ch * K + j] * lin[(size_t)u * C + ch];
        }
        out[(size_t)t * C + ch] = (a + lin[(size_t)t * C + ch]) / div;
    }
}

static void f_depth_attn(const float *q, int nq, const float *states, int n_kv, int H, float *normed,
                         float *scores, float *out) {
    for (int j = 0; j < n_kv; j++) f_unweighted_norm(states + (size_t)j * H, H, 1e-6f, normed + (size_t)j * H);
    const float scale = ink2_inv_sqrt(H);
    for (int i = 0; i < nq; i++) {
        const float *qi = q + (size_t)i * H;
        float mx = -FLT_MAX;
        for (int j = 0; j < n_kv; j++) {
            const float *nj = normed + (size_t)j * H;
            float d = 0.0f;
            for (int h = 0; h < H; h++) d += qi[h] * nj[h];
            scores[j] = d * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n_kv; j++) { scores[j] = p_expf(scores[j] - mx); sum += scores[j]; }
        for (int j = 0; j < n_kv; j++) scores[j] = scores[j] / sum;
        float *oi = out + (size_t)i * H;
        for (int h = 0; h < H; h++) {
            float a = 0.0f;
            for (int j = 0; j < n_kv; j++) a += scores[j] * normed[(size_t)j * H + h];
            oi[h] = a;
        }
    }
}

// ---- Backward primitives (strict ascending fmaf chains; thread-count invariant)

typedef struct { const float *W, *dy; float *dx; int n_out, n_in, T; } mvT_ctx;
static void b_mvT_range(void *cv, int64_t idx0, int64_t idx1) {
    const mvT_ctx *c = cv;
    int n_in = c->n_in, n_out = c->n_out;
    for (int64_t idx = idx0; idx < idx1; idx++) {
        int t = (int)(idx / n_in);
        int i = (int)(idx % n_in);
        const float *dyt = c->dy + (size_t)t * n_out;
        float acc = c->dx[idx];
        for (int o = 0; o < n_out; o++) {
            float d = dyt[o];
            if (d != 0.0f) acc = fmaf(c->W[(size_t)o * n_in + i], d, acc);
        }
        c->dx[idx] = acc;
    }
}
// dx [T][n_in] += W^T [n_in][n_out] @ dy [T][n_out]
static void b_matvec_T_add(const float *W, int n_out, int n_in, int T, const float *dy, float *dx) {
    mvT_ctx c = { W, dy, dx, n_out, n_in, T };
    ink2_par((int64_t)T * n_in, (int64_t)n_out, b_mvT_range, &c);
}

typedef struct {
    const float *dy, *tmp, *dtmp, *x;
    float *gb, *ga;
    int T, r, n_in, n_out;
    float scale;
    int order_nt;
} lora_grad_ctx;

static void b_lora_gb_range(void *cv, int64_t idx0, int64_t idx1) {
    const lora_grad_ctx *c = cv;
    int T = c->T, r = c->r, n_out = c->n_out;
    float s = c->scale;
    int nt = c->order_nt;
    for (int64_t idx = idx0; idx < idx1; idx++) {
        int j = (int)(idx / r);
        int k = (int)(idx % r);
        if (nt > 1) {
            int chunk = (T + nt - 1) / nt;
            float acc = c->gb[idx];
            for (int w = 0; w < nt; w++) {
                int t0 = w * chunk, t1 = t0 + chunk < T ? t0 + chunk : T;
                float part = 0.0f;
                for (int t = t0; t < t1; t++)
                    part = fmaf(c->dy[(size_t)t * n_out + j], s * c->tmp[(size_t)t * r + k], part);
                acc += part;
            }
            c->gb[idx] = acc;
        } else {
            float acc = c->gb[idx];
            for (int t = 0; t < T; t++)
                acc = fmaf(c->dy[(size_t)t * n_out + j], s * c->tmp[(size_t)t * r + k], acc);
            c->gb[idx] = acc;
        }
    }
}

static void b_lora_ga_range(void *cv, int64_t idx0, int64_t idx1) {
    const lora_grad_ctx *c = cv;
    int T = c->T, r = c->r, n_in = c->n_in;
    for (int64_t idx = idx0; idx < idx1; idx++) {
        int k = (int)(idx / n_in);
        int i = (int)(idx % n_in);
        float acc = c->ga[idx];
        for (int t = 0; t < T; t++)
            acc = fmaf(c->dtmp[(size_t)t * r + k], c->x[(size_t)t * n_in + i], acc);
        c->ga[idx] = acc;
    }
}

// Base W^T dy + LoRA ga/gb + A^T B^T dy -> dx (adds += to dx, ga, gb).
static ink2_status b_lora_proj_add(const float *W, int n_out, int n_in, int T, const float *x, const float *dy,
                                   float *dx, model_t *m, int block, int slot, bool skip_lora_grad, int ctrl) {
    b_matvec_T_add(W, n_out, n_in, T, dy, dx);
    const float *a = NULL, *b = NULL;
    int r = 0, ln_in = 0, ln_out = 0;
    float scale = 0.0f;
    if (!model_lora_slot(m, block, slot, &a, &b, &r, &scale, &ln_in, &ln_out)) return INK2_OK;
    if (r <= 0 || scale == 0.0f || ln_in != n_in || ln_out != n_out) return INK2_OK;
    int cnt_a = 0, cnt_b = 0;
    float *ga = model_lora_gradbuf(m, block, slot, 0, &cnt_a);
    float *gb = model_lora_gradbuf(m, block, slot, 1, &cnt_b);
    float *tmp = bwd_alloc_fail("lora_proj") ? NULL : malloc(sizeof(float) * (size_t)T * (size_t)r * 2);
    if (!tmp) {
        fprintf(stderr, "error: inkling2_loop b_lora_proj_add: out of memory allocating intermediate buffer\n");
        return INK2_BAD_INPUT;
    }
    float *dtmp = tmp + (size_t)T * r;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * n_in;
        const float *dyt = dy + (size_t)t * n_out;
        float *tt = tmp + (size_t)t * r;
        float *dtt = dtmp + (size_t)t * r;
        for (int k = 0; k < r; k++) {
            const float *ak = a + (size_t)k * n_in;
            float acc = 0.0f;
            for (int i = 0; i < n_in; i++) acc = fmaf(ak[i], xt[i], acc);
            tt[k] = acc;
            dtt[k] = 0.0f;
        }
        for (int j = 0; j < n_out; j++) {
            float d = dyt[j];
            if (d == 0.0f) continue;
            const float *bj = b + (size_t)j * r;
            for (int k = 0; k < r; k++) dtt[k] = fmaf(bj[k], d, dtt[k]);
        }
        for (int k = 0; k < r; k++) dtt[k] *= scale;
    }
    if (!skip_lora_grad && ga && gb && cnt_a == r * n_in && cnt_b == n_out * r) {
        int order_nt = (ctrl & CTRL_THREAD_ORDER) ? ink2_get_threads() : 1;
        lora_grad_ctx gc = { dy, tmp, dtmp, x, gb, ga, T, r, n_in, n_out, scale, order_nt };
        ink2_par((int64_t)n_out * r, (int64_t)T, b_lora_gb_range, &gc);
        ink2_par((int64_t)r * n_in, (int64_t)T, b_lora_ga_range, &gc);
    }
    // dx += A^T @ dtmp
    for (int t = 0; t < T; t++) {
        const float *dtt = dtmp + (size_t)t * r;
        float *dxt = dx + (size_t)t * n_in;
        for (int k = 0; k < r; k++) {
            float dt = dtt[k];
            if (dt == 0.0f) continue;
            const float *ak = a + (size_t)k * n_in;
            for (int i = 0; i < n_in; i++) dxt[i] = fmaf(ak[i], dt, dxt[i]);
        }
    }
    free(tmp);
    return INK2_OK;
}

static void b_rms_norm_add(const float *x, int n, float eps, const float *wt, const float *dy, float *dx) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    float gu = 0.0f;
    for (int i = 0; i < n; i++) {
        float gi = wt ? dy[i] * wt[i] : dy[i];
        float ui = x[i] * r;
        gu += gi * ui;
    }
    float c = gu / (float)n;
    for (int i = 0; i < n; i++) {
        float gi = wt ? dy[i] * wt[i] : dy[i];
        float ui = x[i] * r;
        dx[i] += r * (gi - c * ui);
    }
}

static void b_l2_clip(const float *x, int n, float thr, const float *dy, float *dx) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float norm = sqrtf(ss);
    float u = norm + 1e-6f;
    float raw = (1.0f / u) * thr;
    if (raw >= 1.0f || norm == 0.0f) {
        float f = raw > 1.0f ? 1.0f : raw;
        for (int i = 0; i < n; i++) dx[i] = dy[i] * f;
        return;
    }
    float sdot = 0.0f;
    for (int i = 0; i < n; i++) sdot += x[i] * dy[i];
    float coef = thr * sdot / (norm * u * u);
    for (int i = 0; i < n; i++) dx[i] = raw * dy[i] - coef * x[i];
}

// Undo scaled_clip on pre_clip vector (which is post-linear before scale/clip/tau): writes d_pre [n * d].
static ink2_status b_scaled_clip(const ink2_shape *s, int g, const float *pre_lin, const float *scale,
                                 int n, int d, float thr, int pos, const float *d_out, float *d_pre) {
    float tau = g ? f_log_tau(s, pos) : 1.0f;
    float tmp_stack[256];
    float *buf = bwd_alloc_fail("scaled_clip") ? NULL
                 : ((size_t)d * 2 <= 256 ? tmp_stack : malloc(sizeof(float) * (size_t)d * 2));
    if (!buf) {
        fprintf(stderr, "error: inkling2_loop b_scaled_clip: out of memory allocating clip buffer\n");
        return INK2_BAD_INPUT;
    }
    float *sh = buf, *dch = buf + d;
    for (int h = 0; h < n; h++) {
        const float *xh = pre_lin + (size_t)h * d;
        const float *sch = scale + (size_t)h * d;
        const float *dyh = d_out + (size_t)h * d;
        float *dxh = d_pre + (size_t)h * d;
        for (int i = 0; i < d; i++) {
            sh[i] = xh[i] * sch[i];
            dch[i] = g ? dyh[i] * tau : dyh[i];
        }
        b_l2_clip(sh, d, thr, dch, dxh);
        for (int i = 0; i < d; i++) dxh[i] *= sch[i];
    }
    if (buf != tmp_stack) free(buf);
    return INK2_OK;
}

// Depth-wise attention backward for T tokens:
//   states_list[j] points to [T][H] tensor (0 <= j < n_kv - 1) and partial [T][H] is slot n_kv - 1
//   d_states_list[j] receives += VJP of d_out [nq][T][H] (when layout_q_first) OR [T][nq][H] (message_input).
static ink2_status b_depth_attn_common(const float *q, int nq, int H, int T, int n_kv,
                                       const float *const *states, float *const *d_states,
                                       const float *d_out, bool layout_q_first) {
    float *buf = bwd_alloc_fail("depth_attn") ? NULL
                 : malloc(sizeof(float) * ((size_t)n_kv * H * 2 + (size_t)n_kv * 2 + (size_t)n_kv * H));
    if (!buf) {
        fprintf(stderr, "error: inkling2_loop b_depth_attn_common: out of memory\n");
        return INK2_BAD_INPUT;
    }
    float *st = buf;
    float *normed = st + (size_t)n_kv * H;
    float *dnorm = normed + (size_t)n_kv * H;
    float *sc = dnorm + (size_t)n_kv * H;
    float *dp = sc + n_kv;
    const float scale = ink2_inv_sqrt(H);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < n_kv; j++)
            memcpy(st + (size_t)j * H, states[j] + (size_t)t * H, sizeof(float) * (size_t)H);
        for (int j = 0; j < n_kv; j++) f_unweighted_norm(st + (size_t)j * H, H, 1e-6f, normed + (size_t)j * H);
        memset(dnorm, 0, sizeof(float) * (size_t)n_kv * H);
        for (int i = 0; i < nq; i++) {
            const float *qi = q + (size_t)i * H;
            const float *doi = layout_q_first
                                   ? d_out + ((size_t)i * T + t) * H
                                   : d_out + ((size_t)t * nq + i) * H;
            float mx = -FLT_MAX;
            for (int j = 0; j < n_kv; j++) {
                float d = 0.0f;
                for (int h = 0; h < H; h++) d += qi[h] * normed[(size_t)j * H + h];
                sc[j] = d * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < n_kv; j++) { sc[j] = p_expf(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j < n_kv; j++) sc[j] /= sum;
            float D = 0.0f;
            for (int j = 0; j < n_kv; j++) {
                float p = sc[j];
                float v = 0.0f;
                for (int h = 0; h < H; h++) {
                    dnorm[(size_t)j * H + h] += p * doi[h];
                    v += doi[h] * normed[(size_t)j * H + h];
                }
                dp[j] = v;
                D += p * v;
            }
            for (int j = 0; j < n_kv; j++) {
                float ds = sc[j] * (dp[j] - D) * scale;
                for (int h = 0; h < H; h++) dnorm[(size_t)j * H + h] += ds * qi[h];
            }
        }
        for (int j = 0; j < n_kv; j++)
            b_rms_norm_add(st + (size_t)j * H, H, 1e-6f, NULL, dnorm + (size_t)j * H,
                           d_states[j] + (size_t)t * H);
    }
    free(buf);
    return INK2_OK;
}

static void b_short_conv_add(const float *cw, int C, int K, int T, const float *d_out, float *d_lin) {
    const float div = (float)sqrt((double)(K + 1));
    for (int u = 0; u < T; u++) {
        for (int ch = 0; ch < C; ch++) {
            float acc = d_out[(size_t)u * C + ch] / div;
            for (int j = 0; j < K; j++) {
                int t = u + (K - 1) - j;
                if (t < T) acc += cw[(size_t)ch * K + j] * (d_out[(size_t)t * C + ch] / div);
            }
            d_lin[(size_t)u * C + ch] += acc;
        }
    }
}

static ink2_status b_kv_conv_add(int g, const float *cw, const float *wd, int H, int nkv, int hd, int K, int T,
                                 const float *lin, const float *dyn, const float *d_out,
                                 float *d_lin, float *d_x, int ctrl) {
    int C = nkv * hd, R = nkv * K;
    const float div = (float)sqrt((double)(g ? 2 * K + 1 : K + 1));
    if (!g) {
        b_short_conv_add(cw, C, K, T, d_out, d_lin);
        return INK2_OK;
    }
    float *d_dyn = bwd_alloc_fail("kv_conv") ? NULL : calloc((size_t)T * R, sizeof(float));
    if (!d_dyn) {
        fprintf(stderr, "error: inkling2_loop b_kv_conv_add: out of memory\n");
        return INK2_BAD_INPUT;
    }
    for (int t = 0; t < T; t++) {
        for (int ch = 0; ch < C; ch++) {
            float dy = d_out[(size_t)t * C + ch] / div;
            d_lin[(size_t)t * C + ch] += dy;
            int kh = ch / hd;
            for (int l = 0; l < K && t - l >= 0; l++) {
                float tap = cw[(size_t)l * C + ch] + dyn[(size_t)t * R + (size_t)kh * K + l];
                d_lin[(size_t)(t - l) * C + ch] += dy * tap;
                d_dyn[(size_t)t * R + (size_t)kh * K + l] += dy * lin[(size_t)(t - l) * C + ch];
            }
        }
    }
    if (!(ctrl & CTRL_DROP_DYNCONV))
        b_matvec_T_add(wd, R, H, T, d_dyn, d_x);
    free(d_dyn);
    return INK2_OK;
}

// ---- Sparse MoE backward (per block, recomputes router & per-expert acts per token)
static ink2_status b_moe_layer(const ink2_shape *s, const ink2_weights *w, int block, int T,
                               const float *x, const float *d_out, float *d_x, int ctrl) {
    char n1[160], n2[160], n13[160], n2e[160];
    snprintf(n1, sizeof n1, "model.layers.%d.mlp.gate.weight", block);
    snprintf(n2, sizeof n2, "model.layers.%d.mlp.gate.e_score_correction_bias", block);
    snprintf(n13, sizeof n13, "model.layers.%d.mlp.experts.gate_up_proj", block);
    snprintf(n2e, sizeof n2e, "model.layers.%d.mlp.experts.down_proj", block);
    const ink2_tensor *wr_t = w->get(w->ctx, n1), *br_t = w->get(w->ctx, n2);
    int H = s->hidden, E = s->n_expert, K = s->n_used, F = s->ff;
    if (!wr_t || !br_t || wr_t->ne[0] != H || wr_t->ne[1] != E || br_t->ne[0] != E) {
        fprintf(stderr, "error: inkling2_loop b_moe_layer: missing or invalid router tensors for block %d\n", block);
        return INK2_NO_TENSOR;
    }
    const float *wr = wr_t->data, *br = br_t->data;
    const ink2_tensor *w13_stack = NULL, *w2_stack = NULL;
    if (!w->get_expert) {
        w13_stack = w->get(w->ctx, n13);
        w2_stack = w->get(w->ctx, n2e);
        if (!w13_stack || !w2_stack) {
            fprintf(stderr, "error: inkling2_loop b_moe_layer: missing expert stack tensors for block %d\n", block);
            return INK2_NO_TENSOR;
        }
    }

    float *scr = bwd_alloc_fail("moe") ? NULL : malloc(sizeof(float) * ((size_t)E * 3 + (size_t)F * 4 + (size_t)H * 2));
    unsigned char *chosen = malloc((size_t)E);
    if (!scr || !chosen) {
        fprintf(stderr, "error: inkling2_loop b_moe_layer: out of memory allocating router/expert scratch\n");
        free(scr); free(chosen);
        return INK2_BAD_INPUT;
    }
    float *sc = scr, *wt = sc + E, *dwt = wt + E;
    float *gu = dwt + E, *act = gu + 2 * F, *dact = act + F;
    float *down = dact + F, *ddown = down + H;

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * H;
        const float *dout = d_out + (size_t)t * H;
        float *dxt = d_x + (size_t)t * H;
        // Router forward
        for (int e = 0; e < E; e++) {
            float l = 0.0f;
            const float *row = wr + (size_t)e * H;
            for (int h = 0; h < H; h++) l += row[h] * xt[h];
            sc[e] = ink2b_sigmoid(l);
            chosen[e] = 0;
            dwt[e] = 0.0f;
        }
        for (int k = 0; k < K; k++) {
            int best = -1; float bv = 0.0f;
            for (int e = 0; e < E; e++) {
                if (chosen[e]) continue;
                float v = sc[e] + br[e];
                if (best < 0 || v > bv) { best = e; bv = v; }
            }
            chosen[best] = 1;
        }
        float sum = 0.0f;
        for (int e = 0; e < E; e++) if (chosen[e]) sum += sc[e];
        float den = sum + 1e-20f;
        for (int e = 0; e < E; e++) wt[e] = chosen[e] ? sc[e] / den * s->route_scale : 0.0f;

        // Experts backward
        for (int e = 0; e < E; e++) {
            if (!chosen[e]) continue;
            const float *g13 = NULL, *g2 = NULL;
            if (w->get_expert) {
                const ink2_tensor *t13 = w->get_expert(w->ctx, n13, e);
                const ink2_tensor *t2 = w->get_expert(w->ctx, n2e, e);
                if (!t13 || !t2) {
                    fprintf(stderr, "error: inkling2_loop b_moe_layer: failed decoding expert %d in block %d\n", e, block);
                    free(scr); free(chosen);
                    return INK2_NO_TENSOR;
                }
                g13 = t13->data; g2 = t2->data;
            } else {
                g13 = w13_stack->data + (size_t)e * 2 * F * H;
                g2 = w2_stack->data + (size_t)e * H * F;
            }
            f_matvec(g13, 2 * F, H, xt, gu);
            for (int j = 0; j < F; j++) {
                float gj = gu[j], uj = gu[F + j];
                float sg_cap = p_tanhf(gj / s->gate_cap) * s->gate_cap;
                float sig = ink2b_sigmoid(gj);
                float su_cap = p_tanhf(uj / s->up_cap) * s->up_cap;
                act[j] = sg_cap * sig * su_cap;
            }
            f_matvec(g2, H, F, act, down);
            float dw = 0.0f;
            for (int h = 0; h < H; h++) {
                dw += dout[h] * down[h];
                ddown[h] = wt[e] * dout[h];
            }
            dwt[e] = dw;

            // d_act = g2^T @ ddown
            for (int j = 0; j < F; j++) dact[j] = 0.0f;
            for (int j = 0; j < F; j++) {
                float acc = 0.0f;
                for (int h = 0; h < H; h++) acc = fmaf(g2[(size_t)h * F + j], ddown[h], acc);
                dact[j] = acc;
            }
            // SoftcappedSwiGLU VJP in place on gu
            for (int j = 0; j < F; j++) {
                float gj = gu[j], uj = gu[F + j];
                float tg = p_tanhf(gj / s->gate_cap);
                float sg_cap = tg * s->gate_cap;
                float sig = ink2b_sigmoid(gj);
                float tu = p_tanhf(uj / s->up_cap);
                float su_cap = tu * s->up_cap;
                float dg_cap = 1.0f - tg * tg;
                float dsig = sig * (1.0f - sig);
                float du_cap = 1.0f - tu * tu;
                float da = dact[j];
                gu[j]     = da * su_cap * (dg_cap * sig + sg_cap * dsig);
                gu[F + j] = da * sg_cap * sig * du_cap;
            }
            // dxt += g13^T @ gu
            for (int i = 0; i < H; i++) {
                float acc = dxt[i];
                for (int j = 0; j < 2 * F; j++) acc = fmaf(g13[(size_t)j * H + i], gu[j], acc);
                dxt[i] = acc;
            }
        }

        // Router weight VJP: wt[e] = sc[e] / den * route_scale
        if (!(ctrl & CTRL_DROP_ROUTER)) {
            float sum_wd = 0.0f;
            for (int e = 0; e < E; e++) if (chosen[e]) sum_wd += wt[e] * dwt[e];
            for (int e = 0; e < E; e++) {
                if (!chosen[e]) continue;
                float d_sc = (s->route_scale * dwt[e] - sum_wd) / den;
                float d_lg = d_sc * sc[e] * (1.0f - sc[e]);
                const float *row = wr + (size_t)e * H;
                for (int h = 0; h < H; h++) dxt[h] = fmaf(row[h], d_lg, dxt[h]);
            }
        }
    }

    free(scr); free(chosen);
    return INK2_OK;
}

// ---- Attention forward recompute (single app) & VJP
enum { W_Q, W_QS, W_K, W_KC, W_KD, W_V, W_VC, W_VD, W_R, W_RS, W_RP, W_S, W_G, W_O, N_ATTN_W };
typedef struct {
    int g, nh, nkv, hd, K, ext, dr, C, Q;
    const float *wt[N_ATTN_W];
} attn_bwd_w;

static ink2_status bind_attn_w(const ink2_shape *s, const ink2_weights *w, int block, attn_bwd_w *L) {
    int g = s->block_global[block], H = s->hidden;
    L->g = g;
    L->nh = g ? s->n_head : s->swa_n_head;
    L->nkv = g ? s->n_kv : s->swa_n_kv;
    L->hd = g ? s->head_dim : s->swa_head_dim;
    L->K = s->conv_k;
    L->ext = g ? s->rel_extent : s->window;
    L->dr = s->d_rel;
    L->C = L->nkv * L->hd;
    L->Q = L->nh * L->hd;
    const char *nm[N_ATTN_W] = {
        "q_proj.linear.weight", "q_proj.norm.scale",
        "k_proj.linear.weight", g ? "k_proj.sconv.static_weight" : "k_proj.sconv.conv1d.weight",
        g ? "k_proj.sconv.weight_proj.weight" : NULL,
        "v_proj.linear.weight", g ? "v_proj.sconv.static_weight" : "v_proj.sconv.conv1d.weight",
        g ? "v_proj.sconv.weight_proj.weight" : NULL,
        "relative_position_bias.linear.weight", "relative_position_bias.norm.scale", "relative_position_bias.proj",
        "sink_proj.weight", "o_proj.gate_proj.weight", "o_proj.linear.weight"
    };
    for (int i = 0; i < N_ATTN_W; i++) {
        L->wt[i] = NULL;
        if (!nm[i]) continue;
        char full[192];
        snprintf(full, sizeof full, "model.layers.%d.self_attn.%s", block, nm[i]);
        const ink2_tensor *t = w->get(w->ctx, full);
        if (!t) {
            fprintf(stderr, "error: inkling2_loop bind_attn_w: missing tensor '%s'\n", full);
            return INK2_NO_TENSOR;
        }
        L->wt[i] = t->data;
    }
    (void)H;
    return INK2_OK;
}

static ink2_status apply_lora_from_model(model_t *m, int block, int slot, int n_out, int n_in, int T,
                                         const float *x, float *y) {
    const float *a = NULL, *b = NULL;
    int r = 0, ln_in = 0, ln_out = 0;
    float scale = 0.0f;
    if (!model_lora_slot(m, block, slot, &a, &b, &r, &scale, &ln_in, &ln_out)) return INK2_OK;
    if (ln_in != n_in || ln_out != n_out) {
        fprintf(stderr, "error: inkling2_loop: LoRA slot %d.%d dims [%d,%d] != [%d,%d]\n",
                block, slot, ln_in, ln_out, n_in, n_out);
        return INK2_BAD_INPUT;
    }
    return f_lora_apply(a, b, r, scale, n_out, n_in, T, x, y);
}

typedef struct {
    attn_bwd_w L;
    float *buf;
    float *q_lin, *q_sc, *att_raw, *at_gate, *ga_tmp;
    float *k_lin, *k_conv, *k_norm, *k_dyn;
    float *v_lin, *v_conv, *v_norm, *v_dyn;
    float *rs_lin, *rs_clip, *rl, *sk, *lg_row;
} attn_fwd_cache;

// Run single-application attention forward once (p0 = 0) and keep intermediates for backward.
static ink2_status b_attn_fwd_save(const ink2_shape *s, const ink2_weights *w, model_t *m,
                                   int app, int T, const float *xq, const float *xk, const float *xv,
                                   attn_fwd_cache *af, float *att_out) {
    memset(af, 0, sizeof *af);
    int block = s->app_block[app];
    ink2_status st = bind_attn_w(s, w, block, &af->L);
    if (st != INK2_OK) return st;
    const attn_bwd_w *L = &af->L;
    int H = s->hidden, nh = L->nh, nkv = L->nkv, hd = L->hd, K = L->K, ext = L->ext, dr = L->dr, C = L->C, Q = L->Q;
    int R = nkv * K;

    size_t n_fw = (size_t)T * (Q * 5 + C * 6 + R * 2 + nh * dr * 2 + nh * ext + nh) + (size_t)T + 1;
    af->buf = bwd_alloc_fail("attn_fwd") ? NULL : calloc(n_fw, sizeof(float));
    if (!af->buf) {
        fprintf(stderr, "error: inkling2_loop b_attn_fwd_save: out of memory\n");
        return INK2_BAD_INPUT;
    }

    float *p = af->buf;
    af->q_lin   = p; p += (size_t)T * Q;
    af->q_sc    = p; p += (size_t)T * Q;
    af->att_raw = p; p += (size_t)T * Q;
    af->at_gate = p; p += (size_t)T * Q;
    af->ga_tmp  = p; p += (size_t)T * Q;
    af->k_lin   = p; p += (size_t)T * C;
    af->k_conv  = p; p += (size_t)T * C;
    af->k_norm  = p; p += (size_t)T * C;
    af->v_lin   = p; p += (size_t)T * C;
    af->v_conv  = p; p += (size_t)T * C;
    af->v_norm  = p; p += (size_t)T * C;
    af->k_dyn   = p; p += (size_t)T * R;
    af->v_dyn   = p; p += (size_t)T * R;
    af->rs_lin  = p; p += (size_t)T * nh * dr;
    af->rs_clip = p; p += (size_t)T * nh * dr;
    af->rl      = p; p += (size_t)T * nh * ext;
    af->sk      = p; p += (size_t)T * nh;
    af->lg_row  = p; p += (size_t)T + 1;

    f_rows(L->wt[W_Q], Q, H, T, xq, af->q_lin);
    if ((st = apply_lora_from_model(m, block, LW_Q, Q, H, T, xq, af->q_lin)) != INK2_OK) goto fail;
    memcpy(af->q_sc, af->q_lin, sizeof(float) * (size_t)T * Q);
    for (int t = 0; t < T; t++)
        f_scaled_clip(s, L->g, af->q_sc + (size_t)t * Q, L->wt[W_QS], nh, hd, s->qk_clip, t);

    f_rows(L->wt[W_K], C, H, T, xk, af->k_lin);
    if ((st = apply_lora_from_model(m, block, LW_K, C, H, T, xk, af->k_lin)) != INK2_OK) goto fail;
    if (L->g) f_rows(L->wt[W_KD], R, H, T, xk, af->k_dyn);
    {
        const float div = (float)sqrt((double)(L->g ? 2 * K + 1 : K + 1));
        for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
            float a = 0.0f;
            if (!L->g) {
                for (int j = 0; j < K; j++) { int u = t - (K - 1) + j; if (u >= 0) a += L->wt[W_KC][(size_t)ch * K + j] * af->k_lin[(size_t)u * C + ch]; }
            } else {
                for (int l = 0; l < K && t - l >= 0; l++) {
                    float tap = L->wt[W_KC][(size_t)l * C + ch] + af->k_dyn[(size_t)t * R + (size_t)(ch / hd) * K + l];
                    a += af->k_lin[(size_t)(t - l) * C + ch] * tap;
                }
            }
            af->k_conv[(size_t)t * C + ch] = (a + af->k_lin[(size_t)t * C + ch]) / div;
        }
    }
    memcpy(af->k_norm, af->k_conv, sizeof(float) * (size_t)T * C);
    for (int t = 0; t < T; t++) for (int h = 0; h < nkv; h++)
        f_l2_clip(af->k_norm + (size_t)t * C + (size_t)h * hd, hd, s->qk_clip);

    f_rows(L->wt[W_V], C, H, T, xv, af->v_lin);
    if ((st = apply_lora_from_model(m, block, LW_V, C, H, T, xv, af->v_lin)) != INK2_OK) goto fail;
    if (L->g) f_rows(L->wt[W_VD], R, H, T, xv, af->v_dyn);
    {
        const float div = (float)sqrt((double)(L->g ? 2 * K + 1 : K + 1));
        for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
            float a = 0.0f;
            if (!L->g) {
                for (int j = 0; j < K; j++) { int u = t - (K - 1) + j; if (u >= 0) a += L->wt[W_VC][(size_t)ch * K + j] * af->v_lin[(size_t)u * C + ch]; }
            } else {
                for (int l = 0; l < K && t - l >= 0; l++) {
                    float tap = L->wt[W_VC][(size_t)l * C + ch] + af->v_dyn[(size_t)t * R + (size_t)(ch / hd) * K + l];
                    a += af->v_lin[(size_t)(t - l) * C + ch] * tap;
                }
            }
            af->v_conv[(size_t)t * C + ch] = (a + af->v_lin[(size_t)t * C + ch]) / div;
        }
    }
    memcpy(af->v_norm, af->v_conv, sizeof(float) * (size_t)T * C);
    for (int t = 0; t < T; t++) for (int h = 0; h < nkv; h++)
        f_unweighted_norm(af->v_norm + (size_t)t * C + (size_t)h * hd, hd, s->eps,
                          af->v_norm + (size_t)t * C + (size_t)h * hd);

    ink2_rel_probe *rp = ink2_active_rel_probe();
    if (rp && app == rp->app && rp->eps != 0.0f && rp->d) {
        size_t n_el = (size_t)T * (size_t)H;
        float *xq_rel = malloc(sizeof(float) * n_el);
        if (!xq_rel) {
            fprintf(stderr, "error: inkling2_loop b_attn_fwd_save: out of memory allocating rel_probe buffer\n");
            st = INK2_BAD_INPUT;
            goto fail;
        }
        for (size_t i = 0; i < n_el; i++) xq_rel[i] = xq[i] + rp->eps * rp->d[i];
        f_rows(L->wt[W_R], nh * dr, H, T, xq_rel, af->rs_lin);
        free(xq_rel);
    } else {
        f_rows(L->wt[W_R], nh * dr, H, T, xq, af->rs_lin);
    }
    memcpy(af->rs_clip, af->rs_lin, sizeof(float) * (size_t)T * nh * dr);
    for (int t = 0; t < T; t++) {
        float *rt = af->rs_clip + (size_t)t * nh * dr;
        f_scaled_clip(s, L->g, rt, L->wt[W_RS], nh, dr, s->rel_clip, t);
        for (int h = 0; h < nh; h++)
            for (int e = 0; e < ext; e++) {
                float a = 0.0f;
                for (int r = 0; r < dr; r++) a += rt[(size_t)h * dr + r] * L->wt[W_RP][(size_t)r * ext + e];
                af->rl[((size_t)t * nh + h) * ext + e] = a;
            }
    }
    f_rows(L->wt[W_S], nh, H, T, xq, af->sk);

    const float qk_scale = ink2_inv_sqrt(hd);
    for (int h = 0; h < nh; h++) {
        int kh = h / (nh / nkv);
        for (int i = 0; i < T; i++) {
            const float *qi = af->q_sc + (size_t)i * Q + (size_t)h * hd;
            int w0 = L->g ? 0 : (i - s->window + 1 > 0 ? i - s->window + 1 : 0);
            float mx = af->sk[(size_t)i * nh + h];
            for (int j = 0; j < T; j++) af->lg_row[j] = 0.0f;
            for (int j = w0; j <= i; j++) {
                const float *kj = af->k_norm + (size_t)j * C + (size_t)kh * hd;
                float a = 0.0f;
                for (int e = 0; e < hd; e++) a += qi[e] * kj[e];
                int d = i - j;
                af->lg_row[j] = a * qk_scale + (d < ext ? af->rl[((size_t)i * nh + h) * ext + d] : 0.0f);
                if (af->lg_row[j] > mx) mx = af->lg_row[j];
            }
            float sum = 0.0f;
            for (int j = w0; j <= i; j++) { af->lg_row[j] = p_expf(af->lg_row[j] - mx); sum += af->lg_row[j]; }
            sum += p_expf(af->sk[(size_t)i * nh + h] - mx);
            for (int j = w0; j <= i; j++) af->lg_row[j] /= sum;
            float *ai = af->att_raw + (size_t)i * Q + (size_t)h * hd;
            for (int e = 0; e < hd; e++) {
                float a = 0.0f;
                for (int j = w0; j <= i; j++) a += af->lg_row[j] * af->v_norm[(size_t)j * C + (size_t)kh * hd + e];
                ai[e] = a;
            }
        }
    }

    for (int t = 0; t < T; t++) {
        f_matvec(L->wt[W_G], Q, H, xq + (size_t)t * H, af->ga_tmp + (size_t)t * Q);
        for (int i = 0; i < Q; i++)
            af->at_gate[(size_t)t * Q + i] = af->att_raw[(size_t)t * Q + i] * ink2b_sigmoid(af->ga_tmp[(size_t)t * Q + i]);
        f_matvec(L->wt[W_O], H, Q, af->at_gate + (size_t)t * Q, att_out + (size_t)t * H);
    }
    st = apply_lora_from_model(m, block, LW_O, H, Q, T, af->at_gate, att_out);
fail:
    if (st != INK2_OK) { free(af->buf); af->buf = NULL; }
    return st;
}

static ink2_status b_attn_bwd_from_saved(const ink2_shape *s, model_t *m,
                                         int app, int T, const float *xq, const float *xk, const float *xv,
                                         attn_fwd_cache *af, const float *d_out,
                                         float *d_xq, float *d_xk, float *d_xv, int ctrl) {
    int block = s->app_block[app], pass = s->app_pass[app];
    bool loop = block >= s->loop_start && block < s->loop_end;
    bool skip_lora_grad = ((ctrl & CTRL_DROP_PASS1) && loop && pass == 1)
                          || ((ctrl & CTRL_DROP_PASS2) && loop && pass == 2);
    const attn_bwd_w *L = &af->L;
    int H = s->hidden, nh = L->nh, nkv = L->nkv, hd = L->hd, ext = L->ext, dr = L->dr, C = L->C, Q = L->Q;

    size_t n_bw = (size_t)T * (Q * 4 + C * 4 + nh * dr + nh * ext + nh);
    float *bw = bwd_alloc_fail("attn_bwd") ? NULL : calloc(n_bw, sizeof(float));
    if (!bw) {
        fprintf(stderr, "error: inkling2_loop b_attn_bwd_from_saved: out of memory\n");
        return INK2_BAD_INPUT;
    }

    float *p = bw;
    float *dq_sc     = p; p += (size_t)T * Q;
    float *d_at_g    = p; p += (size_t)T * Q;
    float *d_att     = p; p += (size_t)T * Q;
    float *d_ga      = p; p += (size_t)T * Q;
    float *dk_norm   = p; p += (size_t)T * C;
    float *dk_conv   = p; p += (size_t)T * C;
    float *dv_norm   = p; p += (size_t)T * C;
    float *dv_conv   = p; p += (size_t)T * C;
    float *d_rl      = p; p += (size_t)T * nh * ext;
    float *d_sk      = p; p += (size_t)T * nh;
    float *d_rs_clip = p; p += (size_t)T * nh * dr;

    ink2_status st = b_lora_proj_add(L->wt[W_O], H, Q, T, af->at_gate, d_out, d_at_g, m, block, LW_O, skip_lora_grad, ctrl);
    if (st != INK2_OK) { free(bw); return st; }

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < Q; i++) {
            float sg = ink2b_sigmoid(af->ga_tmp[(size_t)t * Q + i]);
            float dg = d_at_g[(size_t)t * Q + i];
            d_att[(size_t)t * Q + i] = dg * sg * ((ctrl & CTRL_SCALE_102) ? 1.02f : 1.0f);
            d_ga[(size_t)t * Q + i]  = dg * af->att_raw[(size_t)t * Q + i] * sg * (1.0f - sg);
        }
    }
    b_matvec_T_add(L->wt[W_G], Q, H, T, d_ga, d_xq);

    const float qk_scale = ink2_inv_sqrt(hd);
    float *dp_row = calloc((size_t)T, sizeof(float));
    if (!dp_row) {
        fprintf(stderr, "error: inkling2_loop b_attn_bwd_from_saved: out of memory allocating dp_row\n");
        free(bw);
        return INK2_BAD_INPUT;
    }
    float *lg_row = af->lg_row;
    for (int h = 0; h < nh; h++) {
        int kh = h / (nh / nkv);
        for (int i = 0; i < T; i++) {
            const float *qi = af->q_sc + (size_t)i * Q + (size_t)h * hd;
            const float *dai = d_att + (size_t)i * Q + (size_t)h * hd;
            int w0 = L->g ? 0 : (i - s->window + 1 > 0 ? i - s->window + 1 : 0);
            float mx = af->sk[(size_t)i * nh + h];
            for (int j = w0; j <= i; j++) {
                const float *kj = af->k_norm + (size_t)j * C + (size_t)kh * hd;
                float a = 0.0f;
                for (int e = 0; e < hd; e++) a += qi[e] * kj[e];
                int d = i - j;
                lg_row[j] = a * qk_scale + (d < ext ? af->rl[((size_t)i * nh + h) * ext + d] : 0.0f);
                if (lg_row[j] > mx) mx = lg_row[j];
            }
            float sum = 0.0f;
            for (int j = w0; j <= i; j++) { lg_row[j] = p_expf(lg_row[j] - mx); sum += lg_row[j]; }
            float esink = p_expf(af->sk[(size_t)i * nh + h] - mx);
            sum += esink;
            for (int j = w0; j <= i; j++) lg_row[j] /= sum;
            float p_sink = esink / sum;

            float D = 0.0f;
            for (int j = w0; j <= i; j++) {
                const float *vj = af->v_norm + (size_t)j * C + (size_t)kh * hd;
                float dpj = 0.0f;
                for (int e = 0; e < hd; e++) dpj += dai[e] * vj[e];
                dp_row[j] = dpj;
                D += lg_row[j] * dpj;
            }
            if (!(ctrl & CTRL_DROP_SINK)) d_sk[(size_t)i * nh + h] = -p_sink * D;

            float *dqi = dq_sc + (size_t)i * Q + (size_t)h * hd;
            for (int j = w0; j <= i; j++) {
                float p_j = lg_row[j];
                float *dvj = dv_norm + (size_t)j * C + (size_t)kh * hd;
                for (int e = 0; e < hd; e++) dvj[e] = fmaf(p_j, dai[e], dvj[e]);

                float ds = p_j * (dp_row[j] - D);
                int d = i - j;
                if (d < ext && !(ctrl & CTRL_DROP_RELBIAS))
                    d_rl[((size_t)i * nh + h) * ext + d] += ds;

                float ds_qk = ds * qk_scale;
                const float *kj = af->k_norm + (size_t)j * C + (size_t)kh * hd;
                float *dkj = dk_norm + (size_t)j * C + (size_t)kh * hd;
                for (int e = 0; e < hd; e++) {
                    dqi[e] = fmaf(ds_qk, kj[e], dqi[e]);
                    dkj[e] = fmaf(ds_qk, qi[e], dkj[e]);
                }
            }
        }
    }
    free(dp_row);

    if (!(ctrl & CTRL_DROP_SINK)) b_matvec_T_add(L->wt[W_S], nh, H, T, d_sk, d_xq);
    ink2_rel_probe *rp = ink2_active_rel_probe();
    if (!(ctrl & CTRL_DROP_RELBIAS)) {
        float *d_rs_lin = calloc((size_t)T * nh * dr, sizeof(float));
        if (!d_rs_lin) {
            fprintf(stderr, "error: inkling2_loop b_attn_bwd_from_saved: out of memory\n");
            free(bw);
            return INK2_BAD_INPUT;
        }
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < nh; h++)
                for (int r = 0; r < dr; r++) {
                    float acc = 0.0f;
                    for (int e = 0; e < ext; e++)
                        acc = fmaf(L->wt[W_RP][(size_t)r * ext + e],
                                   d_rl[((size_t)t * nh + h) * ext + e], acc);
                    d_rs_clip[((size_t)t * nh + h) * dr + r] = acc;
                }
            st = b_scaled_clip(s, L->g, af->rs_lin + (size_t)t * nh * dr, L->wt[W_RS], nh, dr, s->rel_clip, t,
                               d_rs_clip + (size_t)t * nh * dr, d_rs_lin + (size_t)t * nh * dr);
            if (st != INK2_OK) { free(d_rs_lin); free(bw); return st; }
        }
        if ((ctrl & CTRL_SCALE_RELBIAS_102) || (rp && rp->app == app && rp->g_rel)) {
            float *g_rel_tmp = calloc((size_t)T * (size_t)H, sizeof(float));
            if (!g_rel_tmp) {
                fprintf(stderr, "error: inkling2_loop b_attn_bwd_from_saved: out of memory allocating rel_probe grad\n");
                free(d_rs_lin);
                free(bw);
                return INK2_BAD_INPUT;
            }
            b_matvec_T_add(L->wt[W_R], nh * dr, H, T, d_rs_lin, g_rel_tmp);
            if (ctrl & CTRL_SCALE_RELBIAS_102)
                for (size_t i = 0; i < (size_t)T * (size_t)H; i++) g_rel_tmp[i] *= 1.02f;
            if (rp && rp->app == app && rp->g_rel)
                memcpy(rp->g_rel, g_rel_tmp, sizeof(float) * (size_t)T * (size_t)H);
            for (size_t i = 0; i < (size_t)T * (size_t)H; i++) d_xq[i] += g_rel_tmp[i];
            free(g_rel_tmp);
        } else {
            b_matvec_T_add(L->wt[W_R], nh * dr, H, T, d_rs_lin, d_xq);
        }
        free(d_rs_lin);
    } else if (rp && rp->app == app && rp->g_rel) {
        memset(rp->g_rel, 0, sizeof(float) * (size_t)T * (size_t)H);
    }

    float *dq_lin = af->att_raw;
    for (int t = 0; t < T; t++) {
        st = b_scaled_clip(s, L->g, af->q_lin + (size_t)t * Q, L->wt[W_QS], nh, hd, s->qk_clip, t,
                           dq_sc + (size_t)t * Q, dq_lin + (size_t)t * Q);
        if (st != INK2_OK) { free(bw); return st; }
    }
    st = b_lora_proj_add(L->wt[W_Q], Q, H, T, xq, dq_lin, d_xq, m, block, LW_Q, skip_lora_grad, ctrl);
    if (st != INK2_OK) { free(bw); return st; }

    for (int t = 0; t < T; t++) for (int h = 0; h < nkv; h++)
        b_l2_clip(af->k_conv + (size_t)t * C + (size_t)h * hd, hd, s->qk_clip,
                  dk_norm + (size_t)t * C + (size_t)h * hd,
                  dk_conv + (size_t)t * C + (size_t)h * hd);
    float *dk_lin = af->k_norm;
    memset(dk_lin, 0, sizeof(float) * (size_t)T * C);
    st = b_kv_conv_add(L->g, L->wt[W_KC], L->wt[W_KD], H, nkv, hd, L->K, T, af->k_lin, af->k_dyn, dk_conv, dk_lin, d_xk, ctrl);
    if (st != INK2_OK) { free(bw); return st; }
    st = b_lora_proj_add(L->wt[W_K], C, H, T, xk, dk_lin, d_xk, m, block, LW_K, skip_lora_grad, ctrl);
    if (st != INK2_OK) { free(bw); return st; }

    memset(dv_conv, 0, sizeof(float) * (size_t)T * C);
    for (int t = 0; t < T; t++) for (int h = 0; h < nkv; h++)
        b_rms_norm_add(af->v_conv + (size_t)t * C + (size_t)h * hd, hd, s->eps, NULL,
                       dv_norm + (size_t)t * C + (size_t)h * hd,
                       dv_conv + (size_t)t * C + (size_t)h * hd);
    float *dv_lin = af->v_norm;
    memset(dv_lin, 0, sizeof(float) * (size_t)T * C);
    st = b_kv_conv_add(L->g, L->wt[W_VC], L->wt[W_VD], H, nkv, hd, L->K, T, af->v_lin, af->v_dyn, dv_conv, dv_lin, d_xv, ctrl);
    if (st != INK2_OK) { free(bw); return st; }
    st = b_lora_proj_add(L->wt[W_V], C, H, T, xv, dv_lin, d_xv, m, block, LW_V, skip_lora_grad, ctrl);

    free(bw);
    return st;
}

// ---- Message step forward & backward
static ink2_status b_msg_step_fwd(const ink2_shape *s, const ink2_weights *w, int block, const char *role,
                                  int T, const float *x, const float *msg, float *out) {
    char n1[160], n2[160], n3[160];
    snprintf(n1, sizeof n1, "model.layers.%d.%s_message_gate.down_proj.weight", block, role);
    snprintf(n2, sizeof n2, "model.layers.%d.%s_message_gate.forget_gate_proj.weight", block, role);
    snprintf(n3, sizeof n3, "model.layers.%d.%s_message_gate.message_gate_proj.weight", block, role);
    const ink2_tensor *wd = w->get(w->ctx, n1), *wf = w->get(w->ctx, n2), *wg = w->get(w->ctx, n3);
    int H = s->hidden, r = s->loop_gate_rank;
    if (!wd || !wf || !wg) {
        fprintf(stderr, "error: inkling2_loop b_msg_step_fwd: missing %s_message_gate tensors for block %d\n", role, block);
        return INK2_NO_TENSOR;
    }
    float *gh = bwd_alloc_fail("msg_fwd") ? NULL : malloc(sizeof(float) * ((size_t)2 * H + (size_t)r));
    if (!gh) {
        fprintf(stderr, "error: inkling2_loop b_msg_step_fwd: out of memory allocating message gate forward buffer\n");
        return INK2_BAD_INPUT;
    }
    float *d = gh + 2 * H;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * H, *mt = msg + (size_t)t * H;
        f_unweighted_norm(xt, H, s->eps, gh);
        f_unweighted_norm(mt, H, s->eps, gh + H);
        f_matvec(wd->data, r, 2 * H, gh, d);
        float *ot = out + (size_t)t * H;
        for (int h = 0; h < H; h++) {
            float fa = 0.0f, ga = 0.0f;
            const float *fr = wf->data + (size_t)h * r, *gr = wg->data + (size_t)h * r;
            for (int k = 0; k < r; k++) { fa += fr[k] * d[k]; ga += gr[k] * d[k]; }
            ot[h] = (1.0f - p_tanhf(fa)) * xt[h] + p_tanhf(ga) * mt[h];
        }
    }
    free(gh);
    return INK2_OK;
}

static ink2_status b_msg_step_bwd(const ink2_shape *s, const ink2_weights *w, int block, const char *role,
                                  int T, const float *x, const float *msg, const float *d_out,
                                  float *d_x, float *d_msg) {
    char n1[160], n2[160], n3[160];
    snprintf(n1, sizeof n1, "model.layers.%d.%s_message_gate.down_proj.weight", block, role);
    snprintf(n2, sizeof n2, "model.layers.%d.%s_message_gate.forget_gate_proj.weight", block, role);
    snprintf(n3, sizeof n3, "model.layers.%d.%s_message_gate.message_gate_proj.weight", block, role);
    const ink2_tensor *wd = w->get(w->ctx, n1), *wf = w->get(w->ctx, n2), *wg = w->get(w->ctx, n3);
    int H = s->hidden, r = s->loop_gate_rank;
    if (!wd || !wf || !wg) {
        fprintf(stderr, "error: inkling2_loop b_msg_step_bwd: missing %s_message_gate tensors for block %d\n", role, block);
        return INK2_NO_TENSOR;
    }
    float *scr = bwd_alloc_fail("msg_bwd") ? NULL : malloc(sizeof(float) * ((size_t)4 * H + (size_t)r * 2 + (size_t)H * 2));
    if (!scr) {
        fprintf(stderr, "error: inkling2_loop b_msg_step_bwd: out of memory allocating message gate backward buffer\n");
        return INK2_BAD_INPUT;
    }
    float *gh = scr, *d_gh = gh + 2 * H, *d = d_gh + 2 * H, *d_d = d + r, *d_fa = d_d + r, *d_ga = d_fa + H;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * H, *mt = msg + (size_t)t * H, *dout = d_out + (size_t)t * H;
        f_unweighted_norm(xt, H, s->eps, gh);
        f_unweighted_norm(mt, H, s->eps, gh + H);
        f_matvec(wd->data, r, 2 * H, gh, d);
        float *dxt = d_x + (size_t)t * H, *dmt = d_msg + (size_t)t * H;
        for (int h = 0; h < H; h++) {
            float fa = 0.0f, ga = 0.0f;
            const float *fr = wf->data + (size_t)h * r, *gr = wg->data + (size_t)h * r;
            for (int k = 0; k < r; k++) { fa += fr[k] * d[k]; ga += gr[k] * d[k]; }
            float fv = p_tanhf(fa), gv = p_tanhf(ga);
            float do_h = dout[h];
            dxt[h] += (1.0f - fv) * do_h;
            dmt[h] += gv * do_h;
            d_fa[h] = (-xt[h] * do_h) * (1.0f - fv * fv);
            d_ga[h] = (mt[h] * do_h) * (1.0f - gv * gv);
        }
        for (int k = 0; k < r; k++) {
            float acc = 0.0f;
            for (int h = 0; h < H; h++) {
                acc = fmaf(wf->data[(size_t)h * r + k], d_fa[h], acc);
                acc = fmaf(wg->data[(size_t)h * r + k], d_ga[h], acc);
            }
            d_d[k] = acc;
        }
        for (int i = 0; i < 2 * H; i++) {
            float acc = 0.0f;
            for (int k = 0; k < r; k++) acc = fmaf(wd->data[(size_t)k * 2 * H + i], d_d[k], acc);
            d_gh[i] = acc;
        }
        b_rms_norm_add(xt, H, s->eps, NULL, d_gh, dxt);
        b_rms_norm_add(mt, H, s->eps, NULL, d_gh + H, dmt);
    }
    free(scr);
    return INK2_OK;
}

// ---- Checkpoint metadata & whole-model backward driver
typedef struct {
    int h_in_id;
    int h_out_id;
    int nb;
    int cur_id[INK2_MAX_APP + 2];
    int has_msg;
    int n_prev;
    int prev_id[INK2_MAX_APP + 2];
    int last_h_id;
} app_ckpt_meta;

static int ink2_lora_cb(void *ctx, int block, int slot, const float **a, const float **b,
                        int *r, float *scale, int *n_in, int *n_out) {
    return model_lora_slot((const model_t *)ctx, block, slot, a, b, r, scale, n_in, n_out) ? 1 : 0;
}

bool model_inkling2_lora_backward(model_t *m, const int32_t *toks, int T, const float *pos_w, double *loss_out) {
    const void *sp = NULL, *wp = NULL;
    if (!model_inkling2_parts(m, &sp, &wp) || !sp || !wp) {
        fprintf(stderr, "error: inkling2_loop backward: model runtime parts unavailable\n");
        return false;
    }
    const ink2_shape *s = (const ink2_shape *)sp;
    const ink2_weights *w = (const ink2_weights *)wp;
    if (!model_lora_grad_alloc(m)) {
        fprintf(stderr, "error: inkling2_loop backward: out of memory allocating LoRA gradient buffers\n");
        return false;
    }

    int H = s->hidden, V = s->vocab_used, K = s->conv_k, n_app = s->n_app;
    if (T < 2 || H < 1 || V < 1) {
        fprintf(stderr, "error: inkling2_loop backward: invalid dimensions T=%d H=%d V=%d\n", T, H, V);
        return false;
    }
    for (int t = 0; t < T; t++) {
        if (toks[t] < 0 || toks[t] >= V) {
            fprintf(stderr, "error: inkling2_loop backward: token id %d at pos %d out of range [0,%d)\n",
                    (int)toks[t], t, V);
            return false;
        }
    }

    int ctrl = read_bwd_ctrl();
    const int nt = model_inkling2_threads(m);
    if (nt != ink2_get_threads()) ink2_set_threads(nt);
    size_t TH = (size_t)T * H;

    // Allocate checkpoint activations ckpt[n_app + 1] and gradient slabs d_ckpt[n_app + 1]
    float **ckpt = calloc((size_t)(n_app + 1), sizeof(float *));
    float **d_ckpt = calloc((size_t)(n_app + 1), sizeof(float *));
    app_ckpt_meta *meta = calloc((size_t)n_app, sizeof(app_ckpt_meta));
    int *ids = malloc(sizeof(int) * (size_t)T);
    ink2_cache fw_c;
    bool c_ok = (ink2_cache_init(&fw_c, s, T) == 0);
    bool ok = !bwd_alloc_fail("ckpt") && ckpt && d_ckpt && meta && ids && c_ok;
    if (ok) {
        for (int k = 0; k <= n_app; k++) {
            ckpt[k] = calloc(TH, sizeof(float));
            d_ckpt[k] = calloc(TH, sizeof(float));
            if (!ckpt[k] || !d_ckpt[k]) { ok = false; break; }
        }
    }
    if (!ok) {
        fprintf(stderr, "error: inkling2_loop backward: out of memory allocating checkpoints\n");
        if (c_ok) ink2_cache_free(&fw_c);
        free(ids); free(meta);
        for (int k = 0; ckpt && k <= n_app; k++) free(ckpt[k]);
        for (int k = 0; d_ckpt && k <= n_app; k++) free(d_ckpt[k]);
        free(ckpt); free(d_ckpt);
        return false;
    }

    for (int t = 0; t < T; t++) ids[t] = (int)toks[t];

    // 1. Forward pass (logits = NULL skips unused [T][V] head pass): record app_states and app_sel
    float *app_out_buf = malloc(sizeof(float) * (size_t)n_app * TH);
    float *row_logits = malloc(sizeof(float) * (size_t)V);
    int n_sel = n_app * T * s->n_used;
    int *sel_buf = n_sel > 0 ? malloc(sizeof(int) * (size_t)n_sel) : NULL;
    if (!app_out_buf || !row_logits || (n_sel > 0 && !sel_buf)) {
        fprintf(stderr, "error: inkling2_loop backward: out of memory allocating forward buffers\n");
        ok = false;
    }

    const ink2_tensor *emb = w->get(w->ctx, "model.embed_tokens.weight");
    const ink2_tensor *emb_nw = w->get(w->ctx, "model.embed_tokens.embed_norm.weight");
    if (!emb || !emb_nw || emb->ne[0] != H || emb_nw->ne[0] != H) {
        fprintf(stderr, "error: inkling2_loop backward: missing or invalid embedding tensors\n");
        ok = false;
    }

    if (ok) {
        for (int t = 0; t < T; t++) {
            const float *erow = emb->data + (size_t)ids[t] * H;
            f_rms_norm(erow, H, s->eps, emb_nw->data, ckpt[0] + (size_t)t * H);
        }
        ink2_lora lcb = { m, ink2_lora_cb };
        ink2_status fst = ink2_forward_chunk_lora(s, w, &lcb, &fw_c, 0, T, ids, NULL, 0, app_out_buf, sel_buf);
        if (fst != INK2_OK) {
            fprintf(stderr, "error: inkling2_loop backward: forward pass failed (status %d)\n", (int)fst);
            ok = false;
        } else if (sel_buf && !model_inkling2_save_sel(m, sel_buf, n_sel)) {
            ok = false;
        }
    }
    free(sel_buf);

    // Build exact DAG checkpoint schedule and populate ckpt[1..n_app] from app_out_buf
    int final_cur[INK2_MAX_APP + 2];
    int final_nb = 0, final_h_id = 0;
    if (ok) {
        for (int a = 0; a < n_app; a++)
            memcpy(ckpt[a + 1], app_out_buf + (size_t)a * TH, sizeof(float) * TH);

        int cur_ids[INK2_MAX_APP + 2], prev_ids[INK2_MAX_APP + 2];
        int nb = 0, n_prev = 0, pre_nb = 0, pre_h_id = 0, last_h_id = 0, h_cur_id = 0;
        for (int a = 0; a < n_app; a++) {
            int b = s->app_block[a], p = s->app_pass[a], loop = b >= s->loop_start && b < s->loop_end;
            if (loop && b == s->loop_start) {
                if (p == 0) {
                    pre_h_id = h_cur_id;
                    pre_nb = nb;
                } else {
                    n_prev = nb - pre_nb - 1;
                    if (n_prev < 0) {
                        fprintf(stderr, "error: inkling2_loop backward: invalid loop pass schedule at app %d (n_prev %d)\n", a, n_prev);
                        ok = false; break;
                    }
                    for (int j = 0; j < n_prev; j++) prev_ids[j] = cur_ids[pre_nb + 1 + j];
                    last_h_id = h_cur_id;
                    h_cur_id = pre_h_id;
                    nb = pre_nb;
                }
            }
            meta[a].h_in_id = h_cur_id;
            meta[a].h_out_id = a + 1;
            meta[a].nb = nb;
            for (int j = 0; j < nb; j++) meta[a].cur_id[j] = cur_ids[j];
            meta[a].has_msg = (loop && p > 0);
            if (meta[a].has_msg) {
                meta[a].n_prev = n_prev;
                for (int j = 0; j < n_prev; j++) meta[a].prev_id[j] = prev_ids[j];
                meta[a].last_h_id = last_h_id;
            }
            if (ink2_block_boundary(s, b)) cur_ids[nb++] = h_cur_id;
            h_cur_id = a + 1;
        }
        final_nb = nb;
        for (int j = 0; j < nb; j++) final_cur[j] = cur_ids[j];
        final_h_id = h_cur_id;
    }
    free(app_out_buf);
    ink2_cache_free(&fw_c);

    // 2. Final block read + model.norm + lm_head forward row-by-row & backward
    double total_loss = 0.0;
    if (ok) {
        const ink2_tensor *fq = w->get(w->ctx, "model.final_input.query");
        const ink2_tensor *nw = w->get(w->ctx, "model.norm.weight");
        const ink2_tensor *wo = w->get(w->ctx, "lm_head.weight");
        int nkv = final_nb + 1;
        float *fin = malloc(sizeof(float) * TH * 4);
        float *raw = malloc(sizeof(float) * (size_t)V * 3);
        if (!fq || !nw || !wo || !fin || !raw) {
            fprintf(stderr, "error: inkling2_loop backward: out of memory allocating head buffers\n");
            ok = false;
        } else {
            float *hn = fin + TH, *d_hn = hn + TH, *d_fin = d_hn + TH;
            float *th = raw + V, *d_raw = th + V;
            memset(d_hn, 0, sizeof(float) * TH);
            memset(d_fin, 0, sizeof(float) * TH);

            const float *fstates[INK2_MAX_APP + 2];
            float *fdstates[INK2_MAX_APP + 2];
            for (int j = 0; j < final_nb; j++) {
                fstates[j] = ckpt[final_cur[j]];
                fdstates[j] = d_ckpt[final_cur[j]];
            }
            fstates[final_nb] = ckpt[final_h_id];
            fdstates[final_nb] = d_ckpt[final_h_id];

            // Recompute final_input & model.norm
            {
                float *bscr = bwd_alloc_fail("bscr") ? NULL : malloc(sizeof(float) * ((size_t)nkv * H * 2 + (size_t)nkv));
                if (!bscr) {
                    fprintf(stderr, "error: inkling2_loop backward: out of memory allocating final_input buffer\n");
                    ok = false;
                } else {
                    float *sn = bscr + (size_t)nkv * H, *bsc = sn + (size_t)nkv * H;
                    for (int t = 0; t < T; t++) {
                        for (int j = 0; j < nkv; j++)
                            memcpy(bscr + (size_t)j * H, fstates[j] + (size_t)t * H, sizeof(float) * (size_t)H);
                        f_depth_attn(fq->data, 1, bscr, nkv, H, sn, bsc, fin + (size_t)t * H);
                        f_rms_norm(fin + (size_t)t * H, H, s->eps, nw->data, hn + (size_t)t * H);
                    }
                    free(bscr);
                }
            }
            // Head forward + softmax loss + head VJP row-by-row
            for (int t = 0; ok && t < T; t++) {
                const float *hnt = hn + (size_t)t * H;
                f_matvec(wo->data, V, H, hnt, raw);
                for (int j = 0; j < V; j++) {
                    float thv = p_tanhf(raw[j] / s->final_cap);
                    th[j] = thv;
                    row_logits[j] = thv * s->final_cap;
                }
                if (m->tape_logits_cb) m->tape_logits_cb(m->tape_logits_ud, t, row_logits);
                if (t < T - 1 && (!pos_w || pos_w[t] != 0.0f)) {
                    int target = ids[t + 1];
                    double wt = pos_w ? (double)pos_w[t] : 1.0;
                    float mx = row_logits[0];
                    for (int j = 1; j < V; j++) if (row_logits[j] > mx) mx = row_logits[j];
                    double sum = 0.0;
                    for (int j = 0; j < V; j++) sum += exp((double)row_logits[j] - (double)mx);
                    total_loss += wt * (((double)mx + log(sum)) - (double)row_logits[target]);
                    for (int j = 0; j < V; j++) {
                        double p = exp((double)row_logits[j] - (double)mx) / sum;
                        double d = wt * (p - (j == target ? 1.0 : 0.0));
                        d_raw[j] = (float)d * (1.0f - th[j] * th[j]);
                    }
                    b_matvec_T_add(wo->data, V, H, 1, d_raw, d_hn + (size_t)t * H);
                }
            }
            // norm -> final_input backward
            if (ok) {
                for (int t = 0; t < T; t++)
                    b_rms_norm_add(fin + (size_t)t * H, H, s->eps, nw->data, d_hn + (size_t)t * H, d_fin + (size_t)t * H);
                if (b_depth_attn_common(fq->data, 1, H, T, nkv, fstates, fdstates, d_fin, true) != INK2_OK)
                    ok = false;
            }
        }
        free(fin); free(raw);
    }
    free(row_logits);

    // 3. Reverse sweep across all applications a = n_app - 1 .. 0
    for (int a = n_app - 1; ok && a >= 0; a--) {
        const app_ckpt_meta *M = &meta[a];
        int b = s->app_block[a], g = s->block_global[b];
        int bound = ink2_block_boundary(s, b);
        const float *x = ckpt[M->h_in_id];
        float *dx = d_ckpt[M->h_in_id];
        const float *d_out = d_ckpt[M->h_out_id];

        // Single-application buffers
        size_t n_rd = (size_t)(g ? 3 : 1);
        float *lbuf = bwd_alloc_fail("layer") ? NULL : calloc(TH * 17, sizeof(float));
        if (!lbuf) {
            fprintf(stderr, "error: inkling2_loop backward: out of memory allocating layer scratch (app %d)\n", a);
            ok = false; break;
        }
        float *rd       = lbuf;               // [3][T][H]
        float *qkv      = rd + 3 * TH;        // [3][T][H]
        float *gt_qkv   = qkv + 3 * TH;       // [3][T][H] pre-norm gated message output
        float *msg_flat = gt_qkv + 3 * TH;    // [T][4][H] if has_msg -> allocates separate if >4*TH
        float *att      = msg_flat;
        float *att_post = att + TH;
        float *res      = att_post + TH;
        float *mi_pre   = res + TH;
        float *mi_gate  = mi_pre + TH;
        float *mi_ln    = mi_gate + TH;
        float *mo       = mi_ln + TH;
        float *recomp   = mo + TH;

        float *msgs = M->has_msg ? calloc(TH * 4, sizeof(float)) : NULL;
        float *d_msgs = M->has_msg ? calloc(TH * 4, sizeof(float)) : NULL;
        float *d_mo = calloc(TH * 10, sizeof(float));
        if ((M->has_msg && (!msgs || !d_msgs)) || !d_mo) {
            fprintf(stderr, "error: inkling2_loop backward: out of memory allocating layer backward buffers (app %d)\n", a);
            free(lbuf); free(msgs); free(d_msgs); free(d_mo);
            ok = false; break;
        }
        float *d_att_post = d_mo + TH;
        float *d_att      = d_att_post + TH;
        float *d_res      = d_att + TH;
        float *d_mi_ln    = d_res + TH;
        float *d_mi_gate  = d_mi_ln + TH;
        float *d_mi_pre   = d_mi_gate + TH;
        float *d_qkv      = d_mi_pre + TH;    // [3][T][H]

        char tname[160];
        const char *roles[4] = { "q", "k", "v", "mlp" };

        // Forward recompute of message_input (if p > 0 loop)
        const ink2_tensor *mq = NULL;
        const float *mstates[INK2_MAX_APP + 2];
        float *mdstates[INK2_MAX_APP + 2];
        if (M->has_msg) {
            mq = w->get(w->ctx, "model.message_input.query");
            if (!mq) {
                fprintf(stderr, "error: inkling2_loop backward: missing tensor 'model.message_input.query'\n");
                ok = false;
            } else {
                for (int j = 0; j < M->n_prev; j++) {
                    mstates[j] = ckpt[M->prev_id[j]];
                    mdstates[j] = d_ckpt[M->prev_id[j]];
                }
                mstates[M->n_prev] = ckpt[M->last_h_id];
                mdstates[M->n_prev] = d_ckpt[M->last_h_id];
                int nkv_m = M->n_prev + 1;
                float *bscr = malloc(sizeof(float) * ((size_t)nkv_m * H * 2 + (size_t)nkv_m));
                if (!bscr) {
                    fprintf(stderr, "error: inkling2_loop backward: out of memory allocating message_input buffer\n");
                    ok = false;
                } else {
                    float *sn = bscr + (size_t)nkv_m * H, *bsc = sn + (size_t)nkv_m * H;
                    const float *ql = mq->data + (size_t)(b - s->loop_start) * 4 * H;
                    for (int t = 0; t < T; t++) {
                        for (int j = 0; j < nkv_m; j++)
                            memcpy(bscr + (size_t)j * H, mstates[j] + (size_t)t * H, sizeof(float) * (size_t)H);
                        f_depth_attn(ql, 4, bscr, nkv_m, H, sn, bsc, msgs + (size_t)t * 4 * H);
                    }
                    free(bscr);
                }
            }
        }

        // Recompute attention_input
        const ink2_tensor *aq = NULL;
        const float *astates[INK2_MAX_APP + 2];
        float *adstates[INK2_MAX_APP + 2];
        for (int j = 0; j < M->nb; j++) { astates[j] = ckpt[M->cur_id[j]]; adstates[j] = d_ckpt[M->cur_id[j]]; }
        astates[M->nb] = x; adstates[M->nb] = dx;
        if (ok && b > 0) {
            snprintf(tname, sizeof tname, "model.layers.%d.attention_input.query", b);
            aq = w->get(w->ctx, tname);
            if (!aq) {
                fprintf(stderr, "error: inkling2_loop backward: missing tensor '%s'\n", tname);
                ok = false;
            } else {
                int nkv_a = M->nb + 1;
                float *bscr = malloc(sizeof(float) * ((size_t)nkv_a * H * 2 + (size_t)nkv_a + n_rd * H));
                if (!bscr) {
                    fprintf(stderr, "error: inkling2_loop backward: out of memory allocating attention_input buffer\n");
                    ok = false;
                } else {
                    float *sn = bscr + (size_t)nkv_a * H, *bsc = sn + (size_t)nkv_a * H, *bo = bsc + nkv_a;
                    for (int t = 0; t < T; t++) {
                        for (int j = 0; j < nkv_a; j++)
                            memcpy(bscr + (size_t)j * H, astates[j] + (size_t)t * H, sizeof(float) * (size_t)H);
                        f_depth_attn(aq->data, (int)n_rd, bscr, nkv_a, H, sn, bsc, bo);
                        for (int i = 0; i < (int)n_rd; i++)
                            memcpy(rd + ((size_t)i * T + t) * H, bo + (size_t)i * H, sizeof(float) * (size_t)H);
                    }
                    free(bscr);
                }
            }
        } else if (ok) {
            memcpy(rd, x, sizeof(float) * TH);
        }

        // Recompute Q, K, V message gates + input norms
        const ink2_tensor *in_norm[3] = {NULL, NULL, NULL};
        float *role_msg_t[4] = {NULL, NULL, NULL, NULL};
        if (M->has_msg) {
            role_msg_t[0] = malloc(sizeof(float) * TH * 4);
            if (!role_msg_t[0]) {
                fprintf(stderr, "error: inkling2_loop backward: out of memory allocating role message buffer\n");
                ok = false;
            } else {
                for (int r = 1; r < 4; r++) role_msg_t[r] = role_msg_t[0] + (size_t)r * TH;
                for (int t = 0; t < T; t++)
                    for (int r = 0; r < 4; r++)
                        memcpy(role_msg_t[r] + (size_t)t * H, msgs + ((size_t)t * 4 + r) * H, sizeof(float) * (size_t)H);
            }
        }
        for (int r = 0; ok && r < 3; r++) {
            const float *src = rd + (size_t)(g ? r : 0) * TH;
            if (M->has_msg) {
                if (b_msg_step_fwd(s, w, b, roles[r], T, src, role_msg_t[r], gt_qkv + (size_t)r * TH) != INK2_OK)
                    ok = false;
                src = gt_qkv + (size_t)r * TH;
            } else if (!g && r > 0) {
                memcpy(qkv + (size_t)r * TH, qkv, sizeof(float) * TH);
                continue;
            }
            if (g) snprintf(tname, sizeof tname, "model.layers.%d.%s_input_layernorm.weight", b, roles[r]);
            else snprintf(tname, sizeof tname, "model.layers.%d.input_layernorm.weight", b);
            in_norm[r] = w->get(w->ctx, tname);
            if (!in_norm[r]) {
                fprintf(stderr, "error: inkling2_loop backward: missing tensor '%s'\n", tname);
                ok = false; break;
            }
            for (int t = 0; t < T; t++)
                f_rms_norm(src + (size_t)t * H, H, s->eps, in_norm[r]->data, qkv + (size_t)r * TH + (size_t)t * H);
        }

        // 1a. Run single-application self_attn forward once and retain intermediates for Step 2 backward.
        attn_fwd_cache af;
        memset(&af, 0, sizeof af);
        if (ok) {
            if (b_attn_fwd_save(s, w, m, a, T, qkv, qkv + TH, qkv + 2 * TH, &af, att) != INK2_OK)
                ok = false;
        }

        const ink2_tensor *pac = NULL, *mq_mlp = NULL, *paln = NULL, *pmc = NULL;
        // 1b. post_attn_conv, residual, mlp_input, mlp_message_gate, post_attn_ln, experts, post_mlp_conv
        if (ok) {
            snprintf(tname, sizeof tname, "model.layers.%d.post_attention_convolution.conv1d.weight", b);
            pac = w->get(w->ctx, tname);
            snprintf(tname, sizeof tname, "model.layers.%d.mlp_input.query", b);
            mq_mlp = w->get(w->ctx, tname);
            snprintf(tname, sizeof tname, "model.layers.%d.post_attention_layernorm.weight", b);
            paln = w->get(w->ctx, tname);
            snprintf(tname, sizeof tname, "model.layers.%d.post_mlp_convolution.conv1d.weight", b);
            pmc = w->get(w->ctx, tname);
            if (!pac || !mq_mlp || !paln || !pmc) {
                fprintf(stderr, "error: inkling2_loop backward: missing layer %d conv/mlp_input/norm tensors\n", b);
                ok = false;
            }
        }
        int nb2 = M->nb + (bound ? 1 : 0);
        const float *m2states[INK2_MAX_APP + 2];
        float *m2dstates[INK2_MAX_APP + 2];
        if (ok) {
            f_short_conv(pac->data, H, K, T, att, att_post);
            if (bound) {
                memcpy(res, att_post, sizeof(float) * TH);
                for (int j = 0; j < M->nb; j++) { m2states[j] = astates[j]; m2dstates[j] = adstates[j]; }
                m2states[M->nb] = x; m2dstates[M->nb] = dx;
            } else {
                for (size_t i = 0; i < TH; i++) res[i] = x[i] + att_post[i];
                for (int j = 0; j < M->nb; j++) { m2states[j] = astates[j]; m2dstates[j] = adstates[j]; }
            }
            m2states[nb2] = res; m2dstates[nb2] = d_res;
            int nkv_m2 = nb2 + 1;
            float *bscr = malloc(sizeof(float) * ((size_t)nkv_m2 * H * 2 + (size_t)nkv_m2));
            if (!bscr) {
                fprintf(stderr, "error: inkling2_loop backward: out of memory allocating mlp_input buffer\n");
                ok = false;
            } else {
                float *sn = bscr + (size_t)nkv_m2 * H, *bsc = sn + (size_t)nkv_m2 * H;
                for (int t = 0; t < T; t++) {
                    for (int j = 0; j < nkv_m2; j++)
                        memcpy(bscr + (size_t)j * H, m2states[j] + (size_t)t * H, sizeof(float) * (size_t)H);
                    f_depth_attn(mq_mlp->data, 1, bscr, nkv_m2, H, sn, bsc, mi_pre + (size_t)t * H);
                }
                free(bscr);
            }
        }
        if (ok) {
            const float *mlp_in = mi_pre;
            if (M->has_msg) {
                if (b_msg_step_fwd(s, w, b, "mlp", T, mi_pre, role_msg_t[3], mi_gate) != INK2_OK) ok = false;
                mlp_in = mi_gate;
            }
            if (ok) {
                for (int t = 0; t < T; t++)
                    f_rms_norm(mlp_in + (size_t)t * H, H, s->eps, paln->data, mi_ln + (size_t)t * H);
                ink2_status est = ink2_expert_layer(s, w, b, T, mi_ln, mo);
                if (est != INK2_OK) {
                    fprintf(stderr, "error: inkling2_loop backward: expert forward failed in block %d (status %d)\n", b, (int)est);
                    ok = false;
                }
            }
            if (ok) {
                float *mo_post = malloc(sizeof(float) * TH);
                if (!mo_post) {
                    fprintf(stderr, "error: inkling2_loop backward: out of memory allocating mo_post buffer\n");
                    ok = false;
                } else {
                    f_short_conv(pmc->data, H, K, T, mo, mo_post);
                    for (size_t i = 0; i < TH; i++) recomp[i] = res[i] + mo_post[i];
                    free(mo_post);
                }
            }
        }

        // Condition 4: BIT-EXACT recompute check against saved checkpoint h_out
        if (ok) {
            if (ctrl & CTRL_DRIFT_CKPT) {
                uint32_t bits;
                memcpy(&bits, &recomp[0], 4);
                bits ^= 1u;
                memcpy(&recomp[0], &bits, 4);
            }
            if (memcmp(recomp, ckpt[M->h_out_id], sizeof(float) * TH) != 0) {
                fprintf(stderr, "error: inkling2_loop backward: recompute at application %d (block %d pass %d) "
                                "differed from saved forward\n", a, b, s->app_pass[a]);
                ok = false;
            }
        }

        // 2. Backward through post_mlp_conv, MoE, post_attention_layernorm, mlp_message_gate, mlp_input
        if (ok) {
            // out = res + post_mlp_conv(mo)
            memcpy(d_res, d_out, sizeof(float) * TH);
            b_short_conv_add(pmc->data, H, K, T, d_out, d_mo);
            if (b_moe_layer(s, w, b, T, mi_ln, d_mo, d_mi_ln, ctrl) != INK2_OK) ok = false;
        }
        const float *mlp_in = M->has_msg ? mi_gate : mi_pre;
        float *d_mlp_in = M->has_msg ? d_mi_gate : d_mi_pre;
        float *d_role_msg_t[4] = {NULL, NULL, NULL, NULL};
        if (ok && M->has_msg) {
            d_role_msg_t[0] = calloc(TH * 4, sizeof(float));
            if (!d_role_msg_t[0]) {
                fprintf(stderr, "error: inkling2_loop backward: out of memory allocating role message gradient buffer\n");
                ok = false;
            } else {
                for (int r = 1; r < 4; r++) d_role_msg_t[r] = d_role_msg_t[0] + (size_t)r * TH;
            }
        }
        if (ok) {
            for (int t = 0; t < T; t++)
                b_rms_norm_add(mlp_in + (size_t)t * H, H, s->eps, paln->data, d_mi_ln + (size_t)t * H,
                               d_mlp_in + (size_t)t * H);
            if (M->has_msg) {
                if (b_msg_step_bwd(s, w, b, "mlp", T, mi_pre, role_msg_t[3], d_mi_gate,
                                   d_mi_pre, d_role_msg_t[3]) != INK2_OK)
                    ok = false;
            }
        }
        if (ok) {
            int nkv_m2 = nb2 + 1;
            if (b_depth_attn_common(mq_mlp->data, 1, H, T, nkv_m2, m2states, m2dstates, d_mi_pre, true) != INK2_OK)
                ok = false;
            if (ok) {
                if (!bound) {
                    for (size_t i = 0; i < TH; i++) dx[i] += d_res[i];
                }
                b_short_conv_add(pac->data, H, K, T, d_res, d_att);
                if (b_attn_bwd_from_saved(s, m, a, T, qkv, qkv + TH, qkv + 2 * TH, &af,
                                          d_att, d_qkv, d_qkv + TH, d_qkv + 2 * TH, ctrl) != INK2_OK)
                    ok = false;
            }
        }
        free(af.buf);
        if (ok) {
            float *d_rd = calloc(TH * 3, sizeof(float));
            if (!d_rd) {
                fprintf(stderr, "error: inkling2_loop backward: out of memory allocating role read gradient buffer\n");
                ok = false;
            } else {
                for (int r = 2; ok && r >= 0; r--) {
                    if (!M->has_msg && !g && r > 0) {
                        for (size_t i = 0; i < TH; i++) d_qkv[i] += d_qkv[(size_t)r * TH + i];
                        continue;
                    }
                    const float *src = M->has_msg ? gt_qkv + (size_t)r * TH : rd + (size_t)(g ? r : 0) * TH;
                    if (!M->has_msg) {
                        for (int t = 0; t < T; t++)
                            b_rms_norm_add(src + (size_t)t * H, H, s->eps, in_norm[r]->data,
                                           d_qkv + (size_t)r * TH + (size_t)t * H,
                                           d_rd + (size_t)(g ? r : 0) * TH + (size_t)t * H);
                    } else {
                        float *d_gt = calloc(TH, sizeof(float));
                        if (!d_gt) {
                            fprintf(stderr, "error: inkling2_loop backward: out of memory allocating gate norm gradient buffer\n");
                            ok = false; break;
                        }
                        for (int t = 0; t < T; t++)
                            b_rms_norm_add(src + (size_t)t * H, H, s->eps, in_norm[r]->data,
                                           d_qkv + (size_t)r * TH + (size_t)t * H,
                                           d_gt + (size_t)t * H);
                        if (b_msg_step_bwd(s, w, b, roles[r], T, rd + (size_t)(g ? r : 0) * TH, role_msg_t[r],
                                           d_gt, d_rd + (size_t)(g ? r : 0) * TH, d_role_msg_t[r]) != INK2_OK)
                            ok = false;
                        free(d_gt);
                    }
                }
                if (ok && b > 0) {
                    int nkv_a = M->nb + 1;
                    if (b_depth_attn_common(aq->data, (int)n_rd, H, T, nkv_a, astates, adstates, d_rd, true) != INK2_OK)
                        ok = false;
                } else if (ok) {
                    for (size_t i = 0; i < TH; i++) dx[i] += d_rd[i];
                }
                free(d_rd);
            }
        }
        if (ok && M->has_msg) {
            for (int t = 0; t < T; t++)
                for (int r = 0; r < 4; r++)
                    memcpy(d_msgs + ((size_t)t * 4 + r) * H, d_role_msg_t[r] + (size_t)t * H, sizeof(float) * (size_t)H);
            const float *ql = mq->data + (size_t)(b - s->loop_start) * 4 * H;
            if (b_depth_attn_common(ql, 4, H, T, M->n_prev + 1, mstates, mdstates, d_msgs, false) != INK2_OK)
                ok = false;
        }

        free(d_role_msg_t[0]);
        free(lbuf); free(msgs); free(d_msgs); free(d_mo); free(role_msg_t[0]);
    }

    if (ok && loss_out) *loss_out = total_loss;
    for (int k = 0; k <= n_app; k++) { free(ckpt[k]); free(d_ckpt[k]); }
    free(ckpt); free(d_ckpt); free(meta); free(ids);
    return ok;
}
