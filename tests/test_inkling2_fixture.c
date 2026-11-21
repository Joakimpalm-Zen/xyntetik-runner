// R4.48: the Inkling2Loop fixture harness. Loads the tiny reference fixture (the lab's record store, R4.48 step 1a: config.json,
// fixture-weights.safetensors, dumps-<seq>.safetensors, inputs-<seq>.safetensors, from transformers pull request
// 51050 at 6173a2f6) and checks:
//   1. the structure: the schedule derived from the config (ink2_shape_derive) against the reference's own call order
//      of the layer modules, application by application;
//   2. every piece of inkling2.h: fed the reference module's recorded inputs, its outputs against the module's
//      recorded outputs, tensor by tensor.
//
// THE TOLERANCE RULE, stated once (fixed 2026-11-21): a piece is compared with the FLOAT64 TRUTH
// (truth-<seq>.safetensors: the reference module re-run in double on the same recorded inputs), element by element,
// under |got - truth| <= bound. The bound is c * n * u * S for one reduction of length n over magnitudes S, with
// u = 2^-24 and c = 2; a chain of steps with no recorded intermediate gets its COMPOSED first-order bound, written out
// per piece below (bound_depth, bound_gate, bound_moe, bound_attn). The harness reports, per piece:
//   - the largest fraction of the bound the piece uses (must be <= 1);
//   - the same fraction for torch's own float32 dump (reported; if torch exceeds the bound, that is a finding about
//     the reference, printed as such);
//   - the perturbation control: one weight of the piece's own tensors + 0.25 must take the piece OUT of bound; its
//     size is printed as a multiple of the bound at the element it moves most.
// Torch's order (its blocked BLAS) is not ours, so bits against the reference are not claimed for reductions; our own
// bits (Runner against Runner, a strict build on two machines) are a separate gate.
//
// A stub returns INK2_NOT_IMPLEMENTED and its line reads RED. The exit status is 1 while any piece is RED or FAILS,
// unless --allow-red names it (for the record of what has landed).
//
// usage: test-inkling2-fixture FIXTURE_DIR [--allow-red piece,piece,...]
#include "inkling2.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- a test-side safetensors reader
typedef struct { char name[192]; int dtype; int n_dims; int64_t shape[6]; const unsigned char *data; size_t nbytes; } st_t;
enum { ST_F32, ST_I64, ST_F64, ST_OTHER };
typedef struct { unsigned char *buf; size_t n; st_t *t; int nt; } st_file;

static int st_load(st_file *f, const char *path) {
    memset(f, 0, sizeof *f);
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    f->buf = malloc((size_t)n); f->n = (size_t)n;
    if (!f->buf || fread(f->buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); return -1; }
    fclose(fp);
    uint64_t hn; memcpy(&hn, f->buf, 8);
    if (8 + hn > f->n) return -1;
    jv *h = json_parse((const char *)f->buf + 8, (size_t)hn);
    if (!h || h->type != J_OBJ) return -1;
    f->t = calloc((size_t)h->n, sizeof *f->t);
    for (int i = 0; i < h->n; i++) {
        if (!strcmp(h->keys[i], "__metadata__")) continue;
        jv *e = h->items[i]; st_t *t = &f->t[f->nt];
        snprintf(t->name, sizeof t->name, "%s", h->keys[i]);
        const char *dt = jv_str(jv_get(e, "dtype"), "");
        t->dtype = !strcmp(dt, "F32") ? ST_F32 : !strcmp(dt, "I64") ? ST_I64 : !strcmp(dt, "F64") ? ST_F64 : ST_OTHER;
        jv *sh = jv_get(e, "shape"), *off = jv_get(e, "data_offsets");
        t->n_dims = sh ? sh->n : 0;
        for (int d = 0; d < t->n_dims && d < 6; d++) t->shape[d] = (int64_t)sh->items[d]->num;
        size_t a = (size_t)off->items[0]->num, b = (size_t)off->items[1]->num;
        t->data = f->buf + 8 + hn + a; t->nbytes = b - a;
        f->nt++;
    }
    jv_free(h);
    return 0;
}
static const st_t *st_get(const st_file *f, const char *name) {
    for (int i = 0; i < f->nt; i++) if (!strcmp(f->t[i].name, name)) return &f->t[i];
    return NULL;
}
static size_t st_count(const st_t *t) { size_t n = 1; for (int d = 0; d < t->n_dims; d++) n *= (size_t)t->shape[d]; return n; }

// the weights, as ink2_tensor (reversed dims: ne[0] fastest)
typedef struct { st_file *f; ink2_tensor tmp[8]; int k; } wctx;
static const ink2_tensor *w_get(void *ctx, const char *name) {
    wctx *c = ctx; const st_t *t = st_get(c->f, name);
    if (!t || t->dtype != ST_F32) return NULL;
    ink2_tensor *o = &c->tmp[c->k++ & 7];
    o->name = t->name; o->data = (const float *)t->data; o->n_dims = t->n_dims;
    for (int d = 0; d < 4; d++) o->ne[d] = d < t->n_dims ? t->shape[t->n_dims - 1 - d] : 1;
    return o;
}

// ---------------------------------------------------------------- the config
// the config's constants as the doubles the float64 truth uses (ink2_shape holds them rounded to float32)
static struct { double alpha, eps, qk, rel; } cfd;
static int cfg_load(ink2_shape *s, const char *path, char *why, size_t wn) {
    FILE *fp = fopen(path, "rb"); if (!fp) { snprintf(why, wn, "cannot open %s", path); return -1; }
    char buf[1 << 16]; size_t n = fread(buf, 1, sizeof buf - 1, fp); fclose(fp); buf[n] = 0;
    jv *c = json_parse(buf, n); if (!c) { snprintf(why, wn, "config.json does not parse"); return -1; }
    memset(s, 0, sizeof *s);
#define I(k) (int)jv_num(jv_get(c, k), -1)
#define F(k) (float)jv_num(jv_get(c, k), -1.0)   // -1: absent (no NAN: the default build is fast-math)
    s->n_block = I("num_hidden_layers"); s->attn_res_block = I("n_layers_per_attn_res_block");
    s->loop_start = I("loop_layer_index_start"); s->loop_end = I("loop_layer_index_end");
    s->loop_passes = I("loop_num_passes"); s->loop_gate_rank = I("loop_gate_rank");
    s->hidden = I("hidden_size"); s->vocab = I("vocab_size"); s->vocab_used = I("unpadded_vocab_size");
    s->n_head = I("num_attention_heads"); s->n_kv = I("num_key_value_heads"); s->head_dim = I("head_dim");
    s->swa_n_head = I("swa_num_attention_heads"); s->swa_n_kv = I("swa_num_key_value_heads");
    s->swa_head_dim = I("swa_head_dim"); s->window = I("sliding_window_size"); s->d_rel = I("d_rel");
    s->rel_extent = I("rel_extent"); s->conv_k = I("conv_kernel_size"); s->n_expert = I("n_routed_experts");
    s->n_used = I("num_experts_per_tok"); s->ff = I("moe_intermediate_size"); s->log_floor = I("log_scaling_n_floor");
    s->log_alpha = F("log_scaling_alpha"); s->eps = F("rms_norm_eps"); s->route_scale = F("route_scale");
    s->gate_cap = F("mlp_gate_softcap"); s->up_cap = F("mlp_up_softcap"); s->final_cap = F("final_logit_softcapping");
    s->qk_clip = F("qk_l2_clip_threshold"); s->rel_clip = F("rel_l2_clip_threshold");
#undef I
#undef F
    cfd.alpha = jv_num(jv_get(c, "log_scaling_alpha"), -1); cfd.eps = jv_num(jv_get(c, "rms_norm_eps"), -1);
    cfd.qk = jv_num(jv_get(c, "qk_l2_clip_threshold"), -1); cfd.rel = jv_num(jv_get(c, "rel_l2_clip_threshold"), -1);
    jv *lt = jv_get(c, "layer_types");
    if (!lt || lt->type != J_ARR || lt->n != s->n_block) { jv_free(c); snprintf(why, wn, "layer_types missing or not %d long", s->n_block); return -1; }
    for (int b = 0; b < s->n_block && b < INK2_MAX_BLOCK; b++) {
        const char *t = jv_str(lt->items[b], "");
        if (strcmp(t, "hybrid") && strcmp(t, "hybrid_sliding")) { jv_free(c); snprintf(why, wn, "layer %d type %s", b, t); return -1; }
        s->block_global[b] = !strcmp(t, "hybrid");
    }
    jv_free(c);
    return ink2_shape_derive(s, why, wn);
}

// ---------------------------------------------------------------- comparison
// ---------------------------------------------------------------- bound mode
#define U  (1.0 / 16777216.0)   // 2^-24
#define CB 2.0                  // c in c * n * u * S
typedef struct { const char *piece; int calls, red, nob, fail; double frac, torch_frac, pert_mult; char note[256]; } bstat;
static void norm_d(const float *x, int n, double eps, double *y) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    double r = 1.0 / sqrt(ss / n + eps); for (int i = 0; i < n; i++) y[i] = x[i] * r;
}
// the depth-wise attention's composed bound at one position, out [nq][H]:
//   e_n = c (H + 5) u                                  each normed element (sum of H squares, mean, eps, sqrt, 1/x, mul)
//   ds_ij = (c (H + 1) u + e_n) A_ij, A_ij = H^-1/2 sum_h |q_ih| |n_jh|        a score (the dot, the scale)
//   dp_ij = p_ij (2 max_k ds_ik + c (nkv + 3) u)        softmax (the shift cancels; exp 1 ulp, the sum, the division)
//   bound_ih = sum_j dp_ij |n_jh| + sum_j p_ij e_n |n_jh| + c (nkv + 1) u sum_j p_ij |n_jh|     the weighted sum
static void bound_depth(const float *q, int nq, const float *states, int nkv, int H, double *bound) {
    double *n = malloc(sizeof(double) * (size_t)nkv * H), *sc = malloc(sizeof(double) * nkv * 3);
    double *A = sc + nkv, *p = A + nkv, e_n = CB * (H + 5) * U, scale = 1.0 / sqrt((double)H);
    for (int j = 0; j < nkv; j++) norm_d(states + (size_t)j * H, H, 1e-6, n + (size_t)j * H);
    for (int i = 0; i < nq; i++) {
        double mx = -1e300, dsmax = 0, sum = 0;
        for (int j = 0; j < nkv; j++) {
            double d = 0, a = 0;
            for (int h = 0; h < H; h++) { d += q[(size_t)i * H + h] * n[(size_t)j * H + h]; a += fabs(q[(size_t)i * H + h] * n[(size_t)j * H + h]); }
            sc[j] = d * scale; A[j] = a * scale; if (sc[j] > mx) mx = sc[j];
            double ds = (CB * (H + 1) * U + e_n) * A[j]; if (ds > dsmax) dsmax = ds;
        }
        for (int j = 0; j < nkv; j++) { p[j] = exp(sc[j] - mx); sum += p[j]; }
        for (int j = 0; j < nkv; j++) p[j] /= sum;
        for (int h = 0; h < H; h++) {
            double b = 0;
            for (int j = 0; j < nkv; j++) {
                double an = fabs(n[(size_t)j * H + h]);
                b += p[j] * (2 * dsmax + CB * (nkv + 3) * U) * an + p[j] * e_n * an + CB * (nkv + 1) * U * p[j] * an;
            }
            bound[(size_t)i * H + h] = b;
        }
    }
    free(n); free(sc);
}
// the message gate's composed bound at one position, out [H]:
//   gh = [norm(x), norm(m)], each element relative e_n = c (H + 5) u
//   dd_k = (c 2H u + e_n) D_k, D_k = sum_i |Wd_ki| |gh_i|                         the rank-r down projection
//   dF_h = c r u sum_k |Wf_hk| |d_k| + sum_k |Wf_hk| dd_k                          (and G with Wg)
//   df_h = dF_h + 2 u |f_h|                                                        tanh' <= 1; tanh within 2 ulp
//   bound_h = |x_h| df_h + |m_h| dg_h + c 3 u (|(1 - f_h) x_h| + |g_h m_h|)        the mix
static void bound_gate(const float *x, const float *m, const float *wd, const float *wf, const float *wg, int H, int r,
                       double eps, double *bound) {
    double *gh = malloc(sizeof(double) * (2 * H + 2 * r)), *d = gh + 2 * H, *dd = d + r, e_n = CB * (H + 5) * U;
    norm_d(x, H, eps, gh); norm_d(m, H, eps, gh + H);
    for (int k = 0; k < r; k++) {
        double a = 0, D = 0;
        for (int i = 0; i < 2 * H; i++) { a += wd[(size_t)k * 2 * H + i] * gh[i]; D += fabs(wd[(size_t)k * 2 * H + i] * gh[i]); }
        d[k] = a; dd[k] = (CB * 2 * H * U + e_n) * D;
    }
    for (int h = 0; h < H; h++) {
        double F = 0, G = 0, aF = 0, aG = 0, pF = 0, pG = 0;
        for (int k = 0; k < r; k++) {
            F += wf[(size_t)h * r + k] * d[k]; G += wg[(size_t)h * r + k] * d[k];
            aF += fabs(wf[(size_t)h * r + k] * d[k]); aG += fabs(wg[(size_t)h * r + k] * d[k]);
            pF += fabs(wf[(size_t)h * r + k]) * dd[k]; pG += fabs(wg[(size_t)h * r + k]) * dd[k];
        }
        double f = tanh(F), g = tanh(G);
        double df = CB * r * U * aF + pF + 2 * U * fabs(f), dg = CB * r * U * aG + pG + 2 * U * fabs(g);
        bound[h] = fabs(x[h]) * df + fabs(m[h]) * dg + CB * 3 * U * (fabs((1 - f) * x[h]) + fabs(g * m[h]));
    }
    free(gh);
}

// The expert layer's composed bound at one position (out [H]), and its float64 value y (the harness's own double
// evaluation, checked against the truth), with the selection sel[K] (ascending, ties lowest index first):
//   dl_e = c H u A_e, A_e = sum |W_eh x_h|                         router logits
//   ds_e = s(1 - s) dl_e + 3 u s                                   sigmoid (exp within 1 ulp, the sum, the division)
//   dw_i = w_i (ds_i / s_i + sum_sel ds_j / S + c (K + 3) u)       the normalized, scaled weights
//   d(gu)_j = c H u G_j, G_j = sum |W13_jh x_h|                    gate and up
//   d(cap) = d(arg) + 3 u |cap|; d(sigmoid) = s'(g) d(g) + 3 u s   softcaps (tanh' <= 1) and the gate's sigmoid
//   d(act) = |sig capu| d(capg) + |capg capu| d(sig) + |capg sig| d(capu) + 2 u |act|
//   d(down)_h = c F u sum |W2_hj act_j| + sum |W2_hj| d(act_j)
//   bound_h = sum_i (|w_i| d(down)_ih + dw_i |down_ih|) + c K u sum_i |w_i down_ih|
static void bound_moe(const float *x, const float *wr, const float *br, const float *w13, const float *w2, int H, int E,
                      int K, int F, double R, double capg, double capu, double *y, double *bound, int *sel) {
    double *l = malloc(sizeof(double) * (size_t)(4 * E + 6 * F + 2 * H)), *A = l + E, *sg = A + E, *ds = sg + E;
    double *gu = ds + E, *G = gu + 2 * F, *act = G + 2 * F, *dact = act + F, *down = dact + F, *ddown = down + H;
    unsigned char *ch = calloc((size_t)E, 1);
    for (int e = 0; e < E; e++) {
        double a = 0, m = 0;
        for (int h = 0; h < H; h++) { a += (double)wr[(size_t)e * H + h] * x[h]; m += fabs((double)wr[(size_t)e * H + h] * x[h]); }
        l[e] = a; A[e] = m; sg[e] = 1.0 / (1.0 + exp(-a));
        ds[e] = sg[e] * (1 - sg[e]) * CB * H * U * m + 3 * U * sg[e];
    }
    for (int k = 0; k < K; k++) {
        int best = -1; double bv = 0;
        for (int e = 0; e < E; e++) if (!ch[e] && (best < 0 || sg[e] + br[e] > bv)) { best = e; bv = sg[e] + br[e]; }
        ch[best] = 1;
    }
    int ns = 0; double S = 0, dS = 0;
    for (int e = 0; e < E; e++) if (ch[e]) { sel[ns++] = e; S += sg[e]; dS += ds[e]; }
    for (int h = 0; h < H; h++) { y[h] = 0; bound[h] = 0; }
    double *acc = calloc((size_t)H, sizeof(double));
    for (int i = 0; i < K; i++) {
        int e = sel[i]; double wi = sg[e] / S * R, dwi = wi * (ds[e] / sg[e] + dS / S + CB * (K + 3) * U);
        const float *g13 = w13 + (size_t)e * 2 * F * H, *g2 = w2 + (size_t)e * H * F;
        for (int j = 0; j < 2 * F; j++) {
            double a = 0, m = 0;
            for (int h = 0; h < H; h++) { a += (double)g13[(size_t)j * H + h] * x[h]; m += fabs((double)g13[(size_t)j * H + h] * x[h]); }
            gu[j] = a; G[j] = m;
        }
        for (int j = 0; j < F; j++) {
            double g = gu[j], u = gu[F + j], dg = CB * H * U * G[j], du = CB * H * U * G[F + j];
            double cg = tanh(g / capg) * capg, cu = tanh(u / capu) * capu, sgm = 1.0 / (1.0 + exp(-g));
            double dcg = dg + 3 * U * fabs(cg), dcu = du + 3 * U * fabs(cu), dsg = sgm * (1 - sgm) * dg + 3 * U * sgm;
            act[j] = cg * sgm * cu;
            dact[j] = fabs(sgm * cu) * dcg + fabs(cg * cu) * dsg + fabs(cg * sgm) * dcu + 2 * U * fabs(act[j]);
        }
        for (int h = 0; h < H; h++) {
            double a = 0, m = 0, p = 0;
            for (int j = 0; j < F; j++) { a += (double)g2[(size_t)h * F + j] * act[j]; m += fabs((double)g2[(size_t)h * F + j] * act[j]); p += fabs((double)g2[(size_t)h * F + j]) * dact[j]; }
            down[h] = a; ddown[h] = CB * F * U * m + p;
            y[h] += wi * a; acc[h] += fabs(wi * a);
            bound[h] += fabs(wi) * ddown[h] + dwi * fabs(a);
        }
    }
    for (int h = 0; h < H; h++) bound[h] += CB * K * U * acc[h];
    free(l); free(ch); free(acc);
}
// The attention's composed first-order bound, both outputs, and its float64 value (the harness's own evaluation,
// checked against the truth), one call. Every quantity z carries its double value and dz, a bound on |computed - true|;
// the inputs and weights are exact (recorded float32). Steps, in the reference's order:
//   any linear, W x of length n:     dz = c n u sum |W x|
//   times the learned scale s:       dy = |s| dz + u |y|
//   L2 clip of y (length d):         ss = sum y^2, dss = c d u ss + 2 sum |y| dy; N = sqrt ss, rN = dss / (2 ss) + u;
//                                    f = thr / (N + 1e-6): df = f (rN + 3.5 u) (the add, 1/x, the product, thr
//                                      rounded to float32: u/2; f32(1e-6) is within u (N + 1e-6) of 1e-6);
//                                    m = min(f, 1): dm = 0 when f - df >= 1 (both sides clamp to exactly 1), else df
//                                      (min is 1-Lipschitz); out = y m: dout = m dy + |y| dm + u |out|
//   log scale (full layers):         A = max((pos + 1) / n_floor, 1), L = log A: dL = 0 when A = 1 (the clamp gives
//                                      an exact 1), else u + 2 u |L| (the quotient's rounding, log within 2 ulp);
//                                    tau = 1 + alpha L: dtau = |alpha| dL + 1.5 u |alpha L| + u tau (alpha rounded to
//                                      float32: u/2); out = z tau: dout = tau dz + |z| dtau + u |out|
//   K/V convolution, sliding:        cv = sum_j w_j lin[t-K+1+j]: dcv = c K u sum |w lin| + sum |w| dlin
//                    full:           tap_l = static_l + dyn_l (dyn a linear of length H): dtap = ddyn + u |tap|;
//                                    cv = sum_l lin[t-l] tap_l: dcv = c K u sum |lin tap| + sum (|tap| dlin + |lin| dtap)
//                    both:           z = (cv + lin) / D, D = sqrt(K + 1) or sqrt(2K + 1) rounded to float32:
//                                      dz = (dcv + dlin + u |cv + lin|) / D + 1.5 u |z|
//   K: the L2 clip above (no scale); V: the unweighted RMS norm of y (length d), r = (sum y^2 / d + eps)^-1/2:
//                                    dz_i = r dy_i + |z_i| r^2 sum_e |y_e| dy_e / d + c (d + 5) u |z_i|
//   relative bias (from xq):         the linear, the scale, the L2 clip at rel_clip, (full) tau; rl_e = sum_r rs_r P_re:
//                                      drl = c d_rel u sum |rs P| + sum |P| drs; the gather and the zero fill are exact
//   score:                           s = (q . k) sc, sc = hd^-0.5 rounded to float32:
//                                      ds = |sc| (c hd u sum |q k| + sum (|q| dk + |k| dq)) + 1.5 u |s|
//   logit:                           l = s + bias (+ the mask's exact 0): dl = ds + dbias + u |l|;
//                                    the sink logit is a linear (length H)
//   softmax over the n admitted keys and the sink, p_j = e_j / sum e: the shift by the max cancels exactly; e_j is
//                                    off relatively by r_j = u |l_j - m| + 2 u (the subtraction, exp within 2 ulp); the
//                                    sum of positive terms c n u; the normalization 2 u (1/sum, then a product):
//                                      dp_j = p_j (dl_j + r_j + sum_i p_i (dl_i + r_i) + c n u + 2 u)  (i over keys and
//                                      sink; dp_j/dl_i = p_j (delta_ij - p_i)); a masked key: p = 0 on both sides
//   attention output:                o = sum_j p_j v_j over the n_k admitted keys:
//                                      do = c n_k u sum |p v| + sum (dp_j |v_j| + p_j dv_j)
//   gate:                            g = sigmoid(G x): dg = g (1 - g) dGx + 3 u g (exp, the sum, the division);
//                                    a = o g: da = g do + |o| dg + u |a|
//   out0 = Wo a (length Q):          bound = c Q u sum |Wo a| + sum |Wo| da;     out1 = p: bound = dp
static void lin_d(const float *W, int n_out, int n_in, const float *x, double *y, double *dy) {
    for (int o = 0; o < n_out; o++) {
        double a = 0, m = 0;
        for (int i = 0; i < n_in; i++) { double p = (double)W[(size_t)o * n_in + i] * x[i]; a += p; m += fabs(p); }
        y[o] = a; dy[o] = CB * n_in * U * m;
    }
}
static void rows_d(const float *W, int n_out, int n_in, int T, const float *x, double *y, double *dy) {
    for (int t = 0; t < T; t++) lin_d(W, n_out, n_in, x + (size_t)t * n_in, y + (size_t)t * n_out, dy + (size_t)t * n_out);
}
// times the scale (if any), the L2 clip, times tau (if tau > 0), in place on (y, dy) of length d
static void clip_d(double *y, double *dy, int d, const float *scale, double thr, double tau, double dtau) {
    double ss = 0, dss = 0;
    for (int i = 0; i < d; i++) {
        if (scale) { y[i] *= scale[i]; dy[i] = fabs((double)scale[i]) * dy[i] + U * fabs(y[i]); }
        ss += y[i] * y[i]; dss += 2 * fabs(y[i]) * dy[i];
    }
    dss += CB * d * U * ss;
    double N = sqrt(ss), rN = (ss > 0 ? dss / (2 * ss) : 0) + U, f = thr / (N + 1e-6), df = f * (rN + 3.5 * U);
    double m = f < 1 ? f : 1, dm = f - df >= 1 ? 0 : df;
    for (int i = 0; i < d; i++) {
        double z = y[i] * m, dz = m * dy[i] + fabs(y[i]) * dm + U * fabs(z);
        if (tau > 0) { double z2 = z * tau; dz = tau * dz + fabs(z) * dtau + U * fabs(z2); z = z2; }
        y[i] = z; dy[i] = dz;
    }
}
// per position: per head the scale, the L2 clip at thr, (full layers) tau(pos); in place on [T][n][d]
static void headclip_d(const ink2_shape *s, int g, int T, int n, int d, const float *scale, double thr, const int *pos,
                       double *y, double *dy) {
    for (int t = 0; t < T; t++) {
        double tau = 0, dtau = 0;
        if (g) {
            double A = (double)(pos[t] + 1) / s->log_floor; if (A < 1) A = 1;
            double L = log(A), dL = A > 1 ? U + 2 * U * fabs(L) : 0;
            tau = 1 + cfd.alpha * L; dtau = fabs(cfd.alpha) * dL + 1.5 * U * fabs(cfd.alpha * L) + U * tau;
        }
        for (int h = 0; h < n; h++) clip_d(y + ((size_t)t * n + h) * d, dy + ((size_t)t * n + h) * d, d, scale + (size_t)h * d, thr, tau, dtau);
    }
}
// one layer's self_attn weights and sizes, for the double evaluation
typedef struct { int g, nh, nkv, hd, K, dr, ext, C, Q; const float *w[14]; } attn_wd;
enum { AQ, AQS, AK, AKC, AKD, AV, AVC, AVD, AR, ARS, ARP, ASK, AG, AO };
static int attn_wd_get(const ink2_shape *s, const st_file *W, int b, int g, attn_wd *A) {
    const char *nm[14] = { "q_proj.linear.weight", "q_proj.norm.scale", "k_proj.linear.weight",
        g ? "k_proj.sconv.static_weight" : "k_proj.sconv.conv1d.weight", g ? "k_proj.sconv.weight_proj.weight" : NULL,
        "v_proj.linear.weight", g ? "v_proj.sconv.static_weight" : "v_proj.sconv.conv1d.weight",
        g ? "v_proj.sconv.weight_proj.weight" : NULL, "relative_position_bias.linear.weight", "relative_position_bias.norm.scale",
        "relative_position_bias.proj", "sink_proj.weight", "o_proj.gate_proj.weight", "o_proj.linear.weight" };
    for (int i = 0; i < 14; i++) {
        char n[192]; A->w[i] = NULL;
        if (!nm[i]) continue;
        snprintf(n, sizeof n, "model.layers.%d.self_attn.%s", b, nm[i]);
        const st_t *t = st_get(W, n); if (!t) return -1;
        A->w[i] = (const float *)(const void *)t->data;
        if (i == ARP) A->ext = (int)t->shape[1];   // the bias's extent, from proj [d_rel][extent]
    }
    A->g = g; A->nh = g ? s->n_head : s->swa_n_head; A->nkv = g ? s->n_kv : s->swa_n_kv; A->hd = g ? s->head_dim : s->swa_head_dim;
    A->K = s->conv_k; A->dr = s->d_rel; A->C = A->nkv * A->hd; A->Q = A->nh * A->hd;
    return 0;
}
// the K or V convolution: lin, dl [T][C] in; x the dynamic input [T][H]
static void conv_d(const attn_wd *A, int kv, int H, int T, const double *lin, const double *dl, const float *x, double *z, double *dz) {
    int g = A->g, C = A->C, K = A->K, R = A->nkv * K, hd = A->hd;
    const float *cw = A->w[kv ? AVC : AKC];
    double *dyn = malloc(sizeof(double) * (size_t)T * 2 * R), *dd = dyn + (size_t)T * R;
    if (g) rows_d(A->w[kv ? AVD : AKD], R, H, T, x, dyn, dd);
    double D = sqrt((double)(g ? 2 * K + 1 : K + 1));   // the double value: its float32 rounding is in the bound
    for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
        double cv = 0, m = 0, p = 0;
        for (int l = 0; l < K && t - l >= 0; l++) {   // lag l: the sliding kernel's tap K-1-l
            double li = lin[(size_t)(t - l) * C + ch], dli = dl[(size_t)(t - l) * C + ch], tp, dtp;
            if (g) { size_t r = (size_t)t * R + (size_t)(ch / hd) * K + l; tp = (double)cw[(size_t)l * C + ch] + dyn[r]; dtp = dd[r] + U * fabs(tp); }
            else { tp = cw[(size_t)ch * K + (K - 1 - l)]; dtp = 0; }
            cv += li * tp; m += fabs(li * tp); p += fabs(tp) * dli + fabs(li) * dtp;
        }
        double sm = cv + lin[(size_t)t * C + ch];
        z[(size_t)t * C + ch] = sm / D;
        dz[(size_t)t * C + ch] = (CB * K * U * m + p + dl[(size_t)t * C + ch] + U * fabs(sm)) / D + 1.5 * U * fabs(sm / D);
    }
    free(dyn);
}
// K: the L2 clip; V: the unweighted RMS norm; in place on [T][n_kv][hd]
static void kvnorm_d(const attn_wd *A, int kv, int T, double *z, double *dz) {
    int hd = A->hd;
    for (int t = 0; t < T; t++) for (int h = 0; h < A->nkv; h++) {
        double *y = z + (size_t)t * A->C + h * hd, *dy = dz + (size_t)t * A->C + h * hd, ss = 0, sd = 0;
        if (!kv) { clip_d(y, dy, hd, NULL, cfd.qk, 0, 0); continue; }
        for (int e = 0; e < hd; e++) { ss += y[e] * y[e]; sd += fabs(y[e]) * dy[e]; }
        double r = 1.0 / sqrt(ss / hd + cfd.eps);
        for (int e = 0; e < hd; e++) { double v = y[e] * r; dy[e] = r * dy[e] + fabs(v) * r * r * sd / hd + CB * (hd + 5) * U * fabs(v); y[e] = v; }
    }
}
// the relative bias [nh][T][T] from rs [T][nh][d_rel] (after its norm) and the recorded distances rd [T][T]
static void bank_d(const attn_wd *A, int T, const double *rs, const double *drs, const int64_t *rd, double *bias, double *db) {
    int nh = A->nh, dr = A->dr, ext = A->ext;
    for (int h = 0; h < nh; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) {
        int64_t e = rd[(size_t)i * T + j]; size_t at = ((size_t)h * T + i) * T + j;
        bias[at] = 0; db[at] = 0;
        if (e < 0 || e >= ext) continue;
        double a = 0, am = 0, ap = 0; const double *r = rs + ((size_t)i * nh + h) * dr, *dr_ = drs + ((size_t)i * nh + h) * dr;
        for (int k = 0; k < dr; k++) { double P = A->w[ARP][(size_t)k * ext + e]; a += r[k] * P; am += fabs(r[k] * P); ap += fabs(P) * dr_[k]; }
        bias[at] = a; db[at] = CB * dr * U * am + ap;
    }
}
// the core: q [T][Q], k, v [T][C], bias [nh][T][T], sinks [T][nh] (each with its bound) -> probs y1, b1 [nh][T][T] and
// the attention output o, dout [T][Q]; mask [T][T] the recorded additive mask (below -1e30: the key is excluded)
static void core_d(const attn_wd *A, int T, const double *q, const double *dq, const double *k, const double *dk, const double *v,
                   const double *dv, const double *bias, const double *db, const double *sk, const double *dsk, const float *mask,
                   double *y1, double *b1, double *o, double *dout) {
    int nh = A->nh, hd = A->hd, C = A->C, Q = A->Q;
    double sc = pow((double)hd, -0.5), *l = malloc(sizeof(double) * 4 * ((size_t)T + 1)), *dl = l + T + 1, *pp = dl + T + 1, *dp = pp + T + 1;
    for (int h = 0; h < nh; h++) {
        int kh = h / (nh / A->nkv);
        for (int i = 0; i < T; i++) {
            const double *qi = q + (size_t)i * Q + h * hd, *dqi = dq + (size_t)i * Q + h * hd;
            double mx = sk[(size_t)i * nh + h], S = 0, mean = 0;
            int nk = 0;
            for (int j = 0; j < T; j++) {
                double mk = mask[(size_t)i * T + j];
                if (mk < -1e30) continue;
                const double *kj = k + (size_t)j * C + kh * hd, *dkj = dk + (size_t)j * C + kh * hd;
                double a = 0, am = 0, ap = 0;
                for (int e = 0; e < hd; e++) { a += qi[e] * kj[e]; am += fabs(qi[e] * kj[e]); ap += fabs(qi[e]) * dkj[e] + fabs(kj[e]) * dqi[e]; }
                double sv = a * sc, ds = sc * (CB * hd * U * am + ap) + 1.5 * U * fabs(sv);
                size_t at = ((size_t)h * T + i) * T + j;
                l[nk] = sv + bias[at] + mk; dl[nk] = ds + db[at] + U * fabs(l[nk]);
                if (l[nk] > mx) mx = l[nk];
                nk++;
            }
            l[nk] = sk[(size_t)i * nh + h]; dl[nk] = dsk[(size_t)i * nh + h];   // the sink, last
            for (int j = 0; j <= nk; j++) { pp[j] = exp(l[j] - mx); S += pp[j]; }
            for (int j = 0; j <= nk; j++) { pp[j] /= S; dp[j] = dl[j] + U * fabs(l[j] - mx) + 2 * U; mean += pp[j] * dp[j]; }
            // + 4 * 2^-149: GRADUAL UNDERFLOW. A probability below 2^-126 is a float32 subnormal (or 0), where a
            // rounding's error is absolute, up to half the subnormal spacing 2^-149, not relative; the exp, the shift and
            // the division each round once, with margin. Found by fixture v4: engaged clips give scores whose softmax
            // tails fall to 1e-48, and both torch and this engine flush them; a purely relative bound cannot hold there.
            for (int j = 0; j <= nk; j++) dp[j] = pp[j] * (dp[j] + mean + CB * (nk + 1) * U + 2 * U) + 4 * 0x1p-149;
            for (int j = 0, jj = 0; j < T; j++) {
                size_t at = ((size_t)h * T + i) * T + j;
                if (mask[(size_t)i * T + j] < -1e30) { y1[at] = 0; b1[at] = 0; } else { y1[at] = pp[jj]; b1[at] = dp[jj]; jj++; }
            }
            for (int e = 0; e < hd; e++) {
                double a = 0, am = 0, ap = 0;
                for (int j = 0, jj = 0; j < T; j++) {
                    if (mask[(size_t)i * T + j] < -1e30) continue;
                    double vv = v[(size_t)j * C + kh * hd + e], dvv = dv[(size_t)j * C + kh * hd + e];
                    a += pp[jj] * vv; am += fabs(pp[jj] * vv); ap += dp[jj] * fabs(vv) + pp[jj] * dvv; jj++;
                }
                o[(size_t)i * Q + h * hd + e] = a; dout[(size_t)i * Q + h * hd + e] = CB * nk * U * am + ap;
            }
        }
    }
    free(l);
}
// o_proj: o, dout [T][Q] (overwritten by the gated values), x [T][H] -> y, b [T][H]
static void oproj_d(const attn_wd *A, int H, int T, const float *x, double *o, double *dout, double *y, double *b) {
    int Q = A->Q; double *gl = malloc(sizeof(double) * 2 * (size_t)Q), *dgl = gl + Q;
    for (int t = 0; t < T; t++) {
        lin_d(A->w[AG], Q, H, x + (size_t)t * H, gl, dgl);
        double *ot = o + (size_t)t * Q, *dt = dout + (size_t)t * Q;
        for (int i = 0; i < Q; i++) {
            double gg = 1.0 / (1.0 + exp(-gl[i])), dg = gg * (1 - gg) * dgl[i] + 3 * U * gg, a = ot[i] * gg;
            dt[i] = gg * dt[i] + fabs(ot[i]) * dg + U * fabs(a); ot[i] = a;
        }
        for (int h = 0; h < H; h++) {
            double a = 0, am = 0, ap = 0;
            for (int i = 0; i < Q; i++) { double w = A->w[AO][(size_t)h * Q + i]; a += w * ot[i]; am += fabs(w * ot[i]); ap += fabs(w) * dt[i]; }
            y[(size_t)t * H + h] = a; b[(size_t)t * H + h] = CB * Q * U * am + ap;
        }
    }
    free(gl);
}
// the whole call: y0, b0 [T][H]; y1, b1 [nh][T][T]
static void bound_attn(const ink2_shape *s, const attn_wd *A, int T, const float *xq, const float *xk, const float *xv,
                       const int *pos, const float *mask, const int64_t *rd, double *y0, double *b0, double *y1, double *b1) {
    int H = s->hidden, nh = A->nh, C = A->C, Q = A->Q;
    size_t nq = (size_t)T * Q, nc = (size_t)T * C, nr = (size_t)T * nh * A->dr, nb = (size_t)nh * T * T, ns = (size_t)T * nh;
    double *m = malloc(sizeof(double) * 2 * (nq + 3 * nc + nr + nb + ns + nq));
    double *q = m, *dq = q + nq, *k = dq + nq, *dk = k + nc, *v = dk + nc, *dv = v + nc, *ln = dv + nc, *dln = ln + nc;
    double *rs = dln + nc, *drs = rs + nr, *bi = drs + nr, *dbi = bi + nb, *sk = dbi + nb, *dsk = sk + ns, *o = dsk + ns, *dout = o + nq;
    rows_d(A->w[AQ], Q, H, T, xq, q, dq); headclip_d(s, A->g, T, nh, A->hd, A->w[AQS], cfd.qk, pos, q, dq);
    rows_d(A->w[AK], C, H, T, xk, ln, dln); conv_d(A, 0, H, T, ln, dln, xk, k, dk); kvnorm_d(A, 0, T, k, dk);
    rows_d(A->w[AV], C, H, T, xv, ln, dln); conv_d(A, 1, H, T, ln, dln, xv, v, dv); kvnorm_d(A, 1, T, v, dv);
    rows_d(A->w[AR], nh * A->dr, H, T, xq, rs, drs); headclip_d(s, A->g, T, nh, A->dr, A->w[ARS], cfd.rel, pos, rs, drs);
    bank_d(A, T, rs, drs, rd, bi, dbi);
    rows_d(A->w[ASK], nh, H, T, xq, sk, dsk);
    core_d(A, T, q, dq, k, dk, v, dv, bi, dbi, sk, dsk, mask, y1, b1, o, dout);
    oproj_d(A, H, T, xq, o, dout, y0, b0);
    free(m);
}
// ---------------------------------------------------------------- the decoder layer's composed bound (line 3a)
// One layer application on its recorded inputs (x, the block states, the messages: exact), against the per-application
// float64 truth. Every quantity carries its double value and d, a first-order bound on |float32 computed - true|, as in
// bound_attn. Each piece's OWN rounding terms are those of bound_depth, bound_gate, bound_attn and bound_moe, written
// out again with exactly one more term each, the first-order propagation of an input's error through the piece:
//   a linear W x of length n:   dy = c n u sum |W x| + sum |W| dx
//   the unweighted norm z = y r, r = (sum y^2 / n + eps)^-1/2:   dz_i = r dy_i + |z_i| r^2 sum_e |y_e| dy_e / n
//                               + c (n + 5) u |z_i|  (its own rounding, e_n of bound_depth)
//   the weighted norm:          w z: |w| dz + u |w z|
//   the depth attention:        the normed states' propagated error enters the scores as H^-1/2 sum |q| dn and the
//                               weighted sum as sum p dn (bound_depth_dx below)
//   the message gate:           the propagated error of norm(x) enters the down projection as sum |Wd| dgh, and the
//                               mix as |1 - f| dx (gate_dx)
//   the attention:              each linear of xq, xk, xv (q, k, v, the dynamic conv taps, the relative bias, the
//                               sinks, the output gate) gains sum |W| dx; everything after is bound_attn's own chain
//                               (clip_d, conv_d's arithmetic, kvnorm_d, bank_d, core_d), which already carries input bounds
//   the experts:                the router logits and gate/up gain sum |W| dx; the SELECTION is not differentiable:
//                               the bound holds only where the chosen set cannot flip, checked per position as
//                               min_chosen (v - dv) > max_other (v + dv), v = sigmoid score + bias, dv its bound plus the
//                               add's u |v|. A position that fails this is NOT bounded and is counted as such.
//   a post convolution:         conv_d's sliding arithmetic on the layer's taps (exact weights)
//   a residual sum a + b:       da + db + u |a + b|
// First order throughout, as every composed bound of this file (no second-order terms).
static void depth_dx(const float *q, int nq, const double *sv, const double *sd, int nkv, int H, double *y, double *dy) {
    double *n = malloc(sizeof(double) * ((size_t)nkv * H * 2 + (size_t)nkv * 2)), *dn = n + (size_t)nkv * H, *sc = dn + (size_t)nkv * H, *p = sc + nkv;
    double e_n = CB * (H + 5) * U, scale = 1.0 / sqrt((double)H);
    for (int j = 0; j < nkv; j++) {
        const double *s = sv + (size_t)j * H, *d = sd + (size_t)j * H; double ss = 0, sdn = 0;
        for (int h = 0; h < H; h++) { ss += s[h] * s[h]; sdn += fabs(s[h]) * d[h]; }
        double r = 1.0 / sqrt(ss / H + 1e-6);
        for (int h = 0; h < H; h++) { double v = s[h] * r; n[(size_t)j * H + h] = v; dn[(size_t)j * H + h] = e_n * fabs(v) + r * d[h] + fabs(v) * r * r * sdn / H; }
    }
    for (int i = 0; i < nq; i++) {
        const float *qi = q + (size_t)i * H; double mx = -1e300, dsmax = 0, sum = 0;
        for (int j = 0; j < nkv; j++) {
            double a = 0, A = 0, P = 0;
            for (int h = 0; h < H; h++) { a += qi[h] * n[(size_t)j * H + h]; A += fabs(qi[h] * n[(size_t)j * H + h]); P += fabs((double)qi[h]) * dn[(size_t)j * H + h]; }
            sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
            double ds = CB * (H + 1) * U * A * scale + scale * P; if (ds > dsmax) dsmax = ds;
        }
        for (int j = 0; j < nkv; j++) { p[j] = exp(sc[j] - mx); sum += p[j]; }
        for (int j = 0; j < nkv; j++) p[j] /= sum;
        for (int h = 0; h < H; h++) {
            double v = 0, b = 0;
            for (int j = 0; j < nkv; j++) {
                double nn = n[(size_t)j * H + h], an = fabs(nn);
                v += p[j] * nn; b += p[j] * (2 * dsmax + CB * (nkv + 3) * U) * an + p[j] * dn[(size_t)j * H + h] + CB * (nkv + 1) * U * p[j] * an;
            }
            y[(size_t)i * H + h] = v; dy[(size_t)i * H + h] = b;
        }
    }
    free(n);
}
// the unweighted norm with input error, one vector (value z, bound dz), its own rounding included
static void unorm_dx(const double *x, const double *dx, int n, double eps, double *z, double *dz) {
    double ss = 0, sd = 0;
    for (int i = 0; i < n; i++) { ss += x[i] * x[i]; sd += fabs(x[i]) * dx[i]; }
    double r = 1.0 / sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) { double v = x[i] * r; z[i] = v; dz[i] = r * dx[i] + fabs(v) * r * r * sd / n + CB * (n + 5) * U * fabs(v); }
}
static void wnorm_dx(const double *x, const double *dx, int n, double eps, const float *wt, double *z, double *dz) {
    unorm_dx(x, dx, n, eps, z, dz);
    for (int i = 0; i < n; i++) { z[i] *= wt[i]; dz[i] = fabs((double)wt[i]) * dz[i] + U * fabs(z[i]); }
}
// the message gate, x with error dx, the message exact (bound_gate's terms plus the two propagation terms)
static void gate_dx(const double *x, const double *dx, const float *m, const float *wd, const float *wf, const float *wg,
                    int H, int r, double eps, double *y, double *dy) {
    double *gh = malloc(sizeof(double) * (4 * (size_t)H + 2 * (size_t)r)), *dg = gh + 2 * H, *d = dg + 2 * H, *dd = d + r;
    double e_n = CB * (H + 5) * U, *mm = calloc(2 * (size_t)H, sizeof(double));
    unorm_dx(x, dx, H, eps, gh, dg);
    for (int i = 0; i < H; i++) dg[i] -= CB * (H + 5) * U * fabs(gh[i]);   // keep the propagated part only: e_n is in dd
    for (int i = 0; i < H; i++) { mm[i] = m[i]; mm[H + i] = 0; }
    unorm_dx(mm, mm + H, H, eps, gh + H, dg + H);
    for (int i = 0; i < H; i++) dg[H + i] = 0;
    for (int k = 0; k < r; k++) {
        double a = 0, D = 0, P = 0;
        for (int i = 0; i < 2 * H; i++) { double w = wd[(size_t)k * 2 * H + i]; a += w * gh[i]; D += fabs(w * gh[i]); P += fabs(w) * dg[i]; }
        d[k] = a; dd[k] = (CB * 2 * H * U + e_n) * D + P;
    }
    for (int h = 0; h < H; h++) {
        double F = 0, G = 0, aF = 0, aG = 0, pF = 0, pG = 0;
        for (int k = 0; k < r; k++) {
            F += wf[(size_t)h * r + k] * d[k]; G += wg[(size_t)h * r + k] * d[k];
            aF += fabs(wf[(size_t)h * r + k] * d[k]); aG += fabs(wg[(size_t)h * r + k] * d[k]);
            pF += fabs(wf[(size_t)h * r + k]) * dd[k]; pG += fabs(wg[(size_t)h * r + k]) * dd[k];
        }
        double f = tanh(F), g = tanh(G);
        double df = CB * r * U * aF + pF + 2 * U * fabs(f), dgg = CB * r * U * aG + pG + 2 * U * fabs(g);
        y[h] = (1 - f) * x[h] + g * m[h];
        dy[h] = fabs(x[h]) * df + fabs((double)m[h]) * dgg + CB * 3 * U * (fabs((1 - f) * x[h]) + fabs(g * m[h])) + fabs(1 - f) * dx[h];
    }
    free(gh); free(mm);
}
static void rows_dx(const float *W, int n_out, int n_in, int T, const double *x, const double *dx, double *y, double *dy) {
    for (int t = 0; t < T; t++) for (int o = 0; o < n_out; o++) {
        double a = 0, m = 0, p = 0; const float *row = W + (size_t)o * n_in; const double *xt = x + (size_t)t * n_in, *dt = dx + (size_t)t * n_in;
        for (int i = 0; i < n_in; i++) { double v = (double)row[i] * xt[i]; a += v; m += fabs(v); p += fabs((double)row[i]) * dt[i]; }
        y[(size_t)t * n_out + o] = a; dy[(size_t)t * n_out + o] = CB * n_in * U * m + p;
    }
}
// conv_d with the dynamic input x carrying dx
static void conv_dx(const attn_wd *A, int kv, int H, int T, const double *lin, const double *dl, const double *x, const double *dxx,
                    double *z, double *dz) {
    int g = A->g, C = A->C, K = A->K, R = A->nkv * K, hd = A->hd;
    const float *cw = A->w[kv ? AVC : AKC];
    double *dyn = malloc(sizeof(double) * (size_t)T * 2 * R), *dd = dyn + (size_t)T * R;
    if (g) rows_dx(A->w[kv ? AVD : AKD], R, H, T, x, dxx, dyn, dd);
    double D = sqrt((double)(g ? 2 * K + 1 : K + 1));
    for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
        double cv = 0, m = 0, p = 0;
        for (int l = 0; l < K && t - l >= 0; l++) {
            double li = lin[(size_t)(t - l) * C + ch], dli = dl[(size_t)(t - l) * C + ch], tp, dtp;
            if (g) { size_t r = (size_t)t * R + (size_t)(ch / hd) * K + l; tp = (double)cw[(size_t)l * C + ch] + dyn[r]; dtp = dd[r] + U * fabs(tp); }
            else { tp = cw[(size_t)ch * K + (K - 1 - l)]; dtp = 0; }
            cv += li * tp; m += fabs(li * tp); p += fabs(tp) * dli + fabs(li) * dtp;
        }
        double sm = cv + lin[(size_t)t * C + ch];
        z[(size_t)t * C + ch] = sm / D;
        dz[(size_t)t * C + ch] = (CB * K * U * m + p + dl[(size_t)t * C + ch] + U * fabs(sm)) / D + 1.5 * U * fabs(sm / D);
    }
    free(dyn);
}
// a post convolution (taps cw [C][K], exact) on a, da [T][C]: conv_d's sliding arithmetic
static void postconv_dx(const float *cw, int C, int K, int T, const double *a, const double *da, double *z, double *dz) {
    double D = sqrt((double)(K + 1));
    for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
        double cv = 0, m = 0, p = 0;
        for (int l = 0; l < K && t - l >= 0; l++) {
            double li = a[(size_t)(t - l) * C + ch], tp = cw[(size_t)ch * K + (K - 1 - l)];
            cv += li * tp; m += fabs(li * tp); p += fabs(tp) * da[(size_t)(t - l) * C + ch];
        }
        double sm = cv + a[(size_t)t * C + ch];
        z[(size_t)t * C + ch] = sm / D;
        dz[(size_t)t * C + ch] = (CB * K * U * m + p + da[(size_t)t * C + ch] + U * fabs(sm)) / D + 1.5 * U * fabs(sm / D);
    }
}
// oproj_d with x carrying dx
static void oproj_dx(const attn_wd *A, int H, int T, const double *x, const double *dxx, double *o, double *dout, double *y, double *b) {
    int Q = A->Q; double *gl = malloc(sizeof(double) * 2 * (size_t)Q), *dgl = gl + Q;
    for (int t = 0; t < T; t++) {
        rows_dx(A->w[AG], Q, H, 1, x + (size_t)t * H, dxx + (size_t)t * H, gl, dgl);
        double *ot = o + (size_t)t * Q, *dt = dout + (size_t)t * Q;
        for (int i = 0; i < Q; i++) {
            double gg = 1.0 / (1.0 + exp(-gl[i])), dg = gg * (1 - gg) * dgl[i] + 3 * U * gg, a = ot[i] * gg;
            dt[i] = gg * dt[i] + fabs(ot[i]) * dg + U * fabs(a); ot[i] = a;
        }
        for (int h = 0; h < H; h++) {
            double a = 0, am = 0, ap = 0;
            for (int i = 0; i < Q; i++) { double w = A->w[AO][(size_t)h * Q + i]; a += w * ot[i]; am += fabs(w * ot[i]); ap += fabs(w) * dt[i]; }
            y[(size_t)t * H + h] = a; b[(size_t)t * H + h] = CB * Q * U * am + ap;
        }
    }
    free(gl);
}
// core_d with the softmax's first-order bound taken without the cancelling j = i terms (used by the composed layer;
// the attention line keeps core_d)
static void core_dx(const attn_wd *A, int T, const double *q, const double *dq, const double *k, const double *dk, const double *v,
                   const double *dv, const double *bias, const double *db, const double *sk, const double *dsk, const float *mask,
                   double *y1, double *b1, double *o, double *dout) {
    int nh = A->nh, hd = A->hd, C = A->C, Q = A->Q;
    double sc = pow((double)hd, -0.5), *l = malloc(sizeof(double) * 4 * ((size_t)T + 1)), *dl = l + T + 1, *pp = dl + T + 1, *dp = pp + T + 1;
    for (int h = 0; h < nh; h++) {
        int kh = h / (nh / A->nkv);
        for (int i = 0; i < T; i++) {
            const double *qi = q + (size_t)i * Q + h * hd, *dqi = dq + (size_t)i * Q + h * hd;
            double mx = sk[(size_t)i * nh + h], S = 0, mean = 0;
            int nk = 0;
            for (int j = 0; j < T; j++) {
                double mk = mask[(size_t)i * T + j];
                if (mk < -1e30) continue;
                const double *kj = k + (size_t)j * C + kh * hd, *dkj = dk + (size_t)j * C + kh * hd;
                double a = 0, am = 0, ap = 0;
                for (int e = 0; e < hd; e++) { a += qi[e] * kj[e]; am += fabs(qi[e] * kj[e]); ap += fabs(qi[e]) * dkj[e] + fabs(kj[e]) * dqi[e]; }
                double sv = a * sc, ds = sc * (CB * hd * U * am + ap) + 1.5 * U * fabs(sv);
                size_t at = ((size_t)h * T + i) * T + j;
                l[nk] = sv + bias[at] + mk; dl[nk] = ds + db[at] + U * fabs(l[nk]);
                if (l[nk] > mx) mx = l[nk];
                nk++;
            }
            l[nk] = sk[(size_t)i * nh + h]; dl[nk] = dsk[(size_t)i * nh + h];   // the sink, last
            for (int j = 0; j <= nk; j++) { pp[j] = exp(l[j] - mx); S += pp[j]; }
            for (int j = 0; j <= nk; j++) { pp[j] /= S; dp[j] = dl[j] + U * fabs(l[j] - mx) + 2 * U; mean += pp[j] * dp[j]; }
            // + 4 * 2^-149: GRADUAL UNDERFLOW. A probability below 2^-126 is a float32 subnormal (or 0), where a
            // rounding's error is absolute, up to half the subnormal spacing 2^-149, not relative; the exp, the shift and
            // the division each round once, with margin. Found by fixture v4: engaged clips give scores whose softmax
            // tails fall to 1e-48, and both torch and this engine flush them; a purely relative bound cannot hold there.
            // dp_j/dl_i = p_j (delta_ij - p_i), so |dp_j| <= p_j sum_{i != j} p_i (e_j + e_i) = p_j ((1 - p_j) e_j +
            // sum_{i != j} p_i e_i), e the logit bound plus the exp's own rounding: core_d's p_j (e_j + sum_i p_i e_i)
            // without the j = i terms that cancel exactly
            for (int j = 0; j <= nk; j++) {
                double others = mean - pp[j] * dp[j]; if (others < 0) others = 0;
                dp[j] = pp[j] * ((1 - pp[j]) * dp[j] + others + CB * (nk + 1) * U + 2 * U) + 4 * 0x1p-149;
            }
            for (int j = 0, jj = 0; j < T; j++) {
                size_t at = ((size_t)h * T + i) * T + j;
                if (mask[(size_t)i * T + j] < -1e30) { y1[at] = 0; b1[at] = 0; } else { y1[at] = pp[jj]; b1[at] = dp[jj]; jj++; }
            }
            for (int e = 0; e < hd; e++) {
                double a = 0, am = 0, ap = 0;
                for (int j = 0, jj = 0; j < T; j++) {
                    if (mask[(size_t)i * T + j] < -1e30) continue;
                    double vv = v[(size_t)j * C + kh * hd + e], dvv = dv[(size_t)j * C + kh * hd + e];
                    a += pp[jj] * vv; am += fabs(pp[jj] * vv); ap += dp[jj] * fabs(vv) + pp[jj] * dvv; jj++;
                }
                o[(size_t)i * Q + h * hd + e] = a; dout[(size_t)i * Q + h * hd + e] = CB * nk * U * am + ap;
            }
        }
    }
    free(l);
}
// bound_attn with input errors on xq, xk, xv; output 0 only
static void bound_attn_dx(const ink2_shape *s, const attn_wd *A, int T, const double *xq, const double *dxq, const double *xk,
                          const double *dxk, const double *xv, const double *dxv, const int *pos, const float *mask,
                          const int64_t *rd, double *y0, double *b0) {
    int H = s->hidden, nh = A->nh, C = A->C, Q = A->Q;
    size_t nq = (size_t)T * Q, nc = (size_t)T * C, nr = (size_t)T * nh * A->dr, nb = (size_t)nh * T * T, ns = (size_t)T * nh;
    double *m = malloc(sizeof(double) * 2 * (nq + 3 * nc + nr + 2 * nb + ns + nq));
    double *q = m, *dq = q + nq, *k = dq + nq, *dk = k + nc, *v = dk + nc, *dv = v + nc, *ln = dv + nc, *dln = ln + nc;
    double *rs = dln + nc, *drs = rs + nr, *bi = drs + nr, *dbi = bi + nb, *sk = dbi + nb, *dsk = sk + ns, *o = dsk + ns, *dout = o + nq;
    double *y1 = dout + nq, *b1 = y1 + nb;
    rows_dx(A->w[AQ], Q, H, T, xq, dxq, q, dq); headclip_d(s, A->g, T, nh, A->hd, A->w[AQS], cfd.qk, pos, q, dq);
    rows_dx(A->w[AK], C, H, T, xk, dxk, ln, dln); conv_dx(A, 0, H, T, ln, dln, xk, dxk, k, dk); kvnorm_d(A, 0, T, k, dk);
    rows_dx(A->w[AV], C, H, T, xv, dxv, ln, dln); conv_dx(A, 1, H, T, ln, dln, xv, dxv, v, dv); kvnorm_d(A, 1, T, v, dv);
    rows_dx(A->w[AR], nh * A->dr, H, T, xq, dxq, rs, drs); headclip_d(s, A->g, T, nh, A->dr, A->w[ARS], cfd.rel, pos, rs, drs);
    bank_d(A, T, rs, drs, rd, bi, dbi);
    rows_dx(A->w[ASK], nh, H, T, xq, dxq, sk, dsk);
    core_dx(A, T, q, dq, k, dk, v, dv, bi, dbi, sk, dsk, mask, y1, b1, o, dout);
    oproj_dx(A, H, T, xq, dxq, o, dout, y0, b0);
    free(m);
}
// bound_moe with an input error dx, one position; returns 1 if the selection is stable under the bounds, else 0
static int moe_dx(const double *x, const double *dx, const float *wr, const float *br, const float *w13, const float *w2, int H, int E,
                  int K, int F, double R, double capg, double capu, double *y, double *bound, int *sel_out, double *v_out,
                  double *dv_out) {
    double *l = malloc(sizeof(double) * (size_t)(4 * E + 6 * F + 2 * H)), *A = l + E, *sg = A + E, *ds = sg + E;
    double *gu = ds + E, *G = gu + 2 * F, *act = G + 2 * F, *dact = act + F, *down = dact + F, *ddown = down + H;
    unsigned char *ch = calloc((size_t)E, 1); int *sel = malloc(sizeof(int) * (size_t)K);
    for (int e = 0; e < E; e++) {
        double a = 0, m = 0, p = 0;
        for (int h = 0; h < H; h++) { double v = (double)wr[(size_t)e * H + h] * x[h]; a += v; m += fabs(v); p += fabs((double)wr[(size_t)e * H + h]) * dx[h]; }
        l[e] = a; A[e] = m; sg[e] = 1.0 / (1.0 + exp(-a));
        ds[e] = sg[e] * (1 - sg[e]) * (CB * H * U * m + p) + 3 * U * sg[e];
    }
    for (int k = 0; k < K; k++) {
        int best = -1; double bv = 0;
        for (int e = 0; e < E; e++) if (!ch[e] && (best < 0 || sg[e] + br[e] > bv)) { best = e; bv = sg[e] + br[e]; }
        ch[best] = 1;
    }
    double lo = 1e300, hi = -1e300;
    for (int e = 0; e < E; e++) {
        double v = sg[e] + br[e], dv = ds[e] + U * fabs(v);
        if (ch[e]) { if (v - dv < lo) lo = v - dv; } else if (v + dv > hi) hi = v + dv;
        if (v_out) { v_out[e] = v; dv_out[e] = dv; }
    }
    int stable = E == K || lo > hi;
    int ns = 0; double S = 0, dS = 0;
    for (int e = 0; e < E; e++) if (ch[e]) { sel[ns++] = e; S += sg[e]; dS += ds[e]; }
    if (sel_out) for (int i = 0; i < K; i++) sel_out[i] = sel[i];
    double *acc = calloc((size_t)H, sizeof(double));
    for (int h = 0; h < H; h++) { y[h] = 0; bound[h] = 0; }
    for (int i = 0; i < K; i++) {
        int e = sel[i]; double wi = sg[e] / S * R, dwi = wi * (ds[e] / sg[e] + dS / S + CB * (K + 3) * U);
        const float *g13 = w13 + (size_t)e * 2 * F * H, *g2 = w2 + (size_t)e * H * F;
        double *dgu = malloc(sizeof(double) * 2 * (size_t)F);
        for (int j = 0; j < 2 * F; j++) {
            double a = 0, m = 0, p = 0;
            for (int h = 0; h < H; h++) { double v = (double)g13[(size_t)j * H + h] * x[h]; a += v; m += fabs(v); p += fabs((double)g13[(size_t)j * H + h]) * dx[h]; }
            gu[j] = a; G[j] = m; dgu[j] = CB * H * U * m + p;
        }
        for (int j = 0; j < F; j++) {
            double g = gu[j], u = gu[F + j], dg = dgu[j], du = dgu[F + j];
            double cg = tanh(g / capg) * capg, cu = tanh(u / capu) * capu, sgm = 1.0 / (1.0 + exp(-g));
            double dcg = dg + 3 * U * fabs(cg), dcu = du + 3 * U * fabs(cu), dsg = sgm * (1 - sgm) * dg + 3 * U * sgm;
            act[j] = cg * sgm * cu;
            dact[j] = fabs(sgm * cu) * dcg + fabs(cg * cu) * dsg + fabs(cg * sgm) * dcu + 2 * U * fabs(act[j]);
        }
        free(dgu);
        for (int h = 0; h < H; h++) {
            double a = 0, m = 0, p = 0;
            for (int j = 0; j < F; j++) { a += (double)g2[(size_t)h * F + j] * act[j]; m += fabs((double)g2[(size_t)h * F + j] * act[j]); p += fabs((double)g2[(size_t)h * F + j]) * dact[j]; }
            down[h] = a; ddown[h] = CB * F * U * m + p;
            y[h] += wi * a; acc[h] += fabs(wi * a);
            bound[h] += fabs(wi) * ddown[h] + dwi * fabs(a);
        }
    }
    for (int h = 0; h < H; h++) bound[h] += CB * K * U * acc[h];
    free(l); free(ch); free(acc); free(sel);
    return stable;
}
static const float *wptr(const st_file *W, const char *fmt, int b) {
    char n[192]; snprintf(n, sizeof n, fmt, b); const st_t *t = st_get(W, n);
    return t ? (const float *)(const void *)t->data : NULL;
}
// The composed layer: the same order as ink2_decoder_layer (inkling2.h). x [T][H], bs [T][nb][H], msg [T][4][H] or
// NULL, exact; y, dy [T][H]. Returns the number of positions whose expert selection is not stable (0: bounded).
// sel [T][K], v, dv [T][E]: the double chain's expert selection and its scores (sigmoid + bias) with their bounds
static int bound_layer(const ink2_shape *s, const st_file *W, int b, int T, const float *x, const float *bs, int nb,
                       const float *msg, const int *pos, const float *mask, const int64_t *rd, double *y, double *dy,
                       int *sel, double *v, double *dv) {
    int H = s->hidden, g = s->block_global[b], bnd = b % s->attn_res_block == 0, unstable = 0;
    size_t TH = (size_t)T * H;
    double *M = calloc(TH * 24 + (size_t)(nb + 2) * H * 2 + (size_t)6 * H, sizeof(double));
    double *rv = M, *rdv = rv + 3 * TH, *qv = rdv + 3 * TH, *qd = qv + 3 * TH, *av = qd + 3 * TH, *ad = av + TH, *cv = ad + TH, *cd = cv + TH;
    double *res = cd + TH, *dres = res + TH, *mi = dres + TH, *dmi = mi + TH, *nv = dmi + TH, *nd = nv + TH, *ev = nd + TH, *ed = ev + TH;
    double *st = ed + TH, *sd = st + (size_t)(nb + 2) * H, *tmp = sd + (size_t)(nb + 2) * H;
    int nr = g ? 3 : 1;
    const char *role[4] = { "q", "k", "v", "mlp" };
    char f1[160];
    // 1. the reads
    for (int t = 0; t < T; t++) {
        if (b > 0) {
            for (int j = 0; j < nb; j++) for (int h = 0; h < H; h++) st[(size_t)j * H + h] = bs[((size_t)t * nb + j) * H + h];
            for (int h = 0; h < H; h++) st[(size_t)nb * H + h] = x[(size_t)t * H + h];
            for (size_t i = 0; i < (size_t)(nb + 1) * H; i++) sd[i] = 0;
            snprintf(f1, sizeof f1, "model.layers.%d.attention_input.query", b);
            const st_t *qt = st_get(W, f1);
            depth_dx((const float *)(const void *)qt->data, nr, st, sd, nb + 1, H, tmp, tmp + 3 * H);
            for (int r = 0; r < nr; r++) for (int h = 0; h < H; h++) {
                rv[(size_t)r * TH + (size_t)t * H + h] = tmp[(size_t)r * H + h]; rdv[(size_t)r * TH + (size_t)t * H + h] = tmp[3 * H + (size_t)r * H + h];
            }
        } else for (int r = 0; r < nr; r++) for (int h = 0; h < H; h++) { rv[(size_t)r * TH + (size_t)t * H + h] = x[(size_t)t * H + h]; rdv[(size_t)r * TH + (size_t)t * H + h] = 0; }
    }
    // 2. the gates and the input norms
    for (int r = 0; r < 3; r++) {
        int src = g ? r : 0;
        for (int t = 0; t < T; t++) {
            const double *xv = rv + (size_t)src * TH + (size_t)t * H, *xd = rdv + (size_t)src * TH + (size_t)t * H;
            double *gv = tmp, *gd = tmp + H;
            if (msg) {
                float mr[4096]; for (int h = 0; h < H; h++) mr[h] = msg[((size_t)t * 4 + r) * H + h];
                char a1[160], a2[160], a3[160];
                snprintf(a1, sizeof a1, "model.layers.%%d.%s_message_gate.down_proj.weight", role[r]);
                snprintf(a2, sizeof a2, "model.layers.%%d.%s_message_gate.forget_gate_proj.weight", role[r]);
                snprintf(a3, sizeof a3, "model.layers.%%d.%s_message_gate.message_gate_proj.weight", role[r]);
                gate_dx(xv, xd, mr, wptr(W, a1, b), wptr(W, a2, b), wptr(W, a3, b), H, s->loop_gate_rank, cfd.eps, gv, gd);
            } else { memcpy(gv, xv, sizeof(double) * H); memcpy(gd, xd, sizeof(double) * H); }
            char nm[160];
            if (g) snprintf(nm, sizeof nm, "model.layers.%%d.%s_input_layernorm.weight", role[r]); else snprintf(nm, sizeof nm, "model.layers.%%d.input_layernorm.weight");
            wnorm_dx(gv, gd, H, cfd.eps, wptr(W, nm, b), qv + (size_t)r * TH + (size_t)t * H, qd + (size_t)r * TH + (size_t)t * H);
        }
    }
    // 3. the attention and the post-attention convolution
    attn_wd A; attn_wd_get(s, W, b, g, &A);
    bound_attn_dx(s, &A, T, qv, qd, qv + TH, qd + TH, qv + 2 * TH, qd + 2 * TH, pos, mask, rd, av, ad);
    postconv_dx(wptr(W, "model.layers.%d.post_attention_convolution.conv1d.weight", b), H, s->conv_k, T, av, ad, cv, cd);
    // 4. the residual and the MLP's block states
    int nb2 = bnd ? nb + 1 : nb;
    for (size_t i = 0; i < TH; i++) {
        if (bnd) { res[i] = cv[i]; dres[i] = cd[i]; }
        else { res[i] = x[i] + cv[i]; dres[i] = cd[i] + U * fabs(res[i]); }
    }
    // 5. mlp_input, the mlp gate, the norm, the experts, the post-MLP convolution, the sum
    const st_t *mq; { char n[160]; snprintf(n, sizeof n, "model.layers.%d.mlp_input.query", b); mq = st_get(W, n); }
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < nb; j++) for (int h = 0; h < H; h++) st[(size_t)j * H + h] = bs[((size_t)t * nb + j) * H + h];
        if (bnd) for (int h = 0; h < H; h++) st[(size_t)nb * H + h] = x[(size_t)t * H + h];
        for (size_t i = 0; i < (size_t)nb2 * H; i++) sd[i] = 0;
        for (int h = 0; h < H; h++) { st[(size_t)nb2 * H + h] = res[(size_t)t * H + h]; sd[(size_t)nb2 * H + h] = dres[(size_t)t * H + h]; }
        depth_dx((const float *)(const void *)mq->data, 1, st, sd, nb2 + 1, H, mi + (size_t)t * H, dmi + (size_t)t * H);
        double *gv = mi + (size_t)t * H, *gd = dmi + (size_t)t * H;
        if (msg) {
            float mr[4096]; for (int h = 0; h < H; h++) mr[h] = msg[((size_t)t * 4 + 3) * H + h];
            gate_dx(gv, gd, mr, wptr(W, "model.layers.%d.mlp_message_gate.down_proj.weight", b),
                    wptr(W, "model.layers.%d.mlp_message_gate.forget_gate_proj.weight", b),
                    wptr(W, "model.layers.%d.mlp_message_gate.message_gate_proj.weight", b), H, s->loop_gate_rank, cfd.eps, tmp, tmp + H);
            memcpy(gv, tmp, sizeof(double) * H); memcpy(gd, tmp + H, sizeof(double) * H);
        }
        wnorm_dx(gv, gd, H, cfd.eps, wptr(W, "model.layers.%d.post_attention_layernorm.weight", b), nv + (size_t)t * H, nd + (size_t)t * H);
        unstable += !moe_dx(nv + (size_t)t * H, nd + (size_t)t * H, wptr(W, "model.layers.%d.mlp.gate.weight", b),
                            wptr(W, "model.layers.%d.mlp.gate.e_score_correction_bias", b), wptr(W, "model.layers.%d.mlp.experts.gate_up_proj", b),
                            wptr(W, "model.layers.%d.mlp.experts.down_proj", b), H, s->n_expert, s->n_used, s->ff, s->route_scale,
                            s->gate_cap, s->up_cap, ev + (size_t)t * H, ed + (size_t)t * H, sel + (size_t)t * s->n_used,
                            v + (size_t)t * s->n_expert, dv + (size_t)t * s->n_expert);
    }
    postconv_dx(wptr(W, "model.layers.%d.post_mlp_convolution.conv1d.weight", b), H, s->conv_k, T, ev, ed, cv, cd);
    for (size_t i = 0; i < TH; i++) { y[i] = res[i] + cv[i]; dy[i] = dres[i] + cd[i] + U * fabs(y[i]); }
    free(M);
    return unstable;
}
// [n][T][d] -> [T][n][d] (the reference's per-head outputs to the piece's layout)
static void tr_ntd(const float *src, int n, int T, int d, float *dst) {
    for (int h = 0; h < n; h++) for (int t = 0; t < T; t++) memcpy(dst + ((size_t)t * n + h) * d, src + ((size_t)h * T + t) * d, sizeof(float) * d);
}
static void f2d(const float *x, size_t n, double *y, double *dy) { for (size_t i = 0; i < n; i++) { y[i] = x[i]; dy[i] = 0; } }
// got (float32) and torch (float32, may be NULL) against truth (float64) under bound; returns the fraction used
// The OUTPUTS DIGEST: FNV-1a 64 over the bytes of every array compared here (our pieces' outputs, the perturbed runs,
// and torch's constant dumps), in call order. Two builds of inkling2.c with the same digest gave the same bits on
// every piece output: the Runner-against-Runner gate (a strict build is the same on every machine and compiler).
// The lines added with the forward (decoder_layer, gate A, gate B) feed a SECOND digest, printed on its own line, so
// the first stays the digest of the 23 piece lines.
static uint64_t g_digest = 1469598103934665603ULL, g_digest2 = 1469598103934665603ULL, *g_dig = &g_digest;
static void digest(const void *p, size_t nbytes) {
    const unsigned char *b = p;
    for (size_t i = 0; i < nbytes; i++) { *g_dig ^= b[i]; *g_dig *= 1099511628211ULL; }
}
static double frac_used(const float *got, const double *truth, const double *bound, size_t n) {
    digest(got, n * sizeof *got);
    double w = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - truth[i]), b = bound[i] > 0 ? bound[i] : 1e-300;
        if (got[i] != got[i]) return INFINITY;   // a NaN (by compare: this file is strict)
        if (e / b > w) w = e / b;
    }
    return w;
}
static void bstat_add(bstat *b, ink2_status st, double frac, double tfrac, double pert) {
    b->calls++;
    if (st == INK2_NOT_IMPLEMENTED) { b->red++; return; }
    if (st != INK2_OK) { b->fail++; return; }
    if (frac > b->frac) b->frac = frac;
    if (tfrac > b->torch_frac) b->torch_frac = tfrac;
    if (pert >= 0 && (b->pert_mult < 0 || pert < b->pert_mult)) b->pert_mult = pert;
    if (frac > 1.0) b->nob++;
}

// the attention's sub-steps, one at a time: ink2_attention_step (inkling2.h)
#define N_ASTEP 13
static const char *astep[N_ASTEP] = { "q_proj", "q_proj.norm", "k_proj", "k_proj.sconv", "k_proj.norm", "v_proj", "v_proj.sconv",
    "v_proj.norm", "relative_position_bias", "relative_position_bias.norm", "sink_proj", "core", "o_proj" };
static long clip_engaged = 0, clip_vectors = 0;
static int moe_sel_diff = 0, moe_tie_done = 0, moe_tie_ok = 0;
static double moe_harness_rel = 0, moe_tie_frac = 0, attn_harness_rel[2] = {0, 0}, attn_frac[2] = {0, 0}, astep_harness_rel[N_ASTEP];

// ---------------------------------------------------------------- expert selection: ours against the truth's
// The truth's selection at one position, in double from the recorded router input, as bound_moe: v_e = sigmoid(W x)_e
// + bias_e, the n_used largest (ties lowest index first); dv_e the score's own error bound in float32 (bound_moe's
// ds_e = s (1 - s) c H u A_e + 3 u s, plus the bias add's u |v_e|). brd: the bias in double (the planted case edits it).
static void select_d(const float *x, const float *wr, const double *brd, int H, int E, int K, int *sel, double *v, double *dv) {
    unsigned char ch[256] = {0};
    for (int e = 0; e < E; e++) {
        double a = 0, m = 0;
        for (int h = 0; h < H; h++) { double p = (double)wr[(size_t)e * H + h] * x[h]; a += p; m += fabs(p); }
        double sg = 1.0 / (1.0 + exp(-a));
        v[e] = sg + brd[e]; dv[e] = sg * (1 - sg) * CB * H * U * m + 3 * U * sg + U * fabs(v[e]);
    }
    for (int k = 0; k < K; k++) {
        int best = -1;
        for (int e = 0; e < E; e++) if (!ch[e] && (best < 0 || v[e] > v[best])) best = e;
        ch[best] = 1;
    }
    for (int e = 0, k = 0; e < E; e++) if (ch[e]) sel[k++] = e;
}
// ours and the truth's selection at one position (each [K], ascending): 0 the same; 1 different at a NEAR-TIE (every
// expert in one set and not the other has a score within its bound of the truth's cut: either side is a correct
// choice for an engine within the bound); 2 different OUTSIDE the bound (a wrong selection). gap: the truth's last
// chosen score minus its first unchosen; gapb: the two scores' bounds summed (gap < gapb: a near-tie).
static int sel_classify(const int *ours, const int *tr, const double *v, const double *dv, int E, int K, double *gap, double *gapb) {
    unsigned char in_t[256] = {0}, in_o[256] = {0};
    for (int k = 0; k < K; k++) { in_t[tr[k]] = 1; in_o[ours[k]] = 1; }
    int last = -1, first = -1; double lo = 1e300, hi = -1e300;
    for (int e = 0; e < E; e++) {
        if (in_t[e]) { if (last < 0 || v[e] < v[last]) last = e; if (v[e] - dv[e] < lo) lo = v[e] - dv[e]; }
        else { if (first < 0 || v[e] > v[first]) first = e; if (v[e] + dv[e] > hi) hi = v[e] + dv[e]; }
    }
    *gap = first < 0 ? INFINITY : v[last] - v[first]; *gapb = first < 0 ? 0 : dv[last] + dv[first];
    int same = 1, near = 1;
    for (int e = 0; e < E; e++) {
        if (in_o[e] == in_t[e]) continue;
        same = 0;
        if (in_o[e] && !(v[e] + dv[e] >= lo)) near = 0;   // we chose it, the truth did not: it must reach the cut
        if (in_t[e] && !(v[e] - dv[e] <= hi)) near = 0;   // the truth chose it, we did not: it must fall to the cut
    }
    return same ? 0 : near ? 1 : 2;
}
typedef struct { int pos, near_differ, hard_differ, near_ties, file_differ; double min_gap, near_gap_max; } selstat;
static void selstat_add(selstat *S, int cls, double gap, double gapb) {
    S->pos++;
    if (gap < S->min_gap) S->min_gap = gap;
    if (gap < gapb) S->near_ties++;
    if (cls == 1) { S->near_differ++; if (gap > S->near_gap_max) S->near_gap_max = gap; }
    if (cls == 2) S->hard_differ++;
}
static selstat SEL_P = {0, 0, 0, 0, 0, INFINITY, 0}, SEL_L = {0, 0, 0, 0, 0, INFINITY, 0};
static int plant_done = 0, plant_cls[2] = {-1, -1};
static double plant_gap[2], plant_gapb[2];

// ---------------------------------------------------------------- the coverage registry (line 4z)
// Every check of a recorded module call marks "<module>#call<k>" with the line that made it.
#define COV_MAX 4096
static char cov_key[COV_MAX][256]; static const char *cov_line[COV_MAX]; static int cov_n = 0;
static void cov_mark(const char *line, const char *key) {
    for (int i = 0; i < cov_n; i++) if (!strcmp(cov_key[i], key)) return;
    if (cov_n < COV_MAX) { snprintf(cov_key[cov_n], sizeof cov_key[0], "%s", key); cov_line[cov_n++] = line; }
}
static const char *cov_find(const char *key) {
    for (int i = 0; i < cov_n; i++) if (!strcmp(cov_key[i], key)) return cov_line[i];
    return NULL;
}
static void cov_markf(const char *line, const char *fmt, int b, int k) { char key[160]; snprintf(key, sizeof key, fmt, b, k); cov_mark(line, key); }

// ---------------------------------------------------------------- single-step lines (4a..4m)
// Each operation on its recorded inputs (exact) against the float64 truth of that call, under its one-step bound;
// torch's float32 dump under the same bound; a perturbation (one weight or input of the step + 0.25) must leave it.
//   linear                y = W x (length n):                 c n u sum |W x|
//   rms_norm (weighted)   z = w (x r), r = (mean x^2 + eps)^-1/2: |w| c (n + 5) u |x r| + u |z|
//   unweighted_norm       z = x r:                             c (n + 5) u |z|
//   short_conv            z = (sum_j w_j x_j + x) / sqrt(K + 1): (c K u sum |w x| + u |sum + x|) / D + 1.5 u |z| (postconv_dx)
//   embed                 the row (exact), then rms_norm
//   l2_clip               clip_d (no scale, no tau)
//   log_scale             z = y tau(pos): |y| dtau + u |z| (headclip_d's dtau)
//   router                logits: linear; weights w_i = s_i / S * R: w_i (ds_i / s_i + sum ds / S + c (K + 3) u),
//                         ds = s (1 - s) dl + 3 u s (bound_moe); the indices by the selection classifier
//   score_fn              s = sigmoid(l): 3 u s
//   experts (given        bound_moe from gate/up on, the weights exact: sum_i |w_i| d(down)_i + c K u sum_i |w_i down_i|
//     indices, weights)
//   act_fn                bound_moe's d(act) with exact gate and up
//   softcap               y = c tanh(x / c): u |x| + 3 u |y| (the quotient's rounding through tanh' <= 1, tanh 2 ulp, the product)
static bstat SL[13];
enum { L_LIN, L_WN, L_UN, L_CONV, L_EMB, L_L2, L_LS, L_RT, L_SF, L_EX, L_ACT, L_CAP, L_RES };
static const char *sl_name[13] = { "linear", "rms_norm", "unweighted_norm", "post_convolution", "embed_tokens", "l2_clip",
    "log_scale", "router", "score_fn", "experts", "act_fn", "softcap", "residual" };
static const char *sl_what[13] = {
    "every nn.Linear call (attention projections, sink, output gate, dynamic conv taps, message gate projections, lm_head alone)",
    "input_layernorm, q/k/v_input_layernorm, post_attention_layernorm, model.norm, embed_tokens.embed_norm",
    "the norms of attention_input, mlp_input, final_input, message_input and the message gates",
    "post_attention_convolution, post_mlp_convolution", "model.embed_tokens: the row, then its norm",
    "q_proj.norm.l2_clip, relative_position_bias.norm.l2_clip", "q_proj.norm.log_scale, relative_position_bias.norm.log_scale",
    "mlp.gate: logits, weights; its indices in the expert_selection line", "mlp.gate.score_fn",
    "mlp.experts on the recorded indices and weights", "mlp.experts.act_fn", "act_fn.gate_softcap, act_fn.up_softcap, logit_softcap alone",
    "the decoder layer's output = residual + post-MLP conv, rebuilt from the recorded submodule outputs (elementwise)" };
static int res_bits_eq = 0, res_bits_n = 0, wire_n = 0, wire_bad = 0;
static int cov_calls = 0, cov_single = 0, cov_composite = 0, cov_unnamed = 0, cov_uncovered = 0, cov_nlines = 0, cov_per_line[64];
static const char *cov_lines[64]; static char cov_first_unnamed[160] = "", cov_first_unc[160] = "";
static uint64_t g_digest3 = 1469598103934665603ULL;
static char wire_first[160] = "";
// the experts on given indices and weights (exact), one row: bound_moe from gate/up on
static void experts_d(const float *x, const int64_t *idx, const float *wts, const float *w13, const float *w2, int H, int K, int F,
                      double capg, double capu, double *y, double *bound) {
    double *gu = malloc(sizeof(double) * (size_t)(4 * F + 3 * H)), *G = gu + 2 * F, *act = G + 2 * F, *dact = act + F, *acc = dact + F;
    for (int h = 0; h < H; h++) { y[h] = 0; bound[h] = 0; acc[h] = 0; }
    for (int i = 0; i < K; i++) {
        int e = (int)idx[i]; double wi = wts[i];
        const float *g13 = w13 + (size_t)e * 2 * F * H, *g2 = w2 + (size_t)e * H * F;
        for (int j = 0; j < 2 * F; j++) {
            double a = 0, m = 0;
            for (int h = 0; h < H; h++) { double p = (double)g13[(size_t)j * H + h] * x[h]; a += p; m += fabs(p); }
            gu[j] = a; G[j] = m;
        }
        for (int j = 0; j < F; j++) {
            double g = gu[j], u = gu[F + j], dg = CB * H * U * G[j], du = CB * H * U * G[F + j];
            double cg = tanh(g / capg) * capg, cu = tanh(u / capu) * capu, sgm = 1.0 / (1.0 + exp(-g));
            double dcg = dg + 3 * U * fabs(cg), dcu = du + 3 * U * fabs(cu), dsg = sgm * (1 - sgm) * dg + 3 * U * sgm;
            act[j] = cg * sgm * cu;
            dact[j] = fabs(sgm * cu) * dcg + fabs(cg * cu) * dsg + fabs(cg * sgm) * dcu + 2 * U * fabs(act[j]);
        }
        for (int h = 0; h < H; h++) {
            double a = 0, m = 0, p = 0;
            for (int j = 0; j < F; j++) { double ww = g2[(size_t)h * F + j]; a += ww * act[j]; m += fabs(ww * act[j]); p += fabs(ww) * dact[j]; }
            y[h] += wi * a; acc[h] += fabs(wi * a); bound[h] += fabs(wi) * (CB * F * U * m + p);
        }
    }
    for (int h = 0; h < H; h++) bound[h] += CB * K * U * acc[h];
    free(gu);
}
static const double *tf64(const st_file *f, const char *key, size_t n) {
    const st_t *t = st_get(f, key); return t && t->dtype == ST_F64 && st_count(t) == n ? (const double *)(const void *)t->data : NULL;
}
static const float *tf32(const st_file *f, const char *key, size_t n) {
    const st_t *t = st_get(f, key); return t && t->dtype == ST_F32 && st_count(t) == n ? (const float *)t->data : NULL;
}
// One recorded call of module mod (#call k) through op, line li: ours, torch and the perturbed run against the truth
static void run_step(const ink2_shape *s, const ink2_weights *w, st_file *W, const st_file *In, const st_file *D, const st_file *TR,
                     const char *mod, int k, const char *op, int li) {
    char key[256], tk[256], dk[256], wn[192];
    int H = s->hidden, E = s->n_expert, K = s->n_used, F = s->ff;
    snprintf(key, sizeof key, "%s#call%d#in0", mod, k); const st_t *i0 = st_get(In, key);
    snprintf(key, sizeof key, "%s#call%d#in1", mod, k); const st_t *i1 = st_get(In, key);
    snprintf(key, sizeof key, "%s#call%d#in2", mod, k); const st_t *i2 = st_get(In, key);
    if (!strcmp(mod, "lm_head") || !strcmp(mod, "logit_softcap")) snprintf(tk, sizeof tk, "%s.alone#call%d", mod, k);
    else snprintf(tk, sizeof tk, "%s#call%d", mod, k);
    snprintf(dk, sizeof dk, "%s#call%d", mod, k);
    if (!i0) { bstat_add(&SL[li], INK2_BAD_INPUT, 0, 0, -1); return; }
    size_t n0 = st_count(i0), no = 0;
    const float *x = i0->dtype == ST_F32 ? (const float *)i0->data : NULL;
    int n = 0, d = 0, *ids = NULL;
    float *wt = NULL;   // the weight the perturbation moves (NULL: the input's first element)
    double *bb = NULL;
    // sizes, weights and the bound
    if (!strcmp(op, "linear")) {
        snprintf(wn, sizeof wn, "%s.weight", mod); const st_t *t = st_get(W, wn);
        if (!t || t->n_dims != 2 || !x) { bstat_add(&SL[li], INK2_NO_TENSOR, 0, 0, -1); return; }
        int no_ = (int)t->shape[0], ni = (int)t->shape[1]; n = (int)(n0 / ni); no = (size_t)n * no_;
        bb = malloc(sizeof(double) * no); wt = (float *)(void *)t->data;
        for (int r = 0; r < n; r++) for (int o = 0; o < no_; o++) {
            double m = 0; for (int i = 0; i < ni; i++) m += fabs((double)wt[(size_t)o * ni + i] * x[(size_t)r * ni + i]);
            bb[(size_t)r * no_ + o] = CB * ni * U * m;
        }
    } else if (!strcmp(op, "rms_norm") || !strcmp(op, "embed")) {
        const char *wm = !strcmp(op, "embed") ? "model.embed_tokens.embed_norm" : mod;
        snprintf(wn, sizeof wn, "%s.weight", wm); const st_t *t = st_get(W, wn);
        const st_t *et = st_get(W, "model.embed_tokens.weight");
        if (!t || !et) { bstat_add(&SL[li], INK2_NO_TENSOR, 0, 0, -1); return; }
        d = (int)t->shape[0];
        float *rows;
        if (!strcmp(op, "embed")) {
            n = (int)n0; ids = malloc(sizeof(int) * (size_t)n); rows = malloc(sizeof(float) * (size_t)n * d);
            for (int i = 0; i < n; i++) { ids[i] = (int)((const int64_t *)(const void *)i0->data)[i]; memcpy(rows + (size_t)i * d, (const float *)et->data + (size_t)ids[i] * d, sizeof(float) * d); }
            wt = (float *)(void *)et->data + (size_t)ids[0] * d;
        } else { n = (int)(n0 / d); rows = (float *)x; wt = (float *)(void *)t->data; }
        no = (size_t)n * d; bb = malloc(sizeof(double) * 2 * no);
        for (int r = 0; r < n; r++) {
            double *z = bb + no + (size_t)r * d; const float *xr = rows + (size_t)r * d; double ss = 0;
            for (int i = 0; i < d; i++) ss += (double)xr[i] * xr[i];
            double rr = 1.0 / sqrt(ss / d + cfd.eps);
            for (int i = 0; i < d; i++) { z[i] = xr[i] * rr; double wz = (double)((const float *)t->data)[i] * z[i];
                bb[(size_t)r * d + i] = fabs((double)((const float *)t->data)[i]) * CB * (d + 5) * U * fabs(z[i]) + U * fabs(wz); }
        }
        if (rows != x) free(rows);
    } else if (!strcmp(op, "unweighted_norm")) {
        double eps = strstr(mod, "_message_gate.norm") ? cfd.eps : 1e-6;
        n = (int)(n0 / H); no = n0; bb = malloc(sizeof(double) * no);
        for (int r = 0; r < n; r++) {
            double ss = 0; for (int i = 0; i < H; i++) ss += (double)x[(size_t)r * H + i] * x[(size_t)r * H + i];
            double rr = 1.0 / sqrt(ss / H + eps);
            for (int i = 0; i < H; i++) bb[(size_t)r * H + i] = CB * (H + 5) * U * fabs(x[(size_t)r * H + i] * rr);
        }
    } else if (!strcmp(op, "short_conv")) {
        snprintf(wn, sizeof wn, "%s.conv1d.weight", mod); const st_t *t = st_get(W, wn);
        if (!t) { bstat_add(&SL[li], INK2_NO_TENSOR, 0, 0, -1); return; }
        n = (int)(n0 / H); no = n0; bb = malloc(sizeof(double) * no * 4);
        double *xv = bb + no, *dz = xv + no, *zv = dz + no;
        for (size_t i = 0; i < no; i++) { xv[i] = x[i]; dz[i] = 0; }
        postconv_dx((const float *)t->data, H, s->conv_k, n, xv, dz, zv, bb);
        wt = (float *)(void *)t->data + (s->conv_k - 1);
    } else if (!strcmp(op, "l2_clip")) {
        d = (int)i0->shape[i0->n_dims - 1]; n = (int)(n0 / d); no = n0; bb = malloc(sizeof(double) * no * 2);
        double thr = strstr(mod, "relative_position_bias") ? cfd.rel : cfd.qk, *yv = bb + no;
        for (size_t i = 0; i < no; i++) { yv[i] = x[i]; bb[i] = 0; }
        for (int r = 0; r < n; r++) clip_d(yv + (size_t)r * d, bb + (size_t)r * d, d, NULL, thr, 0, 0);
    } else if (!strcmp(op, "log_scale")) {
        if (!i1 || i1->dtype != ST_I64) { bstat_add(&SL[li], INK2_BAD_INPUT, 0, 0, -1); return; }
        n = (int)st_count(i1); d = (int)(n0 / n); no = n0; bb = malloc(sizeof(double) * no);
        ids = malloc(sizeof(int) * (size_t)n);
        for (int r = 0; r < n; r++) {
            ids[r] = (int)((const int64_t *)(const void *)i1->data)[r];
            double A = (double)(ids[r] + 1) / s->log_floor; if (A < 1) A = 1;
            double L = log(A), dL = A > 1 ? U + 2 * U * fabs(L) : 0, tau = 1 + cfd.alpha * L;
            double dtau = fabs(cfd.alpha) * dL + 1.5 * U * fabs(cfd.alpha * L) + U * tau;
            for (int j = 0; j < d; j++) { double y = x[(size_t)r * d + j]; bb[(size_t)r * d + j] = fabs(y) * dtau + U * fabs(y * tau); }
        }
    } else if (!strcmp(op, "router")) {
        int b = -1; sscanf(mod, "model.layers.%d.", &b);
        const float *wr = wptr(W, "model.layers.%d.mlp.gate.weight", b), *br = wptr(W, "model.layers.%d.mlp.gate.e_score_correction_bias", b);
        n = (int)(n0 / H); no = (size_t)n * (E + K);   // logits [n][E] then weights [n][K]
        bb = malloc(sizeof(double) * no); wt = (float *)(void *)wr;
        for (int r = 0; r < n; r++) for (int e = 0; e < E; e++) {
            double m = 0; for (int h = 0; h < H; h++) m += fabs((double)wr[(size_t)e * H + h] * x[(size_t)r * H + h]);
            bb[(size_t)r * E + e] = CB * H * U * m;
        }
        (void)br;
    } else if (!strcmp(op, "score_fn")) {
        no = n0; n = (int)n0; bb = malloc(sizeof(double) * no);
        for (size_t i = 0; i < no; i++) bb[i] = 3 * U / (1.0 + exp(-(double)x[i]));
    } else if (!strcmp(op, "experts")) {
        int b = -1; sscanf(mod, "model.layers.%d.", &b);
        if (!i1 || !i2 || i1->dtype != ST_I64) { bstat_add(&SL[li], INK2_BAD_INPUT, 0, 0, -1); return; }
        const float *w13 = wptr(W, "model.layers.%d.mlp.experts.gate_up_proj", b), *w2 = wptr(W, "model.layers.%d.mlp.experts.down_proj", b);
        n = (int)(n0 / H); no = n0; bb = malloc(sizeof(double) * no * 2);
        ids = malloc(sizeof(int) * (size_t)n * K);
        for (int i = 0; i < n * K; i++) ids[i] = (int)((const int64_t *)(const void *)i1->data)[i];
        for (int r = 0; r < n; r++)
            experts_d(x + (size_t)r * H, (const int64_t *)(const void *)i1->data + (size_t)r * K, (const float *)i2->data + (size_t)r * K,
                      w13, w2, H, K, F, s->gate_cap, s->up_cap, bb + no + (size_t)r * H, bb + (size_t)r * H);
        wt = (float *)(void *)w2 + (size_t)ids[0] * H * F;
    } else if (!strcmp(op, "act_fn")) {
        if (!i1) { bstat_add(&SL[li], INK2_BAD_INPUT, 0, 0, -1); return; }
        no = n0; n = (int)n0; bb = malloc(sizeof(double) * no); const float *up = (const float *)i1->data;
        for (size_t i = 0; i < no; i++) {
            double g = x[i], u = up[i], cg = tanh(g / s->gate_cap) * s->gate_cap, cu = tanh(u / s->up_cap) * s->up_cap, sg = 1.0 / (1.0 + exp(-g));
            bb[i] = fabs(sg * cu) * 3 * U * fabs(cg) + fabs(cg * cu) * 3 * U * sg + fabs(cg * sg) * 3 * U * fabs(cu) + 2 * U * fabs(cg * sg * cu);
        }
    } else if (!strcmp(op, "softcap")) {
        double cap = strstr(mod, "gate_softcap") ? s->gate_cap : strstr(mod, "up_softcap") ? s->up_cap : s->final_cap;
        no = n0; n = (int)n0; bb = malloc(sizeof(double) * no);
        for (size_t i = 0; i < no; i++) bb[i] = U * fabs((double)x[i]) + 3 * U * fabs(tanh(x[i] / cap) * cap);
    } else { bstat_add(&SL[li], INK2_BAD_INPUT, 0, 0, -1); return; }
    // ours, the perturbed run, the comparisons
    float *ours = calloc(no + (size_t)n * K + 16, sizeof(float)), *pert = calloc(no + (size_t)n * K + 16, sizeof(float));
    float *xin = malloc(sizeof(float) * (n0 + 1)); int *iout = malloc(sizeof(int) * ((size_t)n * K + 1));
    if (x) memcpy(xin, x, sizeof(float) * n0);
    const float *in[3] = { xin, i1 && i1->dtype == ST_F32 ? (const float *)i1->data : NULL, i2 && i2->dtype == ST_F32 ? (const float *)i2->data : NULL };
    if (!strcmp(op, "experts")) in[1] = in[2];   // experts(hidden, top_k_index, top_k_weights): the weights are #in2
    int ind = d ? d : 1;
    int nrows = !strcmp(op, "short_conv") ? n : n;
    ink2_status st = ink2_op_step(s, w, mod, op, nrows, ind, in, ids, ours, ours + (strcmp(op, "router") ? 0 : (size_t)n * E), iout);
    double frac = 0, tfrac = 0, pf = -1;
    if (st == INK2_OK) {
        // the truth and torch's dump as one array in the layout of ours
        double *tv = malloc(sizeof(double) * no); float *tfv = malloc(sizeof(float) * no); int have_t = 1, have_f = 1;
        if (!strcmp(op, "router")) {
            char k0[256], k1[256], k2[256];
            snprintf(k0, sizeof k0, "%s#call%d#out0", mod, k); snprintf(k1, sizeof k1, "%s#call%d#out1", mod, k); snprintf(k2, sizeof k2, "%s#call%d#out2", mod, k);
            const double *t0 = tf64(TR, k0, (size_t)n * E), *t1 = tf64(TR, k1, (size_t)n * K), *t2 = tf64(TR, k2, (size_t)n * K);
            const float *f0 = tf32(D, k0, (size_t)n * E), *f1 = tf32(D, k1, (size_t)n * K);
            const st_t *f2 = st_get(D, k2);
            int b = -1; sscanf(mod, "model.layers.%d.", &b);
            const float *wr = wptr(W, "model.layers.%d.mlp.gate.weight", b), *br = wptr(W, "model.layers.%d.mlp.gate.e_score_correction_bias", b);
            if (!t0 || !t1 || !t2 || !f0 || !f1 || !f2) have_t = 0;
            else {
                memcpy(tv, t0, sizeof(double) * (size_t)n * E); memcpy(tfv, f0, sizeof(float) * (size_t)n * E);
                double brd[256]; for (int e = 0; e < E; e++) brd[e] = br[e];
                for (int r = 0; r < n; r++) {
                    // the selection: ours against the truth's (the truth file's indices; the harness's double agrees or
                    // is counted), then the weights in ascending expert order where the selections agree
                    int ts[64], hs[64]; double v[256], dv[256], gap, gapb;
                    for (int i = 0; i < K; i++) ts[i] = (int)t2[(size_t)r * K + i];
                    for (int i = 0; i < K; i++) for (int j = i + 1; j < K; j++) if (ts[j] < ts[i]) { int z = ts[i]; ts[i] = ts[j]; ts[j] = z; }
                    select_d(x + (size_t)r * H, wr, brd, H, E, K, hs, v, dv);
                    for (int i = 0; i < K; i++) if (hs[i] != ts[i]) { SEL_P.file_differ++; break; }
                    int cls = sel_classify(iout + (size_t)r * K, ts, v, dv, E, K, &gap, &gapb);
                    selstat_add(&SEL_P, cls, gap, gapb);
                    // the planted case, once: the truth's bias edited (on the truth's side only) so its first unchosen
                    // expert beats its last chosen by 0.5x, then 10x, the two scores' bounds; ours stays on the
                    // recorded router, so the two selections differ, first inside the bound, then outside it
                    if (!plant_done) {
                        int last = ts[0], first = -1;
                        for (int i = 0; i < K; i++) if (v[ts[i]] < v[last]) last = ts[i];
                        for (int e = 0; e < E; e++) { int c = 0; for (int i = 0; i < K; i++) c |= ts[i] == e; if (!c && (first < 0 || v[e] > v[first])) first = e; }
                        if (first >= 0) for (int pi = 0; pi < 2; pi++) {
                            double b2[256], v2[256], dv2[256]; int ts2[64];
                            memcpy(b2, brd, sizeof(double) * E);
                            b2[first] += (v[last] - v[first]) + (pi ? 10.0 : 0.5) * (dv[last] + dv[first]);
                            select_d(x + (size_t)r * H, wr, b2, H, E, K, ts2, v2, dv2);
                            plant_cls[pi] = sel_classify(iout + (size_t)r * K, ts2, v2, dv2, E, K, &plant_gap[pi], &plant_gapb[pi]);
                        }
                        plant_done = 1;
                    }
                    // the weights: dw_i = w_i (ds_i / s_i + sum ds / S + c (K + 3) u), s, ds from the logits (bound_moe)
                    int same_o = 1, same_f = 1; const int64_t *fi = (const int64_t *)(const void *)f2->data + (size_t)r * K;
                    for (int i = 0; i < K; i++) same_o &= iout[(size_t)r * K + i] == ts[i];
                    for (int i = 0; i < K; i++) { int c = 0; for (int j = 0; j < K; j++) c |= fi[j] == ts[i]; same_f &= c; }
                    double S = 0, dS = 0, sg[64], ds[64];
                    for (int i = 0; i < K; i++) {
                        int e = ts[i]; double l = t0[(size_t)r * E + e]; sg[i] = 1.0 / (1.0 + exp(-l));
                        ds[i] = sg[i] * (1 - sg[i]) * bb[(size_t)r * E + e] + 3 * U * sg[i]; S += sg[i]; dS += ds[i];
                    }
                    for (int i = 0; i < K; i++) {
                        size_t at = (size_t)n * E + (size_t)r * K + i; int e = ts[i];
                        double wi = 0; for (int j = 0; j < K; j++) if ((int)t2[(size_t)r * K + j] == e) wi = t1[(size_t)r * K + j];
                        tv[at] = wi; bb[at] = wi * (ds[i] / sg[i] + dS / S + CB * (K + 3) * U);
                        float fw = 0; for (int j = 0; j < K; j++) if (fi[j] == e) fw = f1[(size_t)r * K + j];
                        tfv[at] = same_f ? fw : (float)wi;   // torch's own differing selection is counted by the classifier
                        if (!same_o) ours[at] = (float)wi;   // a differing selection is the classifier's, not the bound's
                    }
                }
            }
        } else {
            const double *t = tf64(TR, tk, no); const float *f = tf32(D, dk, no);
            if (t) memcpy(tv, t, sizeof(double) * no); else have_t = 0;
            if (f) memcpy(tfv, f, sizeof(float) * no); else have_f = 0;
        }
        if (!have_t) st = INK2_BAD_INPUT;
        else {
            frac = frac_used(ours, tv, bb, no);
            tfrac = have_f ? frac_used(tfv, tv, bb, no) : 0;
            float keep;
            if (wt) { keep = *wt; *wt += 0.25f; } else { keep = xin[0]; xin[0] += 0.25f; }
            ink2_op_step(s, w, mod, op, nrows, ind, in, ids, pert, pert + (strcmp(op, "router") ? 0 : (size_t)n * E), iout);
            if (wt) *wt = keep; else xin[0] = keep;
            // the router's perturbation moves the logits; its weights block keeps ours (the same rows)
            if (!strcmp(op, "router")) memcpy(pert + (size_t)n * E, ours + (size_t)n * E, sizeof(float) * (size_t)n * K);
            pf = frac_used(pert, tv, bb, no);
            cov_mark(sl_name[li], dk);
        }
        free(tv); free(tfv);
    }
    bstat_add(&SL[li], st, frac, tfrac, pf);
    free(bb); free(ours); free(pert); free(xin); free(iout); free(ids);
}
// the forward's lines: 3a the decoder layer, 3b gate A (wiring), 3c gate B (chunked = whole, bit for bit)
static bstat BL = {"decoder_layer",0,0,0,0,0,0,-1,""};
static int layer_unstable = 0, layer_positions = 0, layer_nctl = 0, layer_ctl_app[16], layer_ctl_seq[16];
static int layer_nun = 0, layer_un_app[32], layer_un_seq[32], layer_un_n[32];
static double layer_harness_rel = 0;
// gate A, per mode (0 the forward, 1 control: restart from the wrong layer, 2 control: messages from the own pass)
static struct { int st_bad, argmax_bad, positions; double worst_ratio, worst_ours, worst_torch; int worst_app; char worst_seq[8];
                double logit_ours, logit_torch; } GA[3];
static double ga_min_margin = 1e300;
static int gb_rows[3], gb_bad_logits[3], gb_bad_states[3], gb_fail = 0;
static int rw_tries = 0, rw_refused = 0, rw_restart_n = 0, rw_restart_eq = 0;
static const char *gb_name[3] = { "token by token", "split at 1 and 7", "split at 5 and T-2" };
// a sequence through the forward in chunks [cut_i, cut_{i+1}); logits and states vs the whole sequence's, by memcmp
static void gate_b_run(const ink2_shape *s, const ink2_weights *w, int T, const int *ids, const int *cuts, int ncut,
                       const float *L0, const float *S0, int mode) {
    int V = s->vocab_used, H = s->hidden; ink2_cache c;
    if (ink2_cache_init(&c, s, T)) { gb_fail++; return; }
    float *lg = malloc(sizeof(float) * (size_t)T * V), *ap = malloc(sizeof(float) * (size_t)s->n_app * T * H);
    for (int i = 0; i + 1 < ncut; i++) {
        int p0 = cuts[i], n = cuts[i + 1] - cuts[i];
        if (n <= 0) continue;
        if (ink2_forward_chunk(s, w, &c, p0, n, ids + p0, lg, 0, ap) != INK2_OK) { gb_fail++; break; }
        for (int t = 0; t < n; t++) {
            gb_rows[mode]++;
            if (memcmp(lg + (size_t)t * V, L0 + (size_t)(p0 + t) * V, sizeof(float) * V)) gb_bad_logits[mode]++;
            for (int a = 0; a < s->n_app; a++)
                if (memcmp(ap + ((size_t)a * n + t) * H, S0 + ((size_t)a * T + p0 + t) * H, sizeof(float) * H)) { gb_bad_states[mode]++; break; }
        }
    }
    free(lg); free(ap); ink2_cache_free(&c);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s FIXTURE_DIR [--allow-red a,b]\n", argv[0]); return 2; }
    const char *dir = argv[1], *allow = argc > 3 && !strcmp(argv[2], "--allow-red") ? argv[3] : "";
    char path[1024], why[256] = "";
    ink2_shape s;
    snprintf(path, sizeof path, "%s/config.json", dir);
    if (cfg_load(&s, path, why, sizeof why)) { printf("FAIL structure: %s\n", why); return 1; }
    st_file W; snprintf(path, sizeof path, "%s/fixture-weights.safetensors", dir);
    if (st_load(&W, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
    wctx wc = { &W, {{0}}, 0 }; ink2_weights w = { w_get, &wc };
    int bad = 0;
    // the engagement table (engage.py, from the reference run itself): no side of any clip, cap, gate, branch or window
    // may be 0 unless the table names it with a reason
    {
        snprintf(path, sizeof path, "%s/engagement.json", dir);
        FILE *ef = fopen(path, "rb"); static char eb[1 << 20]; size_t en = ef ? fread(eb, 1, sizeof eb - 1, ef) : 0;
        if (ef) fclose(ef);
        eb[en] = 0; jv *ej = en ? json_parse(eb, en) : NULL; jv *uz = ej ? jv_get(ej, "unnamed_zero") : NULL;
        jv *rw = ej ? jv_get(ej, "rows") : NULL, *nm = ej ? jv_get(ej, "named_untested") : NULL;
        if (!uz || uz->type != J_ARR || !rw) { printf("FAIL engagement: %s missing or unreadable\n", path); bad = 1; }
        else if (uz->n) { printf("FAIL engagement: %d unnamed zero sides (first: %s)\n", uz->n, jv_str(uz->items[0], "?")); bad = 1; }
        else printf("GREEN engagement: %d rows, every side engaged except %d named (%s)\n", rw->n, nm ? nm->n : 0,
                    nm && nm->n ? "see engagement.json" : "none");
        if (ej) jv_free(ej);
    }
    for (int i = 0; i < 13; i++) { SL[i].piece = sl_name[i]; SL[i].pert_mult = -1; }
    const char *seqs[] = { "s16", "s24" };
    bstat BM = {"expert_layer",0,0,0,0,0,0,-1,""}, BA = {"attention",0,0,0,0,0,0,-1,""}, SS[N_ASTEP], SC = {"clip",0,0,0,0,0,0,-1,""}, BH = {"head",0,0,0,0,0,0,-1,""};
    for (int i = 0; i < N_ASTEP; i++) { memset(&SS[i], 0, sizeof SS[i]); SS[i].piece = astep[i]; SS[i].pert_mult = -1; }
    bstat B[3] = { {"block_read",0,0,0,0,0,0,-1,""}, {"message_input",0,0,0,0,0,0,-1,""}, {"message_step",0,0,0,0,0,0,-1,""} };
    for (int q = 0; q < 2; q++) {
        st_file D, In;
        cov_n = 0;   // the coverage registry is per sequence
        snprintf(path, sizeof path, "%s/dumps-%s.safetensors", dir, seqs[q]);
        if (st_load(&D, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
        snprintf(path, sizeof path, "%s/inputs-%s.safetensors", dir, seqs[q]);
        if (st_load(&In, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
        // 1. the structure: applicationNN.model.layers.<b>#call<k> against the derived schedule
        int seen = 0, mism = 0, calls[INK2_MAX_BLOCK] = {0};
        for (int a = 0; a < INK2_MAX_APP; a++) {
            char pre[32]; snprintf(pre, sizeof pre, "application%02d.", a);
            const st_t *hit = NULL;
            for (int i = 0; i < D.nt; i++) if (!strncmp(D.t[i].name, pre, strlen(pre))) { hit = &D.t[i]; break; }
            if (!hit) break;
            int b = -1, k = -1;
            if (sscanf(hit->name + strlen(pre), "model.layers.%d#call%d", &b, &k) != 2) { mism++; continue; }
            int want_pass = s.app_pass[a];
            if (a >= s.n_app || s.app_block[a] != b || k != calls[b] ||
                (b >= s.loop_start && b < s.loop_end && k != want_pass)) mism++;
            calls[b < 0 ? 0 : b]++; seen++;
        }
        printf("%s structure (%s): %d applications in the reference, %d derived, %d mismatched: %s\n",
               seen == s.n_app && !mism ? "GREEN" : "FAIL", seqs[q], seen, s.n_app, mism,
               seen == s.n_app && !mism ? "the schedule is the reference's" : "the schedule differs");
        if (seen != s.n_app || mism) bad = 1;
        const st_t *ids = st_get(&In, "model.embed_tokens#call0#in0");
        int Tn = ids ? (int)ids->shape[ids->n_dims - 1] : 0, H = s.hidden;
        float *out = calloc((size_t)Tn * (size_t)(s.vocab > 4 * H ? s.vocab : 4 * H) * 4, sizeof(float));
        float *probs = calloc((size_t)(s.n_head > s.swa_n_head ? s.n_head : s.swa_n_head) * Tn * (Tn + 1), sizeof(float));
        char key[256];
        // 2a. the expert layer in BOUND mode: model.layers.<b>.mlp#call<k>, plus the selection against the reference's
        // recorded indices, the harness's own double against the truth, and the perturbation control
        {
            st_file TRm; snprintf(path, sizeof path, "%s/truth-%s.safetensors", dir, seqs[q]);
            if (st_load(&TRm, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
            int E = s.n_expert, K = s.n_used, F = s.ff;
            double *yd = malloc(sizeof(double) * (size_t)Tn * H), *bd = malloc(sizeof(double) * (size_t)Tn * H);
            float *po = malloc(sizeof(float) * (size_t)Tn * H); int *sel = malloc(sizeof(int) * (size_t)Tn * K);
            for (int b = 0; b < s.n_block; b++) for (int k = 0;; k++) {
                snprintf(key, sizeof key, "model.layers.%d.mlp#call%d#in0", b, k); const st_t *x = st_get(&In, key);
                snprintf(key, sizeof key, "model.layers.%d.mlp#call%d", b, k); const st_t *tr = st_get(&TRm, key), *tf = st_get(&D, key);
                snprintf(key, sizeof key, "model.layers.%d.mlp.gate#call%d#out2", b, k); const st_t *ix = st_get(&D, key);
                if (!x || !tr) break;
                char n1[160], n2[160], n3[160], n4[160];
                snprintf(n1, sizeof n1, "model.layers.%d.mlp.gate.weight", b); snprintf(n2, sizeof n2, "model.layers.%d.mlp.gate.e_score_correction_bias", b);
                snprintf(n3, sizeof n3, "model.layers.%d.mlp.experts.gate_up_proj", b); snprintf(n4, sizeof n4, "model.layers.%d.mlp.experts.down_proj", b);
                const st_t *wr = st_get(&W, n1), *br = st_get(&W, n2), *w13 = st_get(&W, n3), *w2 = st_get(&W, n4);
                ink2_status st = ink2_expert_layer(&s, &w, b, Tn, (const float *)x->data, out);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK && wr && br && w13 && w2) {
                    for (int t = 0; t < Tn; t++)
                        bound_moe((const float *)x->data + (size_t)t * H, (const float *)wr->data, (const float *)br->data,
                                  (const float *)w13->data, (const float *)w2->data, H, E, K, F, s.route_scale, s.gate_cap,
                                  s.up_cap, yd + (size_t)t * H, bd + (size_t)t * H, sel + (size_t)t * K);
                    // the selection against the reference's recorded indices (as sets)
                    if (ix && ix->dtype == ST_I64) for (int t = 0; t < Tn; t++) {
                        const int64_t *ri = (const int64_t *)(const void *)ix->data + (size_t)t * K;
                        for (int a = 0; a < K; a++) { int found = 0; for (int c = 0; c < K; c++) found |= ri[c] == sel[(size_t)t * K + a]; if (!found) moe_sel_diff++; }
                    }
                    // the harness's own double against the truth
                    const double *tv = (const double *)(const void *)tr->data;
                    {   // normwise: the largest |difference| over the largest |truth| of the call
                        double dm = 0, tm = 0;
                        for (size_t i2 = 0; i2 < (size_t)Tn * H; i2++) { double d = fabs(yd[i2] - tv[i2]); if (d > dm) dm = d; if (fabs(tv[i2]) > tm) tm = fabs(tv[i2]); }
                        if (tm > 0 && dm / tm > moe_harness_rel) moe_harness_rel = dm / tm;
                    }
                    frac = frac_used(out, tv, bd, (size_t)Tn * H);
                    tfrac = tf ? frac_used((const float *)tf->data, tv, bd, (size_t)Tn * H) : 0;
                    // perturbation: the down projection's first weight of the first expert position 0 chose
                    float *w0 = (float *)(void *)w2->data + (size_t)sel[0] * H * F; float keep = w0[0]; w0[0] += 0.25f;
                    ink2_expert_layer(&s, &w, b, Tn, (const float *)x->data, po);
                    w0[0] = keep;
                    pert = frac_used(po, tv, bd, (size_t)Tn * H);
                    // the TIE case, once: every router row equal to row 0 and every bias equal, so all scores tie; our
                    // rule must choose experts 0..K-1, and the piece must equal the harness's double with that choice
                    if (!moe_tie_done) {
                        moe_tie_done = 1;
                        float *wrr = (float *)(void *)wr->data, *brr = (float *)(void *)br->data;
                        float *save = malloc(sizeof(float) * (size_t)E * (H + 1));
                        memcpy(save, wrr, sizeof(float) * (size_t)E * H); memcpy(save + (size_t)E * H, brr, sizeof(float) * E);
                        for (int e = 1; e < E; e++) { memcpy(wrr + (size_t)e * H, wrr, sizeof(float) * H); brr[e] = brr[0]; }
                        ink2_expert_layer(&s, &w, b, 1, (const float *)x->data, po);
                        double yt[4096], bt[4096]; int st2[64];
                        bound_moe((const float *)x->data, wrr, brr, (const float *)w13->data, (const float *)w2->data, H, E, K, F,
                                  s.route_scale, s.gate_cap, s.up_cap, yt, bt, st2);
                        moe_tie_ok = 1;
                        for (int a = 0; a < K; a++) moe_tie_ok &= st2[a] == a;
                        moe_tie_frac = frac_used(po, yt, bt, (size_t)H);
                        moe_tie_ok &= moe_tie_frac <= 1.0;
                        memcpy(wrr, save, sizeof(float) * (size_t)E * H); memcpy(brr, save + (size_t)E * H, sizeof(float) * E); free(save);
                    }
                }
                bstat_add(&BM, st, frac, tfrac, pert);
                if (st == INK2_OK) cov_markf("expert_layer", "model.layers.%d.mlp#call%d", b, k);
            }
            free(yd); free(bd); free(po); free(sel);
        }
        // 2b. the attention internals in BOUND mode: model.layers.<b>.self_attn#call<k>, application = the k-th visit
        // of b; both outputs (#out0 [T][H] and the probabilities #out1 [heads][T][keys]) against the truth under
        // bound_attn, torch's float32 dumps the same; the perturbation control q_proj.linear.weight[0] + 0.25. Then each
        // sub-step ONE AT A TIME on its own recorded inputs (ink2_attention_step): the piece's output and torch's float32
        // dump against the harness's double of that step alone, under the step's own bound (the same step function as
        // in bound_attn, its recorded inputs exact: input bounds 0), so an error shows as the step that makes it.
        {
            st_file TRa; snprintf(path, sizeof path, "%s/truth-%s.safetensors", dir, seqs[q]);
            if (st_load(&TRa, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
            ink2_cache cache; if (ink2_cache_init(&cache, &s, Tn)) { printf("FAIL: cache\n"); return 1; }
            int nhm = s.n_head > s.swa_n_head ? s.n_head : s.swa_n_head;
            size_t n0 = (size_t)Tn * H, M = (size_t)nhm * Tn * Tn + (size_t)Tn * (4 * H + nhm * (s.d_rel + 1));   // >= any tensor of the call
            double *dm = malloc(sizeof(double) * M * 14); float *fm = malloc(sizeof(float) * M * 9);
            double *y0 = dm, *b0 = dm + M, *y1 = dm + 2 * M, *b1 = dm + 3 * M;   // also the sub-steps' outputs
            float *po = fm, *pp = fm + M;
            for (int a = 0; a < s.n_app; a++) {
                int b = s.app_block[a], k = 0, g = ink2_app_global(&s, a); for (int j = 0; j < a; j++) k += s.app_block[j] == b;
                const st_t *xq, *xk, *xv, *pos, *mk, *rd, *t0, *t1, *f0, *f1;
#define KW(var, fmt) snprintf(key, sizeof key, fmt, b, k); var = st_get(&In, key)
                KW(xq, "model.layers.%d.self_attn#call%d#kw.hidden_states"); KW(xk, "model.layers.%d.self_attn#call%d#kw.key_hidden_states");
                KW(xv, "model.layers.%d.self_attn#call%d#kw.value_hidden_states"); KW(pos, "model.layers.%d.self_attn#call%d#kw.position_ids");
                KW(mk, "model.layers.%d.self_attn#call%d#kw.attention_mask"); KW(rd, "model.layers.%d.self_attn#call%d#kw.relative_distance");
#undef KW
                snprintf(key, sizeof key, "model.layers.%d.self_attn#call%d#out0", b, k); t0 = st_get(&TRa, key); f0 = st_get(&D, key);
                snprintf(key, sizeof key, "model.layers.%d.self_attn#call%d#out1", b, k); t1 = st_get(&TRa, key); f1 = st_get(&D, key);
                // a sliding layer outside the loop reads one block state for Q, K and V: the reference passes only hidden_states
                if (!xk) xk = xq;
                if (!xv) xv = xq;
                attn_wd A; int nh = g ? s.n_head : s.swa_n_head; size_t n1 = (size_t)nh * Tn * Tn;
                if (!xq || !pos || !mk || !rd || !t0 || !t1 || t0->dtype != ST_F64 || t1->dtype != ST_F64 || st_count(t0) != n0 ||
                    st_count(t1) != n1 || st_count(mk) != (size_t)Tn * Tn || st_count(rd) != (size_t)Tn * Tn || attn_wd_get(&s, &W, b, g, &A)) {
                    bstat_add(&BA, INK2_BAD_INPUT, 0, 0, -1); continue;
                }
                const float *mask = (const float *)mk->data; const int64_t *rdv = (const int64_t *)(const void *)rd->data;
                int P[4096]; for (int t = 0; t < Tn && t < 4096; t++) P[t] = (int)((const int64_t *)(const void *)pos->data)[t];
                ink2_status st = ink2_attention(&s, &w, &cache, a, Tn, P, (const float *)xq->data, (const float *)xk->data,
                                                (const float *)xv->data, out, probs);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK) {
                    bound_attn(&s, &A, Tn, (const float *)xq->data, (const float *)xk->data, (const float *)xv->data, P, mask, rdv, y0, b0, y1, b1);
                    const double *tv0 = (const double *)(const void *)t0->data, *tv1 = (const double *)(const void *)t1->data;
                    for (int o = 0; o < 2; o++) {   // the harness's double vs the truth, normwise
                        const double *y = o ? y1 : y0, *tv = o ? tv1 : tv0; size_t n = o ? n1 : n0; double dmx = 0, tm = 0;
                        for (size_t i = 0; i < n; i++) { double d = fabs(y[i] - tv[i]); if (d > dmx) dmx = d; if (fabs(tv[i]) > tm) tm = fabs(tv[i]); }
                        if (tm > 0 && dmx / tm > attn_harness_rel[o]) attn_harness_rel[o] = dmx / tm;
                    }
                    double fo = frac_used(out, tv0, b0, n0), fp = frac_used(probs, tv1, b1, n1);
                    frac = fmax(fo, fp); attn_frac[0] = fmax(attn_frac[0], fo); attn_frac[1] = fmax(attn_frac[1], fp);
                    if (f0 && f1 && st_count(f0) == n0 && st_count(f1) == n1) {
                        // two statements, not two arguments: frac_used feeds the outputs digest, and C leaves the
                        // order of a call's arguments unspecified (gcc and clang took them in opposite orders)
                        double tq0 = frac_used((const float *)f0->data, tv0, b0, n0);
                        double tq1 = frac_used((const float *)f1->data, tv1, b1, n1);
                        tfrac = fmax(tq0, tq1);
                    }
                    float *w0 = (float *)(void *)A.w[AQ]; float keep = w0[0]; w0[0] += 0.25f;
                    ink2_attention(&s, &w, &cache, a, Tn, P, (const float *)xq->data, (const float *)xk->data,
                                   (const float *)xv->data, po, pp);
                    w0[0] = keep;
                    double q0 = frac_used(po, tv0, b0, n0);   // sequenced, as above
                    double q1 = frac_used(pp, tv1, b1, n1);
                    pert = fmax(q0, q1);
                }
                bstat_add(&BA, st, frac, tfrac, pert);
                if (st == INK2_OK) cov_markf("attention", "model.layers.%d.self_attn#call%d", b, k);
                if (st != INK2_OK) continue;
                // the sub-steps. Buffers: dm[4..13] double inputs and their (zero) bounds, fm[2..8] float inputs and outputs
                for (int si = 0; si < N_ASTEP; si++) {
                    const char *sn = astep[si]; int kv = sn[0] == 'v', H2 = H, C = A.C, Q = A.Q, hd = A.hd;
                    double *u0 = dm + 4 * M, *du0 = dm + 5 * M; float *c0 = fm + 2 * M, *c1 = fm + 3 * M, *tt = fm + 4 * M;
                    const float *in[5] = { 0 }; const st_t *r0, *r1, *tdump, *tdump2 = NULL;
                    size_t n = 0, n2 = 0;
#define SUB(suffix) (snprintf(key, sizeof key, "model.layers.%d.self_attn.%s#call%d%s", b, sn, k, suffix), key)
                    r0 = st_get(&In, SUB("#in0")); r1 = st_get(&In, SUB("#in1")); tdump = st_get(&D, SUB(""));
                    if (!strcmp(sn, "core")) {   // its inputs are the recorded OUTPUTS of q_proj, k_proj, v_proj, the bias, the sinks
                        const char *src[5] = { "q_proj", "k_proj", "v_proj", "relative_position_bias", "sink_proj" };
                        const st_t *e[5]; float *fx[5];
                        for (int i = 0; i < 5; i++) { snprintf(key, sizeof key, "model.layers.%d.self_attn.%s#call%d", b, src[i], k); e[i] = st_get(&D, key); fx[i] = fm + (4 + i) * M; }
                        if (!e[0] || !e[1] || !e[2] || !e[3] || !e[4]) { bstat_add(&SS[si], INK2_BAD_INPUT, 0, 0, -1); continue; }
                        tr_ntd((const float *)e[0]->data, nh, Tn, hd, fx[0]); tr_ntd((const float *)e[1]->data, A.nkv, Tn, hd, fx[1]);
                        tr_ntd((const float *)e[2]->data, A.nkv, Tn, hd, fx[2]);
                        memcpy(fx[3], e[3]->data, sizeof(float) * n1); memcpy(fx[4], e[4]->data, sizeof(float) * (size_t)Tn * nh);
                        size_t sz[5] = { (size_t)Tn * Q, (size_t)Tn * C, (size_t)Tn * C, n1, (size_t)Tn * nh };
                        for (int i = 0; i < 5; i++) { f2d(fx[i], sz[i], dm + (4 + 2 * i) * M, dm + (5 + 2 * i) * M); in[i] = fx[i]; }
                        core_d(&A, Tn, dm + 4 * M, dm + 5 * M, dm + 6 * M, dm + 7 * M, dm + 8 * M, dm + 9 * M, dm + 10 * M, dm + 11 * M,
                               dm + 12 * M, dm + 13 * M, mask, y1, b1, y0, b0);
                        n = (size_t)Tn * Q; n2 = n1;
                        snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj#call%d#in0", b, k); tdump = st_get(&In, key); tdump2 = f1;
                    } else {
                        if (!r0) { bstat_add(&SS[si], INK2_BAD_INPUT, 0, 0, -1); continue; }
                        const float *x0 = (const float *)r0->data, *x1 = r1 && r1->dtype == ST_F32 ? (const float *)r1->data : NULL;
                        in[0] = x0; in[1] = x1;
                        if (!strcmp(sn, "q_proj")) {
                            rows_d(A.w[AQ], Q, H2, Tn, x0, y0, b0); headclip_d(&s, g, Tn, nh, hd, A.w[AQS], cfd.qk, P, y0, b0); n = (size_t)Tn * Q;
                        } else if (!strcmp(sn, "q_proj.norm")) {
                            n = (size_t)Tn * Q; f2d(x0, n, y0, b0); headclip_d(&s, g, Tn, nh, hd, A.w[AQS], cfd.qk, P, y0, b0);
                        } else if (!strcmp(sn, "k_proj") || !strcmp(sn, "v_proj")) {
                            rows_d(A.w[kv ? AV : AK], C, H2, Tn, x0, u0, du0); conv_d(&A, kv, H2, Tn, u0, du0, x0, y0, b0); kvnorm_d(&A, kv, Tn, y0, b0);
                            n = (size_t)Tn * C;
                        } else if (strstr(sn, ".sconv")) {
                            if (!x1) { bstat_add(&SS[si], INK2_BAD_INPUT, 0, 0, -1); continue; }
                            n = (size_t)Tn * C; f2d(x0, n, u0, du0); conv_d(&A, kv, H2, Tn, u0, du0, x1, y0, b0);
                        } else if (strstr(sn, "_proj.norm")) {
                            n = (size_t)Tn * C; f2d(x0, n, y0, b0); kvnorm_d(&A, kv, Tn, y0, b0);
                        } else if (!strcmp(sn, "relative_position_bias")) {
                            rows_d(A.w[AR], nh * A.dr, H2, Tn, x0, u0, du0); headclip_d(&s, g, Tn, nh, A.dr, A.w[ARS], cfd.rel, P, u0, du0);
                            bank_d(&A, Tn, u0, du0, rdv, y0, b0); n = n1;
                        } else if (!strcmp(sn, "relative_position_bias.norm")) {
                            n = (size_t)Tn * nh * A.dr; f2d(x0, n, y0, b0); headclip_d(&s, g, Tn, nh, A.dr, A.w[ARS], cfd.rel, P, y0, b0);
                        } else if (!strcmp(sn, "sink_proj")) {
                            rows_d(A.w[ASK], nh, H2, Tn, x0, y0, b0); n = (size_t)Tn * nh;
                        } else {   // o_proj
                            if (!x1) { bstat_add(&SS[si], INK2_BAD_INPUT, 0, 0, -1); continue; }
                            f2d(x0, (size_t)Tn * Q, u0, du0); oproj_d(&A, H2, Tn, x1, u0, du0, y0, b0); n = n0;
                        }
                    }
#undef SUB
                    // THE REFERENCE: a step that is a module call is compared with the truth file's float64 re-run of
                    // that call (model.layers.<b>.self_attn.<step>#call<k>; q_proj, k_proj, v_proj return [heads][T][hd],
                    // brought to the piece's [T][heads][hd]); the harness's own double (y0 above, the same bound's
                    // value) is kept as a diagnostic. The core is not a module call (eager_attention_forward): it keeps
                    // the harness's double as its reference.
                    int is_core = !strcmp(sn, "core");
                    const double *ref = y0;
                    if (!is_core) {
                        snprintf(key, sizeof key, "model.layers.%d.self_attn.%s#call%d", b, sn, k);
                        const st_t *tt0 = st_get(&TRa, key);
                        double *rt = dm + 4 * M;   // the sub-step's double inputs are no longer needed here
                        if (!tt0 || tt0->dtype != ST_F64 || st_count(tt0) != n) { bstat_add(&SS[si], INK2_BAD_INPUT, 0, 0, -1); continue; }
                        const double *tv = (const double *)(const void *)tt0->data;
                        int nhh = !strcmp(sn, "q_proj") ? nh : (!strcmp(sn, "k_proj") || !strcmp(sn, "v_proj")) ? A.nkv : 0;
                        if (nhh) { for (int h = 0; h < nhh; h++) for (int t = 0; t < Tn; t++) memcpy(rt + ((size_t)t * nhh + h) * hd, tv + ((size_t)h * Tn + t) * hd, sizeof(double) * hd); }
                        else memcpy(rt, tv, sizeof(double) * n);
                        double dmx = 0, tm = 0;
                        for (size_t i = 0; i < n; i++) { double dd = fabs(y0[i] - rt[i]); if (dd > dmx) dmx = dd; if (fabs(rt[i]) > tm) tm = fabs(rt[i]); }
                        if (tm > 0 && dmx / tm > astep_harness_rel[si]) astep_harness_rel[si] = dmx / tm;
                        ref = rt;
                    }
                    ink2_status sst = ink2_attention_step(&s, &w, a, sn, Tn, P, in, c0, c1);
                    double sf = 0, stf = 0, spf = -1;
                    if (sst == INK2_OK) {
                        sf = frac_used(c0, ref, b0, n); if (n2) sf = fmax(sf, frac_used(c1, y1, b1, n2));
                        if (tdump && st_count(tdump) == n) {
                            const float *tf = (const float *)tdump->data;
                            // q_proj, k_proj, v_proj return [heads][T][hd]: to the piece's [T][heads][hd]
                            if (!strcmp(sn, "q_proj")) { tr_ntd(tf, nh, Tn, hd, tt); tf = tt; }
                            else if (!strcmp(sn, "k_proj") || !strcmp(sn, "v_proj")) { tr_ntd(tf, A.nkv, Tn, hd, tt); tf = tt; }
                            stf = frac_used(tf, ref, b0, n);
                            if (n2 && tdump2 && st_count(tdump2) == n2) stf = fmax(stf, frac_used((const float *)tdump2->data, y1, b1, n2));
                        } else sst = INK2_BAD_INPUT;
                    }
                    // THE PERTURBATION CONTROL: + 0.25 on one weight the step reads (its first element), or for the
                    // weightless steps on an input: k_proj.norm and v_proj.norm (an L2 clip, an unweighted norm) the first
                    // element of their input, the core the first element of V; the perturbed outputs feed the
                    // single-step digest, so the pieces' digest keeps its arrays
                    if (sst == INK2_OK) {
                        const float *pw_ = !strcmp(sn, "q_proj") ? A.w[AQ] : !strcmp(sn, "q_proj.norm") ? A.w[AQS] : !strcmp(sn, "k_proj") ? A.w[AK]
                            : !strcmp(sn, "k_proj.sconv") ? A.w[AKC] : !strcmp(sn, "v_proj") ? A.w[AV] : !strcmp(sn, "v_proj.sconv") ? A.w[AVC]
                            : !strcmp(sn, "relative_position_bias") ? A.w[AR] : !strcmp(sn, "relative_position_bias.norm") ? A.w[ARS]
                            : !strcmp(sn, "sink_proj") ? A.w[ASK] : !strcmp(sn, "o_proj") ? A.w[AO] : NULL;
                        float *pt = (float *)(void *)pw_, *xin = fm + 5 * M; const float *pin[5] = { in[0], in[1], in[2], in[3], in[4] };
                        if (!pt && is_core) pt = (float *)(void *)pin[2];   // the core: v[0] (its own buffer fx[2]), which every
                                                                             // query attending key 0 reads (a q element reaches one query only)
                        else if (!pt) { memcpy(xin, in[0], sizeof(float) * n); pin[0] = xin; pt = xin; }   // the norms: a copy
                        float keep = pt[0]; pt[0] += 0.25f;
                        uint64_t *dsave = g_dig; g_dig = &g_digest3;
                        ink2_attention_step(&s, &w, a, sn, Tn, P, pin, po, pp);
                        spf = frac_used(po, ref, b0, n); if (n2) spf = fmax(spf, frac_used(pp, y1, b1, n2));
                        g_dig = dsave;
                        pt[0] = keep;
                    }
                    bstat_add(&SS[si], sst, sf, stf, spf);
                    if (sst == INK2_OK && strcmp(sn, "core")) {
                        static char sline[N_ASTEP][64]; snprintf(sline[si], sizeof sline[si], "attention.%s", sn);
                        char kk[200]; snprintf(kk, sizeof kk, "model.layers.%d.self_attn.%s#call%d", b, sn, k); cov_mark(sline[si], kk);
                    }
                }
                // the three clip steps once more on their recorded inputs times 64 (exact), against the harness's double:
                // the clips engage on the recorded data too (engagement.json: every clip fires on a real share of head
                // vectors), so this adds a heavily clipped case, not the only one; no torch dump here
                for (int ci = 0; ci < 3; ci++) {
                    const char *sn = ci == 0 ? "q_proj.norm" : ci == 1 ? "k_proj.norm" : "relative_position_bias.norm";
                    snprintf(key, sizeof key, "model.layers.%d.self_attn.%s#call%d#in0", b, sn, k);
                    const st_t *r0 = st_get(&In, key); if (!r0) continue;
                    size_t n = st_count(r0); float *x = fm + 2 * M, *c0 = fm + 3 * M; const float *in[5] = { x };
                    for (size_t i = 0; i < n; i++) x[i] = ((const float *)r0->data)[i] * 64.0f;
                    f2d(x, n, y0, b0);
                    if (ci == 0) headclip_d(&s, g, Tn, nh, A.hd, A.w[AQS], cfd.qk, P, y0, b0);
                    else if (ci == 1) kvnorm_d(&A, 0, Tn, y0, b0);
                    else headclip_d(&s, g, Tn, nh, A.dr, A.w[ARS], cfd.rel, P, y0, b0);
                    int d = ci == 1 ? A.hd : ci == 0 ? A.hd : A.dr; const float *sc = ci == 0 ? A.w[AQS] : ci == 2 ? A.w[ARS] : NULL;
                    double thr = ci == 2 ? cfd.rel : cfd.qk;
                    for (size_t v0 = 0; v0 < n; v0 += d) {   // the head vectors the clip shortens
                        double ss = 0;
                        for (int i = 0; i < d; i++) { double z = (double)x[v0 + i] * (sc ? sc[(v0 + i) % ((size_t)(ci == 1 ? A.nkv : nh) * d)] : 1.0f); ss += z * z; }
                        clip_engaged += sqrt(ss) > thr; clip_vectors++;
                    }
                    ink2_status sst = ink2_attention_step(&s, &w, a, sn, Tn, P, in, c0, NULL);
                    bstat_add(&SC, sst, sst == INK2_OK ? frac_used(c0, y0, b0, n) : 0, 0, -1);
                }
            }
            free(dm); free(fm);
            ink2_cache_free(&cache); free(TRa.buf); free(TRa.t);
        }
        // 2c. the head in BOUND mode: lm_head's input to logit_softcap's output.
        //   dl_v = c H u sum_h |W_vh x_h|; bound_v = dl_v + 3 u |cap(l_v)|   (the softcap's tanh' <= 1)
        {
            const st_t *x = st_get(&In, "lm_head#call0#in0"), *tf = st_get(&D, "logit_softcap#call0"), *wo = st_get(&W, "lm_head.weight");
            st_file TRh; snprintf(path, sizeof path, "%s/truth-%s.safetensors", dir, seqs[q]);
            if (st_load(&TRh, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
            const st_t *tr = st_get(&TRh, "logit_softcap#call0");
            if (x && tr && wo) {
                int V = s.vocab_used; size_t cnt = (size_t)Tn * V;
                ink2_status st = ink2_head(&s, &w, Tn, (const float *)x->data, out);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK) {
                    double *bb = malloc(sizeof(double) * cnt); float *po = malloc(sizeof(float) * cnt);
                    for (int t = 0; t < Tn; t++) for (int v = 0; v < V; v++) {
                        double a = 0, m = 0; const float *xt = (const float *)x->data + (size_t)t * H, *row = (const float *)wo->data + (size_t)v * H;
                        for (int h = 0; h < H; h++) { a += (double)row[h] * xt[h]; m += fabs((double)row[h] * xt[h]); }
                        bb[(size_t)t * V + v] = CB * H * U * m + 3 * U * fabs(tanh(a / s.final_cap) * s.final_cap);
                    }
                    float *w0 = (float *)(void *)wo->data; float keep = w0[0]; w0[0] += 0.25f;
                    ink2_head(&s, &w, Tn, (const float *)x->data, po); w0[0] = keep;
                    const double *tv = (const double *)(const void *)tr->data;
                    frac = frac_used(out, tv, bb, cnt); tfrac = tf ? frac_used((const float *)tf->data, tv, bb, cnt) : 0;
                    pert = frac_used(po, tv, bb, cnt);
                    free(bb); free(po);
                }
                bstat_add(&BH, st, frac, tfrac, pert);
                if (st == INK2_OK) cov_mark("head", "logit_softcap#call0");
            }
        }
        // 2d-2f in BOUND mode against the float64 truth: the block-state reads, the cross-pass message read, the
        // message gates. Each call: the piece's output, torch's float32 dump and the perturbed piece, each against
        // the truth under the composed bound.
        st_file TR; snprintf(path, sizeof path, "%s/truth-%s.safetensors", dir, seqs[q]);
        if (st_load(&TR, path)) { printf("FAIL: cannot read %s\n", path); return 1; }
        double *bnd = calloc((size_t)Tn * 4 * H, sizeof(double));
        float *pout = calloc((size_t)Tn * 4 * H, sizeof(float));
        // (d) block reads
        for (int b = -1; b < s.n_block; b++) for (int r = 0; r < (b < 0 ? 1 : 2); r++) {
            char mod[96], qn[128];
            if (b < 0) snprintf(mod, sizeof mod, "model.final_input");
            else snprintf(mod, sizeof mod, "model.layers.%d.%s", b, r ? "mlp_input" : "attention_input");
            snprintf(qn, sizeof qn, "%s.query", mod);
            const st_t *qt = st_get(&W, qn);
            for (int k = 0; qt; k++) {
                snprintf(key, sizeof key, "%s#call%d#in0", mod, k); const st_t *bs = st_get(&In, key);
                snprintf(key, sizeof key, "%s#call%d#in1", mod, k); const st_t *pa = st_get(&In, key);
                if (!bs || !pa) break;
                int nb = (int)bs->shape[bs->n_dims - 2], nkv = nb + 1, n_out = 0, nq = qt->n_dims == 1 ? 1 : (int)qt->shape[0];
                ink2_status st = ink2_block_read(&s, &w, mod, Tn, nb, (const float *)bs->data, (const float *)pa->data, out, &n_out);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK) {
                    // bounds [nq][T][H]
                    float *stk = malloc(sizeof(float) * (size_t)nkv * H); double *bb = malloc(sizeof(double) * (size_t)nq * H);
                    for (int t = 0; t < Tn; t++) {
                        memcpy(stk, (const float *)bs->data + (size_t)t * nb * H, sizeof(float) * (size_t)nb * H);
                        memcpy(stk + (size_t)nb * H, (const float *)pa->data + (size_t)t * H, sizeof(float) * H);
                        bound_depth((const float *)qt->data, nq, stk, nkv, H, bb);
                        for (int i = 0; i < nq; i++) memcpy(bnd + ((size_t)i * Tn + t) * H, bb + (size_t)i * H, sizeof(double) * H);
                    }
                    free(stk); free(bb);
                    // perturbation: the query's first element + 0.25
                    float *q0 = (float *)(void *)qt->data; float keep = q0[0]; q0[0] += 0.25f;
                    int pn = 0; ink2_block_read(&s, &w, mod, Tn, nb, (const float *)bs->data, (const float *)pa->data, pout, &pn);
                    q0[0] = keep;
                    for (int o = 0; o < n_out; o++) {
                        snprintf(key, sizeof key, n_out > 1 ? "%s#call%d#out%d" : "%s#call%d", mod, k, o);
                        const st_t *tr = st_get(&TR, key), *tf = st_get(&D, key);
                        if (!tr || tr->dtype != ST_F64) { st = INK2_BAD_INPUT; break; }
                        size_t off = (size_t)o * Tn * H, cnt = (size_t)Tn * H;
                        double f = frac_used(out + off, (const double *)(const void *)tr->data, bnd + off, cnt);
                        double tf2 = tf ? frac_used((const float *)tf->data, (const double *)(const void *)tr->data, bnd + off, cnt) : 0;
                        double pf = frac_used(pout + off, (const double *)(const void *)tr->data, bnd + off, cnt);
                        if (f > frac) frac = f;
                        if (tf2 > tfrac) tfrac = tf2;
                        if (pf > pert) pert = pf;
                    }
                }
                bstat_add(&B[0], st, frac, tfrac, pert);
                if (st == INK2_OK) { char kk[200]; snprintf(kk, sizeof kk, "%s#call%d", mod, k); cov_mark("block_read", kk); }
            }
        }
        // (e) the cross-pass message read
        {
            const st_t *qt = st_get(&W, "model.message_input.query");
            for (int k = 0; qt; k++) {
                snprintf(key, sizeof key, "model.message_input#call%d#in0", k); const st_t *pb = st_get(&In, key);
                snprintf(key, sizeof key, "model.message_input#call%d#in1", k); const st_t *x = st_get(&In, key);
                snprintf(key, sizeof key, "model.message_input#call%d", k); const st_t *tr = st_get(&TR, key), *tf = st_get(&D, key);
                if (!pb || !x || !tr) break;
                int nprev = (int)pb->shape[pb->n_dims - 2], nkv = nprev + 1, ll = k % (s.loop_end - s.loop_start);
                ink2_status st = ink2_message_input(&s, &w, ll, Tn, nprev, (const float *)pb->data, (const float *)x->data, out);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK) {
                    const float *ql = (const float *)qt->data + (size_t)ll * 4 * H;
                    float *stk = malloc(sizeof(float) * (size_t)nkv * H);
                    for (int t = 0; t < Tn; t++) {
                        memcpy(stk, (const float *)pb->data + (size_t)t * nprev * H, sizeof(float) * (size_t)nprev * H);
                        memcpy(stk + (size_t)nprev * H, (const float *)x->data + (size_t)t * H, sizeof(float) * H);
                        bound_depth(ql, 4, stk, nkv, H, bnd + (size_t)t * 4 * H);   // [T][4][H], as the output
                    }
                    free(stk);
                    float *q0 = (float *)(void *)ql; float keep = q0[0]; q0[0] += 0.25f;
                    ink2_message_input(&s, &w, ll, Tn, nprev, (const float *)pb->data, (const float *)x->data, pout);
                    q0[0] = keep;
                    size_t cnt = (size_t)Tn * 4 * H;
                    frac = frac_used(out, (const double *)(const void *)tr->data, bnd, cnt);
                    tfrac = tf ? frac_used((const float *)tf->data, (const double *)(const void *)tr->data, bnd, cnt) : 0;
                    pert = frac_used(pout, (const double *)(const void *)tr->data, bnd, cnt);
                }
                bstat_add(&B[1], st, frac, tfrac, pert);
                if (st == INK2_OK) { char kk[200]; snprintf(kk, sizeof kk, "model.message_input#call%d", k); cov_mark("message_input", kk); }
            }
        }
        // (f) the message gates
        {
            const char *roles[] = { "q", "k", "v", "mlp" };
            for (int b = s.loop_start; b < s.loop_end; b++) for (int r = 0; r < 4; r++) for (int k = 0;; k++) {
                snprintf(key, sizeof key, "model.layers.%d.%s_message_gate#call%d#in0", b, roles[r], k); const st_t *x = st_get(&In, key);
                snprintf(key, sizeof key, "model.layers.%d.%s_message_gate#call%d#in1", b, roles[r], k); const st_t *m = st_get(&In, key);
                snprintf(key, sizeof key, "model.layers.%d.%s_message_gate#call%d", b, roles[r], k);
                const st_t *tr = st_get(&TR, key), *tf = st_get(&D, key);
                if (!x || !m || !tr) break;
                char n1[160], n2[160], n3[160];
                snprintf(n1, sizeof n1, "model.layers.%d.%s_message_gate.down_proj.weight", b, roles[r]);
                snprintf(n2, sizeof n2, "model.layers.%d.%s_message_gate.forget_gate_proj.weight", b, roles[r]);
                snprintf(n3, sizeof n3, "model.layers.%d.%s_message_gate.message_gate_proj.weight", b, roles[r]);
                const st_t *wd = st_get(&W, n1), *wf = st_get(&W, n2), *wg = st_get(&W, n3);
                ink2_status st = ink2_message_step(&s, &w, b, roles[r], Tn, (const float *)x->data, (const float *)m->data, out);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK && wd && wf && wg) {
                    for (int t = 0; t < Tn; t++)
                        bound_gate((const float *)x->data + (size_t)t * H, (const float *)m->data + (size_t)t * H,
                                   (const float *)wd->data, (const float *)wf->data, (const float *)wg->data, H,
                                   s.loop_gate_rank, s.eps, bnd + (size_t)t * H);
                    float *w0 = (float *)(void *)wd->data; float keep = w0[0]; w0[0] += 0.25f;
                    ink2_message_step(&s, &w, b, roles[r], Tn, (const float *)x->data, (const float *)m->data, pout);
                    w0[0] = keep;
                    size_t cnt = (size_t)Tn * H;
                    frac = frac_used(out, (const double *)(const void *)tr->data, bnd, cnt);
                    tfrac = tf ? frac_used((const float *)tf->data, (const double *)(const void *)tr->data, bnd, cnt) : 0;
                    pert = frac_used(pout, (const double *)(const void *)tr->data, bnd, cnt);
                }
                bstat_add(&B[2], st, frac, tfrac, pert);
                if (st == INK2_OK) { char kk[200]; snprintf(kk, sizeof kk, "model.layers.%d.%s_message_gate#call%d", b, roles[r], k); cov_mark("message_step", kk); }
            }
        }
        g_dig = &g_digest2;   // the forward's lines feed the second digest
        // 3a. the decoder layer, one application at a time on its RECORDED inputs (model.layers.<b>#call<k>: in0 the
        // hidden state, in1 the block states, in2 the messages in later passes; kw.position_ids, kw.attention_mask,
        // kw.relative_distance), against the per-application float64 truth under bound_layer; torch's float32 dump the
        // same; the perturbation control post_mlp_convolution.conv1d.weight[0][K-1] + 0.25 (the current-token tap)
        {
            ink2_cache lc; if (ink2_cache_init(&lc, &s, Tn)) { printf("FAIL: cache\n"); return 1; }
            size_t TH = (size_t)Tn * H;
            double *ly = malloc(sizeof(double) * TH * 2), *lb = ly + TH; float *lo = malloc(sizeof(float) * TH * 2), *lp = lo + TH;
            for (int a = 0; a < s.n_app; a++) {
                int b = s.app_block[a], k = 0; for (int j = 0; j < a; j++) k += s.app_block[j] == b;
                const st_t *x, *bs, *ms, *pz, *mk, *rd, *tr, *tf;
#define LK(var, file, fmt) snprintf(key, sizeof key, fmt, b, k); var = st_get(file, key)
                LK(x, &In, "model.layers.%d#call%d#in0"); LK(bs, &In, "model.layers.%d#call%d#in1"); LK(ms, &In, "model.layers.%d#call%d#in2");
                LK(pz, &In, "model.layers.%d#call%d#kw.position_ids"); LK(mk, &In, "model.layers.%d#call%d#kw.attention_mask");
                LK(rd, &In, "model.layers.%d#call%d#kw.relative_distance"); LK(tr, &TR, "model.layers.%d#call%d"); LK(tf, &D, "model.layers.%d#call%d");
#undef LK
                if (!x || !bs || !pz || !mk || !rd || !tr || tr->dtype != ST_F64 || st_count(tr) != TH) { bstat_add(&BL, INK2_BAD_INPUT, 0, 0, -1); continue; }
                int nb = (int)bs->shape[bs->n_dims - 2], P[4096];
                for (int t = 0; t < Tn; t++) P[t] = (int)((const int64_t *)(const void *)pz->data)[t];
                const float *msg = ms ? (const float *)ms->data : NULL;
                int Kx = s.n_used, Ex = s.n_expert, *osel = malloc(sizeof(int) * (size_t)Tn * Kx * 2), *tsel = osel + (size_t)Tn * Kx;
                double *sv = malloc(sizeof(double) * (size_t)Tn * Ex * 2), *sdv = sv + (size_t)Tn * Ex;
                ink2_status st = ink2_decoder_layer_sel(&s, &w, &lc, a, Tn, P, (const float *)x->data, (const float *)bs->data, nb, msg, lo, osel);
                double frac = 0, tfrac = 0, pert = -1;
                if (st == INK2_OK) {
                    int un = bound_layer(&s, &W, b, Tn, (const float *)x->data, (const float *)bs->data, nb, msg, P,
                                         (const float *)mk->data, (const int64_t *)(const void *)rd->data, ly, lb, tsel, sv, sdv);
                    // our selection against the double chain's (the truth's), classified under the chain's score bounds
                    for (int t = 0; t < Tn; t++) {
                        double gap, gapb;
                        int cls = sel_classify(osel + (size_t)t * Kx, tsel + (size_t)t * Kx, sv + (size_t)t * Ex, sdv + (size_t)t * Ex, Ex, Kx, &gap, &gapb);
                        selstat_add(&SEL_L, cls, gap, gapb);
                    }
                    layer_unstable += un; layer_positions += Tn;
                    const double *tv = (const double *)(const void *)tr->data; double dmx = 0, tm = 0;
                    for (size_t i = 0; i < TH; i++) { double d = fabs(ly[i] - tv[i]); if (d > dmx) dmx = d; if (fabs(tv[i]) > tm) tm = fabs(tv[i]); }
                    if (tm > 0 && dmx / tm > layer_harness_rel) layer_harness_rel = dmx / tm;
                    frac = frac_used(lo, tv, lb, TH);
                    tfrac = tf && st_count(tf) == TH ? frac_used((const float *)tf->data, tv, lb, TH) : 0;
                    float *w0 = (float *)(void *)wptr(&W, "model.layers.%d.post_mlp_convolution.conv1d.weight", b) + (s.conv_k - 1);
                    float keep = *w0; *w0 += 0.25f;
                    ink2_decoder_layer(&s, &w, &lc, a, Tn, P, (const float *)x->data, (const float *)bs->data, nb, msg, lp);
                    *w0 = keep;
                    pert = frac_used(lp, tv, lb, TH);
                    if (pert <= 1.0 && layer_nctl < 16) { layer_ctl_app[layer_nctl] = a; layer_ctl_seq[layer_nctl++] = q; }
                    if (un) { if (layer_nun < 32) { layer_un_app[layer_nun] = a; layer_un_seq[layer_nun] = q; layer_un_n[layer_nun] = un; } layer_nun++; }
                }
                bstat_add(&BL, st, frac, tfrac, pert);
                free(osel); free(sv);
            }
            free(ly); free(lo); ink2_cache_free(&lc);
        }
        // 3b. gate A, WIRING: the free-running forward (its own states fed forward) at every application against the
        // whole-model float64 run (e2e.application<NN>) and torch's float32 run (dumps model.layers.<b>#call<k>, visit
        // order) against the same; the logits' argmax against e2e.logits. Modes 1 and 2 are the deliberate mis-wirings.
        {
            const st_t *idt = st_get(&In, "model.embed_tokens#call0#in0"), *el = st_get(&TR, "e2e.logits"), *tl = st_get(&D, "logits");
            int V = s.vocab_used, *ids = malloc(sizeof(int) * (size_t)Tn);
            for (int t = 0; t < Tn; t++) ids[t] = (int)((const int64_t *)(const void *)idt->data)[t];
            float *L0 = malloc(sizeof(float) * (size_t)Tn * V), *S0 = malloc(sizeof(float) * (size_t)s.n_app * Tn * H);
            float *Lm = malloc(sizeof(float) * (size_t)Tn * V), *Sm = malloc(sizeof(float) * (size_t)s.n_app * Tn * H);
            const double *tlv = el ? (const double *)(const void *)el->data : NULL;
            for (int mode = 0; mode < 3; mode++) {
                int flags = mode == 0 ? 0 : mode == 1 ? INK2_MISWIRE_RESTART : INK2_MISWIRE_OWN_PASS;
                float *Lx = mode ? Lm : L0, *Sx = mode ? Sm : S0;
                ink2_cache fc; if (ink2_cache_init(&fc, &s, Tn)) { printf("FAIL: cache\n"); return 1; }
                ink2_status st = ink2_forward_chunk(&s, &w, &fc, 0, Tn, ids, Lx, flags, Sx);
                ink2_cache_free(&fc);
                if (st != INK2_OK || !tlv || !tl) { GA[mode].st_bad++; continue; }
                if (mode == 0) { digest(L0, sizeof(float) * (size_t)Tn * V); digest(S0, sizeof(float) * (size_t)s.n_app * Tn * H); }
                for (int a = 0; a < s.n_app; a++) {
                    int b = s.app_block[a], k = 0; for (int j = 0; j < a; j++) k += s.app_block[j] == b;
                    snprintf(key, sizeof key, "e2e.application%02d", a); const st_t *ta = st_get(&TR, key);
                    snprintf(key, sizeof key, "model.layers.%d#call%d", b, k); const st_t *da = st_get(&D, key);
                    if (!ta || !da || st_count(ta) != (size_t)Tn * H) { GA[mode].st_bad++; continue; }
                    const double *tv = (const double *)(const void *)ta->data; const float *dv = (const float *)da->data, *ov = Sx + (size_t)a * Tn * H;
                    double eo = 0, et = 0, tm = 0;
                    for (size_t i = 0; i < (size_t)Tn * H; i++) {
                        double d1 = fabs(ov[i] - tv[i]), d2 = fabs(dv[i] - tv[i]);
                        if (!(d1 <= eo)) eo = d1;   // a NaN takes it to NaN, then fails the gate
                        if (d2 > et) et = d2;
                        if (fabs(tv[i]) > tm) tm = fabs(tv[i]);
                    }
                    eo /= tm; et /= tm;
                    double ratio = et > 0 ? eo / et : (eo > 0 ? INFINITY : 0);
                    if (!(ratio <= GA[mode].worst_ratio)) {
                        GA[mode].worst_ratio = ratio; GA[mode].worst_ours = eo; GA[mode].worst_torch = et; GA[mode].worst_app = a;
                        snprintf(GA[mode].worst_seq, sizeof GA[mode].worst_seq, "%s", seqs[q]);
                    }
                }
                for (int t = 0; t < Tn; t++) {
                    const double *tr = tlv + (size_t)t * V; const float *ov = Lx + (size_t)t * V;
                    int at = 0, ao = 0; double m2 = -1e300;
                    for (int v = 1; v < V; v++) { if (tr[v] > tr[at]) at = v; if (ov[v] > ov[ao]) ao = v; }
                    for (int v = 0; v < V; v++) if (v != at && tr[v] > m2) m2 = tr[v];
                    if (mode == 0 && tr[at] - m2 < ga_min_margin) ga_min_margin = tr[at] - m2;
                    GA[mode].positions++; GA[mode].argmax_bad += at != ao;
                }
                for (size_t i = 0; i < (size_t)Tn * V; i++) {
                    double d1 = fabs(Lx[i] - tlv[i]), d2 = fabs(((const float *)tl->data)[i] - tlv[i]);
                    if (!(d1 <= GA[mode].logit_ours)) GA[mode].logit_ours = d1;
                    if (d2 > GA[mode].logit_torch) GA[mode].logit_torch = d2;
                }
            }
            // 3c. gate B: the same forward in chunks through the cache, logits and every application's state against
            // the whole sequence's (mode 0 above) by memcmp
            int c0[64], nc = 0;
            for (int t = 0; t <= Tn; t++) c0[nc++] = t;
            gate_b_run(&s, &w, Tn, ids, c0, nc, L0, S0, 0);
            int c1[4] = { 0, 1, 7, Tn }, c2[4] = { 0, 5, Tn - 2, Tn };
            gate_b_run(&s, &w, Tn, ids, c1, 4, L0, S0, 1);
            gate_b_run(&s, &w, Tn, ids, c2, 4, L0, S0, 2);
            // 3d. a REWIND is refused: on a used cache, a chunk at 0 < p0 < n_past must return INK2_REWIND; a restart
            // at p0 = 0 on the used cache must give the fresh cache's bits (logits and states, memcmp), and so must a
            // restart in two chunks (0..4, then 5..T-1: the continuation after a restart is not a rewind)
            {
                ink2_cache rc; if (ink2_cache_init(&rc, &s, Tn)) { printf("FAIL: cache\n"); return 1; }
                float *lg = malloc(sizeof(float) * (size_t)Tn * V), *ap = malloc(sizeof(float) * (size_t)s.n_app * Tn * H);
                float *ap2 = malloc(sizeof(float) * (size_t)s.n_app * Tn * H);
                ink2_status r0 = ink2_forward_chunk(&s, &w, &rc, 0, Tn, ids, lg, 0, ap);   // the cache now holds 0..T-1
                int pr = Tn / 2;
                ink2_status r1 = ink2_forward_chunk(&s, &w, &rc, pr, Tn - pr, ids + pr, lg, 0, ap);
                rw_tries++; rw_refused += r0 == INK2_OK && r1 == INK2_REWIND;
                ink2_status r2 = ink2_forward_chunk(&s, &w, &rc, 0, Tn, ids, lg, 0, ap);
                rw_restart_n++;
                rw_restart_eq += r2 == INK2_OK && !memcmp(lg, L0, sizeof(float) * (size_t)Tn * V) && !memcmp(ap, S0, sizeof(float) * (size_t)s.n_app * Tn * H);
                int c5 = Tn > 5 ? 5 : 1;
                ink2_status r3 = ink2_forward_chunk(&s, &w, &rc, 0, c5, ids, lg, 0, ap2);
                ink2_status r4 = r3 == INK2_OK ? ink2_forward_chunk(&s, &w, &rc, c5, Tn - c5, ids + c5, lg + (size_t)c5 * V, 0, ap) : r3;
                int eq = r4 == INK2_OK && !memcmp(lg, L0, sizeof(float) * (size_t)Tn * V);
                for (int a = 0; a < s.n_app && eq; a++)
                    eq = !memcmp(ap2 + (size_t)a * c5 * H, S0 + (size_t)a * Tn * H, sizeof(float) * (size_t)c5 * H) &&
                         !memcmp(ap + (size_t)a * (Tn - c5) * H, S0 + ((size_t)a * Tn + c5) * H, sizeof(float) * (size_t)(Tn - c5) * H);
                rw_restart_n++; rw_restart_eq += eq;
                free(lg); free(ap); free(ap2); ink2_cache_free(&rc);
            }
            free(ids); free(L0); free(S0); free(Lm); free(Sm);
        }
        // 4. SINGLE-STEP LINES over every recorded module call of the sequence, by THE COVERAGE TABLE: each module
        // (layer index as B) maps to the op its single-step line runs, or to the piece line that already checks it, or
        // (one row) to a named composite. The table is the whole list: a call whose module is not in it is unnamed.
        g_dig = &g_digest3;
        {
            static const struct { const char *pat, *op, *line; } TAB[] = {
                { "lm_head", "linear", NULL }, { "logit_softcap", "softcap", NULL }, { "model.embed_tokens", "embed", NULL },
                { "model.embed_tokens.embed_norm", "rms_norm", NULL }, { "model.norm", "rms_norm", NULL },
                { "model.final_input", NULL, "block_read" }, { "model.final_input.norm", "unweighted_norm", NULL },
                { "model.message_input", NULL, "message_input" }, { "model.message_input.norm", "unweighted_norm", NULL },
                { "model.layers.B", NULL, "COMPOSITE" },
                { "model.layers.B.attention_input", NULL, "block_read" }, { "model.layers.B.attention_input.norm", "unweighted_norm", NULL },
                { "model.layers.B.mlp_input", NULL, "block_read" }, { "model.layers.B.mlp_input.norm", "unweighted_norm", NULL },
                { "model.layers.B.input_layernorm", "rms_norm", NULL }, { "model.layers.B.q_input_layernorm", "rms_norm", NULL },
                { "model.layers.B.k_input_layernorm", "rms_norm", NULL }, { "model.layers.B.v_input_layernorm", "rms_norm", NULL },
                { "model.layers.B.post_attention_layernorm", "rms_norm", NULL },
                { "model.layers.B.post_attention_convolution", "short_conv", NULL }, { "model.layers.B.post_mlp_convolution", "short_conv", NULL },
                { "model.layers.B.q_message_gate", NULL, "message_step" }, { "model.layers.B.k_message_gate", NULL, "message_step" },
                { "model.layers.B.v_message_gate", NULL, "message_step" }, { "model.layers.B.mlp_message_gate", NULL, "message_step" },
                { "model.layers.B.q_message_gate.norm", "unweighted_norm", NULL }, { "model.layers.B.k_message_gate.norm", "unweighted_norm", NULL },
                { "model.layers.B.v_message_gate.norm", "unweighted_norm", NULL }, { "model.layers.B.mlp_message_gate.norm", "unweighted_norm", NULL },
                { "model.layers.B.q_message_gate.down_proj", "linear", NULL }, { "model.layers.B.k_message_gate.down_proj", "linear", NULL },
                { "model.layers.B.v_message_gate.down_proj", "linear", NULL }, { "model.layers.B.mlp_message_gate.down_proj", "linear", NULL },
                { "model.layers.B.q_message_gate.forget_gate_proj", "linear", NULL }, { "model.layers.B.k_message_gate.forget_gate_proj", "linear", NULL },
                { "model.layers.B.v_message_gate.forget_gate_proj", "linear", NULL }, { "model.layers.B.mlp_message_gate.forget_gate_proj", "linear", NULL },
                { "model.layers.B.q_message_gate.message_gate_proj", "linear", NULL }, { "model.layers.B.k_message_gate.message_gate_proj", "linear", NULL },
                { "model.layers.B.v_message_gate.message_gate_proj", "linear", NULL }, { "model.layers.B.mlp_message_gate.message_gate_proj", "linear", NULL },
                { "model.layers.B.mlp", NULL, "expert_layer" }, { "model.layers.B.mlp.gate", "router", NULL },
                { "model.layers.B.mlp.gate.score_fn", "score_fn", NULL }, { "model.layers.B.mlp.experts", "experts", NULL },
                { "model.layers.B.mlp.experts.act_fn", "act_fn", NULL }, { "model.layers.B.mlp.experts.act_fn.gate_softcap", "softcap", NULL },
                { "model.layers.B.mlp.experts.act_fn.up_softcap", "softcap", NULL },
                { "model.layers.B.self_attn", NULL, "attention" },
                { "model.layers.B.self_attn.q_proj", NULL, "attention.q_proj" }, { "model.layers.B.self_attn.q_proj.linear", "linear", NULL },
                { "model.layers.B.self_attn.q_proj.norm", NULL, "attention.q_proj.norm" },
                { "model.layers.B.self_attn.q_proj.norm.l2_clip", "l2_clip", NULL }, { "model.layers.B.self_attn.q_proj.norm.log_scale", "log_scale", NULL },
                { "model.layers.B.self_attn.k_proj", NULL, "attention.k_proj" }, { "model.layers.B.self_attn.k_proj.linear", "linear", NULL },
                { "model.layers.B.self_attn.k_proj.sconv", NULL, "attention.k_proj.sconv" }, { "model.layers.B.self_attn.k_proj.sconv.weight_proj", "linear", NULL },
                { "model.layers.B.self_attn.k_proj.norm", NULL, "attention.k_proj.norm" },
                { "model.layers.B.self_attn.v_proj", NULL, "attention.v_proj" }, { "model.layers.B.self_attn.v_proj.linear", "linear", NULL },
                { "model.layers.B.self_attn.v_proj.sconv", NULL, "attention.v_proj.sconv" }, { "model.layers.B.self_attn.v_proj.sconv.weight_proj", "linear", NULL },
                { "model.layers.B.self_attn.v_proj.norm", NULL, "attention.v_proj.norm" },
                { "model.layers.B.self_attn.relative_position_bias", NULL, "attention.relative_position_bias" },
                { "model.layers.B.self_attn.relative_position_bias.linear", "linear", NULL },
                { "model.layers.B.self_attn.relative_position_bias.norm", NULL, "attention.relative_position_bias.norm" },
                { "model.layers.B.self_attn.relative_position_bias.norm.l2_clip", "l2_clip", NULL },
                { "model.layers.B.self_attn.relative_position_bias.norm.log_scale", "log_scale", NULL },
                { "model.layers.B.self_attn.sink_proj", NULL, "attention.sink_proj" },
                { "model.layers.B.self_attn.o_proj", NULL, "attention.o_proj" }, { "model.layers.B.self_attn.o_proj.gate_proj", "linear", NULL },
                { "model.layers.B.self_attn.o_proj.linear", "linear", NULL },
            };
            const int NT = (int)(sizeof TAB / sizeof TAB[0]);
            // the calls: every dump tensor's "<module>#call<k>", the applicationNN aliases and the bare logits excluded
            int ncall = 0; char (*calls)[160] = malloc(sizeof *calls * (size_t)D.nt);
            for (int i = 0; i < D.nt; i++) {
                const char *nm = D.t[i].name, *c = strstr(nm, "#call");
                if (!strncmp(nm, "application", 11) || !c) continue;
                char ck[160]; int len = (int)(c - nm) + 5; while (nm[len] >= '0' && nm[len] <= '9') len++;
                snprintf(ck, sizeof ck, "%.*s", len, nm);
                int dup = 0; for (int j = 0; j < ncall && !dup; j++) dup = !strcmp(calls[j], ck);
                if (!dup) snprintf(calls[ncall++], sizeof calls[0], "%s", ck);
            }
            // run every op row's calls
            int *row_of = malloc(sizeof(int) * (size_t)ncall);
            for (int ci = 0; ci < ncall; ci++) {
                char mod[160], pat[160]; int k = -1, b = -1;
                const char *c = strstr(calls[ci], "#call"); snprintf(mod, sizeof mod, "%.*s", (int)(c - calls[ci]), calls[ci]); k = atoi(c + 5);
                if (sscanf(mod, "model.layers.%d", &b) == 1) { const char *rest = strchr(mod + 13, '.'); snprintf(pat, sizeof pat, "model.layers.B%s", rest ? rest : ""); }
                else snprintf(pat, sizeof pat, "%s", mod);
                row_of[ci] = -1;
                for (int r = 0; r < NT; r++) if (!strcmp(TAB[r].pat, pat)) { row_of[ci] = r; break; }
                if (row_of[ci] >= 0 && TAB[row_of[ci]].op) {
                    int li = 0; const char *op = TAB[row_of[ci]].op;
                    for (int j = 0; j < 12; j++) if (!strcmp(sl_name[j], op) || (!strcmp(op, "short_conv") && j == L_CONV) || (!strcmp(op, "embed") && j == L_EMB)) { li = j; break; }
                    run_step(&s, &w, &W, &In, &D, &TR, mod, k, op, li);
                }
            }
            // 4m. the decoder layer's own last step, ELEMENTWISE on the recorded tensors: residual = x + post-attention
            // conv (no add at a block boundary), out = residual + post-MLP conv; BITS against torch's recorded output
            // (the same float32 adds in the same order), then against the truth (the exact sum of the same operands)
            // under its one-rounding bound u |residual| (when added) + u |out|; perturbation: the post-MLP conv
            // output's first element + 0.25. And the WIRING on recorded tensors, by memcmp: each application's hidden
            // input (the previous output; at a pass restart the pre-loop state), its block states (the reference's
            // growth: a boundary layer's input appended after it; restart: the pre-loop states), the MLP read's states
            // and partial, the messages and the message read's inputs (the previous pass's states after the one its
            // first core layer appended, and its last output), the final read's inputs.
            {
                size_t TH = (size_t)Tn * H;
                const float **bsl = malloc(sizeof(float *) * (size_t)(s.n_app + 2)), **pre_bsl = malloc(sizeof(float *) * (size_t)(s.n_app + 2));
                const float **prev_bsl = malloc(sizeof(float *) * (size_t)(s.n_app + 2));
                int nb = 0, pre_nb = 0, n_prev = 0, msg_k = 0;
                const float *h_in = tf32(&D, "model.embed_tokens#call0", TH), *pre_h = NULL, *prev_last = NULL, *last_out = NULL;
                float *pack = malloc(sizeof(float) * TH * (size_t)(s.n_app + 2)), *rb = malloc(sizeof(float) * TH * 3);
                double *rt = malloc(sizeof(double) * TH * 2);
#define WIRE(what, got, want, n) do { wire_n++; if (!(got) || !(want) || memcmp(got, want, sizeof(float) * (n))) { wire_bad++; \
                    if (!wire_first[0]) snprintf(wire_first, sizeof wire_first, "%s %s app %d", seqs[q], what, a); } } while (0)
#define PACK(list, cnt) do { for (int t = 0; t < Tn; t++) for (int j = 0; j < (cnt); j++) memcpy(pack + ((size_t)t * (cnt) + j) * H, (list)[j] + (size_t)t * H, sizeof(float) * H); } while (0)
                for (int a = 0; a < s.n_app; a++) {
                    int b = s.app_block[a], p = s.app_pass[a], k = 0; for (int j = 0; j < a; j++) k += s.app_block[j] == b;
                    int bnd = b % s.attn_res_block == 0;
                    if (b == s.loop_start) {
                        if (p == 0) { pre_h = h_in; pre_nb = nb; for (int j = 0; j < nb; j++) pre_bsl[j] = bsl[j]; }
                        else { n_prev = nb - pre_nb - 1; for (int j = 0; j < n_prev; j++) prev_bsl[j] = bsl[pre_nb + 1 + j]; prev_last = last_out;
                               h_in = pre_h; nb = pre_nb; for (int j = 0; j < nb; j++) bsl[j] = pre_bsl[j]; }
                    }
                    char kk[200];
                    snprintf(kk, sizeof kk, "model.layers.%d#call%d#in0", b, k); const float *x = tf32(&In, kk, TH);
                    snprintf(kk, sizeof kk, "model.layers.%d#call%d#in1", b, k); const float *bsr = tf32(&In, kk, TH * (size_t)nb);
                    WIRE("hidden input", x, h_in, TH);
                    PACK(bsl, nb); WIRE("block states", nb ? bsr : pack, pack, TH * (size_t)nb);
                    if (b >= s.loop_start && b < s.loop_end && p > 0) {   // the message read's inputs and the messages
                        snprintf(kk, sizeof kk, "model.message_input#call%d#in0", msg_k); const float *m0 = tf32(&In, kk, TH * (size_t)n_prev);
                        snprintf(kk, sizeof kk, "model.message_input#call%d#in1", msg_k); const float *m1 = tf32(&In, kk, TH);
                        snprintf(kk, sizeof kk, "model.message_input#call%d", msg_k); const float *mo = tf32(&D, kk, TH * 4);
                        snprintf(kk, sizeof kk, "model.layers.%d#call%d#in2", b, k); const float *mi = tf32(&In, kk, TH * 4);
                        PACK(prev_bsl, n_prev); WIRE("message read states", n_prev ? m0 : pack, pack, TH * (size_t)n_prev);
                        WIRE("message read partial", m1, prev_last, TH); WIRE("messages", mi, mo, TH * 4);
                        msg_k++;
                    }
                    snprintf(kk, sizeof kk, "model.layers.%d.post_attention_convolution#call%d", b, k); const float *pac = tf32(&D, kk, TH);
                    snprintf(kk, sizeof kk, "model.layers.%d.post_mlp_convolution#call%d", b, k); const float *pmc = tf32(&D, kk, TH);
                    snprintf(kk, sizeof kk, "model.layers.%d#call%d", b, k); const float *lout = tf32(&D, kk, TH);
                    int nb2 = bnd ? nb + 1 : nb;
                    snprintf(kk, sizeof kk, "model.layers.%d.mlp_input#call%d#in0", b, k); const float *mis = tf32(&In, kk, TH * (size_t)nb2);
                    snprintf(kk, sizeof kk, "model.layers.%d.mlp_input#call%d#in1", b, k); const float *mip = tf32(&In, kk, TH);
                    if (!x || !pac || !pmc || !lout) { bstat_add(&SL[L_RES], INK2_BAD_INPUT, 0, 0, -1); }
                    else {
                        float *res = rb, *out = rb + TH, *po = rb + 2 * TH; double *rd_ = rt, *bd = rt + TH;
                        for (size_t i = 0; i < TH; i++) res[i] = bnd ? pac[i] : x[i] + pac[i];
                        for (size_t i = 0; i < TH; i++) out[i] = res[i] + pmc[i];
                        res_bits_n++; res_bits_eq += !memcmp(out, lout, sizeof(float) * TH);
                        WIRE("MLP read partial", mip, res, TH);
                        if (bnd) { bsl[nb] = x; PACK(bsl, nb + 1); } else PACK(bsl, nb);
                        WIRE("MLP read states", mis, pack, TH * (size_t)nb2);
                        for (size_t i = 0; i < TH; i++) {
                            double r = bnd ? (double)pac[i] : (double)x[i] + pac[i], o = r + pmc[i];
                            rd_[i] = o; bd[i] = (bnd ? 0 : U * fabs(r)) + U * fabs(o);
                        }
                        double fr = frac_used(out, rd_, bd, TH), tfr = frac_used(lout, rd_, bd, TH);
                        memcpy(po, out, sizeof(float) * TH); po[0] = res[0] + (pmc[0] + 0.25f);
                        double pf = frac_used(po, rd_, bd, TH);
                        bstat_add(&SL[L_RES], INK2_OK, fr, tfr, pf);
                        cov_markf("residual", "model.layers.%d#call%d", b, k);
                    }
                    if (bnd) bsl[nb++] = x;
                    last_out = lout; h_in = lout;
                }
                {
                    int a = s.n_app;
                    const float *f0 = tf32(&In, "model.final_input#call0#in0", TH * (size_t)nb), *f1 = tf32(&In, "model.final_input#call0#in1", TH);
                    PACK(bsl, nb); WIRE("final read states", f0, pack, TH * (size_t)nb); WIRE("final read partial", f1, last_out, TH);
                }
#undef WIRE
#undef PACK
                free(bsl); free(pre_bsl); free(prev_bsl); free(pack); free(rb); free(rt);
            }
            // 4z. the coverage of this sequence's calls
            for (int ci = 0; ci < ncall; ci++) {
                int r = row_of[ci]; cov_calls++;
                if (r < 0) { cov_unnamed++; if (!cov_first_unnamed[0]) snprintf(cov_first_unnamed, sizeof cov_first_unnamed, "%s", calls[ci]); continue; }
                if (TAB[r].line && !strcmp(TAB[r].line, "COMPOSITE")) {
                    // model.layers.<b>: its own last step in the residual line, its children each in theirs (checked
                    // below), its composition in gates A and B
                    const char *ln = cov_find(calls[ci]);
                    if (!ln) { cov_uncovered++; if (!cov_first_unc[0]) snprintf(cov_first_unc, sizeof cov_first_unc, "%s", calls[ci]); }
                    else cov_composite++;
                    continue;
                }
                const char *ln = cov_find(calls[ci]);
                if (!ln) { cov_uncovered++; if (!cov_first_unc[0]) snprintf(cov_first_unc, sizeof cov_first_unc, "%s", calls[ci]); continue; }
                cov_single++;
                int li = -1; for (int j = 0; j < cov_nlines; j++) if (!strcmp(cov_lines[j], ln)) li = j;
                if (li < 0 && cov_nlines < 64) { li = cov_nlines; cov_lines[cov_nlines++] = ln; }
                if (li >= 0) cov_per_line[li]++;
            }
            free(calls); free(row_of);
        }
        g_dig = &g_digest;
        free(bnd); free(pout);
        free(out); free(probs);
    }
    {
        const bstat *b = &BM; int pert_ok = b->pert_mult > 1.0;
        int ok = b->calls && !b->fail && !b->nob && !b->red && pert_ok && !moe_sel_diff && moe_tie_ok;
        printf("%s expert_layer (bound mode, c = %.0f): %d calls, %d failed, %d out of bound; largest fraction %.3g; torch %.3g%s; "
               "perturbation %.3g times the bound; selection vs the reference's indices: %d differ; the harness's double vs the truth: "
               "%.2g normwise; tie case (all scores equal): %s, fraction %.3g\n", b->red ? "RED" : ok ? "GREEN" : "FAIL", CB, b->calls,
               b->fail, b->nob, b->frac, b->torch_frac, b->torch_frac > 1.0 ? " (TORCH OUTSIDE THE BOUND)" : "", b->pert_mult,
               moe_sel_diff, moe_harness_rel, moe_tie_ok ? "experts 0..K-1 chosen, in bound" : "FAILED", moe_tie_frac);
        if (!ok && !(b->red && strstr(allow, "expert_layer"))) bad = 1;
    }
    {
        const bstat *b = &BA; int pert_ok = b->pert_mult > 1.0;
        int ok = b->calls && !b->fail && !b->nob && !b->red && pert_ok;
        printf("%s attention (bound mode, c = %.0f): %d calls, %d failed, %d out of bound; largest fraction %.3g (output %.3g, "
               "probabilities %.3g); torch %.3g%s; perturbation %.3g times the bound%s; the harness's double vs the truth: %.2g "
               "(output), %.2g (probabilities) normwise\n", b->red ? "RED" : ok ? "GREEN" : "FAIL", CB, b->calls, b->fail, b->nob,
               b->frac, attn_frac[0], attn_frac[1], b->torch_frac, b->torch_frac > 1.0 ? " (TORCH OUTSIDE THE BOUND)" : "",
               b->pert_mult, b->red || pert_ok ? "" : " (THE CONTROL DID NOT FAIL: the bound is too loose)", attn_harness_rel[0],
               attn_harness_rel[1]);
        if (!ok && !(b->red && strstr(allow, "attention"))) bad = 1;
    }
    for (int i = 0; i < N_ASTEP; i++) {
        // GREEN: in bound, the control leaves the bound, and torch's own float32 dump inside the same bound
        const bstat *b = &SS[i]; int pert_ok = b->pert_mult > 1.0, tok = b->torch_frac <= 1.0;
        int ok = b->calls && !b->fail && !b->nob && !b->red && pert_ok && tok, stub = BA.red > 0 && !b->calls, core = !strcmp(b->piece, "core");
        char hr[96] = "";
        if (!core) snprintf(hr, sizeof hr, "; the harness's double vs the truth %.2g normwise", astep_harness_rel[i]);
        printf("%s attention.%s (one step on its recorded inputs, its own bound, vs %s): %d calls, %d failed, %d out of bound; "
               "largest fraction %.3g; torch's float32 dump %.3g%s; perturbation %.3g times the bound%s%s\n",
               b->red || stub ? "RED" : ok ? "GREEN" : "FAIL", b->piece,
               core ? "the harness's double (not a module call: eager_attention_forward)" : "the float64 truth of the call",
               b->calls, b->fail, b->nob, b->frac, b->torch_frac, tok ? "" : " (TORCH OUTSIDE THE BOUND)", b->pert_mult,
               pert_ok ? "" : " (THE CONTROL DID NOT FAIL)", hr);
        if (!ok && !(b->red && strstr(allow, "attention"))) bad = 1;
    }
    {
        const bstat *b = &SC; int ok = b->calls && !b->fail && !b->nob && !b->red && clip_engaged > 0, stub = BA.red > 0 && !b->calls;
        printf("%s attention clips engaged (q_proj.norm, k_proj.norm, relative_position_bias.norm on their recorded inputs x 64): "
               "%d calls, %ld of %ld head vectors clipped, %d out of bound; largest fraction %.3g\n", b->red || stub ? "RED" : ok ? "GREEN" : "FAIL",
               b->calls, clip_engaged, clip_vectors, b->nob, b->frac);
        if (!ok && !(b->red && strstr(allow, "attention"))) bad = 1;
    }
    for (int i = 0; i < 3; i++) {
        const bstat *b = &B[i];
        int allowed = strstr(allow, b->piece) != NULL, pert_ok = b->pert_mult > 1.0;
        const char *word = !b->calls || b->fail || b->nob || (!b->red && !pert_ok) ? "FAIL" : b->red ? "RED" : "GREEN";
        printf("%s %s (bound mode, c = %.0f): %d calls, %d not implemented, %d failed, %d out of bound; largest fraction "
               "of the bound used %.3g; torch's float32 dump %.3g%s; perturbation control: the smallest move is %.3g "
               "times the bound%s\n", word, b->piece, CB, b->calls, b->red, b->fail, b->nob, b->frac, b->torch_frac,
               b->torch_frac > 1.0 ? " (TORCH OUTSIDE THE BOUND: a finding about the reference)" : "",
               b->pert_mult, b->red ? "" : pert_ok ? "" : " (THE CONTROL DID NOT FAIL: the bound is too loose)");
        if (!b->calls || b->fail || b->nob || (b->red && !allowed) || (!b->red && !pert_ok)) bad = 1;
    }
    {
        const bstat *b = &BH; int pert_ok = b->pert_mult > 1.0;
        int ok = b->calls && !b->fail && !b->nob && !b->red && pert_ok;
        printf("%s head (bound mode, c = %.0f): %d calls, %d out of bound; largest fraction %.3g; torch %.3g%s; perturbation %.3g "
               "times the bound\n", b->red ? "RED" : ok ? "GREEN" : "FAIL", CB, b->calls, b->nob, b->frac, b->torch_frac,
               b->torch_frac > 1.0 ? " (TORCH OUTSIDE THE BOUND)" : "", b->pert_mult);
        if (!ok) bad = 1;
    }
    {
        // a REPORT line (the coordinator's ruling 2026-11-21): the one-level composition is too loose to gate on; every
        // operation of the layer is gated by its single-step line, the composition by gates A and B
        const bstat *b = &BL;
        char ctl[512] = "";
        for (int i = 0; i < layer_nctl; i++) snprintf(ctl + strlen(ctl), sizeof ctl - strlen(ctl), "%s%s a%02d", i ? ", " : "", seqs[layer_ctl_seq[i]], layer_ctl_app[i]);
        printf("INFO decoder_layer (a REPORT, not a gate: one application on its recorded inputs vs the per-application float64 "
               "truth, the pieces' bounds composed first order, GIVEN THE SAME EXPERT SELECTION): %d calls, %d failed, %d out of "
               "bound; largest fraction %.3g; torch %.3g; the one-level composition lets a 0.25 post-MLP tap through at %d of %d "
               "applications (%s; smallest %.3g times the bound); the operations are each gated by their single-step lines; "
               "selection, ours vs the double chain's: %d of %d positions differ, %d at near-ties (largest gap there %.3g), %d "
               "outside the bound; %d positions have a gap below the chain's propagated score bound (smallest gap %.3g); the "
               "harness's double vs the truth: %.2g normwise\n", b->calls, b->fail, b->nob, b->frac, b->torch_frac, layer_nctl,
               b->calls, layer_nctl ? ctl : "none", b->pert_mult, SEL_L.near_differ + SEL_L.hard_differ, SEL_L.pos,
               SEL_L.near_differ, SEL_L.near_gap_max, SEL_L.hard_differ, SEL_L.near_ties, SEL_L.min_gap, layer_harness_rel);
        (void)layer_nun; (void)layer_un_app; (void)layer_un_seq; (void)layer_un_n;
    }
    {
        int ok0 = !GA[0].st_bad && !GA[0].argmax_bad && GA[0].worst_ratio <= 8.0 && GA[0].positions;
        int f1 = GA[1].st_bad || GA[1].argmax_bad || !(GA[1].worst_ratio <= 8.0), f2 = GA[2].st_bad || GA[2].argmax_bad || !(GA[2].worst_ratio <= 8.0);
        int ok = ok0 && f1 && f2;
        printf("%s forward gate A (a WIRING gate, not an arithmetic bound: every argmax equal and, at every application, our "
               "normwise error vs the whole-model float64 run at most 8x torch's own): argmax %d of %d equal, the truth's smallest "
               "top-1 margin %.3g; worst ratio %.3g (ours %.2g, torch %.2g, %s application %02d); control 1 (restart from the "
               "wrong layer) %s: worst ratio %.3g (ours %.2g, %s application %02d), argmax %d of %d equal; control 2 (messages "
               "from the own pass) %s: worst ratio %.3g (ours %.2g, %s application %02d), argmax %d of %d equal\n",
               ok ? "GREEN" : "FAIL", GA[0].positions - GA[0].argmax_bad, GA[0].positions, ga_min_margin, GA[0].worst_ratio,
               GA[0].worst_ours, GA[0].worst_torch, GA[0].worst_seq, GA[0].worst_app,
               f1 ? "FAILS as it must" : "DID NOT FAIL", GA[1].worst_ratio, GA[1].worst_ours, GA[1].worst_seq, GA[1].worst_app,
               GA[1].positions - GA[1].argmax_bad, GA[1].positions,
               f2 ? "FAILS as it must" : "DID NOT FAIL", GA[2].worst_ratio, GA[2].worst_ours, GA[2].worst_seq, GA[2].worst_app,
               GA[2].positions - GA[2].argmax_bad, GA[2].positions);
        if (!ok) bad = 1;
        printf("INFO forward logits vs e2e.logits (reported, not a gate): largest |ours - truth| %.3g, torch's %.3g, ratio %.3g\n",
               GA[0].logit_ours, GA[0].logit_torch, GA[0].logit_torch > 0 ? GA[0].logit_ours / GA[0].logit_torch : INFINITY);
    }
    {
        int ok = !gb_fail;
        for (int m = 0; m < 3; m++) ok &= gb_rows[m] > 0 && !gb_bad_logits[m] && !gb_bad_states[m];
        printf("%s forward gate B (chunks through the application-indexed cache = the whole sequence, BIT FOR BIT by memcmp, "
               "logits and every application's state): %s: %d positions, %d logit rows and %d state rows differ; %s: %d, %d, %d; "
               "%s: %d, %d, %d%s\n", ok ? "GREEN" : "FAIL", gb_name[0], gb_rows[0], gb_bad_logits[0], gb_bad_states[0],
               gb_name[1], gb_rows[1], gb_bad_logits[1], gb_bad_states[1], gb_name[2], gb_rows[2], gb_bad_logits[2],
               gb_bad_states[2], gb_fail ? " (A CHUNK FAILED TO RUN)" : "");
        if (!ok) bad = 1;
    }
    {
        int ok = rw_tries && rw_refused == rw_tries && rw_restart_eq == rw_restart_n;
        printf("%s forward rewind (a chunk at 0 < p0 < n_past on a used cache must be REFUSED with INK2_REWIND: the convolution "
               "rings hold the newest rows, not the rows before p0; a restart at p0 = 0 on the used cache must give the fresh "
               "cache's bits): refused %d of %d; restart whole and restart in two chunks equal to the fresh cache, by memcmp: "
               "%d of %d\n", ok ? "GREEN" : "FAIL", rw_refused, rw_tries, rw_restart_eq, rw_restart_n);
        if (!ok) bad = 1;
    }
    {
        // the attention and depth-read scale n ** -0.5 as the float32 operand: the engine's route (1 / sqrt in double,
        // no pow) against the CORRECTLY ROUNDED float32 of n ** -0.5, computed offline from a 60-digit decimal value
        // (2026-11-21), for the fixture's head and hidden sizes (8, 32) and real head sizes (96, 128). It cannot fail
        // on a libm whose pow is within an ulp (glibc's is); it pins the operand where one is not.
        static const struct { int n; float want; } SCL[] = { { 8, 0x1.6a09e6p-2f }, { 32, 0x1.6a09e6p-3f }, { 96, 0x1.a20bd8p-4f },
                                                            { 128, 0x1.6a09e6p-4f } };
        int nbad = 0; char txt[256] = "";
        for (int i = 0; i < 4; i++) {
            float got = ink2_inv_sqrt(SCL[i].n); nbad += memcmp(&got, &SCL[i].want, sizeof got) != 0;
            snprintf(txt + strlen(txt), sizeof txt - strlen(txt), "%s%d: %a", i ? ", " : "", SCL[i].n, (double)got);
        }
        printf("%s scale constants (n ** -0.5 as the float32 operand, by 1 / sqrt in double, no pow; against the correctly "
               "rounded float32): %d of 4 differ (%s)\n", nbad ? "FAIL" : "GREEN", nbad, txt);
        if (nbad) bad = 1;
    }
    {
        // the expert selection: ours vs the truth's at every router call, the near-tie branch, the planted cases
        const selstat *S = &SEL_P;
        int planted_ok = plant_cls[0] == 1 && plant_cls[1] == 2;
        int ok = S->pos && !S->hard_differ && !S->file_differ && planted_ok;
        char nt[160] = "";
        if (S->near_differ) snprintf(nt, sizeof nt, "; selection differs at a near-tie of %.3g at %d positions", S->near_gap_max, S->near_differ);
        printf("%s expert_selection (every mlp.gate call on its recorded input: our float32 choice vs the float64 truth's; the "
               "expert_layer, router and experts bounds hold GIVEN THE SAME SELECTION): %d positions, %d differ outside the "
               "bound%s; the harness's double selection vs the truth file's indices: %d differ; smallest gap between the last "
               "chosen and the first unchosen score %.3g; %d positions with the gap below the scores' own error bound (near-ties, "
               "where two correct engines may choose differently); planted (the truth's bias edited, ours on the recorded router): "
               "0.5x the bound apart %s (gap %.3g, bound %.3g), 10x apart %s (gap %.3g, bound %.3g)\n", ok ? "GREEN" : "FAIL",
               S->pos, S->hard_differ, nt, S->file_differ, S->min_gap, S->near_ties,
               plant_cls[0] == 1 ? "reports a near-tie, as it must" : "DID NOT REPORT A NEAR-TIE", plant_gap[0], plant_gapb[0],
               plant_cls[1] == 2 ? "FAILS, as it must" : "DID NOT FAIL", plant_gap[1], plant_gapb[1]);
        if (!ok) bad = 1;
    }
    for (int i = 0; i < 13; i++) {
        bstat *b = &SL[i]; b->piece = sl_name[i];
        int pert_ok = b->pert_mult > 1.0, ok = b->calls && !b->fail && !b->nob && !b->red && pert_ok;
        if (i == L_RES) ok = ok && res_bits_eq == res_bits_n;
        printf("%s %s (single step on its recorded inputs vs the float64 truth of the call, c = %.0f: %s): %d calls, %d failed, "
               "%d out of bound; largest fraction %.3g; torch %.3g%s; perturbation %.3g times the bound%s", ok ? "GREEN" : "FAIL",
               b->piece, CB, sl_what[i], b->calls, b->fail, b->nob, b->frac, b->torch_frac,
               b->torch_frac > 1.0 ? " (TORCH OUTSIDE THE BOUND)" : "", b->pert_mult, pert_ok ? "" : " (THE CONTROL DID NOT FAIL)");
        if (i == L_RES) printf("; BITS: rebuilt from the recorded operands = torch's recorded output at %d of %d applications", res_bits_eq, res_bits_n);
        printf("\n");
        if (!ok) bad = 1;
    }
    {
        printf("%s wiring (recorded tensors by memcmp, the reference's rules restated: hidden inputs, block-state growth with no add "
               "at a boundary, the pass restart, the MLP read's states and partial, the message read's inputs and the messages, the "
               "final read's inputs): %d tensors compared, %d differ%s%s\n", wire_n && !wire_bad ? "GREEN" : "FAIL", wire_n, wire_bad,
               wire_bad ? ", first: " : "", wire_bad ? wire_first : "");
        if (!wire_n || wire_bad) bad = 1;
    }
    {
        int ok = cov_calls && !cov_unnamed && !cov_uncovered;
        char per[2048] = "";
        for (int i = 0; i < cov_nlines; i++) snprintf(per + strlen(per), sizeof per - strlen(per), "%s%s %d", i ? ", " : "", cov_lines[i], cov_per_line[i]);
        printf("%s coverage: %d calls, every one in a single-step line except %d named (composite: model.layers.<b>, its own last "
               "step in the residual line, each child in its own line, its composition in forward gates A and B); steps that are not "
               "module calls, against the harness's own double, named: attention.core (eager_attention_forward), attention clips "
               "engaged (the clip steps on inputs x 64); %d unnamed "
               "(not in the table)%s%s, %d in the table but not checked%s%s; by line: %s\n", ok ? "GREEN" : "FAIL", cov_calls,
               cov_composite, cov_unnamed, cov_unnamed ? ", first: " : "", cov_first_unnamed, cov_uncovered,
               cov_uncovered ? ", first: " : "", cov_first_unc, per);
        if (!ok) bad = 1;
    }
    printf("outputs digest: %016llx\n", (unsigned long long)g_digest);
    printf("outputs digest (forward lines): %016llx\n", (unsigned long long)g_digest2);
    printf("outputs digest (single-step lines): %016llx\n", (unsigned long long)g_digest3);
    return bad;
}
