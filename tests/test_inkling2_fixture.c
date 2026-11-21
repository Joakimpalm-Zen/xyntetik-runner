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
// [n][T][d] -> [T][n][d] (the reference's per-head outputs to the piece's layout)
static void tr_ntd(const float *src, int n, int T, int d, float *dst) {
    for (int h = 0; h < n; h++) for (int t = 0; t < T; t++) memcpy(dst + ((size_t)t * n + h) * d, src + ((size_t)h * T + t) * d, sizeof(float) * d);
}
static void f2d(const float *x, size_t n, double *y, double *dy) { for (size_t i = 0; i < n; i++) { y[i] = x[i]; dy[i] = 0; } }
// got (float32) and torch (float32, may be NULL) against truth (float64) under bound; returns the fraction used
// The OUTPUTS DIGEST: FNV-1a 64 over the bytes of every array compared here (our pieces' outputs, the perturbed runs,
// and torch's constant dumps), in call order. Two builds of inkling2.c with the same digest gave the same bits on
// every piece output: the Runner-against-Runner gate (a strict build is the same on every machine and compiler).
static uint64_t g_digest = 1469598103934665603ULL;
static void digest(const void *p, size_t nbytes) {
    const unsigned char *b = p;
    for (size_t i = 0; i < nbytes; i++) { g_digest ^= b[i]; g_digest *= 1099511628211ULL; }
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
static double moe_harness_rel = 0, moe_tie_frac = 0, attn_harness_rel[2] = {0, 0}, attn_frac[2] = {0, 0};

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
    const char *seqs[] = { "s16", "s24" };
    bstat BM = {"expert_layer",0,0,0,0,0,0,-1,""}, BA = {"attention",0,0,0,0,0,0,-1,""}, SS[N_ASTEP], SC = {"clip",0,0,0,0,0,0,-1,""}, BH = {"head",0,0,0,0,0,0,-1,""};
    for (int i = 0; i < N_ASTEP; i++) { memset(&SS[i], 0, sizeof SS[i]); SS[i].piece = astep[i]; SS[i].pert_mult = -1; }
    bstat B[3] = { {"block_read",0,0,0,0,0,0,-1,""}, {"message_input",0,0,0,0,0,0,-1,""}, {"message_step",0,0,0,0,0,0,-1,""} };
    for (int q = 0; q < 2; q++) {
        st_file D, In;
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
                    ink2_status sst = ink2_attention_step(&s, &w, a, sn, Tn, P, in, c0, c1);
                    double sf = 0, stf = 0;
                    if (sst == INK2_OK) {
                        sf = frac_used(c0, y0, b0, n); if (n2) sf = fmax(sf, frac_used(c1, y1, b1, n2));
                        if (tdump && st_count(tdump) == n) {
                            const float *tf = (const float *)tdump->data;
                            // q_proj, k_proj, v_proj return [heads][T][hd]: to the piece's [T][heads][hd]
                            if (!strcmp(sn, "q_proj")) { tr_ntd(tf, nh, Tn, hd, tt); tf = tt; }
                            else if (!strcmp(sn, "k_proj") || !strcmp(sn, "v_proj")) { tr_ntd(tf, A.nkv, Tn, hd, tt); tf = tt; }
                            stf = frac_used(tf, y0, b0, n);
                            if (n2 && tdump2 && st_count(tdump2) == n2) stf = fmax(stf, frac_used((const float *)tdump2->data, y1, b1, n2));
                        } else sst = INK2_BAD_INPUT;
                    }
                    bstat_add(&SS[si], sst, sf, stf, -1);
                }
                // the clips never engage on the recorded data (head norms far below the thresholds): the three clip
                // steps once more on their recorded inputs times 64 (exact), so the clip branch runs; no torch dump here
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
            }
        }
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
        const bstat *b = &SS[i]; int ok = b->calls && !b->fail && !b->nob && !b->red, stub = BA.red > 0 && !b->calls;
        printf("%s attention.%s (one step on its recorded inputs, its own bound, vs the harness's double): %d calls, %d failed, "
               "%d out of bound; largest fraction %.3g; torch's float32 dump %.3g%s\n", b->red || stub ? "RED" : ok ? "GREEN" : "FAIL",
               b->piece, b->calls, b->fail, b->nob, b->frac, b->torch_frac, b->torch_frac > 1.0 ? " (TORCH OUTSIDE THE BOUND)" : "");
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
    printf("outputs digest: %016llx\n", (unsigned long long)g_digest);
    return bad;
}
