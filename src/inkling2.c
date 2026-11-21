// R4.48: Inkling 2 Mini (inkling2_loop), the text path: the structure as data, the application-indexed cache, and
// the pieces. See inkling2.h.
//
// This file is built STRICT in every build (the Makefile adds -fno-fast-math -ffp-contract=off for it): every
// reduction is a serial float32 loop in index order, nothing is contracted, and the transcendentals are pmath.h's
// portable p_expf and p_tanhf, so a strict build gives the same bits on every machine. Against the reference
// (torch's blocked BLAS order) the pieces are compared under the error bound the fixture harness states.
#include "inkling2.h"
#include "pmath.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ink2_shape_derive(ink2_shape *s, char *why, size_t why_n) {
#define INK2_REFUSE(...) do { if (why && why_n) snprintf(why, why_n, __VA_ARGS__); return -1; } while (0)
    if (s->n_block <= 0 || s->n_block > INK2_MAX_BLOCK) INK2_REFUSE("%d physical layers (1..%d)", s->n_block, INK2_MAX_BLOCK);
    if (s->attn_res_block <= 0) INK2_REFUSE("attention-residual block size %d", s->attn_res_block);
    if (s->loop_passes < 1) INK2_REFUSE("loop passes %d must be at least 1", s->loop_passes);
    if (s->loop_start <= 0) INK2_REFUSE("loop start %d must be greater than 0", s->loop_start);
    if (s->loop_start % s->attn_res_block || s->loop_end % s->attn_res_block)
        INK2_REFUSE("loop range [%d, %d) must be multiples of the block size %d", s->loop_start, s->loop_end,
                    s->attn_res_block);
    if (s->loop_end <= s->loop_start || s->loop_end > s->n_block)
        INK2_REFUSE("loop range [%d, %d) outside the %d layers", s->loop_start, s->loop_end, s->n_block);
    int n_app = s->n_block + (s->loop_passes - 1) * (s->loop_end - s->loop_start);
    if (n_app > INK2_MAX_APP) INK2_REFUSE("%d applications (at most %d)", n_app, INK2_MAX_APP);
    int a = 0;
    for (int b = 0; b < s->loop_start; b++) { s->app_block[a] = b; s->app_pass[a] = 0; a++; }
    for (int p = 0; p < s->loop_passes; p++)
        for (int b = s->loop_start; b < s->loop_end; b++) { s->app_block[a] = b; s->app_pass[a] = p; a++; }
    for (int b = s->loop_end; b < s->n_block; b++) { s->app_block[a] = b; s->app_pass[a] = 0; a++; }
    s->n_app = a;
    return 0;
#undef INK2_REFUSE
}

int ink2_cache_init(ink2_cache *c, const ink2_shape *s, int n_ctx) {
    memset(c, 0, sizeof *c);
    c->n_app = s->n_app; c->n_ctx = n_ctx;
    c->k = calloc((size_t)s->n_app, sizeof *c->k);
    c->v = calloc((size_t)s->n_app, sizeof *c->v);
    c->conv = calloc((size_t)s->n_app, sizeof *c->conv);
    c->conv_floats = calloc((size_t)s->n_app, sizeof *c->conv_floats);
    c->n_past = calloc((size_t)s->n_app, sizeof *c->n_past);
    if (!c->k || !c->v || !c->conv || !c->conv_floats || !c->n_past) { ink2_cache_free(c); return -1; }
    for (int a = 0; a < s->n_app; a++) {
        int g = ink2_app_global(s, a);
        size_t kv = (size_t)(g ? s->n_kv * s->head_dim : s->swa_n_kv * s->swa_head_dim);
        c->k[a] = calloc((size_t)n_ctx * kv, sizeof(float));
        c->v[a] = calloc((size_t)n_ctx * kv, sizeof(float));
        // the six short convolutions of a layer (attention input, MLP input, K, V, post-attention, post-MLP) keep
        // conv_k - 1 rows each; sized generously by the widest (hidden) until the attention piece fixes them
        c->conv_floats[a] = (size_t)6 * (size_t)(s->conv_k > 1 ? s->conv_k - 1 : 0) * (size_t)s->hidden;
        c->conv[a] = c->conv_floats[a] ? calloc(c->conv_floats[a], sizeof(float)) : NULL;
        if (!c->k[a] || !c->v[a] || (c->conv_floats[a] && !c->conv[a])) { ink2_cache_free(c); return -1; }
    }
    return 0;
}

void ink2_cache_free(ink2_cache *c) {
    for (int a = 0; a < c->n_app; a++) {
        if (c->k) free(c->k[a]);
        if (c->v) free(c->v[a]);
        if (c->conv) free(c->conv[a]);
    }
    free(c->k); free(c->v); free(c->conv); free(c->conv_floats); free(c->n_past);
    memset(c, 0, sizeof *c);
}

// x * rsqrt(mean(x^2) + eps), the reference's Inkling2LoopUnweightedRMSNorm: a serial sum of squares, the mean, the
// epsilon, 1/sqrt, one multiply an element
static void ink2_unweighted_norm(const float *x, int n, float eps, float *y) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r;
}

// The depth-wise attention of Inkling2LoopBlockAttentionResidual and Inkling2LoopMessageAttentionResidual at one
// position: states [n_kv][H] are normed (keys = values = the normed states), each of the nq queries scores them
// (dot, then times H^-0.5, as eager_attention_forward), a softmax over the n_kv states, the weighted sum.
static void ink2_depth_attn(const float *q, int nq, const float *states, int n_kv, int H, float *normed,
                            float *scores, float *out) {
    for (int j = 0; j < n_kv; j++) ink2_unweighted_norm(states + (size_t)j * H, H, 1e-6f, normed + (size_t)j * H);
    const float scale = (float)(1.0 / sqrt((double)H));   // H ** -0.5, rounded to float as torch's scalar is
    for (int i = 0; i < nq; i++) {
        const float *qi = q + (size_t)i * H;
        float mx = -INFINITY;
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

// ---- the pieces (a stub returns INK2_NOT_IMPLEMENTED until its piece lands)

// The softcap, tanh(v / cap) * cap: a true division by the float cap (torch's tensor / scalar), then the product
static inline float ink2_softcap(float v, float cap) { return p_tanhf(v / cap) * cap; }
// torch.sigmoid as 1 / (1 + exp(-x)), with the portable exp
static inline float ink2_sigmoid(float x) { return 1.0f / (1.0f + p_expf(-x)); }

// Inkling2LoopSparseMoeBlock at one block (model.layers.<b>.mlp), per position:
//   router: logits = W x (float32), scores = sigmoid(logits); the n_used experts with the largest scores + bias, TIES
//   BROKEN LOWEST INDEX FIRST (the reference's torch.topk(..., sorted=False) leaves ties open; this is our rule, stated
//   and tested by the harness's tie case); weights = score / (sum of the chosen scores + 1e-20) * route_scale, the sum
//   in ascending expert order;
//   experts in ASCENDING expert index (the reference's eager loop order): gate, up = W13 x; act = softcap(gate, gate
//   cap) * sigmoid(gate) * softcap(up, up cap); down = W2 act; out += down * weight.
ink2_status ink2_expert_layer(const ink2_shape *s, const ink2_weights *w, int block, int T,
                              const float *x, float *out) {
    char n1[160], n2[160], n3[160], n4[160];
    snprintf(n1, sizeof n1, "model.layers.%d.mlp.gate.weight", block);
    snprintf(n2, sizeof n2, "model.layers.%d.mlp.gate.e_score_correction_bias", block);
    snprintf(n3, sizeof n3, "model.layers.%d.mlp.experts.gate_up_proj", block);
    snprintf(n4, sizeof n4, "model.layers.%d.mlp.experts.down_proj", block);
    const ink2_tensor *wr = w->get(w->ctx, n1), *br = w->get(w->ctx, n2), *w13 = w->get(w->ctx, n3), *w2 = w->get(w->ctx, n4);
    int H = s->hidden, E = s->n_expert, K = s->n_used, F = s->ff;
    if (!wr || !br || !w13 || !w2 || wr->ne[0] != H || wr->ne[1] != E || br->ne[0] != E ||
        w13->ne[0] != H || w13->ne[1] != 2 * F || w13->ne[2] != E || w2->ne[0] != F || w2->ne[1] != H || w2->ne[2] != E ||
        K < 1 || K > E) return INK2_NO_TENSOR;
    float *sc = malloc(sizeof(float) * ((size_t)E + 2 * (size_t)F + (size_t)H));
    unsigned char *chosen = malloc((size_t)E);
    if (!sc || !chosen) { free(sc); free(chosen); return INK2_BAD_INPUT; }
    float *act = sc + E, *down = act + 2 * F;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * H;
        float *ot = out + (size_t)t * H;
        for (int e = 0; e < E; e++) {
            float l = 0.0f; const float *row = wr->data + (size_t)e * H;
            for (int h = 0; h < H; h++) l += row[h] * xt[h];
            sc[e] = ink2_sigmoid(l);
        }
        memset(chosen, 0, (size_t)E);
        for (int k = 0; k < K; k++) {
            int best = -1; float bv = 0.0f;
            for (int e = 0; e < E; e++) {
                if (chosen[e]) continue;
                float v = sc[e] + br->data[e];
                if (best < 0 || v > bv) { best = e; bv = v; }   // strict >: the lowest index wins a tie
            }
            chosen[best] = 1;
        }
        float sum = 0.0f;
        for (int e = 0; e < E; e++) if (chosen[e]) sum += sc[e];
        float den = sum + 1e-20f;
        for (int h = 0; h < H; h++) ot[h] = 0.0f;
        for (int e = 0; e < E; e++) {
            if (!chosen[e]) continue;
            float wt = sc[e] / den * s->route_scale;
            const float *g13 = w13->data + (size_t)e * 2 * F * H, *g2 = w2->data + (size_t)e * H * F;
            for (int j = 0; j < 2 * F; j++) {
                float a = 0.0f; const float *row = g13 + (size_t)j * H;
                for (int h = 0; h < H; h++) a += row[h] * xt[h];
                act[j] = a;
            }
            for (int j = 0; j < F; j++) {
                float g = act[j], u = act[F + j];
                act[j] = ink2_softcap(g, s->gate_cap) * ink2_sigmoid(g) * ink2_softcap(u, s->up_cap);
            }
            for (int h = 0; h < H; h++) {
                float a = 0.0f; const float *row = g2 + (size_t)h * F;
                for (int j = 0; j < F; j++) a += row[j] * act[j];
                down[h] = a;
            }
            for (int h = 0; h < H; h++) ot[h] += down[h] * wt;
        }
    }
    free(sc); free(chosen);
    return INK2_OK;
}

// y[o] = sum_i W[o][i] x[i], a serial dot in index order per output (nn.Linear without bias)
ink2_status ink2_attention(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                           const int *pos, const float *xq, const float *xk, const float *xv,
                           float *out, float *probs) {
    (void)s; (void)w; (void)c; (void)app; (void)T; (void)pos; (void)xq; (void)xk; (void)xv; (void)out; (void)probs;
    return INK2_NOT_IMPLEMENTED;
}

ink2_status ink2_attention_step(const ink2_shape *s, const ink2_weights *w, int app, const char *step, int T,
                                const int *pos, const float *const *in, float *out, float *out2) {
    (void)s; (void)w; (void)app; (void)step; (void)T; (void)pos; (void)in; (void)out; (void)out2;
    return INK2_NOT_IMPLEMENTED;
}

// The head: lm_head (built UNPADDED by the reference: vocab_used rows) then logit_softcap, tanh(l / cap) * cap
ink2_status ink2_head(const ink2_shape *s, const ink2_weights *w, int T, const float *x, float *logits) {
    const ink2_tensor *wo = w->get(w->ctx, "lm_head.weight");
    int H = s->hidden, V = s->vocab_used;
    if (!wo || wo->ne[0] != H || wo->ne[1] < V) return INK2_NO_TENSOR;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * H;
        for (int v = 0; v < V; v++) {                  // rows from vocab_used on, if a layout keeps them, never enter
            float a = 0.0f; const float *row = wo->data + (size_t)v * H;
            for (int h = 0; h < H; h++) a += row[h] * xt[h];
            logits[(size_t)t * V + v] = ink2_softcap(a, s->final_cap);
        }
    }
    return INK2_OK;
}

ink2_status ink2_block_read(const ink2_shape *s, const ink2_weights *w, const char *which, int T, int n_blocks,
                            const float *block_states, const float *partial, float *out, int *n_out) {
    char name[160];
    snprintf(name, sizeof name, "%s.query", which);
    const ink2_tensor *q = w->get(w->ctx, name);
    if (!q || q->ne[0] != s->hidden) return INK2_NO_TENSOR;
    int H = s->hidden, nq = q->n_dims == 1 ? 1 : (int)q->ne[1], nkv = n_blocks + 1;
    float *st = malloc(sizeof(float) * (size_t)nkv * H * 2 + sizeof(float) * (size_t)nkv + sizeof(float) * (size_t)nq * H);
    if (!st) return INK2_BAD_INPUT;
    float *normed = st + (size_t)nkv * H, *scores = normed + (size_t)nkv * H, *o = scores + nkv;
    for (int t = 0; t < T; t++) {
        memcpy(st, block_states + (size_t)t * n_blocks * H, sizeof(float) * (size_t)n_blocks * H);
        memcpy(st + (size_t)n_blocks * H, partial + (size_t)t * H, sizeof(float) * (size_t)H);
        ink2_depth_attn(q->data, nq, st, nkv, H, normed, scores, o);
        for (int i = 0; i < nq; i++) memcpy(out + ((size_t)i * T + t) * H, o + (size_t)i * H, sizeof(float) * (size_t)H);
    }
    free(st);
    if (n_out) *n_out = nq;
    return INK2_OK;
}

ink2_status ink2_message_input(const ink2_shape *s, const ink2_weights *w, int loop_layer, int T, int n_prev,
                               const float *prev_block_states, const float *x, float *out) {
    const ink2_tensor *q = w->get(w->ctx, "model.message_input.query");   // [loop layers][4][H]
    int H = s->hidden, nloop = s->loop_end - s->loop_start, nkv = n_prev + 1;
    if (!q || q->ne[0] != H || q->ne[1] != 4 || q->ne[2] != nloop || loop_layer < 0 || loop_layer >= nloop)
        return INK2_NO_TENSOR;
    const float *ql = q->data + (size_t)loop_layer * 4 * H;
    float *st = malloc(sizeof(float) * ((size_t)nkv * H * 2 + (size_t)nkv));
    if (!st) return INK2_BAD_INPUT;
    float *normed = st + (size_t)nkv * H, *scores = normed + (size_t)nkv * H;
    for (int t = 0; t < T; t++) {
        memcpy(st, prev_block_states + (size_t)t * n_prev * H, sizeof(float) * (size_t)n_prev * H);
        memcpy(st + (size_t)n_prev * H, x + (size_t)t * H, sizeof(float) * (size_t)H);
        ink2_depth_attn(ql, 4, st, nkv, H, normed, scores, out + (size_t)t * 4 * H);   // [T][4][H]
    }
    free(st);
    return INK2_OK;
}

ink2_status ink2_message_step(const ink2_shape *s, const ink2_weights *w, int block, const char *role, int T,
                              const float *x, const float *message, float *out) {
    char n1[160], n2[160], n3[160];
    snprintf(n1, sizeof n1, "model.layers.%d.%s_message_gate.down_proj.weight", block, role);
    snprintf(n2, sizeof n2, "model.layers.%d.%s_message_gate.forget_gate_proj.weight", block, role);
    snprintf(n3, sizeof n3, "model.layers.%d.%s_message_gate.message_gate_proj.weight", block, role);
    const ink2_tensor *wd = w->get(w->ctx, n1), *wf = w->get(w->ctx, n2), *wg = w->get(w->ctx, n3);
    int H = s->hidden, r = s->loop_gate_rank;
    if (!wd || !wf || !wg || wd->ne[0] != 2 * H || wd->ne[1] != r || wf->ne[0] != r || wf->ne[1] != H ||
        wg->ne[0] != r || wg->ne[1] != H) return INK2_NO_TENSOR;
    float *gh = malloc(sizeof(float) * ((size_t)2 * H + (size_t)r));
    if (!gh) return INK2_BAD_INPUT;
    float *d = gh + 2 * H;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * H, *mt = message + (size_t)t * H;
        ink2_unweighted_norm(xt, H, s->eps, gh);          // cat([norm(x), norm(message)])
        ink2_unweighted_norm(mt, H, s->eps, gh + H);
        for (int k = 0; k < r; k++) {
            float a = 0.0f; const float *row = wd->data + (size_t)k * 2 * H;
            for (int i = 0; i < 2 * H; i++) a += row[i] * gh[i];
            d[k] = a;
        }
        float *ot = out + (size_t)t * H;
        for (int h = 0; h < H; h++) {
            float fa = 0.0f, ga = 0.0f;
            const float *fr = wf->data + (size_t)h * r, *gr = wg->data + (size_t)h * r;
            for (int k = 0; k < r; k++) { fa += fr[k] * d[k]; ga += gr[k] * d[k]; }
            float f = p_tanhf(fa), g = p_tanhf(ga);
            ot[h] = (1.0f - f) * xt[h] + g * mt[h];
        }
    }
    free(gh);
    return INK2_OK;
}

