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
// ---- the pieces (a stub returns INK2_NOT_IMPLEMENTED until its piece lands)

ink2_status ink2_expert_layer(const ink2_shape *s, const ink2_weights *w, int block, int T,
                              const float *x, float *out) {
    (void)s; (void)w; (void)block; (void)T; (void)x; (void)out;
    return INK2_NOT_IMPLEMENTED;
}

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

ink2_status ink2_head(const ink2_shape *s, const ink2_weights *w, int T, const float *x, float *logits) {
    (void)s; (void)w; (void)T; (void)x; (void)logits;
    return INK2_NOT_IMPLEMENTED;
}

ink2_status ink2_block_read(const ink2_shape *s, const ink2_weights *w, const char *which, int T, int n_blocks,
                            const float *block_states, const float *partial, float *out, int *n_out) {
    (void)s; (void)w; (void)which; (void)T; (void)n_blocks; (void)block_states; (void)partial; (void)out;
    if (n_out) *n_out = 0;
    return INK2_NOT_IMPLEMENTED;
}

ink2_status ink2_message_input(const ink2_shape *s, const ink2_weights *w, int loop_layer, int T, int n_prev,
                               const float *prev_block_states, const float *x, float *out) {
    (void)s; (void)w; (void)loop_layer; (void)T; (void)n_prev; (void)prev_block_states; (void)x; (void)out;
    return INK2_NOT_IMPLEMENTED;
}

ink2_status ink2_message_step(const ink2_shape *s, const ink2_weights *w, int block, const char *role, int T,
                              const float *x, const float *message, float *out) {
    (void)s; (void)w; (void)block; (void)role; (void)T; (void)x; (void)message; (void)out;
    return INK2_NOT_IMPLEMENTED;
}
