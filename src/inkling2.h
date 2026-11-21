// R4.48: Inkling 2 Mini (inkling2_loop), the text path. The internal structure as DATA and the signature of every
// piece of the forward, so the pieces can be written apart and joined without argument.
//
// The structure (loop range, pass count, block size, which applications are global) is read from the model's
// config or header, never written here: nothing below names 6, 18, 3, 24 or 48. A layer APPLICATION is one visit of
// a physical layer in forward order; the weights are indexed by physical layer (the passes share every tensor), and
// the cache, the convolution states and every per-application counter by application.
//
// The reference is transformers pull request 51050 at 6173a2f6 (modeling_inkling2_loop.py). Each piece below names
// the reference module it reproduces; the fixture harness (tests/test_inkling2_fixture.c) feeds a piece that
// module's recorded inputs and compares its outputs with the module's recorded outputs.
#ifndef INKLING2_H
#define INKLING2_H

#include <stddef.h>
#include <stdint.h>

#define INK2_MAX_BLOCK 64     // physical layers
#define INK2_MAX_APP   256    // layer applications

typedef enum { INK2_OK = 0, INK2_NOT_IMPLEMENTED = 1, INK2_BAD_INPUT = 2, INK2_NO_TENSOR = 3 } ink2_status;

typedef struct {
    // the loop, as read
    int n_block;              // physical layers (num_hidden_layers / block_count)
    int attn_res_block;       // n_layers_per_attn_res_block: a block state completes after layers b with (b+1) % it == 0
    int loop_start, loop_end; // looped physical layers [start, end)
    int loop_passes;          // total passes over the core (>= 1)
    int loop_gate_rank;
    unsigned char block_global[INK2_MAX_BLOCK];   // 1 = full ("hybrid"), 0 = sliding ("hybrid_sliding")
    // derived: the schedule of applications
    int n_app;
    int app_block[INK2_MAX_APP];  // physical layer of application a
    int app_pass[INK2_MAX_APP];   // 0 outside the core; 0..passes-1 inside
    // dimensions
    int hidden, vocab, vocab_used;
    int n_head, n_kv, head_dim;            // full layers
    int swa_n_head, swa_n_kv, swa_head_dim; // sliding layers
    int window, d_rel, rel_extent, conv_k;
    int n_expert, n_used, ff;
    int log_floor;
    float log_alpha, eps, qk_clip, rel_clip, route_scale, gate_cap, up_cap, final_cap;
} ink2_shape;

// Fills the derived schedule from the fields above and checks the reference's own rules (validate_architecture):
// start > 0, start and end multiples of attn_res_block, end <= n_block, passes >= 1, n_app <= INK2_MAX_APP.
// Returns 0, or -1 with a reason in why.
int ink2_shape_derive(ink2_shape *s, char *why, size_t why_n);
static inline int ink2_app_global(const ink2_shape *s, int a) { return s->block_global[s->app_block[a]]; }

// A tensor as the pieces see it: float32, row-major, ne[0] the fastest dimension (GGUF order). The fixture's reader
// and, later, the GGUF reader both hand pieces this; names are the reference's module paths for now.
typedef struct { const char *name; const float *data; int n_dims; int64_t ne[4]; } ink2_tensor;
typedef const ink2_tensor *(*ink2_lookup_fn)(void *ctx, const char *name);
typedef struct { ink2_lookup_fn get; void *ctx; } ink2_weights;

// The cache, indexed by APPLICATION. Each application owns its K and V rows and its convolution histories; two
// passes over one physical layer never share a slot.
typedef struct {
    int n_app, n_ctx;
    float **k, **v;          // [n_app] -> [n_ctx][n_kv * head_dim] (that application's layer type's sizes)
    float **conv;            // [n_app] -> the layer's short-convolution histories, (conv_k - 1) rows each
    size_t *conv_floats;     // [n_app]
    int *n_past;             // [n_app]
} ink2_cache;
int  ink2_cache_init(ink2_cache *c, const ink2_shape *s, int n_ctx);
void ink2_cache_free(ink2_cache *c);

// ---- the pieces. Each is a stub until it lands; every stub returns INK2_NOT_IMPLEMENTED, so the harness is red
// for it. T is the number of positions; x and out are [T][hidden] unless said otherwise.

// The expert layer: Inkling2LoopSparseMoeBlock (`model.layers.<b>.mlp`): router (sigmoid scores plus the gate bias,
// top n_used, normalized, times route_scale), the softcapped SwiGLU experts, their weighted sum.
ink2_status ink2_expert_layer(const ink2_shape *s, const ink2_weights *w, int block, int T,
                              const float *x, float *out);

// The attention internals: the layer's self_attn (`model.layers.<b>.self_attn`, Inkling2LoopSlidingAttention or
// Inkling2LoopFullAttention): Q, K, V projections with their norms, L2 clips and (full layers) the log-length scale,
// the short and (full layers) dynamic K and V convolutions, the hidden-conditioned relative bias, the per-token
// sinks, the gated output projection. Q, K and V each take their own input (the reference's hidden_states,
// key_hidden_states and value_hidden_states: three block reads). pos[T] are the positions; the cache slot is the
// application's. out is [T][hidden]; probs, if not NULL, gets the attention probabilities [n_head][T][keys], the
// reference's second output (the sink's column excluded, as the reference returns them).
ink2_status ink2_attention(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                           const int *pos, const float *xq, const float *xk, const float *xv,
                           float *out, float *probs);

// The head: `model.norm` output in, `lm_head` then `logit_softcap` out; logits are [T][vocab_used] (the padded rows
// never reach them).
ink2_status ink2_head(const ink2_shape *s, const ink2_weights *w, int T, const float *x, float *logits);

// The block-state read: Inkling2LoopBlockAttentionResidual (`model.layers.<b>.attention_input`, `.mlp_input`,
// `model.final_input`): a depth-wise attention over the completed block states [T][n_blocks][hidden] and the partial
// state [T][hidden], one output per query the module holds (3 for attention_input: Q, K and V inputs; 1 for
// mlp_input and final_input). `which` is the module path; out is [n_out][T][hidden], n_out set on return.
ink2_status ink2_block_read(const ink2_shape *s, const ink2_weights *w, const char *which, int T, int n_blocks,
                            const float *block_states, const float *partial, float *out, int *n_out);

// The cross-pass message read: Inkling2LoopMessageAttentionResidual (`model.message_input`), in passes after the
// first: the previous pass's block states [T][n_prev][hidden] and the current state [T][hidden] in, one message per
// role out, [T][4][hidden] (q, k, v, mlp), for the loop layer at core index `loop_layer`.
ink2_status ink2_message_input(const ink2_shape *s, const ink2_weights *w, int loop_layer, int T, int n_prev,
                               const float *prev_block_states, const float *x, float *out);

// The message step: Inkling2LoopMessageGate (`model.layers.<b>.<role>_message_gate`, role q, k, v or mlp) on a core
// layer in passes after the first: x and message are [T][hidden].
ink2_status ink2_message_step(const ink2_shape *s, const ink2_weights *w, int block, const char *role, int T,
                              const float *x, const float *message, float *out);

// The attention's sub-steps one at a time, on a sub-module's recorded inputs, for the fixture harness (the steps and
// their inputs are listed beside the definition in inkling2.c).
ink2_status ink2_attention_step(const ink2_shape *s, const ink2_weights *w, int app, const char *step, int T,
                                const int *pos, const float *const *in, float *out, float *out2);

#endif
