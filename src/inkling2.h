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
#define INK2_MAX_THREADS 64
#define INK2_MAX_HEAD_DIM 128
#define INK2_MAX_LORA_RANK 256

// INK2_REWIND: a call at 0 < p0 != the cache's n_past (a rewind into, or a gap past, what the cache holds): refused,
// see ink2_cache.
typedef enum { INK2_OK = 0, INK2_NOT_IMPLEMENTED = 1, INK2_BAD_INPUT = 2, INK2_NO_TENSOR = 3, INK2_REWIND = 4 } ink2_status;

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
// n ** -0.5 rounded to float32, without pow (the attention and depth-read scales; see the definition)
float ink2_inv_sqrt(int n);

// Threads over independent OUTPUTS (matvec rows, the experts' rows, the router's experts, vocabulary rows, attention
// heads, message-gate rows): each output is computed by the same serial loop at any thread count, so the bits do not
// depend on it. Model loads bind their house `tpool` onto the calling thread via ink2_set_tpool (CLI's m->tp or the
// server's shared_pool); standalone callers without a model_t fall back to ink2_set_threads / INK2_THREADS (default 1).
// INK2_PAR_MIN: the least work (multiply-adds) a chunk gets, default 16384 (1 forces every loop parallel, for tests).
struct tpool;
struct tpool *ink2_set_tpool(struct tpool *tp);
int ink2_set_threads(int n);
int ink2_get_threads(void);
// What the pool really did, for the tests: the parallel loops that split their range over more than one thread since
// the process began, and the threads (workers + the caller) the last of them ran on. A count of 1 splits nothing.
#define INK2_PAR_STATS 1
void ink2_par_stats(long *split_loops, int *last_pool);
// TEST-ONLY hook called on the caller thread each time ink2_par enters a parallel split; NULL disables.
void ink2_set_test_par_hook(void (*fn)(void *ctx), void *ctx);

// A tensor as the pieces see it: float32, row-major, ne[0] the fastest dimension (GGUF order). The fixture's reader
// and, later, the GGUF reader both hand pieces this; names are the reference's module paths for now.
//
// R4.48.15, the fast route's VIEW: q non-NULL means the tensor is also (or only) its rows as stored in the file, ggml
// type qtype, q_row_bytes apart, ne[0] values a row, and the engine's linear operations run on them with the house
// kernels (src/quants.h vec_dot / vec_dot_multi: float32 activations, the weights decoded in registers, a vectorized
// sum; threads split rows, never a sum). data may then be NULL. q_scale: NULL, or one factor per ne[2] (an expert
// stack's per-expert .scale), applied after the dot. q_act8: TEST-ONLY, the 8-bit control (each activation row
// quantized to 8 bits before the dot, as llama.cpp's CPU product does). A builder zeroes the struct first, so a
// provider that knows nothing of views serves none and the reference code runs.
typedef struct {
    const char *name; const float *data; int n_dims; int64_t ne[4];
    int qtype; const void *q; size_t q_row_bytes; const float *q_scale; int q_act8;
} ink2_tensor;
typedef const ink2_tensor *(*ink2_lookup_fn)(void *ctx, const char *name);
// OPTIONAL: one expert of an expert stack (`model.layers.<b>.mlp.experts.gate_up_proj` [E][2F][H] or `.down_proj`
// [E][H][F]), as a 2-D tensor (ne[0] the stack's ne[0], ne[1] its ne[1]): the same float32 values as row e of the
// stack `get` would return, without the stack. A provider that decodes on lookup offers it so a token touches only
// its chosen experts (a whole real stack is 2.7 GB in float32). NULL: the pieces take the stack from `get`.
// The pieces hold at most two expert results at a time (the expert's gate_up_proj, then its down_proj).
typedef const ink2_tensor *(*ink2_expert_fn)(void *ctx, const char *stack_name, int e);
typedef struct { ink2_lookup_fn get; void *ctx; ink2_expert_fn get_expert; } ink2_weights;

// The cache, indexed by APPLICATION. Each application owns its K and V rows and its convolution histories; two
// passes over one physical layer never share a slot.
// A call over positions p0..p0+T-1 reads the slot's K and V rows below p0 and its convolution histories (only when
// p0 > 0: a call at p0 = 0 is a prefill whatever the slot holds), and writes its own rows and histories; n_past is
// then p0 + T.
// A REWIND (0 < p0 < n_past: a shared prefix, then a different continuation) is REFUSED with INK2_REWIND, and so is
// a gap (p0 > n_past). The K and V rows of 0..p0-1 would still be right, but each convolution history is a ring of
// the NEWEST conv_k - 1 input rows, so after a rewind it holds rows at and past p0, not p0-K+1..p0-1. A real rewind
// will need, per application, the four histories as of p0: either saved with the rows (each history row kept per
// position, as K and V are, or a snapshot at each point a caller may rewind to), or recomputed from the inputs of
// positions p0-K+1..p0-1 (the K and V convolutions' inputs are their linear outputs, so the K/V projection inputs of
// those positions, or the linear outputs themselves, kept per position; the post convolutions' inputs are the
// attention output and the expert output of those positions).
typedef struct {
    int n_app, n_ctx;
    float **k, **v;          // [n_app] -> [n_ctx][n_kv * head_dim] (that application's layer type's sizes)
    float **conv;            // [n_app] -> the layer's four short-convolution histories, by the reference's conv_idx
                             // (0 k_proj.sconv, 1 v_proj.sconv, 2 post_attention_convolution, 3 post_mlp_convolution):
                             // each a ring of conv_k - 1 input rows, position u in row u % (conv_k - 1), rows as wide
                             // as the widest of the four
    size_t *conv_floats;     // [n_app]
    int *n_past;             // [n_app]
    int *rows;               // [n_app] rows each K/V buffer holds: n_ctx, or for a sliding application under
                             // ink2_cache_init_window, window - 1 + max_chunk
    int *k0;                 // [n_app] the position of buffer row 0 (0 for a full-size buffer)
    int max_chunk;           // 0: any T up to n_ctx; else the widest call the sliding buffers take
    int test_drop_key;       // TEST-ONLY (the cap's control, which must change the rows): the windowed path also
                             // drops the first key the mask admits; 0 in use
    float *work_fwd, *work_dl, *work_op;            // reusable forward/layer/op scratch; grown lazily, freed in ink2_cache_free
    size_t work_fwd_cap, work_dl_cap, work_op_cap;
    float *work_route;       // the MoE layer's routing tables over T positions (R4.48.20.4); as work_op
    size_t work_route_cap;
} ink2_cache;
int  ink2_cache_init(ink2_cache *c, const ink2_shape *s, int n_ctx);
// The same with each SLIDING application's K and V held for the window only: window - 1 + max_chunk rows that slide
// forward, instead of n_ctx (40 of Inkling 2 Mini's 48 applications: 0.5 GiB of K/V at any context instead of
// 2.5 GiB at 8K). The keys a call reads are the ones its mask admits, in the same order, so the outputs are the
// full-size cache's bit for bit. A call wider than max_chunk is refused (INK2_BAD_INPUT).
int  ink2_cache_init_window(ink2_cache *c, const ink2_shape *s, int n_ctx, int max_chunk);
// the bytes of K/V, convolution and bookkeeping the cache holds
size_t ink2_cache_bytes(const ink2_cache *c, const ink2_shape *s);
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
// key_hidden_states and value_hidden_states: three block reads). pos[T] are the positions, consecutive from p0 =
// pos[0]; the cache slot is the application's: K and V rows below p0 and the K and V convolution histories are read
// from it (p0 > 0), this call's rows and histories written to it. out is [T][hidden]; probs, if not NULL, gets the
// attention probabilities [n_head][T][p0 + T], the reference's second output (the sink's column excluded, as the
// reference returns them).
ink2_status ink2_attention(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                           const int *pos, const float *xq, const float *xk, const float *xv,
                           float *out, float *probs);

// The head: `model.norm` output in, `lm_head` then `logit_softcap` out; logits are [T][vocab_used] (the padded rows
// never reach them).
ink2_status ink2_head(const ink2_shape *s, const ink2_weights *w, int T, const float *x, float *logits);
// TEST-ONLY: cumulative rows evaluated by ink2_head (across all forwards in the process).
long ink2_head_rows(void);
// TEST-ONLY (R4.48.20.4): the MoE layer's allocations since the process began, its routing tables, the expert body's
// position lists and an instance's scratch growing; a forward after the first at the same width allocates none.
long ink2_moe_allocs(void);
// TEST-ONLY (R4.48.25.1): the decode fusion on or off within a process. RUNNER_INK2_DECODE_FUSE=0 turns it off, and is
// read once, at the engine's first use.
void ink2_set_decode_fuse(int on);
// TEST-ONLY: perturb position p0's post-embed hidden state by delta when p0 > 0, so split invariance tests can
// prove chunk boundaries are live.
void ink2_set_test_chunk_bump(float delta);

// The block-state read: Inkling2LoopBlockAttentionResidual (`model.layers.<b>.attention_input`, `.mlp_input`,
// `model.final_input`): a depth-wise attention over the completed block states [T][n_blocks][hidden] and the partial
// state [T][hidden], one output per query the module holds (3 for attention_input: Q, K and V inputs; 1 for
// mlp_input and final_input). `which` is the module path; out is [n_out][T][hidden], n_out set on return.
ink2_status ink2_block_read(const ink2_shape *s, const ink2_weights *w, const char *which, int T, int n_blocks,
                            const float *block_states, const float *partial, float *out, int *n_out);

// The expert layer's selection alone: sel [T][n_used], the chosen experts in ascending index, by the same router code
// ink2_expert_layer runs (ties lowest index first).
ink2_status ink2_expert_selection(const ink2_shape *s, const ink2_weights *w, int block, int T, const float *x, int *sel);

// The cross-pass message read: Inkling2LoopMessageAttentionResidual (`model.message_input`), in passes after the
// first: the previous pass's block states [T][n_prev][hidden] and the current state [T][hidden] in, one message per
// role out, [T][4][hidden] (q, k, v, mlp), for the loop layer at core index `loop_layer`.
ink2_status ink2_message_input(const ink2_shape *s, const ink2_weights *w, int loop_layer, int T, int n_prev,
                               const float *prev_block_states, const float *x, float *out);

// The message step: Inkling2LoopMessageGate (`model.layers.<b>.<role>_message_gate`, role q, k, v or mlp) on a core
// layer in passes after the first: x and message are [T][hidden].
ink2_status ink2_message_step(const ink2_shape *s, const ink2_weights *w, int block, const char *role, int T,
                              const float *x, const float *message, float *out);

// Whether physical layer `block` is a block boundary, as the reference decides it (is_block_boundary = layer_idx %
// n_layers_per_attn_res_block == 0). At a boundary the layer's INPUT joins the block states: inside the layer, before
// its MLP input reads them (and the attention output starts the new partial block, with no residual add); and in the
// model's loop, after the layer, for every later layer. Block states are per position: a token never reads
// another token's.
static inline int ink2_block_boundary(const ink2_shape *s, int block) { return block % s->attn_res_block == 0; }

// One decoder-layer application (`model.layers.<b>` at its visit `app`: Inkling2LoopSlidingDecoderLayer,
// FullDecoderLayer, SlidingLoopDecoderLayer or FullLoopDecoderLayer by the layer's type and the loop range), in the
// reference's order:
//   reads = attention_input(block_states, x) (1 query sliding, 3 full; layer 0 has none: x);
//   per role r of q, k, v: (messages given) r_message_gate(read_r, message_r); then input_layernorm (sliding, shared)
//     or r_input_layernorm (full), weighted;
//   a = post_attention_convolution(self_attn(Q, K, V inputs));
//   boundary: residual = a and the MLP reads block_states + [x]; otherwise residual = x + a, block_states;
//   m = mlp_input(those, residual); (messages) mlp_message_gate(m, message_mlp); post_attention_layernorm; mlp;
//   post_mlp_convolution; out = residual + that.
// block_states [T][n_blocks][hidden] are the states before this layer; messages [T][4][hidden] (q, k, v, mlp) are
// model.message_input's output in passes after the first on a loop layer, NULL otherwise. The caller grows its block
// states by x after the call when ink2_block_boundary(s, app_block[app]).
ink2_status ink2_decoder_layer(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                               const int *pos, const float *x, const float *block_states, int n_blocks,
                               const float *messages, float *out);

// The same, also returning the layer's own expert selection sel [T][n_used] (the harness's selection check).
ink2_status ink2_decoder_layer_sel(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                                   const int *pos, const float *x, const float *block_states, int n_blocks,
                                   const float *messages, float *out, int *sel);

// The whole text model on token ids, logits [T][vocab_used]: Inkling2LoopNormedEmbedding; the pre-loop layers; the
// first pass over the core; each later pass restarting from the pre-loop hidden and block states, its messages read
// from the PREVIOUS pass's block states (those after the one its first core layer appended) and final hidden state;
// the post-loop layers; final_input; norm; lm_head and the softcap. ink2_forward is one prefill on a fresh cache.
ink2_status ink2_forward(const ink2_shape *s, const ink2_weights *w, int T, const int *ids, float *logits);
// The same over positions p0..p0+T-1 on a cache that holds positions 0..p0-1 (one slot per application): a prefill
// at p0 = 0, a chunk or one token after. ink2_forward is this at p0 = 0: one code path. app_states, if not NULL, gets
// every application's output [n_app][T][hidden]. flags: 0, or the TEST-ONLY mis-wirings below (the harness's
// controls, which must fail its wiring gate).
#define INK2_MISWIRE_RESTART  1   // later passes restart from the input of the last pre-loop layer, not its output
#define INK2_MISWIRE_OWN_PASS 2   // later passes read their own pass's block states for the messages
#define INK2_LOGITS_LAST      4   // when logits != NULL, compute norm + head only for the final row (T - 1) into logits[0..vocab_used-1]
// Optional LoRA adapter callback on attention linear projections (slots 0=Q, 1=K, 2=V, 3=O).
// When NULL or slot returns false, the unadapted base forward runs untouched.
typedef struct {
    void *ctx;
    int (*slot)(void *ctx, int block, int slot, const float **a, const float **b,
                int *r, float *scale, int *n_in, int *n_out);
} ink2_lora;

ink2_status ink2_forward_chunk(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int p0, int T,
                               const int *ids, float *logits, int flags, float *app_states);
// Same forward with an optional LoRA adapter (`lora` may be NULL) and optional expert selection output.
// `logits` may be NULL to skip the final norm and lm_head pass (when caller recomputes head downstream).
ink2_status ink2_forward_chunk_lora(const ink2_shape *s, const ink2_weights *w, const ink2_lora *lora, ink2_cache *c,
                                    int p0, int T, const int *ids, float *logits, int flags, float *app_states,
                                    int *app_sel);

// Shared strict helpers from src/inkling2.c for src/inkling2_bwd.c.
float ink2_inv_sqrt(int n);
void  ink2_par(int64_t n, int64_t work, void (*fn)(void *ctx, int64_t i0, int64_t i1), void *ctx);
// TEST-ONLY (the first-light comparison): the same forward, also returning every application's expert selection
// app_sel [n_app][T][n_used] (each position's chosen experts in ascending index, from ink2_decoder_layer_sel: the
// router run again on the same MLP input, so nothing else changes). app_sel NULL is ink2_forward_chunk exactly.
ink2_status ink2_forward_chunk_sel(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int p0, int T,
                                   const int *ids, float *logits, int flags, float *app_states, int *app_sel);

// One operation of a layer or the model on recorded inputs, for the fixture harness's single-step lines (the ops and
// their layouts are listed beside the definition in inkling2.c); the same code the forward runs.
ink2_status ink2_op_step(const ink2_shape *s, const ink2_weights *w, const char *module, const char *op, int n, int in_d,
                         const float *const *in, const int *iin, float *out, float *out2, int *iout);

// The attention's sub-steps one at a time, on a sub-module's recorded inputs, for the fixture harness (the steps and
// their inputs are listed beside the definition in inkling2.c).
ink2_status ink2_attention_step(const ink2_shape *s, const ink2_weights *w, int app, const char *step, int T,
                                const int *pos, const float *const *in, float *out, float *out2);

// TEST-ONLY isolated relative-branch activation probe (inert unless RUNNER_TEST_INK2_REL_PROBE is set).
typedef struct {
    int app;
    int T, H;
    float eps;
    const float *d;
    float *xq_in;
    float *g_rel;
} ink2_rel_probe;
void ink2_set_rel_probe(ink2_rel_probe *p);
ink2_rel_probe *ink2_active_rel_probe(void);

// TEST-ONLY (R4.48.25, tests/test_inkling2_vmath.c): the fast route's own vector kernels, run as the route runs them,
// for the operation-level gate.
// - ink2_test_vmath_form: the form this build takes, "avx2", "neon" or "scalar" (a build without either takes
//   pmath's scalar forms on the fast route too).
// - ink2_test_vmath: y[i] = exp(x[i]) (fn INK2_VM_EXP) or tanh(x[i]) (fn INK2_VM_TANH), in whole vector blocks and
//   a scalar tail, as the route's callers take them. Returns 0, or -1 for an unknown fn.
// - ink2_test_swiglu: the SwiGLU of n rows of act [n][2F] (the gate's F values then the up's), in place, through the
//   route's ink2_swiglu_range, with the given softcaps.
// - ink2_test_swiglu_ref (R4.48.26): the reference route's scalar SwiGLU (ink2_swiglu) of one gate and up value.
// - ink2_test_attn_core: one query against nk keys of one head (hd wide, keys and values nk x hd, a sink logit, no
//   relative bias, every key admitted), through the fast core (fast = 1) or the reference core (fast = 0); out [hd].
enum { INK2_VM_EXP = 0, INK2_VM_TANH = 1 };
const char *ink2_test_vmath_form(void);
int ink2_test_vmath(int fn, const float *x, float *y, int n);
void ink2_test_swiglu(float gate_cap, float up_cap, float *act, int F, int n);
float ink2_test_swiglu_ref(float gate_cap, float up_cap, float g, float u);
ink2_status ink2_test_attn_core(int fast, int hd, int nk, const float *q, const float *k, const float *v, float sink,
                                float *out);

#endif
