// R4.48.6: the inkling2_loop GGUF as a weight source for src/inkling2.c (see inkling2_gguf.h).
#include "inkling2_gguf.h"
#include "fp16.h"
#include "fpclass.h"
#include "gguf.h"
#include "quants.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OP_RENAME, OP_UNPAD, OP_MSGQ, OP_STATIC, OP_TRANSPOSE };

typedef struct {
    char name[160];      // the reference name
    ink2_tensor t;       // served; data owned here, decoded on first lookup and kept
    float *data;
} slot_t;

typedef struct {
    gguf_tensor *w, *scale;   // the stack and, for NVFP4, its per-expert factor
    char kind;                // 'N', 'B', 'F'
    int64_t rows, cols;
} ink2_stack_t;

struct ink2g {
    gguf_file own;    // the file when ink2g_open opened it
    gguf_file *fp;    // the file read: &own, or the caller's (ink2g_attach)
    int open;
    ink2_shape s;
    slot_t **slots;   // each slot its own allocation: a served ink2_tensor never moves
    int n_slots, cap_slots;
    ink2_stack_t gu[INK2_MAX_BLOCK], dn[INK2_MAX_BLOCK];
    float *ring[2];   // get_expert's results: the pieces hold at most two
    size_t ring_floats;
    int ring_next;
    int ctrl;
    int fast, act8;   // R4.48.15: serve the matrices and the stacks as views (ink2g_set_route)
    ink2_tensor ring_t[2];
    float e2m1[16], e4m3[256];
};

static int refuse(char *why, size_t wn, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why, wn, fmt, ap);
    va_end(ap);
    return -1;
}

// ---------------------------------------------------------------- metadata
#define KEY(k) "inkling2_loop." k

static int get_u32(gguf_file *f, const char *k, int *out, char *why, size_t wn) {
    gguf_kv *kv = gguf_get(f, k);
    if (!kv || (kv->type != GGUF_T_U32 && kv->type != GGUF_T_I32 && kv->type != GGUF_T_U64))
        return refuse(why, wn, "this inkling2_loop file has no integer key %s, which the loop needs", k);
    *out = (int)kv->v.u64;
    return 0;
}
static int get_f32(gguf_file *f, const char *k, float *out, char *why, size_t wn) {
    gguf_kv *kv = gguf_get(f, k);
    if (!kv || kv->type != GGUF_T_F32) return refuse(why, wn, "this inkling2_loop file has no float key %s", k);
    *out = gguf_get_f32(f, k, 0.0f);
    return 0;
}
static int arr_i32(gguf_file *f, const char *k, int n, int *out, char *why, size_t wn) {
    gguf_kv *kv = gguf_get(f, k);
    if (!kv || kv->type != GGUF_T_ARR || (int64_t)kv->arr_n != n || !kv->arr_raw ||
        (kv->arr_type != GGUF_T_I32 && kv->arr_type != GGUF_T_U32))
        return refuse(why, wn, "%s is not an integer array of %d layers", k, n);
    for (int i = 0; i < n; i++) memcpy(&out[i], (const char *)kv->arr_raw + 4 * (size_t)i, 4);
    return 0;
}

static int read_shape(gguf_file *f, ink2_shape *s, char *why, size_t wn) {
    memset(s, 0, sizeof *s);
    const char *arch = gguf_get_str(f, "general.architecture", "");
    if (strcmp(arch, "inkling2_loop"))
        return refuse(why, wn, "general.architecture is '%s', not inkling2_loop", arch);
    int vary = gguf_get_bool(f, KEY("loop.varying"), false);
    if (vary) return refuse(why, wn, "a varying loop (loop.varying true) is not supported by this engine");
    if (get_u32(f, KEY("block_count"), &s->n_block, why, wn) || get_u32(f, KEY("attn_res.block_size"), &s->attn_res_block, why, wn) ||
        get_u32(f, KEY("loop.start"), &s->loop_start, why, wn) || get_u32(f, KEY("loop.end"), &s->loop_end, why, wn) ||
        get_u32(f, KEY("loop.count"), &s->loop_passes, why, wn) || get_u32(f, KEY("loop.gate_rank"), &s->loop_gate_rank, why, wn) ||
        get_u32(f, KEY("embedding_length"), &s->hidden, why, wn) || get_u32(f, KEY("vocab_size"), &s->vocab, why, wn) ||
        get_u32(f, KEY("vocab_size_unpadded"), &s->vocab_used, why, wn) ||
        get_u32(f, KEY("attention.key_length"), &s->head_dim, why, wn) ||
        get_u32(f, KEY("attention.key_length_swa"), &s->swa_head_dim, why, wn) ||
        get_u32(f, KEY("attention.sliding_window"), &s->window, why, wn) ||
        get_u32(f, KEY("attention.relative_length"), &s->d_rel, why, wn) ||
        get_u32(f, KEY("attention.relative_extent"), &s->rel_extent, why, wn) ||
        get_u32(f, KEY("shortconv.l_cache"), &s->conv_k, why, wn) || get_u32(f, KEY("expert_count"), &s->n_expert, why, wn) ||
        get_u32(f, KEY("expert_used_count"), &s->n_used, why, wn) ||
        get_u32(f, KEY("expert_feed_forward_length"), &s->ff, why, wn) ||
        get_u32(f, KEY("attention.log_scale_floor"), &s->log_floor, why, wn) ||
        get_f32(f, KEY("attention.log_scale_alpha"), &s->log_alpha, why, wn) ||
        get_f32(f, KEY("attention.layer_norm_rms_epsilon"), &s->eps, why, wn) ||
        get_f32(f, KEY("attention.qk_clip"), &s->qk_clip, why, wn) ||
        get_f32(f, KEY("attention.relative_clip"), &s->rel_clip, why, wn) ||
        get_f32(f, KEY("expert_weights_scale"), &s->route_scale, why, wn) ||
        get_f32(f, KEY("activation.situ_beta"), &s->gate_cap, why, wn) ||
        get_f32(f, KEY("activation.situ_linear_beta"), &s->up_cap, why, wn) ||
        get_f32(f, KEY("final_logit_softcapping"), &s->final_cap, why, wn))
        return -1;
    if (s->n_block < 1 || s->n_block > INK2_MAX_BLOCK) return refuse(why, wn, "block_count %d is outside 1..%d", s->n_block, INK2_MAX_BLOCK);
    if (s->n_expert < 1) return refuse(why, wn, "expert_count %d must be at least 1", s->n_expert);
    if (s->n_used < 1 || s->n_used > s->n_expert)
        return refuse(why, wn, "expert_used_count %d is outside 1..%d (expert_count)", s->n_used, s->n_expert);
    if (s->head_dim < 1 || s->head_dim > INK2_MAX_HEAD_DIM)
        return refuse(why, wn, "head_dim %d exceeds supported limit %d", s->head_dim, INK2_MAX_HEAD_DIM);
    if (s->swa_head_dim < 1 || s->swa_head_dim > INK2_MAX_HEAD_DIM)
        return refuse(why, wn, "swa_head_dim %d exceeds supported limit %d", s->swa_head_dim, INK2_MAX_HEAD_DIM);
    if (s->d_rel < 1 || s->d_rel > INK2_MAX_HEAD_DIM)
        return refuse(why, wn, "relative_length %d exceeds supported limit %d", s->d_rel, INK2_MAX_HEAD_DIM);
    gguf_kv *pat = gguf_get(f, KEY("attention.sliding_window_pattern"));
    if (!pat || pat->type != GGUF_T_ARR || pat->arr_type != GGUF_T_BOOL || (int)pat->arr_n != s->n_block || !pat->arr_raw)
        return refuse(why, wn, "attention.sliding_window_pattern is not a bool array of %d layers", s->n_block);
    int nh[INK2_MAX_BLOCK], nkv[INK2_MAX_BLOCK];
    if (arr_i32(f, KEY("attention.head_count"), s->n_block, nh, why, wn) ||
        arr_i32(f, KEY("attention.head_count_kv"), s->n_block, nkv, why, wn))
        return -1;
    int have_full = 0, have_swa = 0;
    for (int b = 0; b < s->n_block; b++) {
        const int swa = ((const unsigned char *)pat->arr_raw)[b] != 0;
        s->block_global[b] = (unsigned char)!swa;
        int *h = swa ? &s->swa_n_head : &s->n_head, *k = swa ? &s->swa_n_kv : &s->n_kv, *seen = swa ? &have_swa : &have_full;
        if (*seen && (*h != nh[b] || *k != nkv[b]))
            return refuse(why, wn, "layer %d's head counts (%d, %d) differ from its layer type's (%d, %d)", b, nh[b], nkv[b], *h, *k);
        *h = nh[b]; *k = nkv[b]; *seen = 1;
    }
    if (!have_full || !have_swa) return refuse(why, wn, "the layer pattern needs both full and sliding layers");
    return ink2_shape_derive(s, why, wn);
}

// ---------------------------------------------------------------- the name map
static float e4m3fn(unsigned b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
    return (b & 0x80) ? -v : v;
}

static int is_loop_layer(const ink2_shape *s, int b) { return b >= s->loop_start && b < s->loop_end; }

// the reference's torch dims (outer first) for a name, or 0 for a name the model does not have
static int ref_dims(const ink2_shape *s, const char *name, int64_t *d) {
    const int64_t H = s->hidden, E = s->n_expert, F = s->ff, K = s->conv_k;
    if (!strcmp(name, "model.embed_tokens.weight")) { d[0] = s->vocab; d[1] = H; return 2; }
    if (!strcmp(name, "lm_head.weight")) { d[0] = s->vocab_used; d[1] = H; return 2; }
    if (!strcmp(name, "model.embed_tokens.embed_norm.weight") || !strcmp(name, "model.norm.weight") ||
        !strcmp(name, "model.final_input.query")) { d[0] = H; return 1; }
    if (!strcmp(name, "model.message_input.query")) { d[0] = s->loop_end - s->loop_start; d[1] = 4; d[2] = H; return 3; }
    int b, n = 0;
    if (sscanf(name, "model.layers.%d.%n", &b, &n) != 1 || n == 0 || b < 0 || b >= s->n_block) return 0;
    const char *x = name + n;
    const int full = s->block_global[b];
    const int64_t nh = full ? s->n_head : s->swa_n_head, nkv = full ? s->n_kv : s->swa_n_kv, hd = full ? s->head_dim : s->swa_head_dim;
#define IS(str) (!strcmp(x, str))
    if (IS("self_attn.q_proj.linear.weight") || IS("self_attn.o_proj.gate_proj.weight")) { d[0] = nh * hd; d[1] = H; return 2; }
    if (IS("self_attn.q_proj.norm.scale")) { d[0] = nh; d[1] = hd; return 2; }
    if (IS("self_attn.k_proj.linear.weight") || IS("self_attn.v_proj.linear.weight")) { d[0] = nkv * hd; d[1] = H; return 2; }
    if (!full && (IS("self_attn.k_proj.sconv.conv1d.weight") || IS("self_attn.v_proj.sconv.conv1d.weight"))) { d[0] = nkv * hd; d[1] = 1; d[2] = K; return 3; }
    if (full && (IS("self_attn.k_proj.sconv.static_weight") || IS("self_attn.v_proj.sconv.static_weight"))) { d[0] = K; d[1] = nkv * hd; return 2; }
    if (full && (IS("self_attn.k_proj.sconv.weight_proj.weight") || IS("self_attn.v_proj.sconv.weight_proj.weight"))) { d[0] = nkv * K; d[1] = H; return 2; }
    if (IS("self_attn.o_proj.linear.weight")) { d[0] = H; d[1] = nh * hd; return 2; }
    if (IS("self_attn.sink_proj.weight")) { d[0] = nh; d[1] = H; return 2; }
    if (IS("self_attn.relative_position_bias.proj")) { d[0] = s->d_rel; d[1] = s->rel_extent; return 2; }
    if (IS("self_attn.relative_position_bias.linear.weight")) { d[0] = nh * s->d_rel; d[1] = H; return 2; }
    if (IS("self_attn.relative_position_bias.norm.scale")) { d[0] = nh; d[1] = s->d_rel; return 2; }
    if (IS("mlp.gate.weight")) { d[0] = E; d[1] = H; return 2; }
    if (IS("mlp.gate.e_score_correction_bias")) { d[0] = E; return 1; }
    if (IS("mlp.experts.gate_up_proj")) { d[0] = E; d[1] = 2 * F; d[2] = H; return 3; }
    if (IS("mlp.experts.down_proj")) { d[0] = E; d[1] = H; d[2] = F; return 3; }
    if ((!full && IS("input_layernorm.weight")) || IS("post_attention_layernorm.weight") || IS("mlp_input.query") ||
        (full && (IS("q_input_layernorm.weight") || IS("k_input_layernorm.weight") || IS("v_input_layernorm.weight")))) { d[0] = H; return 1; }
    if (IS("post_attention_convolution.conv1d.weight") || IS("post_mlp_convolution.conv1d.weight")) { d[0] = H; d[1] = 1; d[2] = K; return 3; }
    if (b >= 1 && IS("attention_input.query")) { if (full) { d[0] = 3; d[1] = H; return 2; } d[0] = H; return 1; }
    if (is_loop_layer(s, b)) {
        static const char *roles[4] = { "q", "k", "v", "mlp" };
        for (int r = 0; r < 4; r++) {
            char a[64];
            snprintf(a, sizeof a, "%s_message_gate.down_proj.weight", roles[r]);
            if (IS(a)) { d[0] = s->loop_gate_rank; d[1] = 2 * H; return 2; }
            snprintf(a, sizeof a, "%s_message_gate.forget_gate_proj.weight", roles[r]);
            if (IS(a)) { d[0] = H; d[1] = s->loop_gate_rank; return 2; }
            snprintf(a, sizeof a, "%s_message_gate.message_gate_proj.weight", roles[r]);
            if (IS(a)) { d[0] = H; d[1] = s->loop_gate_rank; return 2; }
        }
    }
#undef IS
    return 0;
}

// the GGUF tensor and op for a reference name (name already known to ref_dims)
static int gguf_name(const ink2_shape *s, const char *name, char *out, size_t on, int *op) {
    *op = OP_RENAME;
    static const struct { const char *ref, *gg; } G[] = {
        { "model.embed_tokens.weight", "token_embd.weight" }, { "model.embed_tokens.embed_norm.weight", "token_embd_norm.weight" },
        { "model.norm.weight", "output_norm.weight" }, { "model.final_input.query", "output_res_query.weight" },
    };
    for (size_t i = 0; i < sizeof G / sizeof G[0]; i++)
        if (!strcmp(name, G[i].ref)) { snprintf(out, on, "%s", G[i].gg); return 0; }
    if (!strcmp(name, "lm_head.weight")) { *op = OP_UNPAD; snprintf(out, on, "output.weight"); return 0; }
    if (!strcmp(name, "model.message_input.query")) { *op = OP_MSGQ; out[0] = 0; return 0; }
    int b, n = 0;
    if (sscanf(name, "model.layers.%d.%n", &b, &n) != 1 || n == 0) return -1;
    const char *x = name + n;
    static const struct { const char *ref, *gg; int op; } L[] = {
        { "self_attn.q_proj.linear.weight", "attn_q", OP_RENAME }, { "self_attn.q_proj.norm.scale", "attn_q_clip", OP_RENAME },
        { "self_attn.k_proj.linear.weight", "attn_k", OP_RENAME }, { "self_attn.v_proj.linear.weight", "attn_v", OP_RENAME },
        { "self_attn.k_proj.sconv.conv1d.weight", "attn_k_conv", OP_RENAME }, { "self_attn.v_proj.sconv.conv1d.weight", "attn_v_conv", OP_RENAME },
        { "self_attn.k_proj.sconv.static_weight", "attn_k_conv", OP_STATIC }, { "self_attn.v_proj.sconv.static_weight", "attn_v_conv", OP_STATIC },
        { "self_attn.k_proj.sconv.weight_proj.weight", "attn_k_conv_proj", OP_RENAME },
        { "self_attn.v_proj.sconv.weight_proj.weight", "attn_v_conv_proj", OP_RENAME },
        { "self_attn.o_proj.gate_proj.weight", "attn_gate", OP_RENAME }, { "self_attn.o_proj.linear.weight", "attn_output", OP_RENAME },
        { "self_attn.sink_proj.weight", "attn_sink_proj", OP_RENAME },
        { "self_attn.relative_position_bias.proj", "attn_rel_bias", OP_TRANSPOSE },
        { "self_attn.relative_position_bias.linear.weight", "attn_rel", OP_RENAME },
        { "self_attn.relative_position_bias.norm.scale", "attn_rel_clip", OP_RENAME },
        { "mlp.gate.weight", "ffn_gate_inp", OP_RENAME }, { "input_layernorm.weight", "attn_norm", OP_RENAME },
        { "post_attention_layernorm.weight", "ffn_norm", OP_RENAME },
        { "post_attention_convolution.conv1d.weight", "attn_post_conv", OP_RENAME },
        { "post_mlp_convolution.conv1d.weight", "ffn_post_conv", OP_RENAME },
        { "attention_input.query", "attn_res_query", OP_RENAME }, { "mlp_input.query", "ffn_res_query", OP_RENAME },
        { "q_input_layernorm.weight", "attn_q_inp_norm", OP_RENAME }, { "k_input_layernorm.weight", "attn_k_inp_norm", OP_RENAME },
        { "v_input_layernorm.weight", "attn_v_inp_norm", OP_RENAME },
        { "q_message_gate.down_proj.weight", "loop_q_ctrl_down", OP_RENAME }, { "q_message_gate.forget_gate_proj.weight", "loop_q_ctrl_forget", OP_RENAME },
        { "q_message_gate.message_gate_proj.weight", "loop_q_ctrl_gain", OP_RENAME },
        { "k_message_gate.down_proj.weight", "loop_k_ctrl_down", OP_RENAME }, { "k_message_gate.forget_gate_proj.weight", "loop_k_ctrl_forget", OP_RENAME },
        { "k_message_gate.message_gate_proj.weight", "loop_k_ctrl_gain", OP_RENAME },
        { "v_message_gate.down_proj.weight", "loop_v_ctrl_down", OP_RENAME }, { "v_message_gate.forget_gate_proj.weight", "loop_v_ctrl_forget", OP_RENAME },
        { "v_message_gate.message_gate_proj.weight", "loop_v_ctrl_gain", OP_RENAME },
        { "mlp_message_gate.down_proj.weight", "loop_ffn_ctrl_down", OP_RENAME },
        { "mlp_message_gate.forget_gate_proj.weight", "loop_ffn_ctrl_forget", OP_RENAME },
        { "mlp_message_gate.message_gate_proj.weight", "loop_ffn_ctrl_gain", OP_RENAME },
    };
    if (!strcmp(x, "mlp.gate.e_score_correction_bias")) { snprintf(out, on, "blk.%d.exp_probs_b.bias", b); return 0; }
    for (size_t i = 0; i < sizeof L / sizeof L[0]; i++)
        if (!strcmp(x, L[i].ref)) { snprintf(out, on, "blk.%d.%s.weight", b, L[i].gg); *op = L[i].op; return 0; }
    (void)s;
    return -1;
}

static int64_t nelem(const gguf_tensor *t) { return (int64_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]); }

// a type this loader decodes for a dense tensor or a stack: F32 and BF16 exactly; any other ggml type Runner's own
// dequantizer takes (src/quants.c dequant_row), each gated against ggml's dequantize; NVFP4 only as an expert stack
static int dense_type_ok(uint32_t type) {
    return type == T_F32 || type == T_BF16 || (type != T_NVFP4 && ggml_type_supported((int)type));
}

// elements off .. off+n-1 of t as float32 (off and n whole blocks for a block type)
static int decode_float(const gguf_tensor *t, int64_t off, int64_t n, float *dst) {
    if (t->type == T_F32) { memcpy(dst, (const float *)t->data + off, sizeof(float) * (size_t)n); return 0; }
    if (t->type == T_BF16) {
        const uint16_t *h = (const uint16_t *)t->data + off;
        for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_f32(h[i]);
        return 0;
    }
    if (t->type == T_NVFP4 || !ggml_type_supported((int)t->type)) return -1;
    const int bs = ggml_block_size((int)t->type);
    if (bs < 1 || off % bs || n % bs) return -1;
    const size_t ts = ggml_type_size((int)t->type);
    const unsigned char *src = (const unsigned char *)t->data + (size_t)(off / bs) * ts;
    // row by row in whole blocks: dequant_row's n is an int
    const int64_t step = (int64_t)bs * 65536;
    for (int64_t k = 0; k < n; k += step) {
        const int64_t m = n - k < step ? n - k : step;
        dequant_row((int)t->type, src + (size_t)(k / bs) * ts, dst + k, (int)m);
    }
    return 0;
}

// the stored tensor's elements, squeezing size-1 dims, must equal the reference's (op: transposed or not)
static int check_tensor(const ink2_shape *s, ink2g *g, const char *ref, char *why, size_t wn) {
    int64_t d[4] = { 1, 1, 1, 1 };
    const int nd = ref_dims(s, ref, d);
    if (!nd) return refuse(why, wn, "%s is not a parameter of this model", ref);
    char gn[160];
    int op;
    if (gguf_name(s, ref, gn, sizeof gn, &op)) return refuse(why, wn, "%s has no GGUF counterpart", ref);
    if (op == OP_MSGQ) {
        for (int b = s->loop_start; b < s->loop_end; b++) {
            snprintf(gn, sizeof gn, "blk.%d.loop_msg_query.weight", b);
            gguf_tensor *t = gguf_find_tensor(g->fp, gn);
            if (!t) return refuse(why, wn, "the file has no %s, which the loop's messages need", gn);
            if (t->ne[0] != (uint64_t)s->hidden || t->ne[1] != 4 || t->ne[2] * t->ne[3] != 1)
                return refuse(why, wn, "%s is %llux%llu, not %dx4", gn, (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], s->hidden);
            if (!dense_type_ok(t->type)) return refuse(why, wn, "%s is type %u, which this loader does not decode", gn, t->type);
        }
        return 0;
    }
    gguf_tensor *t = gguf_find_tensor(g->fp, gn);
    if (!t) return refuse(why, wn, "the file has no %s (the reference's %s)", gn, ref);
    if (!dense_type_ok(t->type))
        return refuse(why, wn, "%s is type %u, which this loader does not decode", gn, t->type);
    if ((op == OP_STATIC || op == OP_TRANSPOSE) && t->type != T_F32 && t->type != T_BF16)
        return refuse(why, wn, "%s is type %u; this loader takes it as BF16 or F32 only (it is transposed on decode)", gn, t->type);
    int64_t want = 1;
    for (int i = 0; i < nd; i++) want *= d[i];
    if (op == OP_UNPAD) {
        if (t->ne[0] != (uint64_t)s->hidden || t->ne[1] < (uint64_t)s->vocab_used || t->ne[2] * t->ne[3] != 1)
            return refuse(why, wn, "%s is %llux%llu, not %d wide with at least %d rows", gn, (unsigned long long)t->ne[0],
                          (unsigned long long)t->ne[1], s->hidden, s->vocab_used);
        return 0;
    }
    // ggml order: ne[0] is the innermost reference dim (squeezed), the op's transposes aside
    int64_t sq[4], k = 0;
    for (int i = nd - 1; i >= 0; i--) if (d[i] != 1 || nd == 1) sq[k++] = d[i];
    if (op == OP_STATIC || op == OP_TRANSPOSE) { int64_t tmp = sq[0]; sq[0] = sq[1]; sq[1] = tmp; }
    for (int i = 0; i < 4; i++) {
        const uint64_t w = i < k ? (uint64_t)sq[i] : 1;
        if (t->ne[i] != w) return refuse(why, wn, "%s has dims %llu,%llu,%llu where the reference's %s needs %lld elements in its layout",
                                         gn, (unsigned long long)t->ne[0], (unsigned long long)t->ne[1], (unsigned long long)t->ne[2], ref, (long long)want);
    }
    return 0;
}

static int check_stacks(ink2g *g, char *why, size_t wn) {
    const ink2_shape *s = &g->s;
    for (int b = 0; b < s->n_block; b++) {
        for (int which = 0; which < 2; which++) {
            ink2_stack_t *st = which ? &g->dn[b] : &g->gu[b];
            char n[96], sn[96];
            snprintf(n, sizeof n, "blk.%d.ffn_%s_exps.weight", b, which ? "down" : "gate_up");
            snprintf(sn, sizeof sn, "blk.%d.ffn_%s_exps.scale", b, which ? "down" : "gate_up");
            st->w = gguf_find_tensor(g->fp, n);
            st->scale = gguf_find_tensor(g->fp, sn);
            st->rows = which ? s->hidden : 2 * s->ff;
            st->cols = which ? s->ff : s->hidden;
            if (!st->w) return refuse(why, wn, "the file has no %s", n);
            if (st->w->ne[0] != (uint64_t)st->cols || st->w->ne[1] != (uint64_t)st->rows || st->w->ne[2] != (uint64_t)s->n_expert || st->w->ne[3] != 1)
                return refuse(why, wn, "%s is %llux%llux%llu, not %lldx%lldx%d", n, (unsigned long long)st->w->ne[0],
                              (unsigned long long)st->w->ne[1], (unsigned long long)st->w->ne[2], (long long)st->cols, (long long)st->rows, s->n_expert);
            if (st->w->type == T_NVFP4) {
                st->kind = 'N';
                if (st->cols % 64) return refuse(why, wn, "%s rows of %lld do not divide into NVFP4 blocks of 64", n, (long long)st->cols);
                if (!st->scale) return refuse(why, wn, "%s is NVFP4 but the file has no %s, its per-expert factor", n, sn);
                if (st->scale->type != T_F32 || nelem(st->scale) != s->n_expert || st->scale->ne[0] != (uint64_t)s->n_expert)
                    return refuse(why, wn, "%s must be F32 with %d values (one per expert); it is type %u with %lld", sn, s->n_expert,
                                  st->scale->type, (long long)nelem(st->scale));
                const float *f = st->scale->data;
                for (int e = 0; e < s->n_expert; e++)
                    if (!(f[e] > 0.0f) || !fp_isfinite(f[e])) return refuse(why, wn, "%s[%d] is %g, not a finite positive factor", sn, e, f[e]);
            } else if (dense_type_ok(st->w->type)) {
                st->kind = st->w->type == T_BF16 ? 'B' : st->w->type == T_F32 ? 'F' : 'Q';
                if (st->kind == 'Q' && st->cols % ggml_block_size((int)st->w->type))
                    return refuse(why, wn, "%s rows of %lld do not divide into its type's blocks", n, (long long)st->cols);
                // a .scale beside it is applied after the decode, as llama.cpp's graph applies it whatever the type
                if (st->scale && (st->scale->type != T_F32 || nelem(st->scale) != s->n_expert))
                    return refuse(why, wn, "%s must be F32 with %d values (one per expert)", sn, s->n_expert);
            } else {
                return refuse(why, wn, "%s is type %u, which this loader does not decode", n, st->w->type);
            }
            if (st->rows * st->cols > (int64_t)g->ring_floats) g->ring_floats = (size_t)(st->rows * st->cols);
        }
    }
    return 0;
}

// every reference name, for the open-time check
static int each_ref(const ink2_shape *s, int (*fn)(void *, const char *), void *ctx) {
    static const char *glob[] = { "model.embed_tokens.weight", "model.embed_tokens.embed_norm.weight", "model.norm.weight",
                                  "model.final_input.query", "model.message_input.query", "lm_head.weight" };
    for (size_t i = 0; i < sizeof glob / sizeof glob[0]; i++) if (fn(ctx, glob[i])) return -1;
    static const char *per[] = {
        "self_attn.q_proj.linear.weight", "self_attn.q_proj.norm.scale", "self_attn.k_proj.linear.weight",
        "self_attn.v_proj.linear.weight", "self_attn.k_proj.sconv.conv1d.weight", "self_attn.v_proj.sconv.conv1d.weight",
        "self_attn.k_proj.sconv.static_weight", "self_attn.v_proj.sconv.static_weight", "self_attn.k_proj.sconv.weight_proj.weight",
        "self_attn.v_proj.sconv.weight_proj.weight", "self_attn.o_proj.gate_proj.weight", "self_attn.o_proj.linear.weight",
        "self_attn.sink_proj.weight", "self_attn.relative_position_bias.proj", "self_attn.relative_position_bias.linear.weight",
        "self_attn.relative_position_bias.norm.scale", "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
        "input_layernorm.weight", "post_attention_layernorm.weight", "post_attention_convolution.conv1d.weight",
        "post_mlp_convolution.conv1d.weight", "attention_input.query", "mlp_input.query", "q_input_layernorm.weight",
        "k_input_layernorm.weight", "v_input_layernorm.weight", "q_message_gate.down_proj.weight",
        "q_message_gate.forget_gate_proj.weight", "q_message_gate.message_gate_proj.weight", "k_message_gate.down_proj.weight",
        "k_message_gate.forget_gate_proj.weight", "k_message_gate.message_gate_proj.weight", "v_message_gate.down_proj.weight",
        "v_message_gate.forget_gate_proj.weight", "v_message_gate.message_gate_proj.weight", "mlp_message_gate.down_proj.weight",
        "mlp_message_gate.forget_gate_proj.weight", "mlp_message_gate.message_gate_proj.weight" };
    for (int b = 0; b < s->n_block; b++)
        for (size_t i = 0; i < sizeof per / sizeof per[0]; i++) {
            char n[160];
            int64_t d[4];
            snprintf(n, sizeof n, "model.layers.%d.%s", b, per[i]);
            if (ref_dims(s, n, d) && fn(ctx, n)) return -1;
        }
    return 0;
}

typedef struct { ink2g *g; char *why; size_t wn; int n; } chk_ctx;
static int chk_one(void *c_, const char *name) {
    chk_ctx *c = c_;
    c->n++;
    return check_tensor(&c->g->s, c->g, name, c->why, c->wn);
}

// the file's tensors this source does not read (other than the vision tower, which is not in this file) are refused:
// an unexpected tensor is a different layout
static int check_no_extra(ink2g *g, char *why, size_t wn) {
    const ink2_shape *s = &g->s;
    for (uint64_t i = 0; i < g->fp->n_tensors; i++) {
        const char *n = g->fp->tensors[i].name;
        int b, k = 0;
        if (!strcmp(n, "token_embd.weight") || !strcmp(n, "token_embd_norm.weight") || !strcmp(n, "output_norm.weight") ||
            !strcmp(n, "output_res_query.weight") || !strcmp(n, "output.weight")) continue;
        if (sscanf(n, "blk.%d.%n", &b, &k) == 1 && k && b >= 0 && b < s->n_block) {
            const char *x = n + k;
            if (!strcmp(x, "ffn_gate_up_exps.weight") || !strcmp(x, "ffn_down_exps.weight") || !strcmp(x, "ffn_gate_up_exps.scale") ||
                !strcmp(x, "ffn_down_exps.scale") || !strcmp(x, "exp_probs_b.bias")) continue;
            if (!strcmp(x, "loop_msg_query.weight") && is_loop_layer(s, b)) continue;
            // every other blk tensor must be a counterpart of a reference name of this layer (checked above by name)
            static const char *ok[] = { "attn_q", "attn_q_clip", "attn_k", "attn_v", "attn_k_conv", "attn_v_conv", "attn_k_conv_proj",
                                        "attn_v_conv_proj", "attn_gate", "attn_output", "attn_sink_proj", "attn_rel_bias", "attn_rel",
                                        "attn_rel_clip", "ffn_gate_inp", "attn_norm", "ffn_norm", "attn_post_conv", "ffn_post_conv",
                                        "attn_res_query", "ffn_res_query", "attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm",
                                        "loop_q_ctrl_down", "loop_q_ctrl_forget", "loop_q_ctrl_gain", "loop_k_ctrl_down",
                                        "loop_k_ctrl_forget", "loop_k_ctrl_gain", "loop_v_ctrl_down", "loop_v_ctrl_forget",
                                        "loop_v_ctrl_gain", "loop_ffn_ctrl_down", "loop_ffn_ctrl_forget", "loop_ffn_ctrl_gain" };
            char base[96];
            snprintf(base, sizeof base, "%s", x);
            char *dot = strstr(base, ".weight");
            if (dot && !dot[7]) {
                *dot = 0;
                int found = 0;
                for (size_t j = 0; j < sizeof ok / sizeof ok[0]; j++) if (!strcmp(base, ok[j])) { found = 1; break; }
                if (found) continue;
            }
        }
        return refuse(why, wn, "the file has a tensor %s that the inkling2_loop text model does not have", n);
    }
    return 0;
}

int ink2g_shape(void *gguf_file_, ink2_shape *s, char *why, size_t wn) {
    return read_shape((gguf_file *)gguf_file_, s, why, wn);
}
static ink2g *ink2g_bind(ink2g *g, ink2_shape *s, char *why, size_t wn);
ink2g *ink2g_open(const char *path, ink2_shape *s, char *why, size_t wn) {
    ink2g *g = calloc(1, sizeof *g);
    if (!g) { refuse(why, wn, "out of memory"); return NULL; }
    if (!gguf_open(&g->own, path)) {
        refuse(why, wn, "%s does not open as a GGUF (a truncated or malformed file is refused by the reader)", path);
        free(g);
        return NULL;
    }
    g->open = 1;
    g->fp = &g->own;
    return ink2g_bind(g, s, why, wn);
}
ink2g *ink2g_attach(void *gguf_file_, ink2_shape *s, char *why, size_t wn) {
    ink2g *g = calloc(1, sizeof *g);
    if (!g) { refuse(why, wn, "out of memory"); return NULL; }
    g->fp = gguf_file_;
    return ink2g_bind(g, s, why, wn);
}
static ink2g *ink2g_bind(ink2g *g, ink2_shape *s, char *why, size_t wn) {
    f16_init();   // the half-float table dequant_row's block scales read (a no-op after the first call)
    if (read_shape(g->fp, &g->s, why, wn) || check_stacks(g, why, wn)) { ink2g_close(g); return NULL; }
    chk_ctx c = { g, why, wn, 0 };
    if (each_ref(&g->s, chk_one, &c) || check_no_extra(g, why, wn)) { ink2g_close(g); return NULL; }
    static const float e2m1[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
    memcpy(g->e2m1, e2m1, sizeof e2m1);
    for (int b = 0; b < 256; b++) g->e4m3[b] = e4m3fn((unsigned)b);
    for (int i = 0; i < 2; i++) {
        g->ring[i] = malloc(sizeof(float) * g->ring_floats);
        if (!g->ring[i]) { refuse(why, wn, "out of memory for the expert buffers"); ink2g_close(g); return NULL; }
    }
    *s = g->s;
    return g;
}

void ink2g_close(ink2g *g) {
    if (!g) return;
    for (int i = 0; i < g->n_slots; i++) { free(g->slots[i]->data); free(g->slots[i]); }
    free(g->slots);
    free(g->ring[0]);
    free(g->ring[1]);
    if (g->open) gguf_close(&g->own);
    free(g);
}

char ink2g_expert_kind(const ink2g *g, int b) { return b >= 0 && b < g->s.n_block ? g->gu[b].kind : 0; }

// ---------------------------------------------------------------- lookups
// The reference names the engine reads only through its linear operation (ink2_lin) or as embedding rows: on the fast
// route these are views of the stored rows, never decoded. Everything else (norms, convolution taps, queries, clip
// scales, the relative bank, the router) stays a float32 decode: small, and read element by element.
static int is_matrix(const char *name) {
    if (!strcmp(name, "lm_head.weight") || !strcmp(name, "model.embed_tokens.weight")) return 1;
    static const char *const suffix[] = {
        ".self_attn.q_proj.linear.weight", ".self_attn.k_proj.linear.weight", ".self_attn.v_proj.linear.weight",
        ".self_attn.k_proj.sconv.weight_proj.weight", ".self_attn.v_proj.sconv.weight_proj.weight",
        ".self_attn.o_proj.gate_proj.weight", ".self_attn.o_proj.linear.weight", ".self_attn.sink_proj.weight",
        ".self_attn.relative_position_bias.linear.weight", "_message_gate.down_proj.weight",
        "_message_gate.forget_gate_proj.weight", "_message_gate.message_gate_proj.weight",
    };
    const size_t n = strlen(name);
    for (size_t i = 0; i < sizeof suffix / sizeof *suffix; i++) {
        const size_t m = strlen(suffix[i]);
        if (n > m && !strcmp(name + n - m, suffix[i])) return 1;
    }
    return 0;
}
static size_t row_bytes(const gguf_tensor *t, int64_t cols) {
    return ggml_type_size((int)t->type) * (size_t)(cols / ggml_block_size((int)t->type));
}
static slot_t *slot_new(ink2g *g, const char *name) {
    if (g->n_slots == g->cap_slots) {
        int cap = g->cap_slots ? 2 * g->cap_slots : 1024;
        slot_t **ns = realloc(g->slots, sizeof(slot_t *) * (size_t)cap);
        if (!ns) return NULL;
        g->slots = ns;
        g->cap_slots = cap;
    }
    slot_t *sl = calloc(1, sizeof *sl);
    if (!sl) return NULL;
    g->slots[g->n_slots++] = sl;
    snprintf(sl->name, sizeof sl->name, "%s", name);
    sl->t.name = sl->name;
    return sl;
}

static const ink2_stack_t *stack_of(const ink2g *g, const char *name);
const ink2_tensor *ink2g_get(void *ctx, const char *name) {
    ink2g *g = ctx;
    const ink2_shape *s = &g->s;
    for (int i = 0; i < g->n_slots; i++) if (!strcmp(g->slots[i]->name, name)) return &g->slots[i]->t;
    int64_t d[4] = { 1, 1, 1, 1 };
    const int nd = ref_dims(s, name, d);
    if (!nd) return NULL;
    if (strstr(name, ".mlp.experts.")) {   // the stacks: per expert on the reference route, whole views on the fast
        const ink2_stack_t *st = g->fast ? stack_of(g, name) : NULL;
        if (!st) return NULL;
        slot_t *sl = slot_new(g, name);
        if (!sl) return NULL;
        sl->t.n_dims = 3;
        sl->t.ne[0] = st->cols; sl->t.ne[1] = st->rows; sl->t.ne[2] = g->s.n_expert; sl->t.ne[3] = 1;
        sl->t.qtype = (int)st->w->type; sl->t.q = st->w->data; sl->t.q_row_bytes = row_bytes(st->w, st->cols);
        sl->t.q_scale = st->scale ? (const float *)st->scale->data : NULL;
        sl->t.q_act8 = g->act8;
        return &sl->t;
    }
    char gn[160];
    int op;
    if (gguf_name(s, name, gn, sizeof gn, &op)) return NULL;
    if (g->fast && is_matrix(name) && (op == OP_RENAME || op == OP_UNPAD)) {
        gguf_tensor *t = gguf_find_tensor(g->fp, gn);
        if (!t) return NULL;
        slot_t *sl = slot_new(g, name);
        if (!sl) return NULL;
        sl->t.n_dims = nd;
        for (int i = 0; i < 4; i++) sl->t.ne[i] = i < nd ? d[nd - 1 - i] : 1;
        sl->t.qtype = (int)t->type; sl->t.q = t->data; sl->t.q_row_bytes = row_bytes(t, sl->t.ne[0]);
        sl->t.q_act8 = g->act8;
        return &sl->t;   // data NULL: nothing decoded (unpad: the first vocab_used rows are the view's rows)
    }
    int64_t n = 1;
    for (int i = 0; i < nd; i++) n *= d[i];
    float *data = malloc(sizeof(float) * (size_t)n);
    if (!data) return NULL;
    if (op == OP_MSGQ) {
        const int64_t per = 4 * (int64_t)s->hidden;
        for (int b = s->loop_start; b < s->loop_end; b++) {
            snprintf(gn, sizeof gn, "blk.%d.loop_msg_query.weight", b);
            if (decode_float(gguf_find_tensor(g->fp, gn), 0, per, data + (size_t)(b - s->loop_start) * per)) { free(data); return NULL; }
        }
    } else {
        gguf_tensor *t = gguf_find_tensor(g->fp, gn);
        if (!t) { free(data); return NULL; }
        if (op == OP_RENAME || op == OP_UNPAD) {
            if (decode_float(t, 0, n, data)) { free(data); return NULL; }   // unpad: the first vocab_used rows
        } else {
            // the stored [A][B] (ggml ne[0] = B): static_weight [K][C] from stored [C][K] with the taps reversed;
            // relative_position_bias.proj [d_rel][extent] from stored [extent][d_rel]
            const int64_t B = (int64_t)t->ne[0], A = (int64_t)t->ne[1];
            float *tmp = malloc(sizeof(float) * (size_t)n);
            if (!tmp || decode_float(t, 0, n, tmp)) { free(tmp); free(data); return NULL; }
            for (int64_t a = 0; a < A; a++)
                for (int64_t bb = 0; bb < B; bb++) {
                    const float v = tmp[a * B + bb];
                    if (op == OP_STATIC) data[((g->ctrl & INK2G_CTRL_NO_FLIP) ? bb : B - 1 - bb) * A + a] = v;   // w[k][c] = g[c][K-1-k]
                    else data[bb * A + a] = v;                             // w[r][e] = g[e][r]
                }
            free(tmp);
        }
    }
    slot_t *sl = slot_new(g, name);
    if (!sl) { free(data); return NULL; }
    sl->data = data;
    sl->t.data = data;
    sl->t.n_dims = nd;
    for (int i = 0; i < 4; i++) sl->t.ne[i] = i < nd ? d[nd - 1 - i] : 1;
    return &sl->t;
}

static const ink2_stack_t *stack_of(const ink2g *g, const char *name) {
    int b, n = 0;
    if (sscanf(name, "model.layers.%d.mlp.experts.%n", &b, &n) != 1 || !n || b < 0 || b >= g->s.n_block) return NULL;
    if (!strcmp(name + n, "gate_up_proj")) return &g->gu[b];
    if (!strcmp(name + n, "down_proj")) return &g->dn[b];
    return NULL;
}

int ink2g_fill_expert(void *ctx, const char *stack, int e, float *dst, int64_t *rows, int64_t *cols) {
    ink2g *g = ctx;
    const ink2_stack_t *st = stack_of(g, stack);
    if (!st || e < 0 || e >= g->s.n_expert) return -1;
    *rows = st->rows;
    *cols = st->cols;
    if (!dst) return 0;
    const int64_t R = st->rows, C = st->cols;
    if (st->kind == 'N') {
        const size_t row_bytes = (size_t)(C / 64) * 36;
        const unsigned char *W = (const unsigned char *)st->w->data + (size_t)e * (size_t)R * row_bytes;
        const float factor = (g->ctrl & INK2G_CTRL_NO_FACTOR) ? 1.0f : ((const float *)st->scale->data)[e];
        const int lo_sh = (g->ctrl & INK2G_CTRL_SWAP_NIBBLE) ? 4 : 0, hi_sh = 4 - lo_sh;
        for (int64_t r = 0; r < R; r++) {
            const unsigned char *blk = W + (size_t)r * row_bytes;
            float *d = dst + (size_t)r * (size_t)C;
            for (int64_t q = 0; q < C / 64; q++, blk += 36)
                for (int sb = 0; sb < 4; sb++) {
                    const float sc = g->e4m3[blk[sb]];
                    const unsigned char *qs = blk + 4 + 8 * sb;
                    float *o = d + q * 64 + sb * 16;
                    for (int j = 0; j < 8; j++) {
                        float v0 = g->e2m1[(qs[j] >> lo_sh) & 15] * sc, v1 = g->e2m1[(qs[j] >> hi_sh) & 15] * sc;   // code x UE4M3
                        o[j] = v0 * factor;                                                  // x the expert's factor
                        o[j + 8] = v1 * factor;
                    }
                }
        }
        return 0;
    }
    if (decode_float(st->w, (int64_t)e * R * C, R * C, dst)) return -1;
    if (st->scale && !(g->ctrl & INK2G_CTRL_NO_FACTOR)) {
        const float factor = ((const float *)st->scale->data)[e];
        for (int64_t i = 0; i < R * C; i++) dst[i] = dst[i] * factor;
    }
    return 0;
}

const ink2_tensor *ink2g_get_expert(void *ctx, const char *stack, int e) {
    ink2g *g = ctx;
    int64_t R, C;
    if (ink2g_fill_expert(g, stack, e, NULL, &R, &C)) return NULL;
    const int k = g->ring_next;
    g->ring_next ^= 1;
    if (ink2g_fill_expert(g, stack, e, g->ring[k], &R, &C)) return NULL;
    ink2_tensor *t = &g->ring_t[k];
    *t = (ink2_tensor){ 0 };
    t->name = stack;
    t->data = g->ring[k];
    t->n_dims = 2;
    t->ne[0] = C; t->ne[1] = R; t->ne[2] = t->ne[3] = 1;
    return t;
}

typedef struct { ink2g *g; int bad; } pre_ctx;
static int pre_one(void *c_, const char *name) {
    pre_ctx *c = c_;
    if (strstr(name, ".mlp.experts.")) return 0;
    if (!ink2g_get(c->g, name)) { c->bad = 1; return -1; }
    return 0;
}
int ink2g_decode_all(ink2g *g, size_t *bytes) {
    pre_ctx c = { g, 0 };
    if (each_ref(&g->s, pre_one, &c) || c.bad) return -1;
    if (bytes) {
        size_t n = 0;
        for (int i = 0; i < g->n_slots; i++) {
            const ink2_tensor *t = &g->slots[i]->t;
            if (t->data) n += sizeof(float) * (size_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
        }
        *bytes = n;
    }
    return 0;
}

void ink2g_set_control(ink2g *g, int flags) {
    for (int i = 0; i < g->n_slots; i++) { free(g->slots[i]->data); free(g->slots[i]); }
    g->n_slots = 0;
    g->ctrl = flags;
}

void ink2g_weights(ink2g *g, ink2_weights *w) {
    w->get = ink2g_get;
    w->ctx = g;
    w->get_expert = g->fast ? NULL : ink2g_get_expert;   // the fast route reads the stacks whole, as views
}

void ink2g_set_route(ink2g *g, int fast, int act8) {
    ink2g_set_control(g, g->ctrl);   // drops what was served on the other route
    g->fast = fast ? 1 : 0;
    g->act8 = fast && act8 ? 1 : 0;
}

size_t ink2g_matrix_bytes(const ink2g *g) {
    size_t n = 0;
    for (int i = 0; i < g->n_slots; i++) {
        const ink2_tensor *t = &g->slots[i]->t;
        if (t->data && is_matrix(g->slots[i]->name)) n += sizeof(float) * (size_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
    }
    return n;
}

int ink2g_each_view(ink2g *g, int (*fn)(void *c, const ink2_tensor *t), void *c) {
    if (!g->fast) return -1;
    char nm[96];
    for (int b = 0; b < g->s.n_block; b++)
        for (int k = 0; k < 2; k++) {
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%s", b, k ? "down_proj" : "gate_up_proj");
            if (!ink2g_get(g, nm)) return -1;
        }
    for (int i = 0; i < g->n_slots; i++)
        if (g->slots[i]->t.q && fn(c, &g->slots[i]->t)) return -1;
    return 0;
}

int ink2g_set_dev(ink2g *g, const ink2_tensor *t, void *dev, int dev_i) {
    for (int i = 0; i < g->n_slots; i++)
        if (&g->slots[i]->t == t) {
            g->slots[i]->t.dev = dev;
            g->slots[i]->t.dev_i = dev_i;
            return 0;
        }
    return -1;
}
