// The per-layer tables a device backend builds from a model's small tensors,
// and the mixer weights it uploads beside them (R6.68.29). The CUDA uploader
// (cuda.c shared_upload_layer_tables, shared_destroy, layer_mixer_tensors)
// allocates, checks and frees its tables by these rows, and --info fit's
// device charge (model.c model_cuda_layer_charge, fit_layer_host_expert_bytes)
// prices them from the same rows: one list, where there were five copies in
// cuda.c and three in model.c, so a table the uploader makes cannot be missing
// from the charge. Header-only, like arch.h: model.c builds without cuda.c.
#ifndef RUNNER_GPU_TABLES_H
#define RUNNER_GPU_TABLES_H

#include <stddef.h>
#include "model.h"

// In the order the uploader checks them, so its first missing table is named
// as it always was.
typedef enum {
    GT_ATTN_NORM, GT_HC_ATTN_NORM, GT_HC_FFN_NORM, GT_FFN_NORM,
    GT_BQ, GT_BK, GT_BV, GT_BO, GT_QNORM, GT_KNORM, GT_MLA_KV_NORM, GT_MLA_KVB,
    GT_POST_ATTN_NORM, GT_POST_FFN_NORM, GT_PLE_POST_NORM, GT_SINKS,
    GT_MOE_ROUTER_B, GT_MOE_GATE_B, GT_MOE_UP_B, GT_MOE_DOWN_B,
    GT_SSM_DT, GT_SSM_A, GT_SSM_NORM, GT_SSM_GNORM, GT_SSM_D, GT_SSM_CONV_B,
    GT_G_POST_NORM1, GT_G_PRE_NORM2, GT_G_POST_NORM2, GT_G_ROUTER_SCALE, GT_G_DOWN_SCALE,
    GT_HOT_BANK,
    GT_COUNT
} gpu_table;

// When a layer's table must have been made, once its pointer table exists.
typedef enum {
    GT_NEED_HOST,       // the layer's host buffer (`host`) is bound
    GT_NEED_HC,         // a hyper-connection model: every layer
    GT_NEED_REC,        // a recurrent layer
    GT_NEED_REC_DN,     // a qwen35-family recurrent layer
    GT_NEED_REC_M2,     // a Mamba-2 recurrent layer (granitehybrid, nemotron_h)
    GT_NEED_MOE_DEV,    // a MoE layer whose bank runs on the device, `host` bound
    GT_NEED_GEMMA_MOE,  // a gemma-4 MoE layer: made, ones when the file has none
    GT_NEED_NONE,       // made on its own path (the hot bank)
} gpu_table_need;

#define GT_NO_HOST ((size_t)-1)

typedef struct {
    gpu_table id;
    const char *table_label;   // the per-layer pointer table, when it cannot be allocated
    const char *upload_label;  // a layer's table, when its upload is missing
    gpu_table_need need;
    size_t host;               // offsetof(layer_t, the host buffer), or GT_NO_HOST
    // the blk.N tensor --info fit prices the table from, as f32; NULL where the
    // charge has a rule of its own (gemma-4's synthesized tables) or where
    // another row prices the same tensor (Mamba-2's gated norm is ssm_norm)
    const char *tensor;
} gpu_table_desc;

#define GT_ROW(i, tl, ul, nd, field, t) { i, tl, ul, nd, offsetof(layer_t, field), t }
#define GT_ROW0(i, tl, ul, nd, t) { i, tl, ul, nd, GT_NO_HOST, t }

static inline const gpu_table_desc *gpu_table_rows(void) {
    static const gpu_table_desc rows[GT_COUNT] = {
        [GT_ATTN_NORM] = GT_ROW(GT_ATTN_NORM, "attn_norm", "attention norm", GT_NEED_HOST, attn_norm_w,
               "attn_norm.weight"),
        [GT_HC_ATTN_NORM] = GT_ROW0(GT_HC_ATTN_NORM, "hc_attn_norm", "hc attention mixer norm", GT_NEED_HC,
                "hc_attn_norm.weight"),
        [GT_HC_FFN_NORM] = GT_ROW0(GT_HC_FFN_NORM, "hc_ffn_norm", "hc FFN mixer norm", GT_NEED_HC,
                "hc_ffn_norm.weight"),
        [GT_FFN_NORM] = GT_ROW(GT_FFN_NORM, "ffn_norm", "FFN norm", GT_NEED_HOST, ffn_norm_w, "ffn_norm.weight"),
        [GT_BQ] = GT_ROW(GT_BQ, "attn_q bias", "Q bias", GT_NEED_HOST, bq, "attn_q.bias"),
        [GT_BK] = GT_ROW(GT_BK, "attn_k bias", "K bias", GT_NEED_HOST, bk, "attn_k.bias"),
        [GT_BV] = GT_ROW(GT_BV, "attn_v bias", "V bias", GT_NEED_HOST, bv, "attn_v.bias"),
        [GT_BO] = GT_ROW(GT_BO, "attn_output bias", "attention output bias", GT_NEED_HOST, bo,
               "attn_output.bias"),
        [GT_QNORM] = GT_ROW(GT_QNORM, "attn_q_norm", "Q norm", GT_NEED_HOST, qnorm_w, "attn_q_norm.weight"),
        [GT_KNORM] = GT_ROW(GT_KNORM, "attn_k_norm", "K norm", GT_NEED_HOST, knorm_w, "attn_k_norm.weight"),
        [GT_MLA_KV_NORM] = GT_ROW(GT_MLA_KV_NORM, "attn_kv_a_norm", "latent norm", GT_NEED_HOST, kv_a_norm_w,
               "attn_kv_a_norm.weight"),
        [GT_MLA_KVB] = GT_ROW(GT_MLA_KVB, "attn_kv_b f32", "attn_kv_b f32", GT_NEED_HOST, mla_kvb,
               "attn_kv_b.weight"),
        [GT_POST_ATTN_NORM] = GT_ROW(GT_POST_ATTN_NORM, "post_attention_norm", "post-attention norm", GT_NEED_HOST,
               post_attn_norm_w, "post_attention_norm.weight"),
        [GT_POST_FFN_NORM] = GT_ROW(GT_POST_FFN_NORM, "post_ffn_norm", "post-FFN norm", GT_NEED_HOST,
               post_ffn_norm_w, "post_ffw_norm.weight"),
        [GT_PLE_POST_NORM] = GT_ROW(GT_PLE_POST_NORM, "per-layer embedding post_norm", "per-layer embedding post-norm",
               GT_NEED_HOST, ple_post_norm, "post_norm.weight"),
        [GT_SINKS] = GT_ROW(GT_SINKS, "attention sinks", "attention sinks", GT_NEED_HOST, attn_sinks,
               "attn_sinks.weight"),
        [GT_MOE_ROUTER_B] = GT_ROW(GT_MOE_ROUTER_B, "MoE router bias", "MoE router bias", GT_NEED_MOE_DEV,
               ffn_gate_inp_b, "ffn_gate_inp.bias"),
        [GT_MOE_GATE_B] = GT_ROW(GT_MOE_GATE_B, "MoE gate bias", "MoE gate bias", GT_NEED_MOE_DEV,
               ffn_gate_exps_b, "ffn_gate_exps.bias"),
        [GT_MOE_UP_B] = GT_ROW(GT_MOE_UP_B, "MoE up bias", "MoE up bias", GT_NEED_MOE_DEV, ffn_up_exps_b,
               "ffn_up_exps.bias"),
        [GT_MOE_DOWN_B] = GT_ROW(GT_MOE_DOWN_B, "MoE down bias", "MoE down bias", GT_NEED_MOE_DEV,
               ffn_down_exps_b, "ffn_down_exps.bias"),
        // a DeltaNet file may spell it ssm_dt (model_cuda_layer_charge reads both)
        [GT_SSM_DT] = GT_ROW0(GT_SSM_DT, "SSM dt", "SSM dt", GT_NEED_REC, "ssm_dt.bias"),
        [GT_SSM_A] = GT_ROW0(GT_SSM_A, "SSM A", "SSM A", GT_NEED_REC, "ssm_a"),
        [GT_SSM_NORM] = GT_ROW0(GT_SSM_NORM, "SSM norm", "SSM norm", GT_NEED_REC_DN, "ssm_norm.weight"),
        [GT_SSM_GNORM] = GT_ROW0(GT_SSM_GNORM, "Mamba gated norm", "Mamba gated norm", GT_NEED_REC_M2, NULL),
        [GT_SSM_D] = GT_ROW0(GT_SSM_D, "Mamba D", "Mamba D", GT_NEED_REC_M2, "ssm_d"),
        [GT_SSM_CONV_B] = GT_ROW0(GT_SSM_CONV_B, "Mamba conv bias", "Mamba conv bias", GT_NEED_REC_M2,
                "ssm_conv1d.bias"),
        [GT_G_POST_NORM1] = GT_ROW0(GT_G_POST_NORM1, "Gemma post_ffw_norm_1", "Gemma post-FFN norm 1",
                GT_NEED_GEMMA_MOE, NULL),
        [GT_G_PRE_NORM2] = GT_ROW0(GT_G_PRE_NORM2, "Gemma pre_ffw_norm_2", "Gemma pre-FFN norm 2",
                GT_NEED_GEMMA_MOE, NULL),
        [GT_G_POST_NORM2] = GT_ROW0(GT_G_POST_NORM2, "Gemma post_ffw_norm_2", "Gemma post-FFN norm 2",
                GT_NEED_GEMMA_MOE, NULL),
        [GT_G_ROUTER_SCALE] = GT_ROW0(GT_G_ROUTER_SCALE, "Gemma router input scale", "Gemma router input scale",
                GT_NEED_GEMMA_MOE, NULL),
        [GT_G_DOWN_SCALE] = GT_ROW0(GT_G_DOWN_SCALE, "Gemma expert down scale", "Gemma expert down scale",
                GT_NEED_GEMMA_MOE, NULL),
        [GT_HOT_BANK] = GT_ROW0(GT_HOT_BANK, "MoE hot bank", NULL, GT_NEED_NONE, NULL),
    };
    // a designated row cannot sit at another id's index; a row left out is
    // all zeros, which test_autofit's table check refuses (id and labels)
    return rows;
}

// The mixer weights a device uploads for a block, in the uploader's order: a
// layer_t field, and the blk.N tensors a family binds it from (a qwen35 or
// lfm2 block, a latent-attention one each spell some of them their own way).
#define GT_MIXER_NAMES 2
typedef struct { size_t field; const char *names[GT_MIXER_NAMES]; } gpu_mixer_desc;
#define GT_MIXER(f, a, b) { offsetof(layer_t, f), { a, b } }
#define GT_N_MIXER 22

static inline const gpu_mixer_desc *gpu_mixer_rows(void) {
    static const gpu_mixer_desc rows[] = {
        GT_MIXER(ple_gate, "inp_gate.weight", NULL), GT_MIXER(ple_proj, "proj.weight", NULL),
        GT_MIXER(wq, "attn_q.weight", "attn_q_b.weight"), GT_MIXER(wk, "attn_k.weight", NULL),
        GT_MIXER(wv, "attn_v.weight", NULL), GT_MIXER(wo, "attn_output.weight", NULL),
        GT_MIXER(wqkv, "attn_qkv.weight", NULL), GT_MIXER(wq_gate, "attn_gate.weight", NULL),
        GT_MIXER(ssm_in, "ssm_in.weight", "shortconv.in_proj.weight"),
        GT_MIXER(ssm_conv, "ssm_conv1d.weight", "shortconv.conv.weight"),
        GT_MIXER(ssm_beta, "ssm_beta.weight", NULL), GT_MIXER(ssm_alpha, "ssm_alpha.weight", NULL),
        GT_MIXER(ssm_out, "ssm_out.weight", "shortconv.out_proj.weight"),
        GT_MIXER(hc_attn_down, "hc_attn_down.weight", NULL),
        GT_MIXER(hc_attn_up, "hc_attn_up.weight", NULL),
        GT_MIXER(hc_attn_inject, "hc_attn_inject.weight", NULL),
        GT_MIXER(hc_ffn_down, "hc_ffn_down.weight", NULL),
        GT_MIXER(hc_ffn_up, "hc_ffn_up.weight", NULL),
        GT_MIXER(hc_ffn_inject, "hc_ffn_inject.weight", NULL),
        // deepseek2's multi-head latent attention: the q-lora down projection
        // and the latent pair; attn_q and attn_output ride wq and wo
        GT_MIXER(wq_a, "attn_q_a.weight", NULL), GT_MIXER(wkv_a, "attn_kv_a_mqa.weight", NULL),
        GT_MIXER(wkv_b, "attn_kv_b.weight", NULL),
    };
    // cuda.c sizes its mixer list by GT_N_MIXER (LAYER_MIXER_MAX)
    _Static_assert(sizeof rows / sizeof rows[0] == GT_N_MIXER,
                   "GT_N_MIXER counts the mixer rows");
    return rows;
}

#endif
