// The architectures this build knows, one row per general.architecture name
// (R6.68.28). Every rule that used to be a list of names (the DeltaNet family,
// the hybrid-typed families, the Mamba-2 families, the NeoX rotation, the
// families no GPU backend runs, the refused families, the admission
// allowlist) is a column here, and a rule about one family compares its id.
// A new architecture is one row, plus its binder's line in model.c
// (bind_arch_meta, a switch over arch_id that -Wswitch holds complete).
//
// Header-only on purpose: quantize.c is linked without model.c into its own
// test binaries, and tokenizer.c and sample.c into many more, so the table
// lives where every unit that includes this header sees the same rows, with
// no object to add to a link line.
#ifndef RUNNER_ARCH_H
#define RUNNER_ARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

typedef enum {
    ARCH_UNKNOWN = 0,   // a name not in the table (RUNNER_ALLOW_UNKNOWN_ARCH)
    ARCH_LLAMA, ARCH_QWEN2, ARCH_QWEN3, ARCH_QWEN35, ARCH_QWEN35MOE, ARCH_QWEN4EXP,
    ARCH_QWEN3MOE, ARCH_STABLELM, ARCH_GEMMA3, ARCH_GEMMA4,
    ARCH_PHI3, ARCH_GPT_OSS, ARCH_APERTUS, ARCH_AFMOE, ARCH_MUSE_GLIMMER, ARCH_GRANITE,
    ARCH_GRANITEHYBRID, ARCH_NEMOTRON_H, ARCH_NEMOTRON_H_MOE, ARCH_CLEF, ARCH_MELLUM,
    ARCH_LFM2, ARCH_LFM2MOE, ARCH_SPARK2_5, ARCH_K2_HORIZON, ARCH_FLINT, ARCH_MISTRAL3,
    ARCH_DEEPSEEK2,
    ARCH_GEMMA, ARCH_GEMMA2,   // refused: they load llama-style and compute wrong
    ARCH_COUNT
} arch_id;

// The state a family's recurrent layers carry: the mixer they run.
typedef enum {
    ARCH_REC_NONE = 0,
    ARCH_REC_DELTANET,   // Gated DeltaNet (the qwen35 family)
    ARCH_REC_MAMBA2,     // Mamba-2 (granitehybrid, nemotron_h)
    ARCH_REC_SHORTCONV,  // short convolution (lfm2)
} arch_rec;

typedef struct {
    const char *name;           // general.architecture
    arch_id id;
    // the family it belongs to when that is another row's (0: its own), the
    // row whose binder it shares: nemotron_h_moe is nemotron_h's, qwen3moe
    // and mellum qwen3's, qwen35moe and clef qwen35's, lfm2moe lfm2's
    arch_id family;
    bool rope_neox;             // NeoX half-split rotation; false: interleaved pairs
    arch_rec rec;
    // DeltaNet layers own no KV rows (R3.44; RUNNER_KV_RECURRENT_ROWS=1 keeps
    // the 1.15.0 layout that gave them the attention shape)
    bool kv_rows_typed;
    // blocks typed by zeros in the per-layer arrays (head_count_kv,
    // feed_forward_length): a zero is a block kind, never a removed sublayer
    bool hybrid_typed;
    // no GPU backend here serves it: the load runs it on the CPU and prints
    // this, and --info fit plans the CPU for it (R3.47)
    const char *cpu_only;
    // CUDA serves it and the other backends do not: the line elsewhere
    const char *cpu_only_noncuda;
    const char *refused;        // non-NULL: the load refuses the family, saying why
} arch_desc;

// Every row: the admitted architectures in the order --caps lists them, then
// the refused ones.
static inline const arch_desc *arch_rows(size_t *n) {
    static const arch_desc arch_table[] = {
        { .name = "llama", .id = ARCH_LLAMA },
        { .name = "qwen2", .id = ARCH_QWEN2, .rope_neox = true },
        { .name = "qwen3", .id = ARCH_QWEN3, .rope_neox = true },
        { .name = "qwen35", .id = ARCH_QWEN35, .rope_neox = true,
          .rec = ARCH_REC_DELTANET, .kv_rows_typed = true },
        { .name = "qwen35moe", .id = ARCH_QWEN35MOE, .family = ARCH_QWEN35, .rope_neox = true,
          .rec = ARCH_REC_DELTANET, .kv_rows_typed = true },
        { .name = "qwen4exp", .id = ARCH_QWEN4EXP, .rope_neox = true,
          .rec = ARCH_REC_DELTANET, .kv_rows_typed = true },
        { .name = "qwen3moe", .id = ARCH_QWEN3MOE, .family = ARCH_QWEN3, .rope_neox = true },
        { .name = "stablelm", .id = ARCH_STABLELM, .rope_neox = true,
          .cpu_only = "this model normalises with LayerNorm and the GPU "
                      "backends implement RMSNorm only" },
        { .name = "gemma3", .id = ARCH_GEMMA3, .rope_neox = true },
        { .name = "gemma4", .id = ARCH_GEMMA4, .rope_neox = true },
        { .name = "phi3", .id = ARCH_PHI3, .rope_neox = true },
        { .name = "gpt-oss", .id = ARCH_GPT_OSS, .rope_neox = true },
        { .name = "apertus", .id = ARCH_APERTUS, .rope_neox = true },
        { .name = "afmoe", .id = ARCH_AFMOE, .rope_neox = true },
        { .name = "muse-glimmer", .id = ARCH_MUSE_GLIMMER, .rope_neox = true },
        { .name = "granite", .id = ARCH_GRANITE, .rope_neox = true },
        { .name = "granitehybrid", .id = ARCH_GRANITEHYBRID, .rope_neox = true,
          .rec = ARCH_REC_MAMBA2, .hybrid_typed = true },
        { .name = "nemotron_h", .id = ARCH_NEMOTRON_H, .rope_neox = true,
          .rec = ARCH_REC_MAMBA2, .hybrid_typed = true },
        { .name = "nemotron_h_moe", .id = ARCH_NEMOTRON_H_MOE, .family = ARCH_NEMOTRON_H, .rope_neox = true,
          .rec = ARCH_REC_MAMBA2, .hybrid_typed = true },
        { .name = "clef", .id = ARCH_CLEF, .family = ARCH_QWEN35, .rope_neox = true,
          .rec = ARCH_REC_DELTANET, .kv_rows_typed = true },
        { .name = "mellum", .id = ARCH_MELLUM, .family = ARCH_QWEN3, .rope_neox = true },
        { .name = "lfm2", .id = ARCH_LFM2, .rope_neox = true,
          .rec = ARCH_REC_SHORTCONV, .hybrid_typed = true,
          .cpu_only = "lfm2 is validated on the CPU only; its GPU path (the "
                      "short-convolution layers and their state) is not "
                      "checked yet" },
        { .name = "lfm2moe", .id = ARCH_LFM2MOE, .family = ARCH_LFM2, .rope_neox = true,
          .rec = ARCH_REC_SHORTCONV, .hybrid_typed = true,
          .cpu_only = "lfm2 is validated on the CPU only; its GPU path (the "
                      "short-convolution layers and their state) is not "
                      "checked yet" },
        { .name = "spark2_5", .id = ARCH_SPARK2_5, .rope_neox = true },
        { .name = "k2-horizon", .id = ARCH_K2_HORIZON, .rope_neox = true },
        { .name = "flint", .id = ARCH_FLINT, .rope_neox = true,
          .cpu_only = "flint is validated on the CPU only; its GPU path "
                      "(sinks with a sliding window, NoPE full layers, "
                      "sigmoid routing with shared experts) is not checked yet" },
        { .name = "mistral3", .id = ARCH_MISTRAL3 },
        // CUDA runs both forms (R4.4.5 c1 naive, c2 absorbed)
        { .name = "deepseek2", .id = ARCH_DEEPSEEK2, .rope_neox = true,
          .cpu_only_noncuda = "deepseek2 (multi-head latent attention) is validated on "
                              "the CPU only; its GPU path is not checked yet" },
        // scalar multipliers and logit softcapping the generic math lacks
        { .name = "gemma", .id = ARCH_GEMMA, .rope_neox = true,
          .refused = "it would load but produce incorrect output without its "
                     "scaling/softcapping" },
        { .name = "gemma2", .id = ARCH_GEMMA2, .family = ARCH_GEMMA, .rope_neox = true,
          .refused = "it would load but produce incorrect output without its "
                     "scaling/softcapping" },
    };
    if (n) *n = sizeof arch_table / sizeof arch_table[0];
    return arch_table;
}

// The family a row belongs to (its own id when it heads one).
static inline arch_id arch_family(const arch_desc *a) {
    return a->family ? a->family : a->id;
}

// The row for `name`; a name not in the table gets the unknown row (id
// ARCH_UNKNOWN, every column off but the NeoX rotation, as the generic path
// treats it), so a caller never tests for NULL.
static inline const arch_desc *arch_find(const char *name) {
    static const arch_desc arch_unknown = { .name = NULL, .id = ARCH_UNKNOWN, .rope_neox = true };
    size_t arch_n;
    const arch_desc *arch_all = arch_rows(&arch_n);
    for (size_t i = 0; name && i < arch_n; i++)
        if (strcmp(arch_all[i].name, name) == 0) return &arch_all[i];
    return &arch_unknown;
}

#endif
