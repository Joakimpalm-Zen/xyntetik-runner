// Pure Metal weight-admission policy. Kept free of Objective-C/Metal types so
// the capacity decision can be tested with device sizes this machine does not
// physically have.
#ifndef RUNNER_METAL_ADMISSION_H
#define RUNNER_METAL_ADMISSION_H

#include <stdint.h>

typedef struct {
    uint64_t working_set;
    uint64_t max_buffer;
} metal_weight_limits;

static inline uint64_t metal_full_weight_budget(metal_weight_limits lim,
                                                uint64_t other_bytes) {
    // max_buffer is deliberately not a full-offload ceiling: metal.m wraps a
    // weight file in as many tensor-boundary MTLBuffers as it needs. Only the
    // aggregate working set competes with KV, scratch, and the safety margin.
    (void)lim.max_buffer;
    return lim.working_set > other_bytes
         ? (uint64_t)((double)(lim.working_set - other_bytes) * 0.94)
         : 0;
}

// The attention score scratch: [n_head][n_ctx] floats a column, for the
// columns one attention dispatch covers. Sized by the whole prompt batch it
// was batch x heads x context floats (2 GiB for a 512-column batch of a
// 32-head model at a 32K context, 8 GiB at 128K, past one buffer's limit on
// a small machine), none of it in the load's budget. It now holds at most
// METAL_ATT_BYTES, and never fewer than METAL_ATT_COLS_MIN columns: a batch
// wider than that runs its attention several dispatches a layer (R6.68.88).
// A context of 8K at 32 heads still fits a 512-column batch in one dispatch,
// as before; measured on a 4B model, each further dispatch a layer costs
// about 1% of prefill speed.
#define METAL_ATT_BYTES (512ull * 1024 * 1024)
#define METAL_ATT_COLS_MIN 64

// The columns the scratch holds for a batch of n_batch: whole kernel tiles
// (ATTN_TILE, 8 columns), so each tile holds the columns it always did.
static inline int metal_att_cols(int n_batch, int n_head, int n_ctx,
                                 uint64_t max_bytes) {
    uint64_t col = 4ull * (uint64_t)(n_head > 0 ? n_head : 1) *
                   (uint64_t)(n_ctx > 0 ? n_ctx : 1);
    uint64_t cols = max_bytes / col;
    cols -= cols % 8;
    if (cols < METAL_ATT_COLS_MIN) cols = METAL_ATT_COLS_MIN;
    if (n_batch > 0 && cols > (uint64_t)n_batch) cols = (uint64_t)n_batch;
    return (int)cols;
}

// What a load holds beside its weights: the KV cache, and a flat allowance
// for the scratch and the margin. The allowance has always carried the
// scores of a default context (256 MiB: 512 columns, 32 heads, 4096
// positions); what a longer context's scores take past that is counted.
static inline uint64_t metal_other_bytes(uint64_t kv_bytes, int n_batch,
                                         int n_head, int n_ctx) {
    const uint64_t carried = 256ull * 1024 * 1024;
    uint64_t scores = 4ull * (uint64_t)metal_att_cols(n_batch, n_head, n_ctx, METAL_ATT_BYTES) *
                      (uint64_t)(n_head > 0 ? n_head : 1) * (uint64_t)(n_ctx > 0 ? n_ctx : 1);
    return kv_bytes + (scores > carried ? scores - carried : 0) + 512ull * 1024 * 1024;
}

#endif // RUNNER_METAL_ADMISSION_H
