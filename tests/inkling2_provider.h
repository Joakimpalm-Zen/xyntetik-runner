// R4.48: a TEST-SIDE weight provider for src/inkling2.c on the real Inkling-2-Mini checkpoint (the publisher's
// safetensors, mmapped read-only). It serves ink2_weights.get and .get_expert under the REFERENCE model's parameter
// names, decoding the checkpoint's sources into float32 ON LOOKUP, in the order of transformers pull request 51050
// at 6173a2f6's loader (core_model_loading.py, with the model's dtype float32):
//   - every source is first cast to float32 (the loader's _materialize_copy to the param dtype; exact for BF16, F32,
//     F8_E4M3 and U8);
//   - "rename": the source as is;
//   - Inkling2UnpadVocab: the first unpadded_vocab_size rows (the target's row count);
//   - Inkling2LoopMergeMessageQueries: queries [2L][H] and kv_queries [2L][H] to [L][4][H] = q, k, v, mlp:
//     out[l][0] = queries[2l], out[l][1] = kv[2l], out[l][2] = kv[2l+1], out[l][3] = queries[2l+1];
//   - Inkling2LoopDequantizeExperts, NVFP4 (amax present): w = fl32(fl32(E2M1[code] * E4M3fn(scale)) * factor[e]),
//     factor[e] = fl32(amax[e] / denominator), element 2j in the LOW nibble of byte j, one scale per 16;
//     MXFP8: w = fl32(E4M3fn(v) * 2^(s - 127)), one E8M0 scale per 32 (torch.exp2 of an integer: a power of two);
//   - Interleave(dim=1) after it on gate_up_proj: output row h * R/2 + i is source row 2i + h (h = 0: the gate rows,
//     the source's even rows; h = 1: the up rows, its odd rows).
// The name map is the loader's own, computed by tests/inkling2_provider_map.py and read from a table.
//
// Lifetime: a returned tensor (and its data) stays valid for at least the next INK2P_PIN_TENSORS `get` calls, and an
// expert for the next INK2P_PIN_EXPERTS `get_expert` calls. Beyond that the decoded data live in two LRU caches
// bounded by bytes (tensors and experts apart); the pinned ones are never evicted, so a cache may run over its
// budget by what is pinned.
#ifndef INKLING2_PROVIDER_H
#define INKLING2_PROVIDER_H

#include "inkling2.h"
#include <stddef.h>
#include <stdint.h>

#define INK2P_PIN_TENSORS 32
#define INK2P_PIN_EXPERTS 16

typedef struct ink2p ink2p;

// TEST-ONLY wrong decodes (the identity test's controls); 0 in use
#define INK2P_CTRL_NO_FACTOR    1   // NVFP4 without the per-expert factor
#define INK2P_CTRL_SWAP_NIBBLE  2   // NVFP4 with element 2j in the HIGH nibble
#define INK2P_CTRL_NO_INTERLEAVE 4  // gate_up_proj rows in the checkpoint's order

// Opens ckpt (the .safetensors) with map (the table). tensor_budget / expert_budget: the caches' sizes in bytes.
// Returns NULL with a reason in why. Every table entry is checked against the file (header length, file size, each
// source's byte range and element count) and against its op chain's shape rules before anything is served.
ink2p *ink2p_open(const char *ckpt, const char *map, size_t tensor_budget, size_t expert_budget, char *why, size_t wn);
void   ink2p_close(ink2p *p);
void   ink2p_weights(ink2p *p, ink2_weights *w);   // get + get_expert, ctx = p
const ink2_tensor *ink2p_get(void *ctx, const char *name);
const ink2_tensor *ink2p_get_expert(void *ctx, const char *stack, int e);

// the map: n entries, each a reference name, its shape (torch order, outer first) and its op chain
int ink2p_count(const ink2p *p);
const char *ink2p_name(const ink2p *p, int i, int *n_dims, int64_t *dims /* [4] */, const char **ops, const char **src_dtype);
int ink2p_find(const ink2p *p, const char *name);   // index or -1
int ink2p_is_stack(const ink2p *p, int i);          // an expert stack (Inkling2LoopDequantizeExperts)

void ink2p_set_control(ink2p *p, int flags);   // also empties both caches

// Decode expert e of stack into dst ([rows][cols] float32, the same bits get_expert returns), bypassing the
// provider's own caches; dst NULL: rows and cols only. 0, or -1 for a name that is not a stack. This is the fill the
// decoded-expert cache (tests/inkling2_ecache.h) takes, with ctx = the provider.
int ink2p_fill_expert(void *ctx, const char *stack, int e, float *dst, int64_t *rows, int64_t *cols);

typedef struct {
    uint64_t gets, get_hits, experts, expert_hits;
    double decode_s_tensor, decode_s_expert;    // wall seconds spent decoding
    uint64_t decoded_floats_tensor, decoded_floats_expert;
    size_t tensor_bytes, tensor_peak, expert_bytes, expert_peak;   // cache residency now and at its largest
} ink2p_stats;
void ink2p_get_stats(const ink2p *p, ink2p_stats *st);

#endif
