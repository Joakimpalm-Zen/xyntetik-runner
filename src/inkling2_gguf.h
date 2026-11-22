// R4.48.6: the inkling2_loop GGUF as a weight source for src/inkling2.c.
//
// The file is llama.cpp's inkling2_loop layout (block_count, attn_res.block_size and the loop.* keys; the tensors
// token_embd ... output, blk.N.*, blk.N.loop_*), with the experts of the NVFP4 layers as GGML NVFP4 blocks plus one
// F32 `blk.N.ffn_{gate_up,down}_exps.scale` per stack ({n_expert}: the publisher's amax / denominator), and every
// other expert stack and small tensor in BF16 or F32 (exact as float32 either way).
//
// It serves ink2_weights.get and .get_expert under the REFERENCE model's parameter names (the names
// tests/inkling2_provider.c serves), as float32, in the reference loader's order, so the forward on the GGUF equals
// the forward on the publisher's safetensors bit for bit:
//   - a rename: the tensor as stored, BF16 or F32 decoded to float32;
//   - lm_head.weight: the first vocab_size_unpadded rows of output.weight;
//   - model.message_input.query [L][4][H]: blk.(loop.start + l).loop_msg_query [4][H], stacked;
//   - self_attn.{k,v}_proj.sconv.static_weight [K][C]: the stored [C][K] read back as w[k][c] = g[c][K-1-k]
//     (llama.cpp's converter wrote flip(0).T);
//   - self_attn.relative_position_bias.proj [d_rel][extent]: the stored [extent][d_rel] transposed back;
//   - an NVFP4 expert: fl32(fl32(E2M1[code] * UE4M3(d)) * scale[e]), the publisher's scale bytes and the file's
//     per-expert factor (bit-equal to amax / denominator); a BF16 expert as stored. gate_up rows are gate first, as
//     the reference loader's Interleave(dim=1) leaves them.
// Every tensor is checked at open (present, shape, a type this source handles, the .scale beside each NVFP4 stack
// and none beside a BF16 one) and anything else is refused with a sentence. The vision tower is not part of this
// file (a separate mmproj in llama.cpp's layout): text path only.
#ifndef INKLING2_GGUF_H
#define INKLING2_GGUF_H
#include "inkling2.h"
#include <stddef.h>
#include <stdint.h>

typedef struct ink2g ink2g;

// Opens path, reads the shape from its metadata into *s (ink2_shape_derive included) and checks every tensor.
// NULL with a sentence in why on any refusal.
ink2g *ink2g_open(const char *path, ink2_shape *s, char *why, size_t wn);
void   ink2g_close(ink2g *g);
void   ink2g_weights(ink2g *g, ink2_weights *w);   // get + get_expert, ctx = g
const ink2_tensor *ink2g_get(void *ctx, const char *name);
const ink2_tensor *ink2g_get_expert(void *ctx, const char *stack, int e);
// Decode expert e of stack into dst ([rows][cols] float32, the bits get_expert returns); dst NULL: rows and cols only.
// The fill the decoded-expert cache (tests/inkling2_ecache.h) takes, with ctx = the source. 0, or -1.
int ink2g_fill_expert(void *ctx, const char *stack, int e, float *dst, int64_t *rows, int64_t *cols);
// TEST-ONLY wrong decodes (the identity test's controls, which must fail it); 0 in use. Also drops the decoded tensors.
#define INK2G_CTRL_NO_FACTOR    1   // NVFP4 experts without the per-expert factor
#define INK2G_CTRL_NO_FLIP      2   // static_weight transposed back without reversing the taps
#define INK2G_CTRL_SWAP_NIBBLE  4   // NVFP4 element j in the high nibble
void ink2g_set_control(ink2g *g, int flags);
// The expert storage of physical layer b: 'N' NVFP4 + .scale, 'B' BF16, 'F' F32.
char ink2g_expert_kind(const ink2g *g, int b);
#endif
