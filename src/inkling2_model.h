// R4.48.6: Inkling 2 Mini (inkling2_loop) on Runner's normal path. The model has its own engine (src/inkling2.c, a
// serial float32 order checked bit for bit against the publisher's reference) and its own weight source
// (src/inkling2_gguf.c); this is the seam between them and model_t, so the engine, the sampler, the server, --score
// and --info see an ordinary model: token ids in, float32 logits rows of n_vocab out.
//
// What this path is: the CPU only (the architecture row says so), one sequence per model_t (each slot its own copy),
// any prompt split (the engine's rows do not depend on how positions are split into calls), K/V held per layer
// application with the sliding applications capped at window + batch. What it is not, yet: GPU offload, speculative
// decoding, MTP, the shared prefix cache, LoRA. A rewind to an earlier position re-prefills from 0 (the engine's
// short-convolution histories are a fold this seam does not snapshot yet).
#ifndef RUNNER_INKLING2_MODEL_H
#define RUNNER_INKLING2_MODEL_H
#include "model.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// The weight side, after gguf_open: the shape from the metadata, every tensor checked, the dense tensors decoded.
// false with a sentence on stderr.
bool   model_inkling2_bind(model_t *m, const model_params *p);
// The per-instance side: context, batch, threads, the engine's cache, the decoded-expert cache, the logits rows.
bool   model_inkling2_runtime(model_t *m, const model_params *p);
// n tokens at pos: pos 0 starts a new sequence, any other pos must be where the last call ended (a rewind is the
// engine's, through the recurrent seam: this model declines snapshots, so the engine re-prefills from 0). Returns the
// last row's logits if want_logits, else NULL (NULL also on a refusal, said on stderr); keep_all: every row is kept
// for model_inkling2_row.
float *model_inkling2_forward(model_t *m, const int32_t *tokens, int n, int pos, bool want_logits, bool keep_all);
float *model_inkling2_row(model_t *m, int b);   // row b of the last keep_all forward
void   model_inkling2_free(model_t *m);
// What the engine holds, for --info fit and the load line (bytes): the dense float32 tensors, the K/V and convolution
// cache at n_ctx, the most the expert cache can hold (its budget, capped at every expert decoded), the file's mapping.
typedef struct { uint64_t dense_f32, cache, expert_cache, mapped, logits; } inkling2_fit;
bool   model_inkling2_fit(model_t *m, inkling2_fit *f);
// --info fit on an inkling2_loop header (no weights read): what a load at n_ctx and n_batch would hold, by the same
// parts, against the RAM available now. Prints the report; returns the exit code (0 fits, 1 does not).
int    model_inkling2_fit_print(gguf_file *g, const char *path, int n_ctx_want, int n_batch);
// LoRA adapter geometry and backward (R4.48.10): attention-only adapters on blk.N.attn_{q,k,v,output}.
int    model_inkling2_n_block(const model_t *m);
bool   model_inkling2_slot_dims(const model_t *m, int layer, int slot, int *n_in, int *n_out, char *name, size_t cap);
// Runtime weights & shape accessor for src/inkling2_bwd.c.
struct ink2_shape;
struct ink2_weights;
bool   model_inkling2_parts(model_t *m, const void **s_out, const void **w_out);
int    model_inkling2_threads(const model_t *m);
bool   model_inkling2_save_sel(model_t *m, const int *sel, int n);
bool   model_inkling2_last_sel(const model_t *m, const int **sel_out, int *n_out);
bool   model_inkling2_lora_backward(model_t *m, const int32_t *toks, int T, const float *pos_w, double *loss_out);
#endif
