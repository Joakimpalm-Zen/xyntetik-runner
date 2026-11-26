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
// What the engine holds, for --info fit and the load line (bytes): the decoded float32 tensors, the K/V and
// convolution cache at n_ctx, the most the expert cache can hold (the reference route: its budget, capped at every
// expert decoded), the file's mapping.
// dense_f32: float32 copies of the matrices (the projections, the head, the embedding), 0 on the fast route;
// small_f32: the small tensors decoded on either route (norms, taps, queries, scales, the router)
typedef struct { uint64_t dense_f32, small_f32, cache, expert_cache, mapped, logits; } inkling2_fit;
bool   model_inkling2_fit(model_t *m, inkling2_fit *f);
// R4.48.15: the route this instance runs: 'F' the fast route (the weights in their file formats, decoded at use,
// float32 activations, the house kernels' vectorized sums) or 'R' the reference route (the serial float32 order of
// step 1). RUNNER_INKLING2_ROUTE=reference selects the reference at load.
char   model_inkling2_route(const model_t *m);
// --info fit on an inkling2_loop header (no weights read): what a load at n_ctx and n_batch would hold, by the same
// parts, against the RAM available now. Prints the report; returns the exit code (0 fits, 1 does not).
int    model_inkling2_fit_print(gguf_file *g, const char *path, int n_ctx_want, const model_params *mp, int n_seq);
// R4.48.9: whether this instance's forward runs on CUDA (false: the CPU engine).
bool   model_inkling2_cuda(const model_t *m);
// R4.48.9, TEST-ONLY (G1): read every device-resident matrix and expert stack back and compare it with the host
// decode bit for bit; *n_tensors read, *n_differ differing, the first one's name in first. false: no device copy.
bool   model_inkling2_cuda_readback(model_t *m, int *n_tensors, int *n_differ, char *first, size_t first_cap);
// R4.48.9, TEST-ONLY: whether this instance holds the tensor named (its view's name) on the device.
bool   model_inkling2_cuda_holds(const model_t *m, const char *name);
// R4.48.9, TEST-ONLY: the device calls this instance ran since its load: single products (gpu_ink2_lin) and G2b
// step 1's expert bodies (gpu_ink2_experts, one per application per forward).
void   model_inkling2_cuda_calls(const model_t *m, long *n_products, long *n_expert_bodies);
// R4.48.9, TEST-ONLY: the bind's placement ('W' whole, 'M' expert stacks on the host, 'C' the CPU, 0 none asked), the
// weights' device bytes its fit counted, and the device handle (instances of one file and placement share it).
// True when the instance holds weights on the device.
bool   model_inkling2_cuda_placement(const model_t *m, char *placed, size_t *need, const void **handle);
// R4.48.9, TEST-ONLY (G2b step 2a): the device attention-core calls this instance ran since its load (one per
// application per forward), -1 when its core runs on the host; *kv_bytes its K/V rings' device bytes.
long   model_inkling2_cuda_attn_calls(const model_t *m, size_t *kv_bytes);
// R4.48.9, TEST-ONLY: the free device memory the fit that decided this instance's placement read (the last
// placement it tried), 0 when none ran
size_t model_inkling2_cuda_fit_free(const model_t *m);
// R4.48.9, TEST-ONLY (G2b step 2b): the device attention-block calls this instance ran since its load, -1 when its
// blocks run 2a's path
long   model_inkling2_cuda_block_calls(const model_t *m);
// R4.48.9, TEST-ONLY (D1a): the device message steps this instance ran since its load, -1 when they run on the host
long   model_inkling2_cuda_msg_calls(const model_t *m);
// R4.48.9, TEST-ONLY (D1b): the device router calls this instance ran since its load, -1 when the router runs on the host
long   model_inkling2_cuda_route_calls(const model_t *m);
// R4.48.9, TEST-ONLY (D1d.3b): the resident decode steps this instance ran since its load, -1 when it has no step
long   model_inkling2_cuda_step_calls(const model_t *m);
// R4.48.9, TEST-ONLY (the stage-3 read's F1): the resident step's bytes as the fit counted them against what was
// allocated, each term by the driver's rounding: out[0] the copy's step part as counted, out[1] as allocated (its small
// tensors + the program's allocations), out[2] one slot's post copies as counted, out[3] as allocated; out[4] 1 when
// every program allocation equals the fit's term in order (the index list: at most its bound). False: no step
bool   model_inkling2_cuda_step_bytes(const model_t *m, size_t out[5]);
// R4.48.9, TEST-ONLY: the device products' output buffer's bytes now (the fit's reservation unless a product grew it),
// 0 off the device
size_t model_inkling2_cuda_lin_out_bytes(const model_t *m);
// R4.48.9, TEST-ONLY: the bytes the fit counted for one slot (its K/V rings and, on the block path, its convolution
// histories, as the driver allocates them), and the attention block's buffers' bytes now; 0 off the device
size_t model_inkling2_cuda_slot_fit_bytes(const model_t *m);
size_t model_inkling2_cuda_block_buf_bytes(const model_t *m);
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
