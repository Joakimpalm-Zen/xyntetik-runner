// Backend contract: CUDA / Metal offload, and the batched decode half.
#ifndef RUNNER_GPU_H
#define RUNNER_GPU_H

#include "tdn_args.h"   // the training DeltaNet calls' shape
#include "train_args.h" // the resident step's kernel arguments (R8.24.1 B); never
                        // kernel_args.h here: metal.m keeps its own copies of its structs
#include "model.h"
#include <stdio.h>
#include <string.h>


enum { GPU_AUTO = 0, GPU_OFF = 1 };
bool   gpu_available(char *name, int name_cap);
// Stable identity of the GPU this backend would use, preferring a UUID: on a
// MIG box the parent card's UUID is the same for every slice, so an identity
// that cannot tell slices apart is worse than useless for VRAM accounting.
// Falls back to a bus id, then to "cuda:0". false when there is no GPU.
bool   gpu_device_id(char *id, int id_cap);
// dedicated GPU memory in bytes; false when there is no discrete GPU
// (Metal's unified memory is governed by the RAM reservation instead)
bool   gpu_mem_info(size_t *free_bytes, size_t *total_bytes);
// True when the device shares one memory pool with the CPU (Apple silicon,
// integrated CUDA devices like the DGX Spark's GB10). A caller that budgets
// "RAM plus VRAM" on such a machine counts the same bytes twice.
bool   gpu_unified_memory(void);
// This loaded model's device as WDDM sees it now: the process's video memory
// budget and current usage on the card, and the non-local (shared, system
// memory) usage. For the per-request paging note, so an eviction names
// itself. false without a loaded device or where the OS publishes none.
bool   gpu_os_vidmem(const model_t *m, uint64_t *budget, uint64_t *usage,
                     uint64_t *nonlocal);
// The offload budget a load would split against right now, computed the way
// the CUDA loader computes it (driver free view, bounded by the OS video
// memory budget and --reserve-vram) with the loader's headroom beside it.
// For `--fit`, which has no model to load. false when there is no discrete
// device: Metal and integrated CUDA share one pool with the CPU, where the
// RAM figure already is the answer.
bool   gpu_offload_budget(int reserve_vram_pct, uint64_t *budget,
                          uint64_t *headroom);
// The GPU-side fit ceiling, which is NOT the same number as free RAM: on
// Metal a single resource allocation larger than the device working-set
// limit (~2/3 of unified memory) fails even when the file would fit in RAM.
// A scheduler that only reads ram_available_bytes sees capability where
// there is a capacity cliff. false when the backend has no such limit
// distinct from what vram_bytes already reports (CUDA), or no GPU.
bool   gpu_max_working_set(size_t *bytes);
// SHA-256 of the shader source compiled INTO this binary, or NULL for a
// backend with no embedded shader source. Published in --caps so a gate can
// prove the running binary carries the current kernels: the header-vs-source
// drift check cannot see a stale binary, and twice on 2026-08-07 a kernel
// change was measured against a build that did not contain it.
const char *gpu_shader_source_sha(void);
// the CUDA device's compute capability as "major.minor" (--caps); NULL elsewhere
const char *gpu_compute_capability(void);
// does this backend's attention read a q8_0 KV cache? The CPU path always
// can; a backend that cannot forces the cache back to f16 rather than
// handing q8_0 blocks to kernels that would read them as fp16.
bool   gpu_kv_q8_ok(void);
bool   gpu_kv_fp4_ok(void);   // fp4 (16-in-9) KV rows readable by the attention kernels
bool   gpu_kv_f32_ok(void);   // f32 KV rows (--kv f32) readable by the attention kernels
// Does this backend have kernels for a tensor format? Answered by the backend
// itself so `--caps` cannot drift from what the loader will actually admit.
// The two used to be a hand-kept literal in main.c and a switch in the backend
// carrying "keep this in sync with gpu_type_ok()" — and they did drift, on
// Metal, in the direction that made --caps advertise formats it could not run.
// A published surface an advisor consumes must not be able to lie by omission
// of a manual edit, so there is now one statement of the fact per backend.
bool   gpu_quant_ok(int type);
// Does this backend run at least one sparse-MoE family on the device? Per-model
// gpu_init() guards still decide the exact router/layout/activation variant.
bool   gpu_moe_ok(void);
// Does this backend run the gemma-4 E-series per-layer-embedding path on the
// device? Same shape of claim as gpu_moe_ok: a scheduler asking "can this box
// serve an E-series model on its GPU" needs an answer before it hands one over,
// and until 0.1.11 the honest answer on Metal was no. Per-model gpu_init()
// guards still decide the rest (E2B's per-layer FFN widths, for one).
bool   gpu_eseries_ok(void);
// True when the backend's layer walk omits a removed attention or FFN
// (--remove-sublayer) per block; false makes the loader refuse the offload.
bool   gpu_removed_sublayers_ok(void);
// R4.40: can an instance change its rope frequency table after init
// (model_rope_for_len)? Metal keeps the table per instance and can; CUDA
// keeps both LongRoPE sets' tables in the shared weight set and each instance
// reads its own set's (R4.40.2).
bool   gpu_rope_sets_ok(void);
// The instance's rope_inv_freq (and its sliding-layer copy) changed: copy it
// to the device. Called between forwards, never during one.
void   gpu_rope_changed(model_t *m);
// R4.40.3: can one device microbatch hold columns on different LongRoPE
// factor sets? CUDA's can (each set's columns rotate with that set's table);
// elsewhere a two-set model's sequences decode one by one (model.c).
bool   gpu_batch_rope_sets_ok(void);
// test hook (R4.40.2): copy back the n = rope_dim/2 frequencies the
// instance's next global-layer rope reads on the device; false where the
// instance has no device or the backend cannot say (tests/test_cuda_longrope.c)
bool   gpu_rope_table_in_use(const model_t *m, float *out, int n);
// The most experts and per-token picks a device bank's router takes (CUDA 512
// and 256, Metal 256 and 256); false where no device routes experts. Past
// them a bank is refused the device, by the words model_moe_route_why gives
// the load and the fit alike (R6.68.67).
bool   gpu_moe_route_caps(int *max_experts, int *max_used);
// Whether this backend runs a qwen35 (Gated DeltaNet) model on the device at
// all; where it does not, the whole model, adapters included, runs on the
// CPU (--adapt-info reports no device kernels for it).
bool   gpu_qwen35_ok(void);
// True when gpu_init would reuse device weights already uploaded for this file
// and configuration (CUDA's shared-weight registry); false elsewhere.
bool   gpu_shared_weights_resident(const model_t *m);
// test hook for the TC tolerance gate: force the tensor-core GEMM opt-in on
// (1) or off (0) regardless of RUNNER_CUDA_TC; -1 returns to the env default.
// A no-op on backends without a TC path (Metal, CPU-only builds).
void   gpu_tc_force(int on);
// How many times the tensor-core GEMM has actually been dispatched. The TC
// tolerance gate used to infer engagement from "the logits differ", which
// cannot tell an unused kernel apart from an exactly-matching one — and Q8_0
// turns out to be the second case. Counting the dispatch removes the guess.
unsigned long gpu_tc_dispatches(void);
// The same count per weight type (CUDA: the tensor-core GEMM; Metal: the
// tiled and tensor GEMMs), so the gate can require that EVERY TC-capable
// format in a model dispatched, not just that something did: a total of a
// mixed Q4_K/IQ file can be all Q4_K. 0 for a type outside the table.
unsigned long gpu_tc_dispatches_type(int type);
// whether the route takes the tensor cores for a weight type of this model
// (RUNNER_CUDA_TC and the promotions included): a record's tc_types
bool gpu_tc_takes(const model_t *m, int type);
// test-prefill-bits' route table (CUDA's scalar route, R6.57, R6.58), asked
// for a dense tensor of a weight type: whether the 16-wide twin of the batch-1
// kernel's route is loaded (false for a type outside the table and on other
// backends), and where a matvec of the type over batch (> 1) columns goes,
// unscaled or carrying a per-tensor scale companion (scaled):
//   NONE        no CUDA decode of the type, a batch under 2, another backend
//   TILE_TWIN   the type's tile kernel, itself the batch-1 kernel's twin
//   TWINS       the batch-1 kernel's width-classed twins
//   PER_COLUMN  the batch-1 kernel once per column
//   TILE        a tile kernel that sums in another order than the batch-1 kernel
enum { GPU_PREFILL_NONE, GPU_PREFILL_TILE_TWIN, GPU_PREFILL_TWINS, GPU_PREFILL_PER_COLUMN,
       GPU_PREFILL_TILE };
bool gpu_prefill_twin16(const model_t *m, int type);
int  gpu_prefill_route(const model_t *m, int type, int batch, bool scaled);
// the same question for a tensor of n_in x n_out (R6.62: a route can depend on
// the shape, as BF16's split-K does on qwen4exp)
int  gpu_prefill_route_shape(const model_t *m, int type, int n_in, int n_out, int batch, bool scaled);
// of those, the ones the pipelined TC2 kernels ran (CUDA, RUNNER_CUDA_TC2)
unsigned long gpu_tc2_dispatches(void);
unsigned long gpu_tc2_reused(void);   // TC2 GEMMs that reused the previous GEMM's staging
// GEMMs the int8 tensor-core route ran (CUDA, RUNNER_CUDA_I8, R3.14)
unsigned long gpu_i8_dispatches(void);
// of those, the ones the compute_80 kernels ran (CUDA, a CC >= 8.0 device, E1)
unsigned long gpu_tc2_80_dispatches(void);
// MoE prefill tiles run by expert (moe_twin), and the test override of its switch
unsigned long gpu_moe_twin_tiles(void);
void gpu_moe_twin_force(int on);
// TC2 GEMMs that ran the 128-column tile, and the test override of its switch
unsigned long gpu_tc2_wide_dispatches(void);
void gpu_tc2_wide_force(int on);
// the prompt window (columns) a CUDA instance was built for; 0 on other
// backends and without a device. Transcripts record it and --verify restores it.
int    gpu_prompt_window(const model_t *m);
// The CUDA routes whose sums differ from the reference kernels' (each is
// tolerance-gated, not byte-identical): which of them are on is part of the
// arithmetic a record was made with, so transcripts and receipts record the
// set (profile.gpu_route) and --verify pins it. Routes that only move bytes
// (the KV gather, the whole-tile recurrent kernels) are not listed: they
// cannot change a token. moe-tc (the tensor-core GEMMs on a MoE family's
// dense projections) and moe-grouped (the grouped expert prefill) count
// only for the MoE families measured for them (qwen3moe and gpt-oss; the
// grouped prefill qwen3moe only): a record of another model never names
// them. moe-limb (R3.81, the default since R3.83): gemma-4's Q4_0 expert
// products through three int8 activation limbs, the same bits at any width,
// in the decode and the prefill alike; named only where the instance takes
// it (every routed expert on the device by default; RUNNER_MOE_LIMB=0 holds
// it off).
enum { GR_TC2 = 1, GR_SM80 = 2, GR_ATTN_TILE = 4, GR_ATTN_DEC = 8, GR_ATTN_TC = 16,
       GR_MOE_TC = 32, GR_MOE_GROUPED = 64, GR_MOE_LIMB = 128, GR_ALL = 255 };
// Pin the set for this process (--gpu-route, --verify): it then overrides
// RUNNER_CUDA_ROUTE (the same set from the environment), the per-route
// RUNNER_CUDA_* diagnostics and the defaults. -1 returns to them.
// No-op on the other backends.
void   gpu_route_pin(int mask);
// The set in force: the pin, else the environment and the defaults; with a
// loaded instance, sm80 only where its compute_80 module loaded. 0 on the
// other backends.
int    gpu_route_active(const model_t *m);
// "reference" (no route) or the names joined by commas, in GR_ order; and
// back. gpu_route_parse returns -1 for an unknown name or an empty item.
static const struct { int bit; const char *name; } GPU_ROUTE_NAMES[] = {
    { GR_TC2, "tc2" }, { GR_SM80, "sm80" }, { GR_ATTN_TILE, "attn-tile" },
    { GR_ATTN_DEC, "attn-dec" }, { GR_ATTN_TC, "attn-tc" },
    { GR_MOE_TC, "moe-tc" }, { GR_MOE_GROUPED, "moe-grouped" },
    { GR_MOE_LIMB, "moe-limb" },
};
static inline void gpu_route_format(int mask, char *buf, size_t n) {
    size_t at = 0;
    buf[0] = 0;
    for (size_t i = 0; i < sizeof GPU_ROUTE_NAMES / sizeof *GPU_ROUTE_NAMES; i++)
        if (mask & GPU_ROUTE_NAMES[i].bit && at < n)
            at += (size_t)snprintf(buf + at, n - at, "%s%s", at ? "," : "",
                                   GPU_ROUTE_NAMES[i].name);
    if (!at) snprintf(buf, n, "reference");
}
static inline int gpu_route_parse(const char *s) {
    if (!s || !*s) return -1;
    if (!strcmp(s, "reference")) return 0;
    int mask = 0;
    while (*s) {
        const char *e = strchr(s, ',');
        size_t len = e ? (size_t)(e - s) : strlen(s);
        int bit = 0;
        for (size_t i = 0; i < sizeof GPU_ROUTE_NAMES / sizeof *GPU_ROUTE_NAMES; i++)
            if (strlen(GPU_ROUTE_NAMES[i].name) == len &&
                !strncmp(GPU_ROUTE_NAMES[i].name, s, len)) bit = GPU_ROUTE_NAMES[i].bit;
        if (!bit) return -1;
        mask |= bit;
        if (!e) break;
        s = e + 1;
        if (!*s) return -1;
    }
    return mask;
}
// A record's gpu_route for an instance: NULL unless it runs on CUDA (the
// backend with a prompt window), else the routes it runs with.
static inline const char *gpu_route_record(const model_t *m) {
    static _Thread_local char buf[96];
    if (gpu_prompt_window(m) <= 0) return NULL;
    gpu_route_format(gpu_route_active(m), buf, sizeof buf);
    return buf;
}
// the whole-tile recurrent prefill kernels (CUDA, RUNNER_CUDA_RECUR_SEQ):
// force them on (1) or off (0) for a test, -1 returns to the switch; and
// the layer-tiles they advanced
void   gpu_recur_seq_force(int on);
unsigned long gpu_recur_seq_dispatches(void);
// speculative verify tiles run with the one-token decode kernels' arithmetic
// per row (CUDA, model_forward_verify), so each row is a solo step's bits
unsigned long gpu_exact_verify_tiles(void);
// Does this backend carry a batched (tensor-core or tiled) GEMM for the
// type? A static fact about the backend's kernel table, answered without a
// device, and the gate's only source for "TC-capable" (its own hand-kept
// list missed Q6_K for five weeks).
bool   gpu_tc_type_has_kernel(int type);
// test hook for the fast-matvec tolerance gate: force the reassociating decode
// matvec on (1) or off (0) regardless of RUNNER_METAL_MV; -1 returns to the
// env default. A no-op on backends without one (CUDA, CPU-only builds).
//
// Why a second kernel rather than a faster first one: the shipped matvec is
// the CPU<->GPU byte-identity contract, which pins which lane owns which
// block and therefore forbids every reassociating transformation worth having
// (see docs/negative-result-metal-multirow-matvec.md). The fast route answers
// to tests/test_mv_tol.c the way the tiled prefill GEMM answers to
// test_tc_tol.c, and the identity route stays reachable and unchanged.
void   gpu_mv_force(int on);
// Test-only (R6.62): y = W x over tn columns through the device encoder's own
// route choice (enc_mv) on m's device context, the way a prompt's product of w
// would run. x holds tn rows of n_in, y tn rows of n_out; w must be resident.
// No CLI path calls it. false = no device, w not resident, or a failed launch.
bool   gpu_test_mv(model_t *m, gguf_tensor *w, const float *x, float *y,
                   int n_in, int n_out, int tn);
// Test-only (R3.81): one gemma-4 Q4_0 expert's product (proj 0 gate_up, 1
// down) over cols columns of x through the production launches: each column
// alone through the decode's expert matvec (prefill 0), or all of them in one
// call through the by-expert prefill's product (1). route 0 today's, 1 the
// grouped prefill's (prefill only), 2 moe-limb, 3 the instance's own (R3.83:
// what its load chose). false: no device, not such a layer, a route this
// build or instance lacks, or a failed launch.
bool   gpu_test_moe_q4_0(model_t *m, int layer, int proj, int expert, int route, int prefill,
                         const float *x, int cols, float *y);
// How many moe-limb prefill tiles (the int8 MMA kernel) have run: a gate that
// compares the prefill with the decode must see that the tile ran, since its
// fallback is the decode kernel itself. 0 on the other backends.
unsigned long gpu_moe_limb_tiles(void);
// Whether m's device has moe-limb's int8 prefill tile (the compute_80 module
// loaded: CC 8.0 or newer, RUNNER_CUDA_SM80 not 0). Without it the prefill
// runs the decode kernel a column at a time. false on the other backends.
bool   gpu_moe_limb_has_tile(const model_t *m);
// How many times the fast matvec has actually been dispatched. Same reason the
// TC path counts: "the logits differ" cannot tell an unused kernel apart from
// an exactly-matching one, and a gate that cannot see its own subject passes
// vacuously.
unsigned long gpu_mv_dispatches(void);
unsigned long gpu_mv2_dispatches(void);   // Metal two-row identity decode matvecs
// test hook for the cooperative-KV attention gate: force the reassociating
// score read on (1) or off (0) regardless of RUNNER_METAL_ATTN_COOP; -1
// returns to the env default. A no-op on backends without one.
void   gpu_attn_coop_force(int on);
unsigned long gpu_attn_coop_dispatches(void);
unsigned long gpu_attn_tile_dispatches(void);  // prefill tiles (Metal), 0 elsewhere
unsigned long gpu_attn_tiled_dispatches(void); // tiled prefill attention (CUDA), 0 elsewhere
// CUDA's attention routes and KV copy-back, so the kernel counters say which
// ran (attn-tc falls back to the tiled kernel where it does not apply);
// 0 on every other backend
unsigned long gpu_attn_tc_dispatches(void);     // tensor-core prefill attention tiles
unsigned long gpu_attn_dec_gqa_dispatches(void); // grouped decode attention launches
unsigned long gpu_attn_dec_mla_dispatches(void); // absorbed MLA decode attention launches (R3.31)
unsigned long gpu_attn_tcw_mla_dispatches(void); // absorbed MLA tensor-core prefill attention launches (R3.32)
unsigned long gpu_kv_gather_tokens(void);       // rows (decoded tokens, forwards of <= 16 rows) whose K/V came back in one copy
unsigned long gpu_attn_gqa_dispatches(void);   // grouped-query decode groups (Metal), 0 elsewhere
// CUDA's BF16 split-K decode GEMV (k_mv_bf16_splitk) launches, so its gate
// can prove the route ran; 0 on every other backend.
unsigned long gpu_splitk_dispatches(void);
// its width-classed twin (k_mv_bf16_splitk_b) launches, one per pass of up to
// 16 prompt columns (R6.62); 0 on every other backend
unsigned long gpu_splitk_twin_dispatches(void);
// Device allocations the engine holds right now (every cuMemAlloc it made
// and has not freed), or -1 where the backend does not count them. A model
// loaded and freed must leave it where it was (tests/test_cuda_teardown.c).
long gpu_live_buffers(void);
// R6.90: NULL, or the first kernel module that did not load on a device that
// was up, with the CUDA error and the driver's JIT log. A gate that finds no
// engaged device asks this before it skips (tests/gate_gpu.h): a module the
// driver rejected is a failure, not "no device". NULL on Metal and the CPU.
const char *gpu_module_failure(void);
// R6.90 (review P2): the oldest compute capability each embedded PTX image
// runs on. k_ptx_src is built for compute_75 (runner.h RUNNER_CUDA_MIN_CC),
// k_ptx80_src for compute_80. A device below an image's minimum cannot JIT it:
// that is no usable device (a skip), not a module that failed to load, so
// cuda.c checks this before the load and records no failure for it.
#define GPU_CUDA_MIN_CC_MAJOR 7
#define GPU_CUDA_MIN_CC_MINOR 5
static inline bool gpu_cc_at_least(int major, int minor, int need_major, int need_minor) {
    return major > need_major || (major == need_major && minor >= need_minor);
}
static inline bool gpu_cc_usable(int major, int minor) {
    return gpu_cc_at_least(major, minor, GPU_CUDA_MIN_CC_MAJOR, GPU_CUDA_MIN_CC_MINOR);
}
// test hook: force (1) or forbid (0) the eager MoE routing path; -1 = env
void   gpu_moe_eager_force(int on);
bool   gpu_init(model_t *m);                     // false = unsupported, use CPU
// process n tokens starting at pos (prompt batches); on success returns true
// and sets *logits to the last token's logits when want_logits (else NULL).
// false = failed, use CPU.
// transposed matvec on the device (training backward, D8 slice 1): dx += W^T
// dy with the CPU trainer's exact accumulation chain; false = no backend /
// no kernel for the type, the caller stays on the CPU path.
bool   gpu_mvt(model_t *m, const gguf_tensor *w, const float *dy, float *dx,
               int n_in, int n_out, int batch);

// Forward matvec in the CANONICAL order (R8.7.1 slice 1): y = W·x associated
// exactly the way vec_dot associates it under RUNNER_CANON_KERNELS, so the
// device answer is bit-identical to the host's instead of close to it. The
// point is the trainer's forward, which is 80% of a training step and stays
// on the CPU today only because moving it would cost the byte contract.
// false = no backend, no kernel for the type (F32, F16, Q8_0, Q4_0, Q4_K, Q5_K
// and Q6_K have a canonical order; R12.1b gave the last four), a scaled
// tensor, or a tensor that is not device-resident.
bool   gpu_mvcanon(model_t *m, const gguf_tensor *w, const float *x, float *y,
                   int n_in, int n_out);

// D8 slice 2: the standalone training context. --train keeps the model
// CPU-resident (m->gpu stays NULL); this context owns its own CUDA state,
// uploads each weight tensor ONCE on first use, and runs the backward's
// dx += W^T dy on the device — the kernel proven bit-identical to the CPU
// chain in slice 1, so a call that returns false (no CUDA, no kernel for
// the type, VRAM budget exhausted) just leaves that tensor on the CPU path
// with the SAME bytes as the result. Opt-in from --train via
// RUNNER_TRAIN_GPU=1. All three are no-ops / false without CUDA.
bool   gpu_train_init(model_t *m);
void   gpu_train_free(model_t *m);
bool   gpu_train_mvt(model_t *m, const gguf_tensor *w, const float *dy,
                     float *dx, int n_in, int n_out, int nb);
// R8.24.1: the same product for a dx the caller holds at zero (+0.0f, every
// element): the device clears its copy instead of receiving the zeros, so
// the result is gpu_train_mvt's bit for bit with one transfer fewer
bool   gpu_train_mvt0(model_t *m, const gguf_tensor *w, const float *dy,
                      float *dx, int n_in, int n_out, int nb);
// R8.19: y[t][j] = sum_i W[j][i] x[t][i] (+ bias[j]) for t < nb on the same
// context, rows x_stride / y_stride apart: a batch-invariant product (column
// t's value does not depend on the other columns of the call), the training
// step's forward products in device mode. Not the CPU's bits. false = no
// context, no kernel for the type, a scaled tensor, or the VRAM budget: the
// caller computes it on the CPU.
bool   gpu_train_mm(model_t *m, const gguf_tensor *w, const float *x,
                    int x_stride, float *y, int y_stride, int n_in, int n_out,
                    const float *bias, int nb);
// R8.24.1: up to GPU_MM_PARTS of those products of ONE input x in one round
// trip (x sent once, one sync); each part is gpu_train_mm's product bit for
// bit, the weights taken up in the parts' order. Returns the parts computed,
// bit s for part s; the caller computes the others (0: none, as when
// gpu_train_mm returns false).
enum { GPU_MM_PARTS = 4 };
typedef struct {
    const gguf_tensor *w;
    float *y;
    int y_stride, n_out;
    const float *bias;
} gpu_mm_part;
unsigned gpu_train_mm_parts(model_t *m, const gpu_mm_part *p, int np,
                            const float *x, int x_stride, int n_in, int nb);
// R8.24.1: pinned host memory for the context's copies; only their speed
// depends on it, never a value. gpu_train_pin registers [p, p + bytes) with
// the driver until gpu_train_free (the range must outlive the context);
// false leaves it pageable. gpu_train_host_buf hands out slot's pinned block
// of at least bytes, owned by the context and the same block from call to
// call (its contents are not kept when it grows); NULL when none can be had.
enum { GPU_TRAIN_HBUF = 4 };
bool   gpu_train_pin(model_t *m, void *p, size_t bytes);
void  *gpu_train_host_buf(model_t *m, int slot, size_t bytes);
// R8.24.1 B1, the kernel gate's entry: the device twins of a gated FFN's
// activation and adjoint over n host floats, h = act(g)*u, dg = dh*u*act'(g),
// du = dh*act(g) (model.c gated_act, act_d, act_f). false: no context, no
// kernel in this PTX, or an activation they do not cover (ACT_SILU only).
// R8.24.1 B2, the kernel gate's entry: the device twins of rmsnorm and its
// adjoint (k_trn_fw, k_trn_bw) over a->rows rows of n host floats: y = the
// forward of x, dx += the adjoint of dy (rows at the strides a gives). w: one
// n-vector for every row, or NULL (the weightless form). false: no context or
// no kernel in this PTX.
bool   gpu_train_rmsnorm_check(model_t *m, const trn_args *a, const float *x,
                               const float *w, const float *dy, float *y, float *dx);
// R8.24.1 B3, the kernel gate's entry: the device twin of rope_apply
// (a->adjoint 0) or rope_unapply (1), k_trope, over a->T host rows of v
// (a->stride floats apart) in place; fr: the layer's a->half_dim inverse
// frequencies. false: no context or no kernel in this PTX.
bool   gpu_train_rope_check(model_t *m, const trope_args *a, const float *fr, float *v);
// R8.24.1 B8, the kernel gate's entry: the device twin of the forward's K/V
// store (k_tkv_store) over a->rows host rows of a->n floats (a->src_stride
// apart) into dst's rows (a->row_bytes apart) in kind 0 (f16, fp16.h's
// rounding) or 3 (f32). dst goes up first: the bytes between rows come back
// unchanged. false: no context, no kernel in this PTX, or another kind.
bool   gpu_train_kv_store_check(model_t *m, const tkv_args *a, const float *src,
                                unsigned char *dst);
// R8.24.1 B R1, the resident step (out/r8241/B-resident-design.md): the
// device bytes its planes may take beside what is held now; its planes for a
// step of T rows (the training K/V rows, the norm weights, the rope tables);
// one taped forward round of n rows at pos on the device (m->x in, the final
// hidden back in m->x, the tape's and the K/V rows kept on the device for the
// resident backward). false: the step fails.
size_t gpu_train_res_avail(model_t *m);
// the bytes the training contexts' scratch planes hold now: 0 after every
// gpu_train_free (#935: the resident planes were never freed)
size_t gpu_train_scratch_bytes(void);
bool   gpu_train_res_begin(model_t *m, int T);
bool   gpu_train_res_round(model_t *m, int n, int pos);
// R2: the resident backward. dx (the head's adjoint, T rows) up once; layer
// l's backward on the device's tape and K/V rows, dx staying on the device;
// every launch done. false: the step fails.
bool   gpu_train_res_bw_begin(model_t *m, const float *dx, int T);   // dx NULL: R3's head wrote it
// R3: the head on the device for a resident step. _begin: its planes for a
// round of `rows`, the output norm and suppressed ids up once, the step's dx
// plane zeroed; NULL, or why the head stays on the host this step. Per round
// after gpu_train_res_round(m, n, pos): the nc scored rows (nruns runs of the
// round's rows, run_b[k] .. run_b[k] + run_len[k] - 1), their targets and
// weights (pos_w NULL: none), the head's adjoint into dx at those positions,
// each row's normalizer (lse) and transformed target logit back. false: the
// step fails.
const char *gpu_train_res_head_begin(model_t *m, int rows);
bool   gpu_train_res_head(model_t *m, int n, int pos, const int *run_b, const int *run_len,
                          int nruns, const int32_t *targets, const float *pos_w, int nc,
                          double *lse, float *z_target);
bool   gpu_train_res_layer_bw(model_t *m, int l);
bool   gpu_train_res_bw_end(model_t *m);
// R8.24.1 B R0, the kernel gate's entry: the residual add's twin (k_tadd),
// x (a->rows host rows, a->x_stride apart) += y (a->y_stride apart) in
// place. false: no context or no kernel in this PTX.
bool   gpu_train_add_check(model_t *m, const tadd_args *a, float *x, const float *y);
// R8.24.1 B4, the kernel gate's entry: the device twin of the forward's
// attention (k_tattn, attn_run for one (t, head) a block) over a->T positions:
// q rows a->q_dim apart in, the K / V rows (a->k_row_b / v_row_b bytes apart,
// kinds a->kind_k / kind_v: 0 f16, 3 f32) in, ao rows out. false: no context,
// no kernel in this PTX, or a shape or kind it does not take.
bool   gpu_train_attn_fw_check(model_t *m, const tattn_args *a, const void *kc,
                               const void *vc, const float *q, float *ao);
// R8.24.1 B5, the kernel gate's entry: the device twin of the head (k_thead)
// over a->rows host rows of a->V logits a->stride apart, in place: for each
// row with targets[r] >= 0, lse[r] and z_target[r] (the transformed row's
// normalizer and target logit) and the row's adjoint (pos_w[r] its weight,
// pos_w may be NULL; suppress: a->n_suppress ids). false: no context or no
// kernel in this PTX.
bool   gpu_train_head_check(model_t *m, const thead_args *a, float *z, const int32_t *targets,
                            const float *pos_w, const int32_t *suppress, double *lse,
                            float *z_target);
// R8.24.1 B7, the kernel gate's entry: the device twin of adam_buf (k_tadam)
// over a->n host parameters, th, mo and vo in place. false: no context or no
// kernel in this PTX.
bool   gpu_train_adam_check(model_t *m, const tadam_args *a, float *th, const float *g,
                            float *mo, float *vo);
// R8.24.1 B R4, the kernel gate's entry: the device twin of
// model_lora_grad_scale's loop (k_tscale), g[i] *= f over n host floats in
// place. false: no context or no kernel in this PTX.
bool   gpu_train_scale_check(model_t *m, float *g, size_t n, float f);
// R4 of the resident step, AdamW on the device (model_lora_res_adam_step,
// which takes the step's norms and their finite check from the host's code on
// the host's copy of the gradients). gpu_train_res_adam_ok: its kernels are in
// this PTX. The moments live in two arenas laid out by the sites the step
// pushed (per layer Q K V O gate up down, each A then B: the host's table
// order), zeroed at the start. _optim_put: a site's moments up (NULL arrays:
// zeros); _optim_get: back. gpu_train_res_adam: each site's own gradients
// times scale where it is not 1 (k_tscale, in place), checked finite
// (k_tfinite), and when all are AdamW (k_tadam: a's constants, a->n per
// segment) on every site's A and B with the arenas' moments. plant_nan, a test
// seam: the device's first gradient made NaN before the check. 1: done; 0: a
// gradient on the device is not finite, nothing updated; -1: the device
// failed. gpu_train_site_ab_pull: a site's A and B back to the host.
bool   gpu_train_res_adam_ok(model_t *m);
bool   gpu_train_res_optim_put(model_t *m, int l, int which, const float *ma, const float *va,
                               const float *mb, const float *vb);
bool   gpu_train_res_optim_get(model_t *m, int l, int which, float *ma, float *va, float *mb,
                               float *vb);
int    gpu_train_res_adam(model_t *m, float scale, const tadam_args *a, bool plant_nan);
bool   gpu_train_site_ab_pull(model_t *m, int l, int which, float *a, float *b);
bool   gpu_train_ffn_check(model_t *m, int act, const float *g, const float *u,
                           const float *dh, float *h, float *dg, float *du, int n);
// R8.24.1 B6, the kernel gate's entry: the device twins of one site's adapter
// on host arrays, in the host's order (lora_apply, then the backward's passes):
// y += s B (A x); ga, gb accumulate; dx += s A^T (B^T dy). x, dx rows n_in
// apart, y, dy rows n_out apart. false: no context or no kernel in this PTX.
bool   gpu_train_lora_check(model_t *m, int r, int n_in, int n_out, int nb, float s,
                            const float *A, const float *B, const float *x,
                            const float *dy, float *y, float *ga, float *gb, float *dx);
// R8.24.1 B: the device-resident FFN of a gated SiLU layer (the products, B1,
// B6), for --train-gpu. gpu_train_ffn_dev_ok: its kernels are in this PTX and
// the device holds every tensor. A backward pushes each FFN site's adapter
// (which 0 gate, 1 up, 2 down; ga and gb continue from the host's values) and
// pulls the gradients back at its end. gpu_train_ffn_fw: the taped forward's
// FFN, x in, the down output into y (n_embd of each row, strides in floats;
// false before y is touched). gpu_train_ffn_recompute: the backward's
// recompute of the gate, up and act(g)*u of xn2, kept on the device for
// gpu_train_ffn_bw: the three sites' backward in the host's order, dbr in,
// dxn2 out (T packed rows).
bool   gpu_train_ffn_dev_ok(model_t *m);
bool   gpu_train_site_push(model_t *m, int l, int which, const float *a,
                           const float *b, const float *ga, const float *gb,
                           int r, int n_in, int n_out, float s);
bool   gpu_train_site_pull(model_t *m, int l, int which, float *ga, float *gb);
bool   gpu_train_ffn_fw(model_t *m, int l, const gguf_tensor *wg, const gguf_tensor *wu,
                        const gguf_tensor *wd, const float *x, int x_stride, float *y,
                        int y_stride, int E, int nff, int nb);
// R8.24.1 B, the layer driver's first slot: as gpu_train_ffn_fw with x the
// residual rows, the FFN's RMS norm (nw, NULL the weightless form, eps) run on
// the device first (B2's k_trn_fw, model.c rmsnorm's twin). false before y is
// touched: the caller then runs the host's norm and FFN.
bool   gpu_train_ffn_fw_norm(model_t *m, int l, const gguf_tensor *wg, const gguf_tensor *wu,
                             const gguf_tensor *wd, const float *nw, float eps,
                             const float *x, int x_stride, float *y, int y_stride, int E,
                             int nff, int nb);
bool   gpu_train_ffn_recompute(model_t *m, int l, const gguf_tensor *wg,
                               const gguf_tensor *wu, const float *xn2, int E, int nff,
                               int T);
bool   gpu_train_ffn_bw(model_t *m, int l, const gguf_tensor *wg, const gguf_tensor *wu,
                        const gguf_tensor *wd, const float *dbr, float *dxn2, int E,
                        int nff, int T);
// RUNNER_TRAIN_PROF: seconds inside gpu_train_mm / gpu_train_mvt (transfers
// and syncs included), their counts, and the kernels' share (the launch and
// its sync) since the last call, which resets them
void   gpu_train_prof(model_t *m, double *t_mm, long *n_mm, double *t_mvt,
                      long *n_mvt, double *k_mm, double *k_mvt);
// the same counters' wall time (gpu_train_mm + gpu_train_mvt) so far, read
// without resetting them: the profile samples it at every phase boundary to
// split a phase's time into the device products' and the host's
double gpu_train_dev_wall(model_t *m);
// R8.21: the Gated DeltaNet recurrence of a --train-gpu step on the device,
// one step function (kernels_train.cuh tdn_rows) for the taped forward's
// rounds, the backward's recompute and its reverse sweep, so all three see
// the same states. Not the CPU's bits; deterministic. gpu_train_dn_ok: the
// context has the kernels and takes this shape (hv == sk, a multiple of
// TDN_ROWS up to TDN_MAX_HV), the same answer for every call of a run.
// gpu_train_dn_fw: y [T][inner] from cv / beta / decay (kernel_args.h
// tdn_args.h). With a state ([nh][hv][hv], read and written back) it is a
// forward round; with state NULL it is the recompute, from a zero state,
// and it keeps its inputs and a checkpoint every C positions on the device
// for gpu_train_dn_bw on the same cv, which writes dq, dk [nh][T][sk], dv
// [T][inner], dbeta and ddecay [T][nh] from dy [T][inner]. false = the
// caller computes that part on the CPU (a failure is noted once on stderr).
bool   gpu_train_dn_ok(model_t *m, const tdn_args *a);
bool   gpu_train_dn_fw(model_t *m, const tdn_args *a, const float *cv,
                       const float *beta, const float *decay, float *state,
                       float *y);
bool   gpu_train_dn_bw(model_t *m, const tdn_args *a, const float *cv,
                       const float *dy, float *dq, float *dk, float *dv,
                       float *dbeta, float *ddecay);
// the record's count of those calls: run on the device, and failed there
// (each failed one ran on the CPU)
void   gpu_train_dn_counts(model_t *m, long *dev, long *failed);
// A forward round's state stays on the device for the next round, and the
// host's copy (the state argument) goes stale. gpu_train_dn_pull brings
// every such state back to its host buffer (false = one could not be, and
// the host's copy is wrong); the trainer calls it where its device mode
// ends and before any CPU code reads a state. gpu_train_dn_reset: the host
// states were zeroed (model_recurrent_reset), so are the device's.
bool   gpu_train_dn_pull(model_t *m);
// A --train-gpu step's attention backward (the score and softmax adjoint,
// model.c attn_bw_run) on the device: dq += .., dk += .., dv += .. from the
// layer's f16 K / V cache rows, q and dao (tdn_args.h tab_args). Within
// rounding of the CPU's (its probabilities are recomputed in another
// order); deterministic. false = the caller computes it on the CPU (a
// failure is noted once on stderr). gpu_train_attn_counts: the record's
// calls run on the device and failed there.
bool   gpu_train_attn_bw(model_t *m, const tab_args *a, const void *kc,
                         const void *vc, const float *q, const float *dao,
                         float *dq, float *dk, float *dv);
void   gpu_train_attn_counts(model_t *m, long *dev, long *failed);
void   gpu_train_dn_reset(model_t *m);
// where the weights went: tensors and bytes cached on the device, the ones
// the budget kept on the CPU, and the budget with whether a flag set it (the
// train record names all of it: the split decides the adapter's bits, and
// --train-gpu-budget with the recorded budget reproduces the split)
void   gpu_train_placement(model_t *m, int *on_dev, size_t *dev_bytes,
                           int *on_cpu, size_t *cpu_bytes, size_t *budget,
                           bool *budget_set);

// D2 on the device (R8.7.2): place the loaded adapter's A/B in VRAM for every
// GPU-resident layer, so an offloaded projection gets the same y += scale*B(Ax)
// the host hook applies to a CPU-resident one. Called once, after the adapter
// is parsed and before any forward. false = this backend cannot apply adapters
// on the device (no kernels, no VRAM, a rank past the kernel's bound), and the
// CALLER MUST REFUSE the combination: serving an unadapted model quietly is
// invisible in the output and reads as a weak adapter. Returns true having
// done nothing when the split left every adapted layer on the host.
bool   gpu_lora_bind(model_t *m);
void   gpu_lora_unbind(model_t *m);

bool   gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                         bool want_logits, float **logits);
void   gpu_free(model_t *m); // releases GPU buffers; KV pointers become invalid
// Stop using the GPU for this model after a runtime failure, releasing
// whatever the backend can release while leaving the model usable on the CPU.
// Distinct from gpu_free: a backend whose KV cache lives in GPU-owned memory
// (Metal's unified buffers) cannot free it here, because the CPU path is about
// to read those very rows. CUDA can and does — the host KV copy is
// authoritative there — so a slot that falls back mid-run hands its VRAM, and
// its reference to the shared weights, straight back.
void   gpu_disable(model_t *m);

// R3.68: rows [0, rows) of the KV cache ready for the device before a forward
// encodes them. false: they do not fit (why says why), and the caller
// continues on the CPU, which reads the same rows from the same host memory.
// A backend whose cache is allocated whole returns true.
bool   gpu_kv_reserve(model_t *m, int rows, char *why, size_t cap);
// R3.68: the KV rows the device maps now and its grow step, for the route
// record; false where the cache is allocated whole.
bool   gpu_kv_rows(const model_t *m, int *mapped, int *step);
// R6.98: last runtime failure detail on the active backend (command buffer
// NSError, scratch allocation, weight-wrap, or KV-map failure), or "" when
// none. Read by gpu_runtime_failure() before gpu_disable() releases the handle.
const char *gpu_last_error(const model_t *m);

// Recurrent-state transfer (turn mark, model_recurrent_mark). A backend that
// keeps the recurrent fold on the device (CUDA: q35_hist/q35_state for
// qwen35, mamba_conv/mamba_state for the Mamba-2 families) copies every
// device-resident recurrent layer's LIVE state to / from the host buffers
// (m->ssm_conv_state / m->ssm_state_mem) at that layer's offset, so the
// host-side snapshot seam sees the whole fold. Host-resident layers (past
// gpu_layers) are untouched: their rows are already live. A backend whose
// recurrent layers all run on the host (Metal, none) moves nothing and
// returns true. false = a copy failed and the caller must treat the fold as
// lost (recompute from position 0).
bool   gpu_recurrent_download(model_t *m);
bool   gpu_recurrent_upload(model_t *m);
// Speculative rollback of a device-resident fold without the host round
// trip: mark before the verify forward, roll back after it (valid for exactly
// one forward in between; false = rebuild the fold from the history).
bool   gpu_recurrent_rollback_mark(model_t *m);
bool   gpu_recurrent_rollback(model_t *m);
bool   gpu_recurrent_restore_row(model_t *m, int r);   // after row r of the last keep-forward

// NextN/MTP head on the device (hyper-connection families, full split). The
// hidden rows the head consumes stay on the device: note_row copies the
// trunk's wide residual row into the pending slot, queue_pending files it
// into the queue slot, run drafts from the queue (or from the chained hidden
// when `chained`) and leaves the logits in m->mtp_logits. false from bound =
// the CPU head (model.c mtp_run) is the one in use.
bool   gpu_mtp_bound(const model_t *m);
bool   gpu_mtp_note_row(model_t *m, int row);
bool   gpu_mtp_queue_pending(model_t *m, int slot);
bool   gpu_mtp_reset_pending(model_t *m);
bool   gpu_mtp_run(model_t *m, bool chained, int slot0, const int32_t *tok, int n, int pos,
                   bool want_logits);   // n <= 64 rows per call

// ------------------------------------------- batched decode (backend half)
//
// One decode step for several independent sequences in a single microbatch.
// Every sequence keeps its own KV cache, its own position and its own logits;
// what is shared is the *work*, because a decode step is weight-bandwidth
// bound and reading the weights once for N tokens costs barely more than
// reading them once for one.
//
// The contract that makes this usable is numerical, not structural: a batched
// step must produce, for each sequence, the bits a lone step would have
// produced. Nothing here reduces across sequences, and the matvecs pick the
// multi-column twin of whatever kernel the batch-1 path would have used, so
// that holds by construction rather than by luck. See gpu_batch_decode.
typedef struct gpu_batch gpu_batch;
// Group n sequences loaded from one file into a reusable microbatch context.
// NULL = this backend/model cannot batch, and the caller decodes one by one.
// Speculative verification on a fully offloaded target needs two things the
// CPU path gets for free: readable post-final-layer hidden state (so the
// per-row head can run) and per-row logits. A unified-memory backend can
// provide both; a discrete one cannot without copies it does not have.
// gpu_spec_keep_ok says whether THIS backend supports it; gpu_spec_logits
// returns row b of the logits the last spec-flagged forward produced.
bool       gpu_spec_keep_ok(const model_t *m);
// The host wrote KV rows from `lo` on behind the backend's back (a prefix
// snapshot or a session image installed into the host cache): a backend that
// mirrors the cache on the device takes them from the host before it reads
// them again. A unified-memory backend's buffer is the host cache.
void       gpu_kv_host_wrote(model_t *m, int lo);
// Whether a full offload copies every row's final residual into m->x
// when m->want_hidden is set (model_hidden_states). A backend that
// says no is run on the CPU for that forward instead.
bool       gpu_hidden_rows_ok(const model_t *m);
float     *gpu_spec_logits(model_t *m, int row);

// The backend's own facts for the route record (model_route_note, plan P1):
// CUDA whether single-token decode replays its captured graph, Metal whether
// decode fusion is on; "" when the backend has none to add.
const char *gpu_route_detail(const model_t *m);

gpu_batch *gpu_batch_create(model_t **seqs, int n);
void       gpu_batch_free(gpu_batch *b);
// Evaluate one token for each of the n sequences named by idx (indices into
// the create() array). tok[i]/pos[i] are that sequence's token and KV write
// position; out[i] receives its logits, owned by the batch and valid until
// the next call. false = nothing was evaluated, decode these sequences singly.
bool       gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                            const int *pos, int n, float **out);
// R3.82: the last gpu_batch_decode returned false after its step ran: a
// member's recurrent state on the device may have advanced, so the caller must
// not decode those sequences again on the device. It recovers each one with
// gpu_recurrent_rescue and releases its backend (model_batch_decode). False
// after a plain refusal ("nothing was evaluated").
bool       gpu_batch_failed_stepped(const gpu_batch *b);
// The sequence's pre-forward recurrent snapshot to the host's fold buffers,
// every copy checked; true when the backend holds none to read. False: the
// host has no verified pre-step state.
bool       gpu_recurrent_rescue(model_t *m);
// A batch holds its own reference to the device data its members share, so
// its graphs and buffers outlive the members' backends safely; this releases
// them, and that reference, once the batch can no longer step (its lead gave
// up its backend). The batch then declines every step. No-op otherwise.
void       gpu_batch_release_orphaned(gpu_batch *b);
// Did the last successful decode use non-identity Metal GEMM tiles for any
// weight product? A policy switch alone is not evidence that a tile ran.
bool       gpu_batch_tiles_used(const gpu_batch *b);

// R4.48.9: Inkling 2 Mini's weights on the device: each tensor its rows as the file stores them (the type and row
// stride of the fast route's views; an expert stack's per-expert factors beside its rows, never folded in). A handle
// owns its device buffers. gpu_ink2_put copies one tensor up: its index, or -1 (no device, or no device memory).
// gpu_ink2_get copies tensor i back (rows: the bytes put; scale: the factors put, or NULL). The test-only readback
// (model_inkling2_cuda_readback) decodes both sides with dequant_row and compares them bit for bit.
typedef struct gpu_ink2 gpu_ink2;
gpu_ink2  *gpu_ink2_bind(void);
int        gpu_ink2_put(gpu_ink2 *h, const void *rows, size_t bytes, const float *scale, int n_scale);
bool       gpu_ink2_get(gpu_ink2 *h, int i, void *rows, size_t bytes, float *scale, int n_scale);
size_t     gpu_ink2_bytes(const gpu_ink2 *h);   // the device bytes held
// whether the handle's products (gpu_ink2_lin) take tensors of this ggml type: its module has the type's kernel
bool       gpu_ink2_runs(const gpu_ink2 *h, int type);
// the products' activation and output buffers allocated now at these sizes (the largest product's), so a fit check
// made after it counts them; gpu_ink2_lin still grows them for a larger product. False: no device memory.
bool       gpu_ink2_reserve(gpu_ink2 *h, size_t x_bytes, size_t y_bytes);
// G2a: y[t][o] = scale * dot(row row0 + o of tensor i, x[t]) for t < T, o < n_out, on the device: the activations up,
// one batch-1 matvec launch per column (each column's sums in that kernel's fixed order whatever T is), the rows back.
// x is T rows n_in apart, y T rows n_out apart (contiguous; the caller gathers). type: the tensor's ggml type,
// row_bytes its row stride. scale other than 1 only for NVFP4 (the kernel's own factor). False: a device failure.
bool       gpu_ink2_lin(gpu_ink2 *h, int i, int type, size_t row_bytes, int64_t row0, int n_out, int n_in, int T,
                        const float *x, float *y, float scale);
// G2b step 1: one application's expert body on the device, the router's choice given (chosen [T][E] 0/1, wt [T][E]):
// for each used expert in ascending index, its positions' x gathered, gate_up (tensor i13, rows e*2F.., type13,
// rb13 a row) a column at a time with factor s13[e] (NULL: 1), Inkling's SwiGLU (gate and up capped by gate_cap and
// up_cap, times the gate's sigmoid), down (i2) with s2[e], and out[t] += wt[t][e] * down in ascending expert index.
// x and out [T][H]. Factors only on NVFP4. False: a device failure, or the kernels are not in the module
// (gpu_ink2_experts_ok). gpu_ink2_reserve_experts allocates its buffers for T columns now (for the fit check).
bool       gpu_ink2_experts(gpu_ink2 *h, int i13, int i2, int type13, int type2, size_t rb13, size_t rb2, int H, int F,
                            int E, int T, const float *x, const unsigned char *chosen, const float *wt,
                            const float *s13, const float *s2, float gate_cap, float up_cap, float *out);
bool       gpu_ink2_experts_ok(const gpu_ink2 *h);
bool       gpu_ink2_reserve_experts(gpu_ink2 *h, int T, int H, int F);
void       gpu_ink2_free(gpu_ink2 *h);

#endif // RUNNER_GPU_H
