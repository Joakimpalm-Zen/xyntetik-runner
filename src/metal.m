// Metal GPU backend: full single-token forward pass on Apple GPUs.
// Compiled without ARC; every object lives for the process lifetime.
#import <Metal/Metal.h>

#include "runner.h"
#include "kernels_metal.h"
#include "kernels_tensor_metal.h"
#include "compat.h"   // plat_ram_available_bytes: the residency guard below
#include "metal_admission.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Sized past the HIGHEST type the loader can admit, not the highest Metal
// serves: these arrays are indexed by ggml type, and a CPU-supported type
// with no Metal kernel (T_NVFP4) must land on a NULL slot and decline —
// never index past the table. The sval review's lesson, applied here before
// it fires: capacity that silently un-guards is the bug class.
enum { METAL_TYPE_SLOTS = T_NVFP4 + 1 };
// Weight-mmap wraps. Enough for a 64 GB monolithic file at the M1's 4.29 GB
// per-buffer ceiling, with room to spare; exceeding it is reported, not
// silently truncated.
enum { METAL_MAX_WBUF = 24 };

typedef struct {
    id<MTLDevice>       dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_rmsnorm, p_qknorm, p_headnorm, p_rope, p_store, p_attn;
    id<MTLComputePipelineState> p_attn_chunk, p_attn_comb;
    id<MTLComputePipelineState> p_attn_coop, p_attn_chunk_coop;  // cooperative KV score read
    id<MTLComputePipelineState> p_silu, p_gelu, p_add, p_scale;
    id<MTLComputePipelineState> p_sigmul;   // attention output gate (x *= sigmoid(g))
    id<MTLComputePipelineState> p_moe_route, p_moe_actmul, p_moe_sum, p_trace_copy;
    id<MTLComputePipelineState> p_mv[METAL_TYPE_SLOTS];       // indexed by ggml type
    id<MTLComputePipelineState> p_mm[METAL_TYPE_SLOTS];       // tiled prefill GEMM
    id<MTLComputePipelineState> p_tensor[METAL_TYPE_SLOTS];   // Metal 4 MPP prefill
    id<MTLComputePipelineState> p_mvf[METAL_TYPE_SLOTS];      // fast decode matvec
    id<MTLComputePipelineState> p_moe_mv[METAL_TYPE_SLOTS];   // indexed by ggml type
    id<MTLComputePipelineState> p_moe_mv_em[METAL_TYPE_SLOTS]; // expert-major twin
    id<MTLComputePipelineState> p_moe_mv_em8[METAL_TYPE_SLOTS]; // + 8-column tiling
    id<MTLComputePipelineState> p_moe_gua[METAL_TYPE_SLOTS]; // fused gate+up+act
    id<MTLComputePipelineState> p_rope_store, p_add_rmsnorm; // decode fusion
    id<MTLComputePipelineState> p_attn_front_q8;             // front megakernel
    id<MTLComputePipelineState> p_attn_front_q40, p_attn_front_q4k;
    id<MTLComputePipelineState> p_attn_front_q6k, p_attn_front_q4k6;
    id<MTLComputePipelineState> p_moe_mm[METAL_TYPE_SLOTS];  // grouped MMA, f32-staged
    id<MTLComputePipelineState> p_moe_mmh[METAL_TYPE_SLOTS];  // half-staged twins
    id<MTLComputePipelineState> p_moe_group;                 // slot-by-expert sort
    // The model mmap, wrapped zero-copy. Usually ONE buffer; more when the file
    // exceeds maxBufferLength, which on an M1 is 4.29 GB against a 5.73 GB
    // working set (0.75x) and is the ceiling that actually bites for a big
    // monolithic GGUF.
    //
    // The split is at TENSOR boundaries, so no tensor ever straddles two
    // buffers and no kernel had to learn about this: a dispatch reads exactly
    // one weight tensor, so binding the containing buffer and passing an
    // offset within it is all that changes. Adjacent wraps may overlap by one
    // page, because a no-copy wrap must start page-aligned and tensors are
    // not; wrapping the same read-only pages twice costs a page of mapping and
    // keeps every tensor whole.
    id<MTLBuffer> wbuf[METAL_MAX_WBUF];
    uint64_t      wbuf_base[METAL_MAX_WBUF];   // host address of each wrap's byte 0
    uint64_t      wbuf_end[METAL_MAX_WBUF];    // one past its last byte's address
    int           n_wbuf;
    bool          weights_copied;
    // Sticky: a weight range no wrap holds ends the offload for this model.
    bool          bind_failed;
    id<MTLBuffer> kc, vc;
    id<MTLBuffer> x, xb, xb2, q, kt, vt, hb, hb2, att, logits;
    id<MTLBuffer> agate;             // [n][q_dim] attention output gate scratch
    id<MTLBuffer> att_acc, att_ms;   // chunked-decode partials
    id<MTLBuffer> moe_logits, moe_sel, moe_selw, moe_hb, moe_hb2, moe_eout;
    id<MTLBuffer> moe_colmap, moe_eoff;   // grouped-MMA slot map + offsets
    id<MTLBuffer> moe_trace_logits;
    id<MTLBuffer> inv_freq, inv_freq_local, out_norm, dummy;
    id<MTLBuffer> *ppn;                 // gemma4 E-series per-layer post_norm
    id<MTLBuffer> ple, ple_tmp;         // [n][n_layer][P] slices, [n][P] gate
    id<MTLBuffer> *attn_norm, *ffn_norm;        // per layer
    id<MTLBuffer> *bq, *bk, *bv, *bo;           // per layer, may be nil
    id<MTLBuffer> *qn, *kn;                     // qwen3 per-head q/k norms
    id<MTLBuffer> *sinks;                       // gpt-oss attention sinks
    id<MTLBuffer> *gib, *geb, *ueb, *deb;       // gpt-oss MoE biases
    id<MTLBuffer> *pan, *pfn;                   // gemma sandwich norms
    id<MTLBuffer> *gpn1, *gprn2, *gpn2;         // gemma4 MoE branch norms
    id<MTLBuffer> *ggis, *gdsc;                 // gemma4 router/down scales
    id<MTLBuffer> suppress;                     // gemma never-emit token ids
    int batch_cap;                              // scratch rows allocated
} gpu_t;

// mirrors struct mv_args in kernels.metal — keep the field order in step
typedef struct { int n_in, n_out; uint64_t w_off; int has_bias;
                 int n_col, x_stride, y_stride, col_tile; } mv_args;
typedef struct { int n_in, n_out, n_col; uint64_t w_off;
                 int has_bias, x_stride, y_stride; } mm_args;
typedef struct { int n, x_stride, y_stride; float eps; } norm_args;
typedef struct { int kv_dim, q8, stride, pos, kv_rows; uint64_t off, row_b; } store_args;
typedef struct { int n_head, n_kv, head_dim, half_dim, pos, neox;
                 float mscale; int q_off_e, k_off_e, v_off_e;
                 uint64_t kc_off, vc_off; } rope_store_args;
typedef struct { int n; float eps; } add_norm_args;
typedef struct { int n_embd, n_head, n_kv, head_dim, half_dim, pos, neox;
                 float mscale, eps;
                 uint64_t wq_off, wk_off, wv_off;
                 int has_bq, has_bk, has_bv;
                 uint64_t kc_off, vc_off;
                 int has_qn, has_kn; } attn_front_args;
typedef struct { int n_in, n_out; uint64_t g_off, u_off, estride_g, estride_u;
                 int xs, has_bias, act, slots_per_token, ys; } moe_gua_args;
typedef struct { int head_dim, n_heads, half_dim, pos, neox; float mscale; int stride; } rope_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale; int q8, window, has_sinks;
                 int q_stride, att_stride, out_stride, kv_rows; } attn_args;
typedef struct { int head_dim, n_head, n_head_kv, n_ctx, pos; uint64_t l_off; float scale;
                 int q8, window, chunk, n_chunks, kv_rows; } attn_chunk_args;
typedef struct { int head_dim, n_head, n_chunks, has_sinks; } attn_comb_args;

// Chunked decode attention. The split is chosen from the HEAD COUNT, not a
// fixed span: the point of chunking is to fill the GPU, so what matters is
// n_head * n_chunks (the threadgroup count), and n_head varies per model.
// Targeting ~64 threadgroups put an 8-head model at 8 chunks, which is where
// a measured sweep on an M1 landed too -- 3137 tokens of context: single-pass
// 7.32, chunk 256 -> 7.86, 384 -> 7.94, 512 -> 7.85, 1024 -> 7.51, and 384 is
// exactly span/8. Fewer, larger chunks starve the GPU; more, smaller ones pay
// combine cost for parallelism it cannot use.
// RUNNER_METAL_ATTN_CHUNK sets an explicit chunk size for measurement, and 0
// disables the path entirely (the single-pass kernel is the A/B baseline).
// Revised 2026-08-07. The sweep above never tested 128, and 128 wins: at 3259
// tokens on gemma-3-4b, chunk 128 gives 8.130 tok/s against 8.060 for the
// span/8 the target above produces (medians of 3 interleaved rounds, arms
// disjoint: [8.13,8.14] vs [8.06,8.07]). Every point the two sweeps share
// agrees -- 256 and 512 land at 7.96/7.95 here against 7.86/7.85 there -- so
// this is a new point on the curve, not a contradiction of the old one.
//
// The target stays expressed in threadgroups rather than a flat chunk size,
// because the right answer does depend on n_head: a 32-head model already gets
// 32 threadgroups from the single-pass kernel and needs little splitting, while
// an 8-head model on an 8-core GPU is starved. Raising the target to 256 leaves
// many-head models where they were and only makes few-head models finer.
//
// HONEST SIZE OF THIS TUNE. Against the previous target, interleaved, 3 rounds:
// GPU time 119.6-119.7 vs 120.7-120.9 ms/tok, disjoint, ~1.0%. End to end the
// medians differ by +0.87% but the arms OVERLAP, so it is not separable from
// noise on this machine. Kept because the direct measure of the thing changed
// is clean and nothing regresses (short-context decode 9.46 tok/s, unmoved).
// A second A/B meant to settle it was ABANDONED, not reported: macOS began
// rendering an animated aerial wallpaper on the same GPU partway through and
// both arms fell ~7%. Watch for that before trusting any GPU number on an idle
// Mac -- WallpaperAerialsExtension and WindowServer are the processes to check.
//
// For scale, the split itself is worth +9.4% over RUNNER_METAL_ATTN_CHUNK=0
// (7.975 vs 7.290 tok/s at 3259 ctx, arms disjoint). This tunes it, it does
// not replace it.
//
// Provenance note: this change was pushed inside 5d7808b, whose message
// describes only a Makefile fix -- `git add -A` swept it in. The measurement
// record lives here rather than in that commit for exactly that reason.
#define METAL_ATTN_MAX_CHUNKS 64
#define METAL_ATTN_TARGET_GROUPS 256
// Below this many positions per chunk the split is all overhead: at 40 tokens
// of context an 8-head model would otherwise be cut into 8 chunks of 5, and
// short-context decode measured 9.28 -> 9.20 tok/s doing exactly that.
//
// This is a FLOOR on the chunk size, not a veto on chunking. It used to be the
// latter -- a target implying chunks below this disabled the split entirely --
// which is why raising the target alone would have turned the path off at
// exactly the contexts it helps. Short context still falls back to the
// single-pass kernel, but via the n_chunks < 2 test below, which is the
// condition that actually means "nothing to split".
#define METAL_ATTN_MIN_CHUNK 128
static int metal_attn_chunk_override(void) {
    static int v = -2;
    if (v == -2) {
        const char *e = getenv("RUNNER_METAL_ATTN_CHUNK");
        v = -1;
        if (e && *e) { long n = strtol(e, NULL, 10); if (n >= 0) v = (int)n; }
    }
    return v;
}

typedef struct { int n_in, n_out; uint64_t w_off, estride;
                 int xs, ys, has_bias, bias_stride, slots_per_token;
                 int n_slots; } moe_args;   // n_slots: expert-major only

static void gpu_release_state(gpu_t *g, int n_layer) {
    if (!g) return;
    for (int l = 0; l < n_layer; l++) {
        if (g->attn_norm) [g->attn_norm[l] release];
        if (g->ffn_norm)  [g->ffn_norm[l] release];
        if (g->bq) [g->bq[l] release];
        if (g->bk) [g->bk[l] release];
        if (g->bv) [g->bv[l] release];
        if (g->bo) [g->bo[l] release];
        if (g->qn) [g->qn[l] release];
        if (g->kn) [g->kn[l] release];
        if (g->sinks) [g->sinks[l] release];
        if (g->gib) [g->gib[l] release];
        if (g->geb) [g->geb[l] release];
        if (g->ueb) [g->ueb[l] release];
        if (g->deb) [g->deb[l] release];
        if (g->pan) [g->pan[l] release];
        if (g->ppn) [g->ppn[l] release];
        if (g->pfn) [g->pfn[l] release];
        if (g->gpn1) [g->gpn1[l] release];
        if (g->gprn2) [g->gprn2[l] release];
        if (g->gpn2) [g->gpn2[l] release];
        if (g->ggis) [g->ggis[l] release];
        if (g->gdsc) [g->gdsc[l] release];
    }
    free(g->attn_norm); free(g->ffn_norm);
    free(g->bq); free(g->bk); free(g->bv); free(g->bo);
    free(g->qn); free(g->kn);
    free(g->sinks); free(g->gib); free(g->geb); free(g->ueb); free(g->deb);
    free(g->ppn); free(g->pan); free(g->pfn);
    free(g->gpn1); free(g->gprn2); free(g->gpn2); free(g->ggis); free(g->gdsc);
    [g->p_attn_coop release]; [g->p_attn_chunk_coop release];
    for (int i = 0; i < g->n_wbuf; i++) [g->wbuf[i] release];
    id<MTLBuffer> bufs[] = { g->kc, g->vc, g->x, g->xb, g->xb2,
                             g->q, g->kt, g->vt, g->hb, g->hb2, g->att,
                             g->logits, g->att_acc, g->att_ms,
                             g->moe_logits, g->moe_sel, g->moe_selw,
                             g->moe_trace_logits,
                             g->moe_hb, g->moe_hb2, g->moe_eout,
                             g->moe_colmap, g->moe_eoff,
                             g->inv_freq, g->inv_freq_local, g->out_norm,
                             g->dummy, g->suppress, g->ple, g->ple_tmp,
                             g->agate };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) [bufs[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mv[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) {
        [g->p_mm[i] release];
        [g->p_tensor[i] release];
    }
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_mvf[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mv[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mv_em[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mv_em8[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_gua[i] release];
    [g->p_rope_store release]; [g->p_add_rmsnorm release];
    [g->p_attn_front_q8 release];
    [g->p_attn_front_q40 release]; [g->p_attn_front_q4k release];
    [g->p_attn_front_q6k release]; [g->p_attn_front_q4k6 release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mm[i] release];
    for (int i = 0; i < METAL_TYPE_SLOTS; i++) [g->p_moe_mmh[i] release];
    [g->p_moe_group release];
    [g->p_rmsnorm release]; [g->p_qknorm release]; [g->p_headnorm release];
    [g->p_rope release]; [g->p_store release]; [g->p_attn release];
    [g->p_attn_chunk release]; [g->p_attn_comb release];
    [g->p_silu release]; [g->p_gelu release]; [g->p_add release];
    [g->p_sigmul release];
    [g->p_scale release];
    [g->p_moe_route release]; [g->p_moe_actmul release];
    [g->p_moe_sum release]; [g->p_trace_copy release];
    [g->queue release];
    [g->dev release];
    free(g);
}

static bool metal_init_injected(const char *point) {
    const char *inject = getenv("RUNNER_METAL_INIT_INJECT_FAILURE");
    return inject && *inject && strcmp(inject, "0") &&
           (!strcmp(inject, "always") || !strcmp(inject, point));
}

static bool metal_env_on(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0");
}

static bool metal_moe_batch_on(void) {
    const char *v = getenv("RUNNER_METAL_MOE_BATCH");
    return !v || !*v || (strcmp(v, "0") && strcmp(v, "off"));
}

// RUNNER_METAL_MOE_EM=1: expert-major expert FFN kernels for large batches.
// The slot-major default re-reads an expert's weight rows once per slot that
// selected it; expert-major reads them once per threadgroup and walks the
// expert's token list from threadgroup memory. Outputs are BIT-IDENTICAL to
// slot-major (one shared dot body, one writer per (slot, row)) — the gate is
// byte comparison, not tolerance. Opt-in until its measured prefill win
// clears the promotion bar; it only engages when a dispatch carries at least
// as many slots as experts, so decode always stays slot-major.
static bool metal_moe_em_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RUNNER_METAL_MOE_EM");
        on = v && *v && strcmp(v, "0") && strcmp(v, "off") ? 1 : 0;
    }
    return on > 0;
}

// RUNNER_METAL_MOE_MM=1: grouped simdgroup-MMA expert FFNs for prefill —
// sort the batch's slots by expert on-GPU, run each expert's token group
// through the dense k_mm tile structure with gathered columns. Same numerics
// class as dense RUNNER_METAL_MM (half-staged operands, reassociated k-sum):
// answers to the tolerance gates, never to byte identity. Opt-in until the
// measured prefill win clears the bar recorded in
// docs/negative-result-metal-moe-expert-major.md.
// DEFAULT ON since 2026-09-01, when the owner ratified the house fidelity
// bar (margin-qualified top-1 + mean KLD, test-moe-mm-ab) as the instrument
// that judges reassociating prefill on sparse-routing models — the dense
// logit-identity bound measures routing near-tie flips, which the flip
// account (scripts/moe-mm-flips.py) shows are staging-invariant and which
// the fidelity bar shows leave the output distribution at the model's own
// noise floor. RUNNER_METAL_MOE_MM=0 restores the matvec path;
// "half" selects the half-staged comparison twins. NOT cached: the
// mv-vs-mm harness toggles this between loads in one process, and a getenv
// per MoE layer encode is noise.
// Decode fusion (RUNNER_METAL_FUSE, default ON): fold the per-layer decode
// chain — rope(q)+rope(k)+store, residual-add+rmsnorm, MoE gate+up+act —
// into single dispatches. Every fused kernel reproduces the unfused
// arithmetic element for element, so the contract is BYTE IDENTITY against
// RUNNER_METAL_FUSE=0, held by test-metal-fuse. Justified by the measured
// dispatch budget (686/token, 3.4-4.8 us chain cost each on M5 Max), not by
// any other engine's shape.
static bool metal_fuse_on(void) {
    const char *v = getenv("RUNNER_METAL_FUSE");
    return !v || !*v || (strcmp(v, "0") && strcmp(v, "off"));
}

static void metal_fuse_announce(void) {
    static bool told = false;
    if (!told) {
        told = true;
        fprintf(stderr, "gpu: decode fusion on "
                "(byte-identical; RUNNER_METAL_FUSE=0 disables)\n");
    }
}

static void metal_front_announce(id<MTLComputePipelineState> pipe,
                                 gpu_t *g) {
    static bool told = false;
    if (told) return;
    told = true;
    const char *name =
        pipe == g->p_attn_front_q8   ? "q8_0"    :
        pipe == g->p_attn_front_q40  ? "q4_0"    :
        pipe == g->p_attn_front_q4k  ? "q4_k"    :
        pipe == g->p_attn_front_q6k  ? "q6_k"    :
        pipe == g->p_attn_front_q4k6 ? "q4k+q6k" : "?";
    fprintf(stderr, "gpu: attention front on (%s)\n", name);
}

static int metal_moe_mm_on(void) {
    const char *v = getenv("RUNNER_METAL_MOE_MM");
    if (!v || !*v) return 1;
    if (!strcmp(v, "0") || !strcmp(v, "off")) return 0;
    return strcmp(v, "half") == 0 ? 2 : 1;
}


static bool gpu_init_fail(model_t *m, gpu_t *g, id<MTLLibrary> lib,
                          const char *why) {
    if (why && *why)
        fprintf(stderr, "gpu: Metal initialization failed (%s) — using CPU\n", why);
    else
        fprintf(stderr, "gpu: Metal initialization failed — using CPU\n");
    if (lib) [lib release];
    gpu_release_state(g, m ? m->n_layer : 0);
    return false;
}

static bool metal_buffer_ok(id<MTLBuffer> b) {
    return b != nil && b.contents != NULL;
}

static bool metal_command_failed(id<MTLCommandBuffer> cb) {
    const char *inject = getenv("RUNNER_METAL_INJECT_FAILURE");
    static int injected_once = 0;
    bool injected = inject && *inject && strcmp(inject, "0") &&
                    (!injected_once || strcmp(inject, "always") == 0);
    if (injected) injected_once = 1;
    return injected || cb.status == MTLCommandBufferStatusError;
}

static id<MTLBuffer> new_f32_scratch(id<MTLDevice> dev, size_t n) {
    if (n > SIZE_MAX / sizeof(float)) return nil;
    return [dev newBufferWithLength:n * sizeof(float)
                            options:MTLResourceStorageModeShared];
}

static void release_buf(id<MTLBuffer> b) {
    [b release];
}

// The ONE place per-column scratch is sized. gpu_init calls it for n = 1 and
// gpu_forward_batch for every larger batch, so a buffer added here is covered
// in both directions. It used to be sized twice — a hand-rolled n = 1 block in
// gpu_init and this — and the two lists drifted: g->att_acc, g->att_ms and
// g->agate existed only here, so a session whose every forward is a single
// token (batch_cap starts at 1, this returns early) ran with them nil and the
// kernels dereferenced a nil binding. That produced garbage decode past the
// chunk threshold; see `make test-metal-decode-only`.
static bool metal_ensure_batch(gpu_t *g, model_t *m, int n) {
    if (!g || n <= g->batch_cap) return true;
    // eligibility keeps heterogeneous models off this path, but size off the
    // per-layer maxima anyway so the two sizing sites cannot drift apart
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    for (int l = 0; l < m->n_layer; l++) {
        if (model_q_dim(m, l)  > q_dim)  q_dim  = model_q_dim(m, l);
        if (model_kv_dim(m, l) > kv_dim) kv_dim = model_kv_dim(m, l);
    }
    int xdim   = q_dim > m->n_embd ? q_dim : m->n_embd;
    size_t nb = (size_t)n;

    id<MTLBuffer> x      = new_f32_scratch(g->dev, nb * (size_t)m->n_embd);
    id<MTLBuffer> xb     = new_f32_scratch(g->dev, nb * (size_t)xdim);
    id<MTLBuffer> xb2    = new_f32_scratch(g->dev, nb * (size_t)xdim);
    // xdim, not q_dim: the gemma-4 MoE branch reuses q as its routed-branch
    // scratch and writes n_embd floats at offset 0 (enc_gemma_moe_ffn), and
    // gemma geometries decouple head_dim from n_embd in the direction that
    // makes q the SMALLER of the two -- gemma-3-4b is n_embd 2560 against
    // n_head 8 x head_dim 256 = 2048. At decode (nb == 1) the difference is
    // written straight past the end of the buffer.
    id<MTLBuffer> q      = new_f32_scratch(g->dev, nb * (size_t)xdim);
    id<MTLBuffer> kt     = new_f32_scratch(g->dev, nb * (size_t)kv_dim);
    id<MTLBuffer> vt     = new_f32_scratch(g->dev, nb * (size_t)kv_dim);
    id<MTLBuffer> hb     = new_f32_scratch(g->dev, nb * (size_t)m->n_ff);
    id<MTLBuffer> hb2    = new_f32_scratch(g->dev, nb * (size_t)m->n_ff);
    id<MTLBuffer> att    = new_f32_scratch(g->dev, nb * (size_t)m->n_head *
                                                   (size_t)m->n_ctx);
    id<MTLBuffer> logits = new_f32_scratch(g->dev, nb * (size_t)m->n_vocab);
    int P = m->n_embd_ple;
    id<MTLBuffer> ple = nil, ple_tmp = nil;
    if (P > 0) {
        ple     = new_f32_scratch(g->dev, nb * (size_t)m->n_layer * P);
        ple_tmp = new_f32_scratch(g->dev, nb * (size_t)P);
    }
    id<MTLBuffer> agate = nil;
    if (m->attn_out_gate)
        agate = new_f32_scratch(g->dev, nb * (size_t)q_dim);
    id<MTLBuffer> moe_logits = nil, moe_sel = nil, moe_selw = nil;
    id<MTLBuffer> moe_trace_logits = nil;
    id<MTLBuffer> moe_hb = nil, moe_hb2 = nil, moe_eout = nil;
    id<MTLBuffer> moe_colmap = nil, moe_eoff = nil;
    if (m->n_expert > 0) {
        size_t route_slots = nb * (size_t)m->n_expert_used *
                             (size_t)m->n_layer;
        size_t slots = nb * (size_t)m->n_expert_used;
        size_t moe_ff = (size_t)m->n_ff_exp * (m->moe_gemma ? 2u : 1u);
        moe_logits = new_f32_scratch(g->dev, nb * (size_t)m->n_expert);
        moe_sel    = [g->dev newBufferWithLength:sizeof(int) * route_slots
                                         options:MTLResourceStorageModeShared];
        moe_selw   = new_f32_scratch(g->dev, route_slots);
        moe_hb     = new_f32_scratch(g->dev, slots * moe_ff);
        moe_hb2    = new_f32_scratch(g->dev, slots * moe_ff);
        moe_eout   = new_f32_scratch(g->dev, slots * (size_t)m->n_embd);
        moe_colmap = [g->dev newBufferWithLength:sizeof(int) * slots
                                         options:MTLResourceStorageModeShared];
        moe_eoff   = [g->dev newBufferWithLength:sizeof(int) *
                                                 ((size_t)m->n_expert + 1)
                                         options:MTLResourceStorageModeShared];
        if (getenv("RUNNER_MOE_TRACE"))
            moe_trace_logits = new_f32_scratch(g->dev, nb * (size_t)m->n_layer *
                                                        (size_t)m->n_expert);
    }
    if (!metal_buffer_ok(x) || !metal_buffer_ok(xb) || !metal_buffer_ok(xb2) ||
        !metal_buffer_ok(q) || !metal_buffer_ok(kt) || !metal_buffer_ok(vt) ||
        !metal_buffer_ok(hb) || !metal_buffer_ok(hb2) || !metal_buffer_ok(att) ||
        !metal_buffer_ok(logits) ||
        (P > 0 && (!metal_buffer_ok(ple) || !metal_buffer_ok(ple_tmp))) ||
        (m->attn_out_gate && !metal_buffer_ok(agate)) ||
        (m->n_expert > 0 &&
         (!metal_buffer_ok(moe_logits) || !metal_buffer_ok(moe_sel) ||
          !metal_buffer_ok(moe_selw) || !metal_buffer_ok(moe_hb) ||
          !metal_buffer_ok(moe_hb2) || !metal_buffer_ok(moe_eout) ||
          !metal_buffer_ok(moe_colmap) || !metal_buffer_ok(moe_eoff) ||
          (getenv("RUNNER_MOE_TRACE") && !metal_buffer_ok(moe_trace_logits))))) {
        release_buf(x); release_buf(xb); release_buf(xb2); release_buf(q);
        release_buf(kt); release_buf(vt); release_buf(hb); release_buf(hb2);
        release_buf(att); release_buf(logits);
        release_buf(ple); release_buf(ple_tmp); release_buf(agate);
        release_buf(moe_logits); release_buf(moe_sel); release_buf(moe_selw);
        release_buf(moe_hb); release_buf(moe_hb2); release_buf(moe_eout);
        release_buf(moe_colmap); release_buf(moe_eoff);
        release_buf(moe_trace_logits);
        return false;
    }

    release_buf(g->x); release_buf(g->xb); release_buf(g->xb2);
    release_buf(g->q); release_buf(g->kt); release_buf(g->vt);
    release_buf(g->hb); release_buf(g->hb2); release_buf(g->att);
    release_buf(g->logits);
    g->x = x; g->xb = xb; g->xb2 = xb2; g->q = q; g->kt = kt; g->vt = vt;
    g->hb = hb; g->hb2 = hb2; g->att = att; g->logits = logits;
    if (P > 0) {
        release_buf(g->ple); release_buf(g->ple_tmp);
        g->ple = ple; g->ple_tmp = ple_tmp;
    }
    if (m->attn_out_gate) {
        release_buf(g->agate);
        g->agate = agate;
    }
    if (m->n_expert > 0) {
        release_buf(g->moe_logits); release_buf(g->moe_sel);
        release_buf(g->moe_selw); release_buf(g->moe_hb);
        release_buf(g->moe_hb2); release_buf(g->moe_eout);
        release_buf(g->moe_colmap); release_buf(g->moe_eoff);
        release_buf(g->moe_trace_logits);
        g->moe_logits = moe_logits; g->moe_sel = moe_sel;
        g->moe_selw = moe_selw; g->moe_hb = moe_hb;
        g->moe_hb2 = moe_hb2; g->moe_eout = moe_eout;
        g->moe_colmap = moe_colmap; g->moe_eoff = moe_eoff;
        g->moe_trace_logits = moe_trace_logits;
    }
    g->batch_cap = n;
    return true;
}

// A device that exists is not a backend that works: if the shader library does
// not compile, gpu_init falls back and every run is CPU-speed. --caps exists so
// a scheduler can place work BEFORE dispatching, so reporting a usable Metal
// backend in that state would be a lie that costs a whole run. Compiling the
// library here is the same work gpu_init does moments later and only happens
// on the --caps path. tests/test_metal_shaders.m is the build-time gate.
bool gpu_available(char *name, int cap) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        fprintf(stderr, "gpu: no Metal device is available — using CPU\n");
        return false;
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
                              [NSString stringWithUTF8String:k_metal_src]
                                           options:nil
                                             error:&err];
    if (!lib) {
        fprintf(stderr, "gpu: Metal device present but its shader library does "
                "not compile — reporting no GPU backend (every run would be "
                "CPU-speed): %s\n",
                err ? err.localizedDescription.UTF8String : "(no diagnostic)");
        [dev release];
        return false;
    }
    if (name) snprintf(name, cap, "%s", dev.name.UTF8String);
    [lib release];
    [dev release];
    return true;
}

bool gpu_unified_memory(void) { return true; }

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    // unified memory: the RAM reservation governs; no separate VRAM pool
    (void)free_bytes; (void)total_bytes;
    return false;
}

// k_metal_sha is generated into kernels_metal.h alongside the source string
// itself, so it can only agree with the shaders this binary actually holds.
const char *gpu_shader_source_sha(void) { return k_metal_sha; }

bool gpu_max_working_set(size_t *bytes) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    uint64_t ws = dev.recommendedMaxWorkingSetSize;
    [dev release];
    if (ws == 0) return false;
    if (bytes) *bytes = (size_t)ws;
    return true;
}

// UNVERIFIED — written without a macOS machine to run it on. Nobody on this
// project has executed this function.
//
// Returning false disables the VRAM registry on Metal, which is the correct
// behaviour anyway rather than a placeholder: Apple GPUs share one unified
// memory pool with the CPU, gpu_mem_info already declines to report a separate
// VRAM figure, and the registry's whole arithmetic is "device free bytes minus
// pending claims". With no device-private pool there is no VRAM to account for
// and the RAM reservation (--reserve-ram) governs instead.
//
// If a Metal build ever does want registry accounting, the identity to return
// is [MTLDevice registryID] rendered as a string, and gpu_mem_info would have
// to start reporting recommendedMaxWorkingSetSize / currentAllocatedSize. Both
// need a real Mac to verify before anyone trusts them.
bool gpu_device_id(char *id, int cap) {
    (void)id; (void)cap;
    return false;
}

static bool gpu_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_BF16: case T_Q8_0: case T_Q4_0:
        case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q2_K: case T_Q3_K:
        case T_Q4_K:
        case T_Q5_K: case T_Q6_K: case T_IQ4_NL: case T_IQ4_XS: case T_MXFP4:
            return true;
        default:
            return false;
    }
}

// The --caps answer for this backend, sourced from the admission test above so
// the advertised list and the loader agree by construction.
bool gpu_quant_ok(int type) { return gpu_type_ok(type); }

static bool metal_moe_type_ok(int type) {
    switch (type) {
        case T_F32: case T_F16: case T_Q8_0: case T_Q4_0:
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_MXFP4:
            return true;
        default:
            return false;
    }
}

static bool metal_tensor_type_ok(const gguf_tensor *t, bool moe) {
    if (!t || (moe ? metal_moe_type_ok(t->type) : gpu_type_ok(t->type)))
        return true;
    fprintf(stderr, "gpu: tensor %s uses %s, which has no Metal%s kernel — "
            "using CPU\n", t->name, ggml_type_name(t->type),
            moe ? " MoE" : "");
    return false;
}

static bool metal_moe_supported(const model_t *m) {
    if (m->n_expert <= 0) return true;
    if (m->n_expert > 256 || m->n_expert_used < 1 ||
        m->n_expert_used > 256) {
        fprintf(stderr, "gpu: this MoE geometry is outside the Metal router limit — using CPU\n");
        return false;
    }
    if (!model_moe_router_is_plain(m)) {
        fprintf(stderr, "gpu: this model's MoE router has no Metal kernel — using CPU\n");
        return false;
    }
    if (m->n_ff_shexp > 0) {
        fprintf(stderr, "gpu: shared-expert MoE has no Metal path yet — using CPU\n");
        return false;
    }
    if (m->ffn_act != ACT_SILU &&
        !(m->gptoss && m->ffn_act == ACT_SWIGLU_OAI) &&
        !(m->moe_gemma && m->ffn_act == ACT_GELU)) {
        fprintf(stderr, "gpu: this MoE activation has no Metal kernel yet — using CPU\n");
        return false;
    }
    for (int l = 0; l < m->n_layer; l++) {
        const layer_t *ly = &m->layers[l];
        if (!ly->is_moe) continue;
        if (ly->moe_split) {
            fprintf(stderr, "gpu: split expert layout is not on the metal backend yet — using CPU\n");
            return false;
        }
        if (ly->exp_probs_b) {
            fprintf(stderr, "gpu: MoE router/expert bias has no Metal path yet — using CPU\n");
            return false;
        }
        if (ly->moe_gemma) {
            if (!ly->ffn_gate_inp || !ly->ffn_gate_up_exps ||
                !ly->ffn_down_exps || !ly->w_gate || !ly->w_up || !ly->w_down) {
                fprintf(stderr, "gpu: unsupported Gemma MoE tensor layout for Metal — using CPU\n");
                return false;
            }
            if (!metal_tensor_type_ok(ly->ffn_gate_inp, false) ||
                !metal_tensor_type_ok(ly->ffn_gate_up_exps, true) ||
                !metal_tensor_type_ok(ly->ffn_down_exps, true)) {
                return false;
            }
        } else {
            if (!ly->ffn_gate_inp || !ly->ffn_gate_exps ||
                !ly->ffn_up_exps || !ly->ffn_down_exps) {
                fprintf(stderr, "gpu: unsupported MoE tensor layout for Metal — using CPU\n");
                return false;
            }
            if (!metal_tensor_type_ok(ly->ffn_gate_inp, false) ||
                !metal_tensor_type_ok(ly->ffn_gate_exps, true) ||
                !metal_tensor_type_ok(ly->ffn_up_exps, true) ||
                !metal_tensor_type_ok(ly->ffn_down_exps, true)) {
                return false;
            }
        }
    }
    return true;
}

static id<MTLComputePipelineState> mk_pipeline(id<MTLDevice> dev,
                                               id<MTLLibrary> lib,
                                               NSString *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (!p) fprintf(stderr, "gpu: pipeline %s failed: %s\n",
                    name.UTF8String, err.localizedDescription.UTF8String);
    // Threadgroup memory is the occupancy currency: an Apple GPU core has
    // 32 KB of it, so a kernel asking 14 KB gets 2 resident threadgroups and
    // one asking 8 KB gets 4. That ratio is invisible from throughput alone,
    // and a tile-shape sweep that cannot see it is guessing.
    if (p && metal_env_on("RUNNER_METAL_STATS"))
        fprintf(stderr, "metal-pipeline %-16s tgmem=%5lu B  max_threads=%lu\n",
                name.UTF8String,
                (unsigned long)p.staticThreadgroupMemoryLength,
                (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return p;
}

// Kept as a distinct lookup surface: test_metal_shaders scans mk_pipeline
// calls against the baseline library, while test_metal_tensor owns this
// separately compiled Metal 4 library.
static id<MTLComputePipelineState> mk_tensor_pipeline(id<MTLDevice> dev,
                                                      id<MTLLibrary> lib,
                                                      NSString *name) {
    return mk_pipeline(dev, lib, name);
}

// Absolute admission anchor: 64 rows of 256 unit Q4_K weights times 32
// columns of unit activations must produce exactly 256 (within fp16/MPP
// rounding). A pipeline that merely compiles is not enough—the LM Studio M5
// regression was precisely a functional tensor self-test that failed later.
// Per-type self-test rows: unit weights whose dot with an all-ones input
// is exactly NI, the same bar for every admitted type.
static size_t metal_tensor_test_row(int type, uint8_t *b, int ni) {
    if (type == T_Q4_K) {                      // 144 B / 256 weights
        b[0] = 0x00; b[1] = 0x3c;              // fp16 d = 1
        for (int j = 0; j < 4; j++) b[4 + j] = 1;
        for (int j = 0; j < 4; j++) b[12 + j] = 1;
        memset(b + 16, 0x11, 128);             // both nibbles = 1
        return 144;
    }
    if (type == T_Q8_0) {                      // 34 B / 32 weights
        for (int k = 0; k < ni / 32; k++) {
            uint8_t *q = b + k * 34;
            q[0] = 0x00; q[1] = 0x3c;
            memset(q + 2, 1, 32);              // int8 quant = 1
        }
        return (size_t)(ni / 32) * 34;
    }
    // T_Q4_0: 18 B / 32 weights, nibble 9 -> 9 - 8 = 1
    for (int k = 0; k < ni / 32; k++) {
        uint8_t *q = b + k * 18;
        q[0] = 0x00; q[1] = 0x3c;
        memset(q + 2, 0x99, 16);
    }
    return (size_t)(ni / 32) * 18;
}

static bool metal_tensor_self_test(id<MTLDevice> dev,
                                   id<MTLComputePipelineState> p, int type) {
    enum { NI = 256, NOUT = 64, NC = 32, ROW_MAX = 288, SHMEM = 128*64*2 };
    uint8_t *wh = calloc(NOUT, ROW_MAX);
    float *xh = malloc((size_t)NC * NI * sizeof(float));
    if (!wh || !xh) { free(wh); free(xh); return false; }
    size_t row_b = metal_tensor_test_row(type, wh, NI);
    for (int r = 1; r < NOUT; r++)
        metal_tensor_test_row(type, wh + (size_t)r * row_b, NI);
    for (int i = 0; i < NC * NI; i++) xh[i] = 1.0f;
    id<MTLBuffer> wb = [dev newBufferWithBytes:wh length:NOUT * ROW_MAX
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> xb = [dev newBufferWithBytes:xh length:(size_t)NC * NI * 4
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [dev newBufferWithLength:(size_t)NC * NOUT * 4
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:NOUT * 4
                                        options:MTLResourceStorageModeShared];
    free(wh); free(xh);
    if (!wb || !xb || !yb || !bb) {
        [wb release]; [xb release]; [yb release]; [bb release]; return false;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    mm_args a = { NI, NOUT, NC, 0, 0, NI, NOUT };
    [e setComputePipelineState:p];
    [e setBuffer:wb offset:0 atIndex:0]; [e setBuffer:xb offset:0 atIndex:1];
    [e setBuffer:yb offset:0 atIndex:2]; [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:bb offset:0 atIndex:4];
    [e setThreadgroupMemoryLength:SHMEM atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    bool ok = cb.status == MTLCommandBufferStatusCompleted;
    if (ok) {
        float *out = yb.contents;
        for (int i = 0; i < NC * NOUT; i++)
            if (!isfinite(out[i]) || fabsf(out[i] - 256.0f) > 0.5f) { ok = false; break; }
    }
    if (!ok) fprintf(stderr, "gpu: Metal 4 tensor self-test failed%s%s — using simdgroup GEMM\n",
                     cb.error ? ": " : "", cb.error ? cb.error.localizedDescription.UTF8String : "");
    [q release]; [wb release]; [xb release]; [yb release]; [bb release];
    return ok;
}

static id<MTLBuffer> f32_buf(id<MTLDevice> dev, const float *src, size_t n) {
    if (!src) return nil;
    return [dev newBufferWithBytes:src length:n * sizeof(float)
                           options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> f32_buf_ones(id<MTLDevice> dev, const float *src, size_t n) {
    if (src) return f32_buf(dev, src, n);
    if (n > SIZE_MAX / sizeof(float)) return nil;
    float *tmp = malloc(n * sizeof(float));
    if (!tmp) return nil;
    for (size_t i = 0; i < n; i++) tmp[i] = 1.0f;
    id<MTLBuffer> b = [dev newBufferWithBytes:tmp length:n * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    free(tmp);
    return b;
}


// Static half of the attention-front admission: everything knowable
// without resolving wrap buffers, answered as the pipeline that will run
// (nil = not front-capable). Shared by the engage site and by F2b's defer
// decision (a deferred add would starve the megakernel of a final x).
// qk-norm models are served (the kernel replays k_qknorm in-dispatch);
// v_rmsnorm and gated attention are not. The staged row needs n_embd inside
// the 32 KB threadgroup budget (<= 7936). Admission is MoE-only, from the
// measurement, not the type roster: the front recovers dispatch-chain
// latency, which is where MoE decode's milliseconds sit (120B +5.6%%, 30B
// +6.0%%, both byte-identical), while a fast dense model is matvec-bound
// and the head-partitioned grid starves the GPU that the split mv grid
// fills (Qwen3-8B measured -8.4%%). RUNNER_METAL_FRONT=all (test-only)
// lifts the MoE requirement so fixture-scale gates can pin every kernel.
static bool metal_front_all(void) {
    const char *v = getenv("RUNNER_METAL_FRONT");
    return v && (!strcmp(v, "all") || !strcmp(v, "1"));
}

static bool metal_front_off(void) {
    const char *v = getenv("RUNNER_METAL_FRONT");
    return v && (!strcmp(v, "0") || !strcmp(v, "off"));
}

static id<MTLComputePipelineState> metal_front_pipe(gpu_t *g, model_t *m,
                                                    int l) {
    if (l >= m->n_layer || metal_front_off()) return nil;
    layer_t *ly = &m->layers[l];
    if (m->kv_q8 || model_kv_owner(m, l) != l || !model_layer_ropes(m, l) ||
        m->v_rmsnorm || (m->attn_out_gate && ly->wq_gate) ||
        !ly->wq || !ly->wk || !ly->wv ||
        (m->n_expert <= 0 && !metal_front_all()) || m->n_embd > 7936)
        return nil;
    int tq = ly->wq->type, tk = ly->wk->type, tv = ly->wv->type;
    if (tq == T_Q8_0 && tk == T_Q8_0 && tv == T_Q8_0 && m->n_embd % 32 == 0)
        return g->p_attn_front_q8;
    if (tq == T_Q4_0 && tk == T_Q4_0 && tv == T_Q4_0 && m->n_embd % 32 == 0)
        return g->p_attn_front_q40;
    if (m->n_embd % 256) return nil;
    if (tq == T_Q4_K && tk == T_Q4_K && tv == T_Q4_K)
        return g->p_attn_front_q4k;
    if (tq == T_Q6_K && tk == T_Q6_K && tv == T_Q6_K)
        return g->p_attn_front_q6k;
    if (tq == T_Q4_K && tk == T_Q4_K && tv == T_Q6_K)
        return g->p_attn_front_q4k6;
    return nil;
}

static bool metal_front_capable(gpu_t *g, model_t *m, int l) {
    return metal_front_pipe(g, m, l) != nil;
}

static float *gpu_forward_native_batch(model_t *m, const int32_t *tokens,
                                       int n, int pos);

// Tiled prefill GEMM (k_mm_*): simdgroup matrix units instead of one output
// element per simdgroup. Not bit-identical to the matvec path by construction
// — the weight is dequantized before the multiply and the sum is reassociated
// into 8-element matrix steps — so it answers to tests/test_tc_tol.c, the same
// tolerance gate the CUDA tensor-core prefill answers to, through this hook.
// -1 restores the env default (RUNNER_METAL_MM), 0 pins the matvec path on,
// 1 forces the tiled path wherever a kernel exists for the weight type.
enum { MM_ENV_UNSET = -2 };
static int g_mm_state = MM_ENV_UNSET;
static int g_tensor_state = MM_ENV_UNSET;
static unsigned long g_tc_dispatches = 0;

void gpu_tc_force(int on) {
    g_mm_state = on < 0 ? MM_ENV_UNSET : (on != 0);
    g_tensor_state = on < 0 ? MM_ENV_UNSET : (on != 0);
}

unsigned long gpu_tc_dispatches(void) { return g_tc_dispatches; }

static bool metal_mm_on(void) {
    if (g_mm_state == MM_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_MM");
        if (!e || !*e) g_mm_state = -1;                  // promoted default
        else g_mm_state = strcmp(e, "0") && strcmp(e, "off");
    }
    if (g_mm_state >= 0) return g_mm_state != 0;
    return true;   // promoted: every type with a k_mm_* kernel, prefill only
}

// Metal 4 tensor operations are a separate, fail-closed rung above the
// established simdgroup GEMM. They are M5-only: the API exists on M4 but is a
// slower software path there. The tracer bullet stays opt-in until its real
// model tolerance and >=1.2x prefill promotion bars pass.
static bool metal_tensor_requested(void) {
    if (g_tensor_state != MM_ENV_UNSET) return g_tensor_state != 0;
    const char *e = getenv("RUNNER_METAL_TENSOR");
    return e && *e && strcmp(e, "0") && strcmp(e, "off");
}

static bool metal_tensor_device_ok(id<MTLDevice> dev) {
    if (!metal_tensor_requested()) return false;
    if (![dev.name hasPrefix:@"Apple M5"] && ![dev.name hasPrefix:@"Apple M6"])
        return false;
    NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
    return v.majorVersion > 26 || (v.majorVersion == 26 && v.minorVersion >= 2);
}
// Fast decode matvec (k_mvf_*): the reassociating twin of the identity matvec,
// selected at n_col == 1 only. See the kernel-family comment in kernels.metal
// for what it does that identity forbids, and tests/test_mv_tol.c for the gate.
//
// Default: OFF. It ships opt-in until a measurement promotes it, exactly as
// the fused CPU int8 dot does — a lever that has not cleared its bar on a
// machine is not a default anywhere. `RUNNER_METAL_MV=1` opts in; -1 through
// gpu_mv_force restores the env default.
enum { MV_ENV_UNSET = -2 };
static int g_mv_state = MV_ENV_UNSET;

void gpu_mv_force(int on) {
    g_mv_state = on < 0 ? MV_ENV_UNSET : (on != 0);
}

// Cooperative KV score read (k_attn_coop / k_attn_chunk_coop). It reassociates
// the per-row dot into a simd_sum over 32 lane partials, so it cannot live on
// the identity route -- it answers to tests/test_attn_tol.c the way the tiled
// prefill GEMM answers to test_tc_tol.c.
//
// PROMOTED 2026-08-17, and it is opt-OUT now: `RUNNER_METAL_ATTN_COOP=0` pins
// the byte-identical kernel back, exactly as `RUNNER_METAL_MM=0` does for
// prefill. The cert that bought this is in the suite's
// m1-night-results-2026-08-17.md -- 0/64 teacher-forced flips on every local
// model that engages the route, both KV cache formats, at a measured
// +3.0-4.3% decode across 2.3k-8.1k spans.
enum { COOP_ENV_UNSET = -2 };
static int g_coop_state = COOP_ENV_UNSET;
static unsigned long g_coop_dispatches = 0;

void gpu_attn_coop_force(int on) {
    g_coop_state = on < 0 ? COOP_ENV_UNSET : (on != 0);
}
unsigned long gpu_attn_coop_dispatches(void) { return g_coop_dispatches; }

static bool metal_attn_coop_on(void) {
    if (g_coop_state == COOP_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_ATTN_COOP");
        if (!e || !*e) g_coop_state = -1;                 // promoted default
        else g_coop_state = strcmp(e, "0") && strcmp(e, "off");
    }
    if (g_coop_state >= 0) return g_coop_state != 0;
    return true;   // promoted: decode attention, every arch that reaches it
}

static unsigned long g_mv_dispatches = 0;
unsigned long gpu_mv_dispatches(void) { return g_mv_dispatches; }

static bool metal_mv_fast_on(void) {
    // Cache the env read: this sits in the per-matvec dispatch path, and
    // gpu_mv_force(-1) resets to UNSET so the next call re-reads it.
    if (g_mv_state == MV_ENV_UNSET) {
        const char *e = getenv("RUNNER_METAL_MV");
        g_mv_state = e && *e && strcmp(e, "0") && strcmp(e, "off");
    }
    return g_mv_state > 0;
}

void gpu_moe_eager_force(int on) { (void)on; }

bool gpu_moe_ok(void) {
    return true;    // plain fused sparse-MoE routes and experts run on Metal
}

bool gpu_eseries_ok(void) {
    // The per-layer-embedding branch landed in 0.1.11 and is gated by
    // `make test-metal-eseries`. Before that this was honestly false, which is
    // exactly the window a scheduler had no way to see.
    return true;
}

bool gpu_kv_q8_ok(void) {
    return true;
}

// Largest K such that the prologue plus layers [0,K) fit `budget` bytes, and
// whether a prefix wrap is legal for this file at all.
//
// A zero-copy wrap must be one contiguous range, and every kernel indexes
// g->weights by a FILE offset, so the only shape needing no arithmetic change
// is a prefix: [0, end of blk.K-1). That requires non-layer tensors to sit
// before blk.0 and blocks to be ordered -- true of gemma-4-26B, gemma-3-4b,
// e2b, Trinity-Nano and gpt-oss, five architectures, but a property of how
// converters write files rather than of the format. So it is CHECKED, and a
// file that violates it gets full offload or CPU, never a truncated wrap
// pointed at the wrong weights.
// Would a partial split actually help? Only if the weights the CPU keeps still
// fit in available RAM. Wiring K layers into the Metal working set makes them
// non-evictable, which is the whole point -- but it also removes that memory
// from the page cache the remaining layers stream through, so on a heavily
// oversubscribed machine it makes the tail page harder than before. Explicit
// --gpu-layers bypasses this: an operator asking for a specific split gets it.
static bool metal_wrap_check(gpu_t *g, model_t *m, uint64_t wlen);

// Per-buffer ceiling for the weight wraps. RUNNER_METAL_MAX_BUF overrides the
// device's maxBufferLength so the multi-buffer path can be exercised on a
// machine whose models all fit one buffer -- which is every machine this was
// developed on. Without a knob the split path would ship untested.
static uint64_t metal_wbuf_cap(id<MTLDevice> dev) {
    const char *e = getenv("RUNNER_METAL_MAX_BUF");
    if (e && *e) {
        long long v = atoll(e);
        if (v > 0) return (uint64_t)v;
    }
    return (uint64_t)dev.maxBufferLength;
}

// Wrap [0, wlen) of the model mmap in as few zero-copy buffers as the
// per-buffer ceiling allows, cutting only at TENSOR boundaries so no tensor is
// ever split across two of them. That is what lets every kernel stay unchanged:
// a dispatch reads exactly one weight tensor, so the caller binds the
// containing buffer and passes an offset within it.
//
// Wraps may overlap by up to a page. A no-copy wrap must begin page-aligned
// and tensors are not page-aligned, so a cut rounds its base DOWN to a page,
// which re-includes the tail of the previous group. Those bytes are read-only
// and never addressed through the second wrap; the cost is one page of extra
// mapping per cut, and the alternative (cutting mid-tensor) is a correctness
// bug.
static bool metal_wrap_weights(gpu_t *g, model_t *m, id<MTLDevice> dev,
                               uint64_t wlen, size_t page) {
    uint64_t cap = metal_wbuf_cap(dev);
    g->n_wbuf = 0;
    if (wlen == 0 || cap == 0) return false;

    // Extents are HOST ADDRESSES: a split GGUF has one mapping per part, and
    // an address is the one key valid across all of them. A wrap can never
    // span two mappings (their address ranges are unrelated), so each extent
    // remembers its part and the grouping below cuts on a part change.
    uint32_t n_parts = gguf_map_count(&m->gf);
    uint64_t pbase[64], pend[64];
    if (n_parts == 0 || n_parts > 64) return false;
    for (uint32_t p = 0; p < n_parts; p++) {
        size_t psz;
        void *pm = gguf_map_part(&m->gf, p, &psz);
        if (!pm || !psz) return false;
        pbase[p] = (uint64_t)(uintptr_t)pm;
        pend[p]  = pbase[p] + psz;
    }

    // Tensor extents inside the wrapped prefix, in file order. GGUF writes
    // them ascending, but nothing in the format promises it, and a descending
    // file would silently produce overlapping groups -- so sort. The wlen
    // prefix (a partial layer split) is a part-0 file-offset concept and a
    // multi-part set never takes one: gpu_init refuses that combination.
    uint64_t n = m->gf.n_tensors;
    typedef struct { uint64_t beg, end; uint32_t part; } ext;
    ext *ex = calloc(n ? n : 1, sizeof(ext));
    if (!ex) return false;
    uint64_t n_ex = 0;
    for (uint64_t i = 0; i < n; i++) {
        gguf_tensor *t = &m->gf.tensors[i];
        if (!t->data) continue;
        uint64_t beg = (uint64_t)(uintptr_t)t->data;
        uint64_t end = beg + t->nbytes;
        if (n_parts == 1) {
            uint64_t off = beg - pbase[0];
            if (off >= wlen) continue;      // beyond the wrapped prefix
            if (off + t->nbytes > wlen) end = pbase[0] + wlen;
            ex[n_ex].part = 0;
        } else {
            uint32_t p = 0;
            while (p < n_parts && !(beg >= pbase[p] && end <= pend[p])) p++;
            if (p == n_parts) { free(ex); return false; }  // outside every map
            ex[n_ex].part = p;
        }
        ex[n_ex].beg = beg;
        ex[n_ex].end = end;
        n_ex++;
    }
    if (n_ex == 0) { free(ex); return false; }
    for (uint64_t i = 1; i < n_ex; i++) {   // insertion sort: near-sorted input
        ext k = ex[i];
        uint64_t j = i;
        while (j > 0 && ex[j - 1].beg > k.beg) { ex[j] = ex[j - 1]; j--; }
        ex[j] = k;
    }

    uint64_t gi = 0;
    while (gi < n_ex) {
        if (g->n_wbuf >= METAL_MAX_WBUF) {
            fprintf(stderr, "gpu: this model needs more than %d weight buffers "
                    "at a %.2f GB per-buffer limit — not wrapping\n",
                    METAL_MAX_WBUF, cap / 1e9);
            free(ex);
            return false;
        }
        uint32_t part = ex[gi].part;
        uint64_t base = ex[gi].beg & ~(uint64_t)(page - 1);
        if (base < pbase[part]) base = pbase[part];   // maps are page-aligned
        uint64_t last = ex[gi].end;
        if (last - base > cap) {
            // One tensor bigger than the ceiling. Nothing here can split it,
            // and pretending otherwise would wrap it short.
            fprintf(stderr, "gpu: a single %.2f GB tensor exceeds the %.2f GB "
                    "per-buffer limit — not wrapping\n",
                    (double)(ex[gi].end - ex[gi].beg) / 1e9, cap / 1e9);
            free(ex);
            return false;
        }
        uint64_t gj = gi + 1;
        while (gj < n_ex && ex[gj].part == part &&
               ex[gj].end - base <= cap) { last = ex[gj].end; gj++; }

        uint64_t len = (last - base + page - 1) & ~(uint64_t)(page - 1);
        if (base + len > pend[part]) len = pend[part] - base;
        id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:(uint8_t *)(uintptr_t)base
                                                 length:(NSUInteger)len
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
        if (!b) {
            for (int i = 0; i < g->n_wbuf; i++) { [g->wbuf[i] release]; g->wbuf[i] = nil; }
            g->n_wbuf = 0;
            free(ex);
            return false;
        }
        g->wbuf[g->n_wbuf] = b;
        g->wbuf_base[g->n_wbuf] = base;
        g->wbuf_end[g->n_wbuf] = base + len;
        g->n_wbuf++;
        gi = gj;
    }
    free(ex);
    if (!metal_wrap_check(g, m, wlen)) {
        for (int i = 0; i < g->n_wbuf; i++) { [g->wbuf[i] release]; g->wbuf[i] = nil; }
        g->n_wbuf = 0;
        return false;
    }
    if (g->n_wbuf > 1)
        fprintf(stderr, "gpu: weights wrapped in %d buffers (%.2f GB "
                "per-buffer limit)\n", g->n_wbuf, cap / 1e9);
    return true;
}

// Which wrap holds [off, off+len), and where inside it. The LENGTH is part of
// the question: a kernel reads a whole tensor, and a wrap that contains the
// first byte but not the last would read past its own mapping. Linear over at
// most METAL_MAX_WBUF entries, and one compare in the single-buffer case.
//
// Returns nil only when no single wrap contains the range, which
// metal_wrap_check() has already ruled out at init -- so a nil here is a
// programming error, and the callers say so rather than binding something
// plausible and returning wrong numbers.
static id<MTLBuffer> metal_wbuf_for(gpu_t *g, uint64_t off, uint64_t len,
                                    uint64_t *within) {
    for (int i = 0; i < g->n_wbuf; i++)
        if (off >= g->wbuf_base[i] && off + len <= g->wbuf_end[i]) {
            *within = off - g->wbuf_base[i];
            return g->wbuf[i];
        }
    *within = 0;
    return nil;
}

// RUNNER_METAL_INJECT_BIND_FAILURE=1: make the next weight binding
// unresolvable. The situation is otherwise unreachable by construction — a
// split GGUF is refused at load and a single-file wrap is checked at init —
// so without a hook the recovery path below could only be reasoned about, not
// run. Same argument as the command-buffer hook in docs/metal-fallback.md.
static bool metal_bind_inject(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RUNNER_METAL_INJECT_BIND_FAILURE");
        on = v && *v && strcmp(v, "0") ? 1 : 0;
    }
    return on > 0;
}

// Bind helper for the three sites that address the weight mmap. Keeps the
// fallback in ONE place: with a single wrap the resolver always succeeds, so
// this degrades to exactly the old behaviour and the diagnostic can never fire.
//
// When it does fire, the offload ENDS. This used to bind buffer 0 at the
// unresolvable offset, announce that "results from this model are not
// trustworthy", and compute — which is what a split GGUF got for as long as
// that path existed. A range no wrap holds cannot be read, and an internal bug
// is the last place to keep going: the flag is sticky, gpu_forward_native_batch
// discards the forward, and the caller finishes on the CPU. The encoder still
// needs a well-formed binding, so buffer 0 at offset 0 goes in and the result
// is thrown away rather than an out-of-range offset being handed to a kernel.
static uint64_t metal_bind_weights(gpu_t *g, id<MTLComputeCommandEncoder> e,
                                   uint64_t off, uint64_t len) {
    uint64_t within = 0;
    id<MTLBuffer> wb = metal_wbuf_for(g, off, len, &within);
    if (!wb || metal_bind_inject()) {
        if (!g->bind_failed) {
            g->bind_failed = true;
            fprintf(stderr, "gpu: no weight wrap holds [%llu,+%llu) — "
                    "falling back to CPU\n",
                    (unsigned long long)off, (unsigned long long)len);
        }
        wb = g->wbuf[0];
        within = 0;
    }
    [e setBuffer:wb offset:0 atIndex:0];
    return within;
}

// Every range a kernel will address must sit inside ONE wrap. The split is
// made at tensor boundaries so this holds by construction; checking it turns a
// silent wrong-answer bug into a refusal that falls back to a copied buffer.
// MoE expert banks are checked at full tensor length because the kernel strides
// across experts from the base offset, so the whole stack has to be reachable.
static bool metal_wrap_check(gpu_t *g, model_t *m, uint64_t wlen) {
    for (uint64_t i = 0; i < m->gf.n_tensors; i++) {
        gguf_tensor *t = &m->gf.tensors[i];
        if (!t->data) continue;
        uint64_t addr = (uint64_t)(uintptr_t)t->data;
        uint64_t len = t->nbytes;
        if (gguf_map_count(&m->gf) == 1) {
            // the wlen prefix (partial layer split) exists only single-map
            uint64_t off = addr - (uint64_t)(uintptr_t)m->gf.map;
            if (off >= wlen) continue;
            if (off + len > wlen) len = wlen - off;
        }
        uint64_t within;
        if (!metal_wbuf_for(g, addr, len, &within)) {
            fprintf(stderr, "gpu: tensor %s straddles a weight-buffer boundary "
                    "— not wrapping\n", t->name);
            return false;
        }
    }
    return true;
}

static bool metal_partial_pays(const model_t *m, uint64_t wrapped) {
    uint64_t have = plat_ram_available_bytes();
    if (!have) return false;                 // cannot tell: do not gamble
    // The WRAPPED part must fit in RAM. This is the condition that matters and
    // it is not the one this guard originally checked.
    //
    // The design behind partial offload argued the win was residency: wiring
    // weights into the Metal working set makes them non-evictable, so an
    // oversubscribed model would stop paging. That is wrong, and physics says
    // why -- you cannot pin more bytes than the machine has. Measured twice on
    // an 8 GB M1 with ~2.8 GB available:
    //   gemma-4-26B, 3.8 GB wrapped:  1.5-1.7 -> 0.21 tok/s   (8x slower)
    //   e4b-q4km,    4.3 GB wrapped:  5.6     -> 0.15 tok/s   (35x slower)
    // Both passed the old tail-fits check. Neither could hold its wrap
    // resident, so the pinned pages thrashed against everything else.
    //
    // What survives is narrow and honest: partial offload helps only when the
    // WHOLE model fits in RAM but its weights plus KV/scratch exceed Metal's
    // aggregate working-set budget. maxBufferLength no longer enters this
    // decision: the multi-buffer wrapper removes that per-resource ceiling.
    // In the fits-in-RAM case the split costs nothing in residency and gains
    // the ordinary compute blend. An oversubscribed model is refused however
    // the split is drawn, because whatever is pinned is taken from what the
    // rest must stream through.
    uint64_t total = m->gf.map_size;
    uint64_t margin = have / 8;              // KV, scratch, the OS
    return total + margin <= have && wrapped <= total;
}

static int metal_fit_layers(model_t *m, uint64_t budget, int cap,
                            bool *layout_ok, uint64_t *wrap_end) {
    *layout_ok = false;
    *wrap_end = 0;
    uint64_t *end_of = calloc((size_t)m->n_layer + 1, sizeof(uint64_t));
    if (!end_of) return 0;
    uint64_t prologue_end = 0;
    int prev_layer = -1;
    bool ok = true;
    for (uint64_t t = 0; t < m->gf.n_tensors && ok; t++) {
        gguf_tensor *w = &m->gf.tensors[t];
        uint64_t beg = (uint64_t)((uint8_t *)w->data - (uint8_t *)m->gf.map);
        uint64_t end = beg + w->nbytes;
        int li = strncmp(w->name, "blk.", 4) ? -1 : atoi(w->name + 4);
        if (li < 0) {
            if (prev_layer >= 0) { ok = false; break; }   // non-layer after a layer
            if (end > prologue_end) prologue_end = end;
        } else if (li >= m->n_layer || li < prev_layer) {
            ok = false;                                    // out of range or unordered
        } else {
            prev_layer = li;
            if (end > end_of[li + 1]) end_of[li + 1] = end;
        }
    }
    int K = 0;
    if (ok && end_of[m->n_layer] > 0) {
        end_of[0] = prologue_end;
        for (int i = 1; i <= m->n_layer; i++)
            if (end_of[i] < end_of[i - 1]) end_of[i] = end_of[i - 1];
        *layout_ok = true;
        int top = (cap > 0 && cap < m->n_layer) ? cap : m->n_layer;
        for (int i = 1; i <= top; i++)
            if (end_of[i] <= budget) { K = i; *wrap_end = end_of[i]; } else break;
    }
    free(end_of);
    return K;
}

bool gpu_init(model_t *m) {
    // A split GGUF has one mapping per part. The weight wraps are keyed by
    // HOST ADDRESS (not file offset), so each part's mapping gets its own
    // tensor-boundary wraps and a multi-part set takes a full offload like a
    // single file. What the separate mappings cannot express is a partial
    // LAYER split: the prefix arithmetic in metal_fit_layers is file-offset
    // arithmetic over one mapping, so that case refuses to CPU below rather
    // than guessing.
    if (m->qwen35) {
        fprintf(stderr, "gpu: qwen35 hybrid path is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (m->attn_out_gate && m->n_expert > 0) {
        // the dense output gate (muse-glimmer) is implemented below; gate +
        // sparse MoE (afmoe) has never been run against the identity gates,
        // so it keeps its CPU fallback rather than gaining silent support
        fprintf(stderr, "gpu: gated attention with sparse MoE (afmoe) is not on the metal backend yet — using CPU\n");
        return false;
    }
    if (!metal_moe_supported(m))
        return false;
    if (m->ffn_act != ACT_SILU && m->ffn_act != ACT_GELU &&
        !(m->n_expert > 0 && m->gptoss && m->ffn_act == ACT_SWIGLU_OAI)) {
        fprintf(stderr, "gpu: '%s' FFN activation is not on the metal backend yet — using CPU\n",
                m->arch);
        return false;
    }
    // every weight matmul must have a kernel for its quant type
    if (!metal_tensor_type_ok(m->output, false)) return false;
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        if (!ly->wv && !m->v_rmsnorm) {
            fprintf(stderr, "gpu: '%s' layer layout is not on the metal backend yet — using CPU\n",
                    m->arch);
            return false;
        }
        if (model_is_swa(m, l) && !m->rope_inv_freq_local) {
            fprintf(stderr, "gpu: '%s' sliding-window rope table is missing — using CPU\n",
                    m->arch);
            return false;
        }
        // EVERY tensor this layer will hand to enc_mv_n, because that indexes
        // g->p_mv[] / g->p_mm[] by the tensor's type: an unadmitted type is a
        // nil pipeline at best and an out-of-range read of the table at worst.
        // ple_gate and ple_proj (the gemma-4 E-series per-layer embedding
        // projections) were missing, so an E-series file quantizing them
        // outside this list reached the dispatch instead of the refusal.
        // Bound off the array, not a literal 8 — that literal is the same kind
        // of hand-kept second fact the list itself is.
        gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                              ly->w_gate, ly->w_up, ly->w_down, ly->wq_gate,
                              ly->ple_gate, ly->ple_proj };
        for (size_t i = 0; i < sizeof(ws) / sizeof(*ws); i++)
            if (!metal_tensor_type_ok(ws[i], false)) return false;
    }

    id<MTLDevice> dev = metal_init_injected("device")
                      ? nil : MTLCreateSystemDefaultDevice();
    if (!dev) {
        fprintf(stderr, "gpu: no Metal device is available — using CPU\n");
        return false;
    }

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
                              [NSString stringWithUTF8String:k_metal_src]
                                           options:nil
                                             error:&err];
    if (!lib) {
        fprintf(stderr, "gpu: shader compile failed: %s\n",
                err.localizedDescription.UTF8String);
        [dev release];
        return false;
    }

    gpu_t *g = calloc(1, sizeof(gpu_t));
    if (!g) {
        fprintf(stderr, "gpu: Metal backend state allocation failed — using "
                "CPU\n");
        [lib release];
        [dev release];
        return false;
    }
    g->dev = dev;
    g->queue = [dev newCommandQueue];
    g->p_rmsnorm      = mk_pipeline(dev, lib, @"k_rmsnorm");
    g->p_qknorm       = mk_pipeline(dev, lib, @"k_qknorm");
    g->p_headnorm     = mk_pipeline(dev, lib, @"k_head_rmsnorm");
    g->p_rope         = mk_pipeline(dev, lib, @"k_rope");
    g->p_store        = mk_pipeline(dev, lib, @"k_store_kv");
    g->p_attn         = mk_pipeline(dev, lib, @"k_attn");
    g->p_attn_coop    = mk_pipeline(dev, lib, @"k_attn_coop");
    g->p_attn_chunk_coop = mk_pipeline(dev, lib, @"k_attn_chunk_coop");
    g->p_attn_chunk   = mk_pipeline(dev, lib, @"k_attn_chunk");
    g->p_attn_comb    = mk_pipeline(dev, lib, @"k_attn_combine");
    g->p_silu         = mk_pipeline(dev, lib, @"k_silu_mul");
    g->p_gelu         = mk_pipeline(dev, lib, @"k_gelu_mul");
    g->p_add          = mk_pipeline(dev, lib, @"k_add");
    g->p_sigmul       = mk_pipeline(dev, lib, @"k_sigmoid_mul");
    g->p_scale        = mk_pipeline(dev, lib, @"k_scale");
    g->p_moe_route    = mk_pipeline(dev, lib, @"k_moe_route");
    g->p_moe_actmul   = mk_pipeline(dev, lib, @"k_moe_actmul");
    g->p_moe_sum      = mk_pipeline(dev, lib, @"k_moe_sum");
    g->p_trace_copy   = mk_pipeline(dev, lib, @"k_trace_copy_f32");
    g->p_mm[T_F32]   = mk_pipeline(dev, lib, @"k_mm_f32");
    g->p_mm[T_F16]   = mk_pipeline(dev, lib, @"k_mm_f16");
    g->p_mm[T_Q8_0]  = mk_pipeline(dev, lib, @"k_mm_q8_0");
    g->p_mm[T_Q4_0]  = mk_pipeline(dev, lib, @"k_mm_q4_0");
    g->p_mm[T_Q2_K]  = mk_pipeline(dev, lib, @"k_mm_q2_K");
    g->p_mm[T_Q3_K]  = mk_pipeline(dev, lib, @"k_mm_q3_K");
    g->p_mm[T_BF16]  = mk_pipeline(dev, lib, @"k_mm_bf16");
    g->p_mm[T_IQ4_NL] = mk_pipeline(dev, lib, @"k_mm_iq4_nl");
    g->p_mm[T_IQ4_XS] = mk_pipeline(dev, lib, @"k_mm_iq4_xs");
    g->p_mm[T_Q4_K]  = mk_pipeline(dev, lib, @"k_mm_q4_K");
    g->p_mm[T_Q6_K]  = mk_pipeline(dev, lib, @"k_mm_q6_K");
    g->p_mm[T_MXFP4] = mk_pipeline(dev, lib, @"k_mm_mxfp4");
    g->p_mv[T_F32]    = mk_pipeline(dev, lib, @"k_mv_f32");
    g->p_mv[T_F16]    = mk_pipeline(dev, lib, @"k_mv_f16");
    g->p_mv[T_Q8_0]   = mk_pipeline(dev, lib, @"k_mv_q8_0");
    g->p_mv[T_Q4_0]   = mk_pipeline(dev, lib, @"k_mv_q4_0");
    g->p_mv[T_Q4_1]   = mk_pipeline(dev, lib, @"k_mv_q4_1");
    g->p_mv[T_Q5_0]   = mk_pipeline(dev, lib, @"k_mv_q5_0");
    g->p_mv[T_Q5_1]   = mk_pipeline(dev, lib, @"k_mv_q5_1");
    g->p_mv[T_Q2_K]   = mk_pipeline(dev, lib, @"k_mv_q2_K");
    g->p_mv[T_Q3_K]   = mk_pipeline(dev, lib, @"k_mv_q3_K");
    g->p_mv[T_BF16]   = mk_pipeline(dev, lib, @"k_mv_bf16");
    g->p_mv[T_IQ4_NL] = mk_pipeline(dev, lib, @"k_mv_iq4_nl");
    g->p_mv[T_IQ4_XS] = mk_pipeline(dev, lib, @"k_mv_iq4_xs");
    g->p_mv[T_Q4_K]   = mk_pipeline(dev, lib, @"k_mv_q4_K");
    g->p_mv[T_Q5_K]   = mk_pipeline(dev, lib, @"k_mv_q5_K");
    g->p_mv[T_Q6_K]   = mk_pipeline(dev, lib, @"k_mv_q6_K");
    g->p_mv[T_MXFP4]  = mk_pipeline(dev, lib, @"k_mv_mxfp4");

    // Fast (reassociating) decode matvec, gated by RUNNER_METAL_MV. Only the
    // two types that carry real decode traffic on the models this was measured
    // against; a type with no entry here simply keeps the identity kernel, so
    // the coverage can grow one measured format at a time.
    g->p_mvf[T_Q4_0]  = mk_pipeline(dev, lib, @"k_mvf_q4_0");
    g->p_mvf[T_Q8_0]  = mk_pipeline(dev, lib, @"k_mvf_q8_0");
    g->p_moe_mv[T_F32]  = mk_pipeline(dev, lib, @"k_moe_mv_f32");
    g->p_moe_mv[T_F16]  = mk_pipeline(dev, lib, @"k_moe_mv_f16");
    g->p_moe_mv[T_Q8_0] = mk_pipeline(dev, lib, @"k_moe_mv_q8_0");
    g->p_moe_mv[T_Q4_0] = mk_pipeline(dev, lib, @"k_moe_mv_q4_0");
    g->p_moe_mv[T_Q2_K] = mk_pipeline(dev, lib, @"k_moe_mv_q2_K");
    g->p_moe_mv[T_Q3_K] = mk_pipeline(dev, lib, @"k_moe_mv_q3_K");
    g->p_moe_mv[T_Q4_K] = mk_pipeline(dev, lib, @"k_moe_mv_q4_K");
    g->p_moe_mv[T_Q5_K] = mk_pipeline(dev, lib, @"k_moe_mv_q5_K");
    g->p_moe_mv[T_Q6_K] = mk_pipeline(dev, lib, @"k_moe_mv_q6_K");
    g->p_moe_mv[T_MXFP4] = mk_pipeline(dev, lib, @"k_moe_mv_mxfp4");
    g->p_moe_mv_em[T_F32]   = mk_pipeline(dev, lib, @"k_moe_mv_em_f32");
    g->p_moe_mv_em[T_F16]   = mk_pipeline(dev, lib, @"k_moe_mv_em_f16");
    g->p_moe_mv_em[T_Q8_0]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q8_0");
    g->p_moe_mv_em[T_Q4_0]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q4_0");
    g->p_moe_mv_em[T_Q2_K]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q2_K");
    g->p_moe_mv_em[T_Q3_K]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q3_K");
    g->p_moe_mv_em[T_Q4_K]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q4_K");
    g->p_moe_mv_em[T_Q5_K]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q5_K");
    g->p_moe_mv_em[T_Q6_K]  = mk_pipeline(dev, lib, @"k_moe_mv_em_q6_K");
    g->p_moe_mv_em[T_MXFP4] = mk_pipeline(dev, lib, @"k_moe_mv_em_mxfp4");
    g->p_moe_mv_em8[T_Q8_0]  = mk_pipeline(dev, lib, @"k_moe_mv_em8_q8_0");
    g->p_moe_mv_em8[T_MXFP4] = mk_pipeline(dev, lib, @"k_moe_mv_em8_mxfp4");
    g->p_rope_store       = mk_pipeline(dev, lib, @"k_rope_store");
    g->p_attn_front_q8    = mk_pipeline(dev, lib, @"k_attn_front_q8_0");
    g->p_attn_front_q40   = mk_pipeline(dev, lib, @"k_attn_front_q4_0");
    g->p_attn_front_q4k   = mk_pipeline(dev, lib, @"k_attn_front_q4_k");
    g->p_attn_front_q6k   = mk_pipeline(dev, lib, @"k_attn_front_q6_k");
    g->p_attn_front_q4k6  = mk_pipeline(dev, lib, @"k_attn_front_q4k_q6k");
    g->p_add_rmsnorm      = mk_pipeline(dev, lib, @"k_add_rmsnorm");
    g->p_moe_gua[T_F32]   = mk_pipeline(dev, lib, @"k_moe_gua_f32");
    g->p_moe_gua[T_F16]   = mk_pipeline(dev, lib, @"k_moe_gua_f16");
    g->p_moe_gua[T_Q8_0]  = mk_pipeline(dev, lib, @"k_moe_gua_q8_0");
    g->p_moe_gua[T_Q4_0]  = mk_pipeline(dev, lib, @"k_moe_gua_q4_0");
    g->p_moe_gua[T_Q2_K]  = mk_pipeline(dev, lib, @"k_moe_gua_q2_K");
    g->p_moe_gua[T_Q3_K]  = mk_pipeline(dev, lib, @"k_moe_gua_q3_K");
    g->p_moe_gua[T_Q4_K]  = mk_pipeline(dev, lib, @"k_moe_gua_q4_K");
    g->p_moe_gua[T_Q5_K]  = mk_pipeline(dev, lib, @"k_moe_gua_q5_K");
    g->p_moe_gua[T_Q6_K]  = mk_pipeline(dev, lib, @"k_moe_gua_q6_K");
    g->p_moe_gua[T_MXFP4] = mk_pipeline(dev, lib, @"k_moe_gua_mxfp4");
    g->p_moe_group        = mk_pipeline(dev, lib, @"k_moe_group");
    g->p_moe_mm[T_Q8_0]   = mk_pipeline(dev, lib, @"k_moe_mm_q8_0");
    g->p_moe_mm[T_MXFP4]  = mk_pipeline(dev, lib, @"k_moe_mm_mxfp4");
    g->p_moe_mm[T_F32]    = mk_pipeline(dev, lib, @"k_moe_mm_f32");
    g->p_moe_mm[T_F16]    = mk_pipeline(dev, lib, @"k_moe_mm_f16");
    g->p_moe_mm[T_Q4_0]   = mk_pipeline(dev, lib, @"k_moe_mm_q4_0");
    g->p_moe_mm[T_Q2_K]   = mk_pipeline(dev, lib, @"k_moe_mm_q2_K");
    g->p_moe_mm[T_Q3_K]   = mk_pipeline(dev, lib, @"k_moe_mm_q3_K");
    g->p_moe_mm[T_Q4_K]   = mk_pipeline(dev, lib, @"k_moe_mm_q4_K");
    g->p_moe_mm[T_Q6_K]   = mk_pipeline(dev, lib, @"k_moe_mm_q6_K");
    g->p_moe_mmh[T_Q8_0]  = mk_pipeline(dev, lib, @"k_moe_mmh_q8_0");
    g->p_moe_mmh[T_MXFP4] = mk_pipeline(dev, lib, @"k_moe_mmh_mxfp4");
    [lib release];
    lib = nil;

    if (metal_tensor_device_ok(dev)) {
#if defined(MAC_OS_VERSION_26_0)
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError *terr = nil;
        id<MTLLibrary> tlib = [dev newLibraryWithSource:
            [NSString stringWithUTF8String:k_metal_tensor_src]
            options:opts error:&terr];
        if (!tlib) {
            fprintf(stderr, "gpu: Metal 4 tensor library unavailable: %s — using simdgroup GEMM\n",
                    terr ? terr.localizedDescription.UTF8String : "unknown error");
        } else {
            id<MTLComputePipelineState> tp = mk_tensor_pipeline(dev, tlib, @"k_tensor_q4_K");
            if (tp && metal_tensor_self_test(dev, tp, T_Q4_K)) {
                g->p_tensor[T_Q4_K] = tp;
                id<MTLComputePipelineState> t8 =
                    mk_tensor_pipeline(dev, tlib, @"k_tensor_q8_0");
                if (t8 && metal_tensor_self_test(dev, t8, T_Q8_0))
                    g->p_tensor[T_Q8_0] = t8;
                else
                    [t8 release];
                id<MTLComputePipelineState> t4 =
                    mk_tensor_pipeline(dev, tlib, @"k_tensor_q4_0");
                if (t4 && metal_tensor_self_test(dev, t4, T_Q4_0))
                    g->p_tensor[T_Q4_0] = t4;
                else
                    [t4 release];
                fprintf(stderr, "gpu: Metal 4 tensor GEMM admitted for Q4_K%s%s\n",
                        g->p_tensor[T_Q8_0] ? "/Q8_0" : "",
                        g->p_tensor[T_Q4_0] ? "/Q4_0" : "");
            } else {
                [tp release];
            }
            [tlib release];
        }
        [opts release];
#else
        fprintf(stderr, "gpu: this build SDK has no Metal 4 compiler — using simdgroup GEMM\n");
#endif
    } else if (metal_tensor_requested()) {
        fprintf(stderr, "gpu: Metal 4 tensor GEMM requires M5+ and macOS 26.2+ — using simdgroup GEMM\n");
    }
    if (!g->p_rmsnorm || !g->p_rope || !g->p_store || !g->p_attn ||
        !g->p_attn_chunk || !g->p_attn_comb ||
        !g->p_silu || !g->p_gelu || !g->p_add || !g->p_scale || !g->p_sigmul ||
        !g->queue || !g->p_qknorm || !g->p_headnorm ||
        !g->p_moe_route || !g->p_moe_actmul || !g->p_moe_sum ||
        !g->p_trace_copy)
        return gpu_init_fail(m, g, lib, "pipeline allocation");
    // Admission and the pipeline tables must agree BY CONSTRUCTION rather than
    // by two hand-kept lists. enc_mv and enc_moe_mv index p_mv[]/p_moe_mv[] by
    // tensor type with no nil check at the use site, and
    // -setComputePipelineState:nil raises rather than failing a return value:
    // a type added to an admission allowlist without its kernel would crash
    // mid-forward, on whatever model first carried that type. Driving the
    // check from the same predicates the loader admits with turns that into a
    // loud refusal at load, and keeps working when a type is added to either
    // list. (This is what the two literal lists here used to assert by hand,
    // and they had already fallen behind: the dense list checked 11 of the 16
    // types gpu_type_ok admits, leaving BF16, Q2_K, Q3_K, IQ4_NL and IQ4_XS
    // unasserted. All 16 do have kernels today -- the list had drifted, not
    // the kernels.)
    for (int t = 0; t < METAL_TYPE_SLOTS; t++) {
        if (gpu_type_ok(t) && !g->p_mv[t]) {
            fprintf(stderr, "gpu: no Metal matvec kernel for admitted type %s\n",
                    ggml_type_name(t));
            return gpu_init_fail(m, g, lib, "pipeline allocation");
        }
        if (m->n_expert > 0 && metal_moe_type_ok(t) && !g->p_moe_mv[t]) {
            fprintf(stderr, "gpu: no Metal MoE kernel for admitted type %s\n",
                    ggml_type_name(t));
            return gpu_init_fail(m, g, lib, "MoE pipeline allocation");
        }
    }
    if (metal_init_injected("state"))
        return gpu_init_fail(m, g, lib, "injected state allocation failure");

    // weights: wrap the mmap zero-copy (page aligned; length page-rounded —
    // mmap always maps whole pages, so the rounded tail is valid memory).
    // wlen is a prefix limit in part-0 file-offset space; for a multi-part
    // set it is never a prefix (full offload only) and wtotal carries the
    // budget question across every part's mapping.
    size_t page = 16384;
    size_t wlen = (m->gf.map_size + page - 1) & ~(page - 1);
    uint64_t wtotal = gguf_mapped_size(&m->gf);

    // Choose K. Full offload stays the default and the fast path; a partial
    // split is only entered when the whole file cannot be wrapped, because a
    // model that fits should never pay a CPU tail.
    //
    // The ceiling covers KV and scratch as well as weights, not weights alone
    // -- measured: a 5.17 GB wrap was refused under a 5.73 GB limit. So the
    // budget subtracts what this load will also allocate, and keeps a margin
    // on top rather than trusting the remainder to be exact.
    int gpu_K = m->n_layer;
    uint64_t ws_limit = dev.recommendedMaxWorkingSetSize;
    {
        // recommendedMaxWorkingSetSize is the aggregate residency ceiling.
        // The budget subtracts KV and scratch and keeps a safety margin.
        // maxBufferLength is smaller (4.29 GB against 5.73 GB on an M1), but
        // it is only the size of each individual wrap: metal_wrap_weights()
        // cuts a larger file into several tensor-boundary buffers. Clamping
        // this budget to maxBufferLength preserved the old ceiling in the
        // admission decision even after the multi-buffer implementation had
        // removed it from allocation.
        uint64_t other = (uint64_t)model_kv_byte_off(m, m->n_layer) * 2
                       + 512ull * 1024 * 1024;
        metal_weight_limits limits = {
            .working_set = ws_limit,
            .max_buffer = (uint64_t)dev.maxBufferLength,
        };
        uint64_t budget = metal_full_weight_budget(limits, other);
        int cap = m->gpu_layers_override;
        bool need_split = wtotal > budget || (cap > 0 && cap < m->n_layer);
        if (need_split && m->gf.n_maps > 1) {
            // Either the whole set exceeds the budget or a layer split was
            // forced; both need the prefix arithmetic one mapping cannot span.
            return gpu_init_fail(m, g, lib,
                "a split GGUF takes a full Metal offload only — a partial "
                "layer split cannot span its separate part mappings (merge "
                "to one file with --quantize --quant keep for a split)");
        }
        if (need_split) {
            bool layout_ok = false;
            uint64_t wrap_end = 0;
            int fit = metal_fit_layers(m, budget, cap, &layout_ok, &wrap_end);
            if (!layout_ok) {
                if (wlen > budget)
                    return gpu_init_fail(m, g, lib,
                        "model exceeds the Metal working set and its tensor "
                        "layout forbids a partial split (non-layer tensors "
                        "after blk.0, or blocks out of order)");
                fprintf(stderr, "gpu: --gpu-layers ignored — tensor layout "
                        "does not allow a partial split\n");
            } else if (fit < 1) {
                return gpu_init_fail(m, g, lib,
                    "not even one layer fits the Metal working set");
            } else if (wlen > budget && cap <= 0 &&
                       !metal_partial_pays(m, wrap_end)) {
                // Auto-selected partial offload that would make things WORSE.
                // Measured on an 8 GB M1 with gemma-4-26B: wrapping 3.8 GB of
                // 14.4 left the remaining 10.6 GB of CPU-side weights fighting
                // an already-starved page cache, and decode went 1.5-1.7 ->
                // 0.21 tok/s. An 8x regression, enabled by default, on the
                // exact model this feature was built for.
                //
                // Residency only pays if the tail the CPU still owns fits in
                // RAM. When it does not, pinning part of the model just takes
                // memory from the part that is streaming.
                return gpu_init_fail(m, g, lib,
                    "model is larger than available RAM, so no split can be "
                    "held resident — CPU-only is faster (--gpu-layers forces "
                    "a split anyway)");
            } else {
                gpu_K = fit;
                if (gpu_K < m->n_layer) {
                    wlen = (size_t)((wrap_end + page - 1) & ~(uint64_t)(page - 1));
                    if (wlen > m->gf.map_size)
                        wlen = (m->gf.map_size + page - 1) & ~(size_t)(page - 1);
                }
            }
        }
    }

    if (!metal_wrap_weights(g, m, dev, wlen, page)) {
        if (m->gf.n_maps > 1)
            // One copied buffer cannot stand in for several mappings, and
            // copying parts separately would just re-implement the wrap that
            // already failed. Refuse; the run continues on the CPU.
            return gpu_init_fail(m, g, lib,
                "could not wrap the split GGUF's part mappings");
        // fallback: copy (costs RAM but still works). One buffer by
        // definition -- a copy this large only happens for a model well under
        // the per-buffer ceiling, since the ceiling is what forced the split.
        g->wbuf[0] = [dev newBufferWithBytes:m->gf.map
                                      length:m->gf.map_size
                                     options:MTLResourceStorageModeShared];
        g->wbuf_base[0] = (uint64_t)(uintptr_t)m->gf.map;
        g->wbuf_end[0] = g->wbuf_base[0] + m->gf.map_size;
        g->n_wbuf = g->wbuf[0] ? 1 : 0;
        g->weights_copied = true;
        if (!g->wbuf[0]) {
            // say what was asked for and what the device allows — the RAM
            // warning two lines below this in a load gives numbers, and a
            // bare "allocation failed" gives a scheduler nothing to reason
            // with (16 GB-Mac field report, 2026-08-05)
            char why[192];
            uint64_t ws = dev.recommendedMaxWorkingSetSize;
            // Name BOTH limits: a message quoting only the working set reads
            // as self-contradictory whenever maxBufferLength is what refused
            // the allocation, and that is the common case (it is 0.75x the
            // working set). A field report spent its entire run trying to
            // reconcile "10.3 requested, 12.7 limit, failed".
            uint64_t mb = (uint64_t)dev.maxBufferLength;
            snprintf(why, sizeof why,
                     "weight buffer allocation: %.1f GB requested; device max "
                     "single buffer %.1f GB, working set %.1f GB%s",
                     wlen / 1e9, mb / 1e9, ws / 1e9,
                     (uint64_t)wlen > mb ? " — exceeds the per-buffer limit"
                     : ((uint64_t)wlen > ws ? " — exceeds the working set" : ""));
            return gpu_init_fail(m, g, lib, why);
        }
    }

    // Q-bias width off the per-layer MAXIMUM: gemma4 varies q width per layer,
    // and a global-scalar size would overrun on the widest one. The per-column
    // scratch takes the same maxima inside metal_ensure_batch().
    int q_dim = m->n_head * m->head_dim;
    for (int l = 0; l < m->n_layer; l++)
        if (model_q_dim(m, l) > q_dim) q_dim = model_q_dim(m, l);
    size_t kv_bytes = model_kv_byte_off(m, m->n_layer);

    #define NEWBUF(n) [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
    g->kc = NEWBUF(kv_bytes);
    g->vc = NEWBUF(kv_bytes);
    if (!metal_buffer_ok(g->kc) || !metal_buffer_ok(g->vc))
        return gpu_init_fail(m, g, lib, "KV buffer allocation");
    memset(g->kc.contents, 0, kv_bytes);
    memset(g->vc.contents, 0, kv_bytes);
    if (metal_init_injected("after-kv"))
        return gpu_init_fail(m, g, lib, "injected post-KV allocation failure");

    // Per-column scratch (x, xb, xb2, q, kt, vt, hb, hb2, att, logits, the
    // E-series PLE slices and the attention output gate) through the same
    // function a prompt batch widens it with, so the two cannot list different
    // buffers. g->batch_cap is still 0 from the calloc, so this allocates.
    if (!metal_ensure_batch(g, m, 1))
        return gpu_init_fail(m, g, lib, "batch scratch allocation");
    g->dummy  = NEWBUF(4);
    // Chunked decode attention partials: one (max, sum, hd-vector) per
    // (head, chunk). They carry NO batch dimension -- the split is taken only
    // at n == 1 -- so they are sized once here rather than in
    // metal_ensure_batch(). Sizing them there left them nil for the whole run
    // of a session whose every forward is a single token, because
    // metal_ensure_batch returns early at n <= batch_cap and batch_cap starts
    // at 1: the chunked kernels then wrote their partials through a nil
    // binding and decode produced garbage past the chunk threshold, silently
    // (`make test-metal-decode-only`).
    g->att_acc = NEWBUF(sizeof(float) * (size_t)m->n_head *
                        METAL_ATTN_MAX_CHUNKS * (size_t)model_head_dim(m, 0));
    g->att_ms  = NEWBUF(sizeof(float) * (size_t)m->n_head *
                        METAL_ATTN_MAX_CHUNKS * 2);
    #undef NEWBUF
    if (!metal_buffer_ok(g->dummy) ||
        !metal_buffer_ok(g->att_acc) || !metal_buffer_ok(g->att_ms))
        return gpu_init_fail(m, g, lib, "scratch buffer allocation");

    g->inv_freq = f32_buf(dev, m->rope_inv_freq, m->rope_dim / 2);
    g->inv_freq_local = f32_buf(dev, m->rope_inv_freq_local,
                                m->rope_dim_local / 2);
    g->out_norm = f32_buf(dev, m->out_norm_w, m->n_embd);
    if (m->n_suppress > 0)
        g->suppress = [dev newBufferWithBytes:m->suppress
                                       length:(size_t)m->n_suppress * sizeof(int32_t)
                                      options:MTLResourceStorageModeShared];
    if (!metal_buffer_ok(g->inv_freq) || !metal_buffer_ok(g->out_norm) ||
        (m->rope_inv_freq_local && !metal_buffer_ok(g->inv_freq_local)) ||
        (m->n_suppress > 0 && !metal_buffer_ok(g->suppress)))
        return gpu_init_fail(m, g, lib, "shared constant allocation");
    g->attn_norm = calloc(m->n_layer, sizeof(id));
    g->ffn_norm  = calloc(m->n_layer, sizeof(id));
    g->bq = calloc(m->n_layer, sizeof(id));
    g->bk = calloc(m->n_layer, sizeof(id));
    g->bv = calloc(m->n_layer, sizeof(id));
    g->bo = calloc(m->n_layer, sizeof(id));
    g->qn = calloc(m->n_layer, sizeof(id));
    g->kn = calloc(m->n_layer, sizeof(id));
    g->sinks = calloc(m->n_layer, sizeof(id));
    g->gib = calloc(m->n_layer, sizeof(id));
    g->geb = calloc(m->n_layer, sizeof(id));
    g->ueb = calloc(m->n_layer, sizeof(id));
    g->deb = calloc(m->n_layer, sizeof(id));
    g->ppn = calloc(m->n_layer, sizeof(id));
    g->pan = calloc(m->n_layer, sizeof(id));
    g->pfn = calloc(m->n_layer, sizeof(id));
    g->gpn1 = calloc(m->n_layer, sizeof(id));
    g->gprn2 = calloc(m->n_layer, sizeof(id));
    g->gpn2 = calloc(m->n_layer, sizeof(id));
    g->ggis = calloc(m->n_layer, sizeof(id));
    g->gdsc = calloc(m->n_layer, sizeof(id));
    if (!g->attn_norm || !g->ffn_norm || !g->bq || !g->bk || !g->bv ||
        !g->ppn ||
        !g->bo || !g->qn || !g->kn || !g->sinks || !g->gib || !g->geb ||
        !g->ueb || !g->deb || !g->pan || !g->pfn || !g->gpn1 ||
        !g->gprn2 || !g->gpn2 || !g->ggis || !g->gdsc)
        return gpu_init_fail(m, g, lib, "per-layer table allocation");
    for (int l = 0; l < m->n_layer; l++) {
        layer_t *ly = &m->layers[l];
        g->attn_norm[l] = f32_buf(dev, ly->attn_norm_w, m->n_embd);
        g->ffn_norm[l]  = f32_buf(dev, ly->ffn_norm_w, m->n_embd);
        int l_hd = model_head_dim(m, l);
        int l_kv_dim = model_kv_dim(m, l);
        g->bq[l] = f32_buf(dev, ly->bq, q_dim);
        g->bk[l] = f32_buf(dev, ly->bk, l_kv_dim);
        g->bv[l] = f32_buf(dev, ly->bv, l_kv_dim);
        g->bo[l] = f32_buf(dev, ly->bo, m->n_embd);
        g->qn[l] = f32_buf(dev, ly->qnorm_w, l_hd);
        g->kn[l] = f32_buf(dev, ly->knorm_w, l_hd);
        g->sinks[l] = f32_buf(dev, ly->attn_sinks, m->n_head);
        g->pan[l] = f32_buf(dev, ly->post_attn_norm_w, m->n_embd);
        g->ppn[l] = f32_buf(dev, ly->ple_post_norm, m->n_embd);
        g->pfn[l] = f32_buf(dev, ly->post_ffn_norm_w, m->n_embd);
        g->gib[l] = f32_buf(dev, ly->ffn_gate_inp_b, m->n_expert);
        g->geb[l] = f32_buf(dev, ly->ffn_gate_exps_b,
                            (size_t)m->n_expert * (size_t)m->n_ff_exp);
        g->ueb[l] = f32_buf(dev, ly->ffn_up_exps_b,
                            (size_t)m->n_expert * (size_t)m->n_ff_exp);
        g->deb[l] = f32_buf(dev, ly->ffn_down_exps_b,
                            (size_t)m->n_expert * (size_t)m->n_embd);
        if (ly->moe_gemma) {
            g->gpn1[l] = f32_buf_ones(dev, ly->ffn_post_norm1_w, m->n_embd);
            g->gprn2[l] = f32_buf_ones(dev, ly->ffn_pre_norm2_w, m->n_embd);
            g->gpn2[l] = f32_buf_ones(dev, ly->ffn_post_norm2_w, m->n_embd);
            float *gs = malloc(sizeof(float) * (size_t)m->n_embd);
            if (!gs) return gpu_init_fail(m, g, lib, "Gemma router scale allocation");
            float inv = 1.0f / sqrtf((float)m->n_embd);
            for (int i = 0; i < m->n_embd; i++)
                gs[i] = (ly->gate_inp_scale ? ly->gate_inp_scale[i] : 1.0f) * inv;
            g->ggis[l] = f32_buf(dev, gs, m->n_embd);
            free(gs);
            g->gdsc[l] = f32_buf_ones(dev, ly->down_exps_scale, m->n_expert);
        }
        if (!metal_buffer_ok(g->attn_norm[l]) ||
            !metal_buffer_ok(g->ffn_norm[l]) ||
            (ly->bq && !metal_buffer_ok(g->bq[l])) ||
            (ly->bk && !metal_buffer_ok(g->bk[l])) ||
            (ly->bv && !metal_buffer_ok(g->bv[l])) ||
            (ly->bo && !metal_buffer_ok(g->bo[l])) ||
            (ly->qnorm_w && !metal_buffer_ok(g->qn[l])) ||
            (ly->knorm_w && !metal_buffer_ok(g->kn[l])) ||
            (ly->attn_sinks && !metal_buffer_ok(g->sinks[l])) ||
            (ly->post_attn_norm_w && !metal_buffer_ok(g->pan[l])) ||
            (ly->post_ffn_norm_w && !metal_buffer_ok(g->pfn[l])) ||
            (ly->ffn_gate_inp_b && !metal_buffer_ok(g->gib[l])) ||
            (ly->ffn_gate_exps_b && !metal_buffer_ok(g->geb[l])) ||
            (ly->ffn_up_exps_b && !metal_buffer_ok(g->ueb[l])) ||
            (ly->ffn_down_exps_b && !metal_buffer_ok(g->deb[l])) ||
            (ly->moe_gemma &&
             (!metal_buffer_ok(g->gpn1[l]) || !metal_buffer_ok(g->gprn2[l]) ||
              !metal_buffer_ok(g->gpn2[l]) || !metal_buffer_ok(g->ggis[l]) ||
              !metal_buffer_ok(g->gdsc[l]))))
            return gpu_init_fail(m, g, lib, "per-layer buffer allocation");
    }

    // CPU batch prompt processing writes the same cache through these pointers,
    // but do not detach the malloc-owned cache until every Metal allocation has
    // succeeded. Any failure above falls back with the CPU KV still intact.
    free(m->kcache); free(m->vcache);
    m->kcache = (f16_t *)g->kc.contents;
    m->vcache = (f16_t *)g->vc.contents;
    m->kv_owner = KV_OWNER_GPU_BACKEND;
    m->gpu = g;
    m->gpu_owner = g;
    // K layers on the device. Full offload (K == n_layer) returns logits and
    // never reaches the CPU loop; a partial split leaves the boundary
    // activation in the host x buffer and the CPU finishes from layer K.
    // Setting this wrong in either direction is silent: too high and the
    // dispatcher asks for logits the device never computed, too low and every
    // layer is re-run on the CPU, discarding the GPU's work.
    m->gpu_layers = gpu_K;
    fprintf(stderr, "gpu: Metal backend on %s%s\n", dev.name.UTF8String,
            g->weights_copied ? " (weights copied)" : " (zero-copy weights)");
    if (gpu_K < m->n_layer)
        fprintf(stderr, "gpu: partial offload — %d of %d layers on the GPU "
                "(%.1f GB wrapped of %.1f GB; working set %.1f GB). The "
                "remaining %d run on the CPU.\n",
                gpu_K, m->n_layer, wlen / 1e9, m->gf.map_size / 1e9,
                ws_limit / 1e9, m->n_layer - gpu_K);
    return true;

}

// ---------------------------------------------------------------- encoding

// ------------------------------------------------------- dispatch census
// "Where does the batch dimension collapse?" is not answerable from
// throughput: a kernel that takes the batch in grid.y and a kernel encoded n
// times cost the same to WRITE and look identical from outside. This counts
// dispatches by kind, so a prefill census at two batch sizes shows directly
// which kinds scale with n (collapsed) and which do not (batched).
//
// Off unless RUNNER_METAL_STATS is set... except that the counters themselves
// are unconditional. They are non-atomic increments of a static struct on the
// encoding thread, which is the same thread for a whole command buffer, and
// the alternative — a branch per dispatch — costs more than the increment.
static struct {
    unsigned long tensor, mm, mv, mvf, rmsnorm, qknorm, headnorm, rope, store,
                  attn, attn_chunk, elem, moe, ple;
    // KV bytes an attention dispatch will read, accumulated for DECODE only
    // (n == 1), split by whether the layer slides. Decode-only because at
    // n > 1 each column attends over its own growing range and a single
    // `pos` does not describe the read; counting prefill here would produce
    // an underestimate that looks authoritative.
    unsigned long long kv_global, kv_swa;
    unsigned long      kv_layers_global, kv_layers_swa;
    // `ah` (the per-head attention scores) is device memory the decode kernel
    // round-trips ~4x per head over the attended span (write, max scan,
    // exp-sum, weighted read). Counted here so "is the scores buffer worth
    // moving to threadgroup memory" is a measurement, not the plan's
    // arithmetic — same decode-only framing as the KV bytes above.
    unsigned long long scores_bytes;
} g_disp;

static void enc_rmsnorm_n(gpu_t *g, id<MTLComputeCommandEncoder> e,
                          id<MTLBuffer> x, NSUInteger x_off,
                          id<MTLBuffer> y, NSUInteger y_off, id<MTLBuffer> w,
                          int n, float eps,
                          int n_col, int x_stride, int y_stride) {
    norm_args a = { n, x_stride, y_stride, eps };
    [e setComputePipelineState:g->p_rmsnorm];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBuffer:y offset:y_off atIndex:1];
    [e setBuffer:w offset:0 atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    g_disp.rmsnorm++;
    [e dispatchThreadgroups:MTLSizeMake(n_col, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Decode fusion F2: residual add + the following rmsnorm, one dispatch.
// Byte-identical to k_add then k_rmsnorm (same strided sum, same tree).
static void enc_add_rmsnorm(gpu_t *g, id<MTLComputeCommandEncoder> e,
                            id<MTLBuffer> x, id<MTLBuffer> d,
                            id<MTLBuffer> y, id<MTLBuffer> w,
                            int n, float eps) {
    add_norm_args a = { n, eps };
    [e setComputePipelineState:g->p_add_rmsnorm];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:d offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBuffer:w offset:0 atIndex:3];
    [e setBytes:&a length:sizeof(a) atIndex:4];
    g_disp.rmsnorm++;
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_rmsnorm(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> x, NSUInteger x_off,
                        id<MTLBuffer> y, NSUInteger y_off, id<MTLBuffer> w,
                        int n, float eps) {
    enc_rmsnorm_n(g, e, x, x_off, y, y_off, w, n, eps, 1, n, n);
}

// n_col columns in one dispatch (see the MV macros in kernels.metal): the
// weight row is walked once per column and stays in cache after the first, so
// a prompt batch stops re-streaming the whole weight matrix per token. Output
// is bit-identical to n_col separate enc_mv calls.
// Columns per threadgroup in the batched matmul. Tunable so the tradeoff can
// be re-measured on other Apple GPUs without a rebuild.
static int metal_col_tile(void) {
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RUNNER_METAL_COL_TILE");
        v = s && *s ? atoi(s) : 8;
        if (v < 1) v = 1;
    }
    return v;
}

static void enc_mv_cols(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                        id<MTLBuffer> y, NSUInteger y_off,
                        int n_in, int n_out, id<MTLBuffer> bias,
                        int n_col, int x_stride, int y_stride);

static void enc_mv_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                     gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                     id<MTLBuffer> y, NSUInteger y_off,
                     int n_in, int n_out, id<MTLBuffer> bias,
                     int n_col, int x_stride, int y_stride) {
    // Small batches take the multi-column IDENTITY matvec, not the GEMM
    // tiles: a 32-wide mm tile carrying 2 real columns is a full weight
    // sweep through mostly-padding (measured on the 70B speculative verify:
    // K=1 11.7 tok/s, K=2 through the mm path 2.2 tok/s). The identity
    // matvec serves the weight row from cache across the columns instead —
    // the same reasoning as CUDA's gemvb twins — and is bit-identical to
    // n_col solo dispatches into the bargain. 16 covers the speculative
    // window (spec_batch); real prefill batches go to the tiles as before.
    if (n_col > 1 && n_col <= 16) {
        enc_mv_cols(g, e, w, x, x_off, y, y_off, n_in, n_out, bias,
                    n_col, x_stride, y_stride);
        return;
    }
    if (n_col > 1 && g->p_tensor[w->type] && n_in % 256 == 0) {
        [e setComputePipelineState:g->p_tensor[w->type]];
        mm_args ma = { n_in, n_out, n_col,
                       metal_bind_weights(g, e,
                           (uint64_t)(uintptr_t)w->data,
                           w->nbytes),
                       bias != nil, x_stride, y_stride };
        [e setBuffer:x offset:x_off atIndex:1];
        [e setBuffer:y offset:y_off atIndex:2];
        [e setBytes:&ma length:sizeof(ma) atIndex:3];
        [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
        [e setThreadgroupMemoryLength:128 * 64 * sizeof(uint16_t)
                              atIndex:0];
        g_disp.tensor++;
        g_tc_dispatches++;
        [e dispatchThreadgroups:MTLSizeMake((n_out + 127) / 128,
                                            (n_col + 255) / 256, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    // Tiled GEMM when this is a real batch and a kernel exists for the weight
    // type. n_in must be a whole number of k-steps (every real model's is) and
    // K-quant kernels index a 256-superblock, so require that too.
    if (n_col > 1 && metal_mm_on() && g->p_mm[w->type] &&
        n_in % 32 == 0 &&
        !((w->type == T_Q4_K || w->type == T_Q6_K || w->type == T_Q2_K ||
           w->type == T_Q3_K || w->type == T_IQ4_XS) && n_in % 256 != 0)) {
        [e setComputePipelineState:g->p_mm[w->type]];
        mm_args ma = { n_in, n_out, n_col,
                       metal_bind_weights(g, e,
                           (uint64_t)(uintptr_t)w->data,
                           w->nbytes),
                       bias != nil, x_stride, y_stride };
        [e setBuffer:x offset:x_off atIndex:1];
        [e setBuffer:y offset:y_off atIndex:2];
        [e setBytes:&ma length:sizeof(ma) atIndex:3];
        [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
        // MM_TILE_M/MM_TILE_N mirror MM_TM/MM_TN in kernels.metal -- this file
        // cannot see that #define (the shader is compiled from an embedded
        // source string at runtime, not by clang alongside this file), so the
        // two are kept in step by hand, the same way mv_args/mm_args already
        // are (see the struct comment above).
        enum { MM_TILE_M = 64, MM_TILE_N = 32 };
        g_disp.mm++;
        g_tc_dispatches++;
        [e dispatchThreadgroups:MTLSizeMake((n_out + MM_TILE_M - 1) / MM_TILE_M,
                                            (n_col + MM_TILE_N - 1) / MM_TILE_N, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    enc_mv_cols(g, e, w, x, x_off, y, y_off, n_in, n_out, bias,
                n_col, x_stride, y_stride);
}

// The identity matvec tail of enc_mv_n, callable directly: the microbatch
// decode path (fwd_micro below) must stay BIT-IDENTICAL to sequential solo
// decode, so it may never take the mm/tensor branches above — each column
// here runs the exact k_mv dot the solo path runs, weights served from cache
// across the columns. This is the same twin discipline CUDA's enc_mv_batch
// enforces, arrived at from the other side.
static void enc_mv_cols(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                        id<MTLBuffer> y, NSUInteger y_off,
                        int n_in, int n_out, id<MTLBuffer> bias,
                        int n_col, int x_stride, int y_stride) {
    int col_tile = metal_col_tile();
    if (col_tile > n_col) col_tile = n_col;
    mv_args a = { n_in, n_out, 0,
                  bias != nil, n_col, x_stride, y_stride, col_tile };
    // Fast decode matvec: single column only. At n_col > 1 either the tiled
    // GEMM above already took the work, or this type has no mm kernel and the
    // identity matvec is what the batch falls back to — in both cases the
    // fast kernel would be answering a question nobody asked, and it would put
    // a non-identity result on the PREFILL path, where the tolerance gate
    // (decode-only by construction) has never looked.
    id<MTLComputePipelineState> mv = g->p_mv[w->type];
    if (n_col == 1 && metal_mv_fast_on() && g->p_mvf[w->type]) {
        mv = g->p_mvf[w->type];
        g_mv_dispatches++;
        g_disp.mvf++;
    } else {
        g_disp.mv++;
    }
    [e setComputePipelineState:mv];
    a.w_off = metal_bind_weights(g, e,
                  (uint64_t)(uintptr_t)w->data,
                  w->nbytes);
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
    // 128 threads = 4 simdgroups = 4 rows per threadgroup
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4,
                                        (n_col + col_tile - 1) / col_tile, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                   gguf_tensor *w, id<MTLBuffer> x, NSUInteger x_off,
                   id<MTLBuffer> y, NSUInteger y_off,
                   int n_in, int n_out, id<MTLBuffer> bias) {
    enc_mv_n(g, e, m, w, x, x_off, y, y_off, n_in, n_out, bias,
             1, n_in, n_out);
}

// n_col columns in one dispatch (grid.y), stride elements apart. Each
// (head, column) pair is an independent reduction, so this is bit-identical to
// n_col separate enc_qknorm calls -- see the kernel comment. It exists because
// the per-token encoding of this op was 50n of the 240n per-token dispatches
// measured in docs/metal-dispatch-census-2026-08-13.md.
static void enc_qknorm_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                         id<MTLBuffer> v, NSUInteger v_off, id<MTLBuffer> w,
                         int n_heads, int hd, int n_col, int stride) {
    float eps = m->rms_eps;
    [e setComputePipelineState:g->p_qknorm];
    [e setBuffer:v offset:v_off atIndex:0];
    [e setBuffer:w offset:0 atIndex:1];
    [e setBytes:&hd length:4 atIndex:2];
    [e setBytes:&eps length:4 atIndex:3];
    [e setBytes:&stride length:4 atIndex:4];
    g_disp.qknorm++;
    [e dispatchThreadgroups:MTLSizeMake(n_heads, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}


// Batched twin of enc_qknorm_n, same bit-identity argument. src and dst share
// one stride: the K and V staging buffers are both strided by kv_dim.
static void enc_headnorm_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                           id<MTLBuffer> src, NSUInteger src_off,
                           id<MTLBuffer> dst, NSUInteger dst_off,
                           id<MTLBuffer> w, int n_heads, int hd,
                           int n_col, int stride) {
    float eps = m->rms_eps;
    int has_weight = w != nil;
    [e setComputePipelineState:g->p_headnorm];
    [e setBuffer:src offset:src_off atIndex:0];
    [e setBuffer:dst offset:dst_off atIndex:1];
    [e setBuffer:w ? w : g->dummy offset:0 atIndex:2];
    [e setBytes:&hd length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e setBytes:&has_weight length:4 atIndex:5];
    [e setBytes:&stride length:4 atIndex:6];
    g_disp.headnorm++;
    [e dispatchThreadgroups:MTLSizeMake(n_heads, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}


static void enc_rope_n(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       id<MTLBuffer> v, NSUInteger v_off, int n_heads, int pos,
                       int layer, int n_col, int stride) {
    bool local = model_is_swa(m, layer);
    int hd = model_head_dim(m, layer);
    int rope_dim = model_rope_dim(m, layer);
    rope_args a = { hd, n_heads, rope_dim / 2, pos,
                    m->rope_neox, model_rope_mscale(m, layer), stride };
    [e setComputePipelineState:g->p_rope];
    [e setBuffer:v offset:v_off atIndex:0];
    [e setBuffer:(local && g->inv_freq_local) ? g->inv_freq_local : g->inv_freq
          offset:0 atIndex:1];
    [e setBytes:&a length:sizeof(a) atIndex:2];
    g_disp.rope++;
    [e dispatchThreads:MTLSizeMake(a.half_dim, n_heads, n_col)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}


// n_col columns in one dispatch (grid.y), a_stride/b_stride elements apart.
// Bit-identical to n_col separate calls -- every element was already
// independent, so only the encoding changes. See the kernel-family comment in
// kernels.metal for why this exists.
static void enc_elem_n(gpu_t *g, id<MTLComputeCommandEncoder> e,
                       id<MTLComputePipelineState> p,
                       id<MTLBuffer> a, NSUInteger a_off,
                       id<MTLBuffer> b, NSUInteger b_off, int n,
                       int n_col, int a_stride, int b_stride) {
    [e setComputePipelineState:p];
    [e setBuffer:a offset:a_off atIndex:0];
    [e setBuffer:b offset:b_off atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e setBytes:&a_stride length:4 atIndex:3];
    [e setBytes:&b_stride length:4 atIndex:4];
    g_disp.elem++;
    [e dispatchThreads:MTLSizeMake(n, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Single column. Still used by the MoE FFN, which stays per-token because
// routing picks different experts per token.
static void enc_elem(gpu_t *g, id<MTLComputeCommandEncoder> e,
                     id<MTLComputePipelineState> p,
                     id<MTLBuffer> a, NSUInteger a_off,
                     id<MTLBuffer> b, NSUInteger b_off, int n) {
    enc_elem_n(g, e, p, a, a_off, b, b_off, n, 1, 0, 0);
}

static void enc_scale_n(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> x, NSUInteger x_off,
                        int n, float scale, int n_col, int stride) {
    [e setComputePipelineState:g->p_scale];
    [e setBuffer:x offset:x_off atIndex:0];
    [e setBytes:&scale length:4 atIndex:1];
    [e setBytes:&n length:4 atIndex:2];
    [e setBytes:&stride length:4 atIndex:3];
    g_disp.elem++;
    [e dispatchThreads:MTLSizeMake(n, n_col, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_scale(gpu_t *g, id<MTLComputeCommandEncoder> e,
                      id<MTLBuffer> x, NSUInteger x_off,
                      int n, float scale) {
    enc_scale_n(g, e, x, x_off, n, scale, 1, 0);
}

// Head transforms (logit softcap / suppressed tokens) deliberately do NOT
// live on the backend: model_forward_batch applies them on the host for both
// the batched and the solo path, so the two cannot drift apart. A Metal-side
// copy existed here and double-applied the softcap.
static void enc_moe_route(gpu_t *g, id<MTLComputeCommandEncoder> e,
                          int ne, int used, int tokens,
                          NSUInteger logits_off, NSUInteger sel_off) {
    int ls = ne;
    [e setComputePipelineState:g->p_moe_route];
    [e setBuffer:g->moe_logits offset:logits_off atIndex:0];
    [e setBuffer:g->moe_sel offset:sel_off atIndex:1];
    [e setBuffer:g->moe_selw offset:sel_off atIndex:2];
    [e setBytes:&ne length:4 atIndex:3];
    [e setBytes:&used length:4 atIndex:4];
    [e setBytes:&tokens length:4 atIndex:5];
    [e setBytes:&ls length:4 atIndex:6];
    g_disp.moe++;
    [e dispatchThreads:MTLSizeMake(tokens, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void enc_moe_trace_logits(gpu_t *g, id<MTLComputeCommandEncoder> e,
                                 int layer, int tokens, int ne) {
    if (!g->moe_trace_logits) return;
    int count = tokens * ne;
    [e setComputePipelineState:g->p_trace_copy];
    [e setBuffer:g->moe_logits offset:0 atIndex:0];
    [e setBuffer:g->moe_trace_logits
          offset:(NSUInteger)((size_t)layer * (size_t)count * sizeof(float))
          atIndex:1];
    [e setBytes:&count length:sizeof(count) atIndex:2];
    g_disp.moe++;
    [e dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_moe_mv(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       gguf_tensor *base, uint64_t estride,
                       id<MTLBuffer> x, NSUInteger x_off,
                       id<MTLBuffer> y, NSUInteger y_off,
                       int n_in, int n_out, int nslots, int xs, int ys,
                       id<MTLBuffer> bias, int bias_stride,
                       NSUInteger sel_off, int slots_per_token) {
    // Expert-major pays only when experts are shared across slots: at
    // nslots >= n_expert every expert averages a full token list, and the
    // one-read-per-expert weight traffic beats slot-major's re-reads. Below
    // that (decode: nslots == n_expert_used) slot-major stays.
    id<MTLComputePipelineState> emp = nil;
    if (metal_moe_em_on() && m->n_expert > 0 && nslots >= m->n_expert) {
        emp = g->p_moe_mv_em8[base->type];       // column-tiled where it exists
        if (!emp) emp = g->p_moe_mv_em[base->type];
    }
    bool em = emp != nil;
    if (em) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            fprintf(stderr, "gpu: MoE expert-major prefill kernels on "
                    "(RUNNER_METAL_MOE_EM)\n");
        }
    }
    [e setComputePipelineState:em ? emp : g->p_moe_mv[base->type]];
    moe_args a = { n_in, n_out,
                   metal_bind_weights(g, e,
                       (uint64_t)(uintptr_t)base->data,
                       base->nbytes),
                   estride, xs, ys, bias != nil, bias_stride,
                   slots_per_token, nslots };
    [e setBuffer:x offset:x_off atIndex:1];
    [e setBuffer:y offset:y_off atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:g->moe_sel offset:sel_off atIndex:4];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:5];
    g_disp.moe++;
    [e dispatchThreadgroups:MTLSizeMake((n_out + 3) / 4,
                                        em ? m->n_expert : nslots, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// Group this layer's slots by expert: colmap gets the slot ids ordered by
// expert, eoff the per-expert [begin, end) offsets. One threadgroup; the
// buffers are hazard-tracked, so the mm dispatches below wait on it and the
// next layer's grouping waits on them.
static void enc_moe_group(gpu_t *g, id<MTLComputeCommandEncoder> e,
                          int ne, int nslots, NSUInteger sel_off) {
    [e setComputePipelineState:g->p_moe_group];
    [e setBuffer:g->moe_sel offset:sel_off atIndex:0];
    [e setBuffer:g->moe_colmap offset:0 atIndex:1];
    [e setBuffer:g->moe_eoff offset:0 atIndex:2];
    [e setBytes:&ne length:4 atIndex:3];
    [e setBytes:&nslots length:4 atIndex:4];
    g_disp.moe++;
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Grouped simdgroup-MMA expert matmul: the dense k_mm tile shape over one
// expert's gathered token columns, grid z spanning experts. Column tiles
// beyond an expert's count exit on two int reads. Requires enc_moe_group to
// have run for this layer's sel window.
static void enc_moe_mm(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                       id<MTLComputePipelineState> pso,
                       gguf_tensor *base, uint64_t estride,
                       id<MTLBuffer> x, id<MTLBuffer> y,
                       int n_in, int n_out, int nslots, int xs, int ys,
                       id<MTLBuffer> bias, int bias_stride,
                       int slots_per_token) {
    enum { MM_TILE_M = 64, MM_TILE_N = 32 };   // mirrors MM_TM/MM_TN
    [e setComputePipelineState:pso];
    struct { int n_in, n_out; uint64_t w_off, estride;
             int xs, ys, has_bias, bias_stride, slots_per_token; } a = {
        n_in, n_out,
        metal_bind_weights(g, e, (uint64_t)(uintptr_t)base->data, base->nbytes),
        estride, xs, ys, bias != nil, bias_stride, slots_per_token };
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&a length:sizeof(a) atIndex:3];
    [e setBuffer:bias ? bias : g->dummy offset:0 atIndex:4];
    [e setBuffer:g->moe_colmap offset:0 atIndex:5];
    [e setBuffer:g->moe_eoff offset:0 atIndex:6];
    g_disp.moe++;
    [e dispatchThreadgroups:MTLSizeMake((n_out + MM_TILE_M - 1) / MM_TILE_M,
                                        (nslots + MM_TILE_N - 1) / MM_TILE_N,
                                        m->n_expert)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void enc_moe_actmul(gpu_t *g, id<MTLComputeCommandEncoder> e,
                           id<MTLBuffer> gbuf, NSUInteger goff,
                           id<MTLBuffer> ubuf, NSUInteger uoff,
                           int nff, int nslots, int gss, int uss, int act) {
    int args[4] = { nff, gss, uss, act };
    [e setComputePipelineState:g->p_moe_actmul];
    [e setBuffer:gbuf offset:goff atIndex:0];
    [e setBuffer:ubuf offset:uoff atIndex:1];
    [e setBytes:args length:sizeof(args) atIndex:2];
    g_disp.moe++;
    [e dispatchThreads:MTLSizeMake(nff, nslots, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_moe_sum(gpu_t *g, id<MTLComputeCommandEncoder> e,
                        id<MTLBuffer> out, NSUInteger out_off,
                        int n, int nslots, int es, id<MTLBuffer> dscale,
                        NSUInteger sel_off, int tokens, int out_stride,
                        int fuse_add) {
    int has_dscale = dscale != nil;
    [e setComputePipelineState:g->p_moe_sum];
    [e setBuffer:out offset:out_off atIndex:0];
    [e setBuffer:g->moe_eout offset:0 atIndex:1];
    [e setBuffer:g->moe_selw offset:sel_off atIndex:2];
    [e setBuffer:dscale ? dscale : g->dummy offset:0 atIndex:3];
    [e setBuffer:g->moe_sel offset:sel_off atIndex:4];
    [e setBytes:&n length:4 atIndex:5];
    [e setBytes:&nslots length:4 atIndex:6];
    [e setBytes:&es length:4 atIndex:7];
    [e setBytes:&has_dscale length:4 atIndex:8];
    [e setBytes:&tokens length:4 atIndex:9];
    [e setBytes:&out_stride length:4 atIndex:10];
    [e setBytes:&fuse_add length:4 atIndex:11];
    g_disp.moe++;
    [e dispatchThreads:MTLSizeMake(n, tokens, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// Metal's KV cache *is* the backend buffer (unified memory: m->kcache points at
// g->kc.contents), and the CPU path is about to read and overwrite those rows.
// Releasing the buffers here would pull them out from under the fallback. Keep
// gpu_owner reachable for model_free(), but make m->gpu NULL so no future
// forward tries to submit Metal work.
void gpu_disable(model_t *m) {
    if (!m) return;
    m->gpu = NULL;
    m->gpu_layers = 0;
}

// Metal has no batched-decode kernels yet, so it declines the microbatch and
// model_batch_decode decodes sequentially. The port is the same shape as the
// CUDA one (per-column position and per-sequence KV buffer, batched twins of
// the batch-1 matvec kernels) and is tracked in FUTURE.md Phase 6; unified
// memory removes the KV upload/copyback half of it entirely.
bool gpu_mvt(model_t *mm, const gguf_tensor *w, const float *dy, float *dx,
             int n_in, int n_out, int batch) {
    (void)mm; (void)w; (void)dy; (void)dx; (void)n_in; (void)n_out;
    (void)batch;
    return false;   // no Metal training path yet
}

bool gpu_train_init(model_t *mm) { (void)mm; return false; }
void gpu_train_free(model_t *mm) { (void)mm; }
bool gpu_train_mvt(model_t *mm, const gguf_tensor *w, const float *dy,
                   float *dx, int n_in, int n_out, int nb) {
    (void)mm; (void)w; (void)dy; (void)dx; (void)n_in; (void)n_out; (void)nb;
    return false;
}

void gpu_free(model_t *m) {
    if (!m) return;
    gpu_t *g = m->gpu_owner ? (gpu_t *)m->gpu_owner : (gpu_t *)m->gpu;
    m->gpu = NULL;
    m->gpu_owner = NULL;
    if (!g) return;
    // The KV cache pointers alias MTLBuffer.contents. Detach before releasing
    // the buffers so model_free() never calls free() on borrowed memory.
    if (m->kv_owner == KV_OWNER_GPU_BACKEND) {
        m->kcache = NULL;
        m->vcache = NULL;
        m->kv_owner = KV_OWNER_MALLOC;
    }
    gpu_release_state(g, m->n_layer);
}

// The native prompt-batch encoder implements exactly the plain llama shape.
// Every feature beyond it — MoE, GELU dense FFN, embedding scale, weightless
// V norm / V-less layers, sandwich norms, per-layer output scale, logit
// softcap or suppressed tokens, heterogeneous per-layer geometry — is
// implemented only by the per-token path, so a model carrying any of them
// must take that path or the batch would be silently wrong (the gemma3/4
// families hit several of these at once).

// Partial split hand-off: the CPU loop resumes at layer gpu_layers and reads
// the boundary hidden state from the host x buffer. On Metal this is a copy
// within one address space -- g->x is shared storage, so it IS host memory --
// and it is small (n * n_embd floats; 11 KB at decode). The KV needs nothing:
// m->kcache/m->vcache already point at g->kc/g->vc contents, so the layers
// the device just wrote are already visible to the CPU.
static void metal_handoff_boundary(model_t *m, int n) {
    gpu_t *g = (gpu_t *)m->gpu;
    if (!g || !m->x) return;
    memcpy(m->x, g->x.contents, sizeof(float) * (size_t)n * m->n_embd);
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    // One encoder for every n, decode included. There used to be a second,
    // per-token encoder here; keeping two implementations of the same layer
    // loop cost three defects (features silently missing behind an
    // eligibility check, a double-applied logit softcap, and wrong output on
    // real gemma-4 E2B weights) before it was removed. A scratch allocation
    // failure now falls back to the CPU loudly instead of to a second path.
    if (!metal_ensure_batch((gpu_t *)m->gpu, m, n)) {
        fprintf(stderr, "gpu: Metal batch scratch allocation failed — "
                "releasing the backend, continuing on CPU\n");
        return false;
    }
    bool partial = m->gpu_layers > 0 && m->gpu_layers < m->n_layer;
    float *lg = gpu_forward_native_batch(m, tokens, n, pos);
    if (!lg) return false;
    if (partial) {
        metal_handoff_boundary(m, n);
        if (logits) *logits = NULL;
        return true;
    }
    if (logits) *logits = want_logits ? lg : NULL;
    return true;
}


static NSUInteger foff(size_t elems) {
    return (NSUInteger)(elems * sizeof(float));
}

static void enc_moe_experts_batch(gpu_t *g, id<MTLComputeCommandEncoder> e,
                                  model_t *m, layer_t *ly,
                                  int n, int xdim, NSUInteger sel_off,
                                  bool sum_add) {
    int n_embd = m->n_embd;
    int used = m->n_expert_used, slots = n * used;
    int nff = m->n_ff_exp;
    uint64_t gstride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_gate_exps->type, n_embd);
    uint64_t ustride = (uint64_t)nff *
                       ggml_row_size(ly->ffn_up_exps->type, n_embd);
    uint64_t dstride = (uint64_t)n_embd *
                       ggml_row_size(ly->ffn_down_exps->type, nff);

    int l = (int)(ly - m->layers);
    // Grouped-MMA route (RUNNER_METAL_MOE_MM): sort slots by expert once,
    // then any expert tensor whose type has a k_moe_mm kernel runs as a real
    // GEMM over its token group; the rest stay on the slot-major matvec.
    int mm_mode = n > 1 ? metal_moe_mm_on() : 0;
    id<MTLComputePipelineState> *mmp =
        mm_mode == 2 ? g->p_moe_mmh : g->p_moe_mm;
    bool mm = mm_mode &&
              (mmp[ly->ffn_gate_exps->type] ||
               mmp[ly->ffn_up_exps->type] ||
               mmp[ly->ffn_down_exps->type]);
    if (mm) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            fprintf(stderr, "gpu: MoE grouped-MMA prefill kernels on "
                    "(%s-staged, RUNNER_METAL_MOE_MM)\n",
                    mm_mode == 2 ? "half" : "f32");
        }
        enc_moe_group(g, e, m->n_expert, slots, sel_off);
    }
    // Decode fusion F3: gate-mv + up-mv + actmul in ONE dispatch per layer
    // (budget: -2 x n_layer per token). Same dot bodies, same bias points,
    // same activation code — byte identity is the gate. Needs gate and up
    // to share a quant type and to resolve inside one weight wrap.
    bool gua = n == 1 && metal_fuse_on() && !mm &&
               ly->ffn_gate_exps->type == ly->ffn_up_exps->type &&
               g->p_moe_gua[ly->ffn_gate_exps->type] &&
               (g->geb[l] != nil) == (g->ueb[l] != nil);
    if (gua) {
        uint64_t gw = 0, uw = 0;
        id<MTLBuffer> gb = metal_wbuf_for(g,
            (uint64_t)(uintptr_t)ly->ffn_gate_exps->data,
            ly->ffn_gate_exps->nbytes, &gw);
        id<MTLBuffer> ub = metal_wbuf_for(g,
            (uint64_t)(uintptr_t)ly->ffn_up_exps->data,
            ly->ffn_up_exps->nbytes, &uw);
        if (gb && gb == ub) {
            metal_fuse_announce();
            moe_gua_args a = { n_embd, nff, gw, uw, gstride, ustride,
                               xdim, g->geb[l] != nil, m->ffn_act, used, nff };
            [e setComputePipelineState:g->p_moe_gua[ly->ffn_gate_exps->type]];
            [e setBuffer:gb offset:0 atIndex:0];
            [e setBuffer:g->xb offset:0 atIndex:1];
            [e setBuffer:g->moe_hb offset:0 atIndex:2];
            [e setBytes:&a length:sizeof(a) atIndex:3];
            [e setBuffer:g->moe_sel offset:sel_off atIndex:4];
            [e setBuffer:g->geb[l] ? g->geb[l] : g->dummy offset:0 atIndex:5];
            [e setBuffer:g->ueb[l] ? g->ueb[l] : g->dummy offset:0 atIndex:6];
            g_disp.moe++;
            [e dispatchThreadgroups:MTLSizeMake((nff + 3) / 4, slots, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            goto gua_done;
        }
    }
    if (mm && mmp[ly->ffn_gate_exps->type])
        enc_moe_mm(g, e, m, mmp[ly->ffn_gate_exps->type],
                   ly->ffn_gate_exps, gstride, g->xb, g->moe_hb,
                   n_embd, nff, slots, xdim, nff, g->geb[l], nff, used);
    else
        enc_moe_mv(g, e, m, ly->ffn_gate_exps, gstride, g->xb, 0,
                   g->moe_hb, 0, n_embd, nff, slots, xdim, nff,
                   g->geb[l], nff, sel_off, used);
    if (mm && mmp[ly->ffn_up_exps->type])
        enc_moe_mm(g, e, m, mmp[ly->ffn_up_exps->type],
                   ly->ffn_up_exps, ustride, g->xb, g->moe_hb2,
                   n_embd, nff, slots, xdim, nff, g->ueb[l], nff, used);
    else
        enc_moe_mv(g, e, m, ly->ffn_up_exps, ustride, g->xb, 0,
                   g->moe_hb2, 0, n_embd, nff, slots, xdim, nff,
                   g->ueb[l], nff, sel_off, used);
    enc_moe_actmul(g, e, g->moe_hb, 0, g->moe_hb2, 0,
                   nff, slots, nff, nff, m->ffn_act);
    gua_done: ;
    // A down-projection + weighted-sum fold (F6) was built and measured
    // 2026-09-01: byte-identical (after pinning its accumulate to fma —
    // see the per-inlining-site contraction note on the score gates) and a
    // CONSISTENT -0.5..-0.9%% on both MoE flagships: collapsing the down
    // matvec's (n_embd x slots) grid to n_embd rows narrows the heaviest
    // weight sweep of the layer 4x, and that costs more than the one
    // dispatch it saves. The mild corollary of the F4 rule. Removed.
    if (mm && mmp[ly->ffn_down_exps->type])
        enc_moe_mm(g, e, m, mmp[ly->ffn_down_exps->type],
                   ly->ffn_down_exps, dstride, g->moe_hb,
                   g->moe_eout, nff, n_embd, slots, nff, n_embd,
                   g->deb[l], n_embd, 0);
    else
        enc_moe_mv(g, e, m, ly->ffn_down_exps, dstride, g->moe_hb, 0,
                   g->moe_eout, 0, nff, n_embd, slots, nff, n_embd,
                   g->deb[l], n_embd, sel_off, 0);
    // Decode fusion F5: fold the post-FFN residual add into the sum —
    // write the residual stream directly (x[i] + s, k_add's expression).
    if (sum_add) {
        metal_fuse_announce();
        enc_moe_sum(g, e, g->x, 0, n_embd, used, n_embd, nil,
                    sel_off, n, n_embd, 1);
    } else
        enc_moe_sum(g, e, g->xb, 0, n_embd, used, n_embd, nil,
                    sel_off, n, xdim, 0);
}

static int  metal_nan_stage(void);
static bool metal_scan_bad(const float *p, int n, const char *what, int layer);

static void enc_gemma_moe_experts_batch(gpu_t *g,
                                        id<MTLComputeCommandEncoder> e,
                                        model_t *m, layer_t *ly, int l,
                                        int n, int xdim,
                                        NSUInteger sel_off,
                                        id<MTLCommandBuffer> *cbp,
                                        id<MTLComputeCommandEncoder> *ep) {
#define XP(BUF, CNT, WHAT) do { if (cbp && metal_nan_stage() >= 4) { \
        [e endEncoding]; [*cbp commit]; [*cbp waitUntilCompleted]; \
        metal_scan_bad((const float *)(BUF).contents, (int)(CNT), (WHAT), l); \
        *cbp = [g->queue commandBuffer]; e = [*cbp computeCommandEncoder]; *ep = e; \
    } } while (0)
    int n_embd = m->n_embd;
    int used = m->n_expert_used, slots = n * used;
    int nff = m->n_ff_exp;
    uint64_t gustride = (uint64_t)(2 * (size_t)nff) *
                        ggml_row_size(ly->ffn_gate_up_exps->type, n_embd);
    uint64_t dstride = (uint64_t)n_embd *
                       ggml_row_size(ly->ffn_down_exps->type, nff);

    enc_moe_mv(g, e, m, ly->ffn_gate_up_exps, gustride, g->xb2, 0,
               g->moe_hb, 0, n_embd, 2 * nff, slots, xdim, 2 * nff,
               nil, 0, sel_off, used);
    XP(g->xb2, (size_t)n * xdim, "exp:in-xb2");
    XP(g->moe_hb, (size_t)slots * 2 * nff, "exp:gate_up");
    enc_moe_actmul(g, e, g->moe_hb, 0, g->moe_hb, foff(nff),
                   nff, slots, 2 * nff, 2 * nff, ACT_GELU);
    XP(g->moe_hb, (size_t)slots * 2 * nff, "exp:actmul");
    enc_moe_mv(g, e, m, ly->ffn_down_exps, dstride, g->moe_hb, 0,
               g->moe_eout, 0, nff, n_embd, slots, 2 * nff, n_embd,
               nil, 0, sel_off, 0);
    XP(g->moe_eout, (size_t)slots * n_embd, "exp:down");
    enc_moe_sum(g, e, g->q, 0, n_embd, used, n_embd, g->gdsc[l],
                sel_off, n, xdim, 0);
#undef XP
}

static void enc_moe_ffn_batch(gpu_t *g, id<MTLComputeCommandEncoder> e,
                              model_t *m, layer_t *ly, int l,
                              int n, int xdim, bool norm_done,
                              bool sum_add) {
    int ne = m->n_expert, used = m->n_expert_used;
    if (!norm_done)
        enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->ffn_norm[l],
                      m->n_embd, m->rms_eps, n, m->n_embd, xdim);
    NSUInteger sel_off = foff((size_t)l * n * used);
    // A fused router matvec + route (single threadgroup) was built and
    // measured 2026-09-01: byte-identical, and 30B decode COLLAPSED 77 -> 40
    // tok/s — one threadgroup means one GPU core streaming the ~1 MB router
    // matrix that the split mv grid spreads across the chip. Same lesson as
    // the dense-model front admission, at matvec scale: never trade a
    // parallel weight sweep for a dispatch. Removed; the split pair stays.
    for (int b = 0; b < n; b++)
        enc_mv(g, e, m, ly->ffn_gate_inp, g->xb,
               foff((size_t)b * xdim), g->moe_logits,
               foff((size_t)b * ne), m->n_embd, ne, g->gib[l]);
    enc_moe_trace_logits(g, e, l, n, ne);
    if (metal_moe_batch_on()) {
        enc_moe_route(g, e, ne, used, n, 0, sel_off);
    } else {
        for (int b = 0; b < n; b++)
            enc_moe_route(g, e, ne, used, 1,
                          foff((size_t)b * ne),
                          sel_off + foff((size_t)b * used));
    }
    enc_moe_experts_batch(g, e, m, ly, n, xdim, sel_off, sum_add);
}

static void enc_gemma_moe_ffn_batch(gpu_t *g, id<MTLComputeCommandEncoder> e,
                                    model_t *m, layer_t *ly, int l,
                                    int n, int xdim,
                                    id<MTLCommandBuffer> *cbp,
                                    id<MTLComputeCommandEncoder> *ep) {
#define GP(BUF, CNT, WHAT) do { if (cbp && metal_nan_stage() >= 3) { \
        [e endEncoding]; [*cbp commit]; [*cbp waitUntilCompleted]; \
        metal_scan_bad((const float *)(BUF).contents, (int)(CNT), (WHAT), l); \
        *cbp = [g->queue commandBuffer]; e = [*cbp computeCommandEncoder]; *ep = e; \
    } } while (0)
    int ne = m->n_expert, used = m->n_expert_used;
    int n_embd = m->n_embd, dff = m->n_ff;

    // Every token retains the old reduction and matvec arithmetic. Only the
    // independent normalization columns share dispatches in this first slice.
    enc_rmsnorm_n(g, e, g->x, 0, g->xb2, 0, g->ffn_norm[l],
                  n_embd, m->rms_eps, n, n_embd, xdim);
    for (int b = 0; b < n; b++) {
        NSUInteger xbo = foff((size_t)b * xdim);
        NSUInteger hbo = foff((size_t)b * dff);
        enc_mv(g, e, m, ly->w_gate, g->xb2, xbo, g->hb, hbo,
               n_embd, dff, nil);
        enc_mv(g, e, m, ly->w_up, g->xb2, xbo, g->hb2, hbo,
               n_embd, dff, nil);
        enc_elem(g, e, g->p_gelu, g->hb, hbo, g->hb2, hbo, dff);
        enc_mv(g, e, m, ly->w_down, g->hb, hbo, g->xb, xbo,
               dff, n_embd, nil);
    }
    GP(g->xb, (size_t)n * xdim, "moe:dense-shared");
    enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->gpn1[l],
                  n_embd, m->rms_eps, n, xdim, xdim);
    GP(g->xb, (size_t)n * xdim, "moe:dense-gpn1");

    enc_rmsnorm_n(g, e, g->x, 0, g->xb2, 0, g->gprn2[l],
                  n_embd, m->rms_eps, n, n_embd, xdim);
    enc_rmsnorm_n(g, e, g->x, 0, g->q, 0, g->ggis[l],
                  n_embd, m->rms_eps, n, n_embd, xdim);
    for (int b = 0; b < n; b++)
        enc_mv(g, e, m, ly->ffn_gate_inp, g->q,
               foff((size_t)b * xdim), g->moe_logits,
               foff((size_t)b * ne), n_embd, ne, nil);
    enc_moe_trace_logits(g, e, l, n, ne);
    NSUInteger sel_off = foff((size_t)l * n * used);
    if (metal_moe_batch_on()) {
        enc_moe_route(g, e, ne, used, n, 0, sel_off);
    } else {
        for (int b = 0; b < n; b++)
            enc_moe_route(g, e, ne, used, 1,
                          foff((size_t)b * ne),
                          sel_off + foff((size_t)b * used));
    }
    GP(g->moe_logits, (size_t)n * ne, "moe:router-logits");
    enc_gemma_moe_experts_batch(g, e, m, ly, l, n, xdim, sel_off, cbp, ep);
    if (cbp && metal_nan_stage() >= 4) e = *ep;
    GP(g->q, (size_t)n * xdim, "moe:experts-out");
    enc_rmsnorm_n(g, e, g->q, 0, g->q, 0, g->gpn2[l],
                  n_embd, m->rms_eps, n, xdim, xdim);
    enc_elem_n(g, e, g->p_add, g->xb, 0, g->q, 0,
               n_embd, n, xdim, xdim);
    // GP may end the encoder supplied by the caller and replace it. Hand the
    // live encoder back before the caller emits the post-FFN probes/norms;
    // otherwise trace levels 3 and 4 continue through an already-ended
    // encoder even though ordinary inference (trace level 0) is unaffected.
    if (ep && metal_nan_stage() >= 3) *ep = e;
#undef GP
}

// gemma-4 E-series per-layer embeddings, mirroring the CPU tail in model.c:
// gate(x) -> GELU-gated against this layer's slice of the per-layer embedding
// table -> project back to n_embd -> own RMS norm -> into the residual. Runs
// on the post-FFN residual and BEFORE the layer output scale.
//
// m->ple is filled on the host by model_ple_prepass before the backend is
// called, so the slice is copied into a shared buffer once per forward rather
// than recomputed here.
static void enc_ple(gpu_t *g, id<MTLComputeCommandEncoder> e, model_t *m,
                    layer_t *ly, int l, int n, int xdim) {
    int P = m->n_embd_ple, n_embd = m->n_embd;
    enc_mv_n(g, e, m, ly->ple_gate, g->x, 0, g->ple_tmp, 0,
             n_embd, P, nil, n, n_embd, P);
    // gate *= gelu-gated slice: p_gelu computes g[i] = gelu(g[i]) * u[i],
    // which is exactly gated_act(ACT_GELU, gate, slice)
    // ple is [token][layer][P], so this layer's slice starts at l*P and the
    // per-column stride is the whole layer row.
    enc_elem_n(g, e, g->p_gelu, g->ple_tmp, 0,
               g->ple, foff((size_t)l * P), P, n, P, m->n_layer * P);
    enc_mv_n(g, e, m, ly->ple_proj, g->ple_tmp, 0, g->xb, 0,
             P, n_embd, nil, n, P, xdim);
    enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->ppn[l],
                  n_embd, m->rms_eps, n, xdim, xdim);
    enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);
}

// RUNNER_METAL_NAN_TRACE=1: submit after every layer and report the first one
// whose residual carries a NaN/Inf. The GPU path has no equivalent of
// RUNNER_DEBUG_ACT — without this, a backend that silently produces NaN logits
// can only be bisected by guesswork. Costs one command buffer per layer, so it
// is opt-in and read once.
static bool metal_nan_trace(void) {
    static int on = -1;
    if (on < 0) { const char *v = getenv("RUNNER_METAL_NAN_TRACE"); on = v && *v && strcmp(v, "0"); }
    return on > 0;
}

// RUNNER_METAL_NAN_TRACE=2 additionally probes INSIDE the layer. The
// per-layer form names the layer that first carries a NaN; it cannot say which
// stage of that layer produced it, which is where a wrong op actually hides.
// Costs one command buffer per probe, so it is strictly opt-in.
static int metal_nan_stage(void) {
    static int lvl = -1;
    if (lvl < 0) { const char *v = getenv("RUNNER_METAL_NAN_TRACE"); lvl = v && *v ? atoi(v) : 0; }
    return lvl;
}

static bool metal_scan_bad(const float *p, int n, const char *what, int layer) {
    for (int i = 0; i < n; i++) {
        uint32_t u; memcpy(&u, &p[i], 4);
        if ((u & 0x7f800000u) == 0x7f800000u) {   // NaN or Inf, -ffast-math safe
            fprintf(stderr, "metal-nan: L%d %s[%d] = %s\n", layer, what, i,
                    (u & 0x7fffffu) ? "NaN" : "Inf");
            return true;
        }
    }
    return false;
}

static void metal_moe_route_trace(gpu_t *g, model_t *m, int n) {
    if (!metal_env_on("RUNNER_METAL_MOE_ROUTE_TRACE") || m->n_expert <= 0)
        return;
    const int *sel = (const int *)g->moe_sel.contents;
    const float *selw = (const float *)g->moe_selw.contents;
    int used = m->n_expert_used;
    for (int l = 0; l < m->gpu_layers; l++) {
        if (!m->layers[l].is_moe && !m->layers[l].moe_gemma) continue;
        size_t layer = (size_t)l * (size_t)n * (size_t)used;
        for (int t = 0; t < n; t++) {
            size_t row = layer + (size_t)t * (size_t)used;
            fprintf(stderr, "metal-moe-route l=%d t=%d", l, t);
            for (int s = 0; s < used; s++) {
                uint32_t bits;
                memcpy(&bits, &selw[row + (size_t)s], sizeof(bits));
                fprintf(stderr, " %d:%08x", sel[row + (size_t)s], bits);
            }
            fputc('\n', stderr);
        }
    }
}

static FILE *metal_moe_trace_file(void) {
    static FILE *fp;
    static int opened;
    if (!opened) {
        opened = 1;
        const char *path = getenv("RUNNER_MOE_TRACE");
        if (path && *path) {
            fp = fopen(path, "a");
            if (!fp) fprintf(stderr, "warning: RUNNER_MOE_TRACE=%s: could not open for append\n", path);
        }
    }
    return fp;
}

static void metal_moe_full_trace(gpu_t *g, model_t *m, int n, int pos) {
    FILE *fp = metal_moe_trace_file();
    if (!fp || !g->moe_trace_logits || m->n_expert <= 0) return;
    const float *logits = (const float *)g->moe_trace_logits.contents;
    const int *sel = (const int *)g->moe_sel.contents;
    const float *selw = (const float *)g->moe_selw.contents;
    int ne = m->n_expert, used = m->n_expert_used;
    for (int l = 0; l < m->gpu_layers; l++) {
        if (!m->layers[l].is_moe && !m->layers[l].moe_gemma) continue;
        for (int t = 0; t < n; t++) {
            size_t lr = ((size_t)l * n + t) * ne;
            size_t sr = ((size_t)l * n + t) * used;
            fprintf(fp, "{\"pos\":%d,\"layer\":%d,\"experts\":[", pos + t, l);
            for (int s = 0; s < used; s++) fprintf(fp, "%s%d", s ? "," : "", sel[sr + s]);
            fprintf(fp, "],\"gates\":[");
            for (int s = 0; s < used; s++) fprintf(fp, "%s%.9g", s ? "," : "", selw[sr + s]);
            fprintf(fp, "],\"norms\":[],\"logits\":[");
            for (int x = 0; x < ne; x++) fprintf(fp, "%s%.9g", x ? "," : "", logits[lr + x]);
            fputs("]}\n", fp);
        }
    }
    fflush(fp);
}

static float *gpu_forward_native_batch(model_t *m, const int32_t *tokens,
                                       int n, int pos) {
    gpu_t *g = m->gpu;
    int n_embd = m->n_embd;
    int q_dim  = m->n_head * m->head_dim;
    int kv_dim = m->n_head_kv * m->head_dim;
    int xdim   = q_dim > n_embd ? q_dim : n_embd;

    size_t ers = ggml_row_size(m->tok_embd->type, n_embd);
    for (int b = 0; b < n; b++) {
        float *xp = (float *)g->x.contents + (size_t)b * n_embd;
        dequant_row(m->tok_embd->type,
                    (uint8_t *)m->tok_embd->data + (size_t)tokens[b] * ers,
                    xp, n_embd);
        model_embd_transform(m, xp);
    }
    // E-series per-layer embedding table. model_forward_batch deliberately
    // skips its own prepass under full offload (CUDA stages the table on the
    // device), so Metal builds it here from the scaled embeddings it just
    // wrote, then hands it over — shared storage, so a copy into unified
    // memory rather than a transfer.
    if (m->n_embd_ple > 0 && m->ple && g->ple) {
        model_ple_prepass(m, tokens, n, (const float *)g->x.contents,
                          m->ple, m->ple_tmp);
        memcpy(g->ple.contents, m->ple,
               sizeof(float) * (size_t)n * m->n_layer * m->n_embd_ple);
    }

    bool nantrace = metal_nan_trace();
    int  nanstage = metal_nan_stage() >= 2;
    // RUNNER_METAL_TIMING=1: split a forward into the CPU time spent encoding
    // dispatches and the GPU time actually executing them. Decode on Metal is
    // slower than CPU on an M1 and the two candidate explanations -- encode
    // overhead at ~420 dispatches per token, versus the kernels themselves --
    // are indistinguishable from throughput alone.
    static int timing = -1;
    if (timing < 0) {
        const char *t = getenv("RUNNER_METAL_TIMING");
        timing = t && *t && strcmp(t, "0") ? 1 : 0;
    }
    double t_enc0 = timing ? CFAbsoluteTimeGetCurrent() : 0;
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

#define NAN_PROBE(BUF, CNT, WHAT) do { if (nanstage) { \
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; \
        if (metal_scan_bad((const float *)(BUF).contents, (int)(CNT), (WHAT), l)) return NULL; \
        cb = [g->queue commandBuffer]; e = [cb computeCommandEncoder]; \
    } } while (0)

    // Partial split: only the leading gpu_layers run here.
    int n_gpu_layers = m->gpu_layers > 0 && m->gpu_layers < m->n_layer
                     ? m->gpu_layers : m->n_layer;
    bool fuse_pending_add = false;   // F2b: post-FFN add deferred into the
                                     // next layer's attention norm
    for (int l = 0; l < n_gpu_layers; l++) {
        layer_t *ly = &m->layers[l];
        int hd = model_head_dim(m, l);
        int n_kv = model_n_head_kv(m, l);
        int q_dim_l = model_q_dim(m, l);
        int kv_dim_l = model_kv_dim(m, l);
        int window = model_is_swa(m, l) ? m->swa_window : 0;
        // Weight-heavy projections run ONCE for the whole batch: the simdgroup
        // walks its weight row per column, so the row is fetched from device
        // memory on the first token and cached for the rest. Per-token submits
        // instead re-streamed every weight matrix n times, which is why Metal
        // prefill used to run at decode speed. Bit-identical either way.
        bool owns_kv = model_kv_owner(m, l) == l;
        // The attention-front megakernel: norm + q/k/v + rope + store in ONE
        // dispatch (kernels.metal explains why the walk is cut exactly
        // here). Models it cannot serve byte-exactly stay on the split path.
        id<MTLComputePipelineState> fpipe =
            n == 1 && metal_fuse_on() ? metal_front_pipe(g, m, l) : nil;
        bool front = fpipe != nil;
        uint64_t qw = 0, kw = 0, vw = 0;
        id<MTLBuffer> qb = nil;
        if (front) {
            uint64_t kwo = 0, vwo = 0;
            qb = metal_wbuf_for(g, (uint64_t)(uintptr_t)ly->wq->data,
                                ly->wq->nbytes, &qw);
            id<MTLBuffer> kb = metal_wbuf_for(g,
                (uint64_t)(uintptr_t)ly->wk->data, ly->wk->nbytes, &kwo);
            id<MTLBuffer> vb = metal_wbuf_for(g,
                (uint64_t)(uintptr_t)ly->wv->data, ly->wv->nbytes, &vwo);
            kw = kwo; vw = vwo;
            if (!(qb && qb == kb && qb == vb)) front = false;
        }
        if (fuse_pending_add) {
            // Fusion F2b: the previous layer deferred its post-FFN residual
            // add into this layer's attention norm (one dispatch, not two).
            // F2b never defers INTO a front-capable layer, so front and
            // pending are mutually exclusive here by construction.
            enc_add_rmsnorm(g, e, g->x, g->xb, g->xb, g->attn_norm[l],
                            n_embd, m->rms_eps);
            fuse_pending_add = false;
        } else if (!front)
            enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->attn_norm[l],
                          n_embd, m->rms_eps, n, n_embd, xdim);
        if (front) {
            metal_fuse_announce();
            metal_front_announce(fpipe, g);
            size_t frow_b = model_kv_row_bytes(m, l);
            uint64_t roff = model_kv_byte_off(m, l) +
                (uint64_t)model_kv_row_at(m, l, pos) * frow_b;
            attn_front_args fa = {
                n_embd, m->n_head, n_kv, hd, model_rope_dim(m, l) / 2,
                pos, m->rope_neox, model_rope_mscale(m, l), m->rms_eps,
                qw, kw, vw,
                g->bq[l] != nil, g->bk[l] != nil, g->bv[l] != nil,
                roff, roff,
                g->qn[l] != nil, g->kn[l] != nil };
            bool local = model_is_swa(m, l);
            [e setComputePipelineState:fpipe];
            [e setBuffer:qb offset:0 atIndex:0];
            [e setBuffer:g->x  offset:0 atIndex:1];
            [e setBuffer:g->attn_norm[l] offset:0 atIndex:2];
            [e setBuffer:g->q  offset:0 atIndex:3];
            [e setBuffer:g->kt offset:0 atIndex:4];
            [e setBuffer:g->vt offset:0 atIndex:5];
            [e setBuffer:g->kc offset:0 atIndex:6];
            [e setBuffer:g->vc offset:0 atIndex:7];
            [e setBuffer:(local && g->inv_freq_local) ? g->inv_freq_local
                                                      : g->inv_freq
                  offset:0 atIndex:8];
            [e setBuffer:g->bq[l] ? g->bq[l] : g->dummy offset:0 atIndex:9];
            [e setBuffer:g->bk[l] ? g->bk[l] : g->dummy offset:0 atIndex:10];
            [e setBuffer:g->bv[l] ? g->bv[l] : g->dummy offset:0 atIndex:11];
            [e setBytes:&fa length:sizeof(fa) atIndex:12];
            [e setBuffer:g->qn[l] ? g->qn[l] : g->dummy offset:0 atIndex:13];
            [e setBuffer:g->kn[l] ? g->kn[l] : g->dummy offset:0 atIndex:14];
            g_disp.mv++;
            [e dispatchThreadgroups:MTLSizeMake(m->n_head + 2 * n_kv, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
        enc_mv_n(g, e, m, ly->wq, g->xb, 0, g->q,  0,
                 n_embd, q_dim_l,  g->bq[l], n, xdim, q_dim);
        // the output gate projects the same normed input as Q; xb is
        // overwritten by the wo matvec below, so project it here
        if (m->attn_out_gate && ly->wq_gate)
            enc_mv_n(g, e, m, ly->wq_gate, g->xb, 0, g->agate, 0,
                     n_embd, q_dim_l, nil, n, xdim, q_dim);
        if (owns_kv) {
            enc_mv_n(g, e, m, ly->wk, g->xb, 0, g->kt, 0,
                     n_embd, kv_dim_l, g->bk[l], n, xdim, kv_dim);
            // gemma-4 global layers publish no V projection: V is the raw K
            // projection, taken before K is normed/roped (as on the CPU path).
            if (ly->wv)
                enc_mv_n(g, e, m, ly->wv, g->xb, 0, g->vt, 0,
                         n_embd, kv_dim_l, g->bv[l], n, xdim, kv_dim);
        }
        }

        // Batched in grid.y rather than encoded once per token. Bit-identical:
        // each (head, column) pair was always an independent reduction, so
        // only the encoding changes. This was 65n of the 240n per-token
        // dispatches the census measured; the elementwise ops are the rest.
        if (g->qn[l] && !front)
            enc_qknorm_n(g, e, m, g->q, 0, g->qn[l], m->n_head, hd, n, q_dim);
        if (owns_kv) {
            if (m->v_rmsnorm)
                enc_headnorm_n(g, e, m, ly->wv ? g->vt : g->kt, 0,
                               g->vt, 0, nil, n_kv, hd, n, kv_dim);
            if (g->kn[l] && !front)
                enc_qknorm_n(g, e, m, g->kt, 0, g->kn[l], n_kv, hd, n, kv_dim);
        }
        {
            size_t row_b = model_kv_row_bytes(m, l);
            int q8 = m->kv_q8;
            int kv_units = q8 ? kv_dim_l / 32 : kv_dim_l;

            // rope/store/attention each take the batch in one dispatch: the
            // kernels derive their column's position from pos + col, so every
            // token still rotates at, writes to, and attends over exactly the
            // range a per-token submit gave it.
            if (front) goto rope_store_done;   // the megakernel already did it
            // Decode fusion F1: rope(q)+rope(k)+f16 store in ONE dispatch
            // (budget: -2 dispatches x n_layer per token). q8 caches and
            // prefill keep the split path; byte identity is the gate.
            bool fused_rs = n == 1 && !q8 && owns_kv && metal_fuse_on() &&
                            g->p_rope_store && model_layer_ropes(m, l);
            if (fused_rs) {
                metal_fuse_announce();
                int half = model_rope_dim(m, l) / 2;
                uint64_t roff = model_kv_byte_off(m, l) +
                                (uint64_t)model_kv_row_at(m, l, pos) * row_b;
                rope_store_args ra = {
                    m->n_head, n_kv, hd, half, pos, m->rope_neox,
                    model_rope_mscale(m, l), 0, 0, 0, roff, roff };
                bool local = model_is_swa(m, l);
                [e setComputePipelineState:g->p_rope_store];
                [e setBuffer:g->q  offset:0 atIndex:0];
                [e setBuffer:g->kt offset:0 atIndex:1];
                [e setBuffer:g->vt offset:0 atIndex:2];
                [e setBuffer:g->kc offset:0 atIndex:3];
                [e setBuffer:g->vc offset:0 atIndex:4];
                [e setBuffer:(local && g->inv_freq_local) ? g->inv_freq_local
                                                          : g->inv_freq
                      offset:0 atIndex:5];
                [e setBytes:&ra length:sizeof(ra) atIndex:6];
                int threads = m->n_head * half + n_kv * half +
                              n_kv * (hd - 2 * half) + (kv_dim_l + 1) / 2;
                g_disp.rope++;
                [e dispatchThreads:MTLSizeMake(threads, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                goto rope_store_done;
            }
            if (model_layer_ropes(m, l)) {
                enc_rope_n(g, e, m, g->q,  0, m->n_head, pos, l, n, q_dim);
            } else if (m->attn_temp_scale != 0.0f) {
                // NoPE layer: no rotation, and THIS is where the llama-4
                // attention temperature applies -- on the layers that skipped
                // rope, not on every layer (matching the CPU path and
                // llama.cpp). The factor is per POSITION, so unlike everything
                // else here it cannot be one batched dispatch; it is one scale
                // per column, the same shape the CUDA backend uses.
                for (int b = 0; b < n; b++) {
                    float ts = model_attn_temp(m, pos + b);
                    if (ts != 1.0f)
                        enc_scale(g, e, g->q, foff((size_t)b * q_dim),
                                  q_dim_l, ts);
                }
            }
            if (owns_kv) {
                if (model_layer_ropes(m, l))
                    enc_rope_n(g, e, m, g->kt, 0, n_kv, pos, l, n, kv_dim);

                store_args sa = { kv_dim_l, q8, kv_dim, pos,
                                  model_kv_is_ring(m, l) ? m->kv_ring : 0,
                                  model_kv_byte_off(m, l), (uint64_t)row_b };
                [e setComputePipelineState:g->p_store];
                [e setBuffer:g->kt offset:0 atIndex:0];
                [e setBuffer:g->vt offset:0 atIndex:1];
                [e setBuffer:g->kc offset:0 atIndex:2];
                [e setBuffer:g->vc offset:0 atIndex:3];
                [e setBytes:&sa length:sizeof(sa) atIndex:4];
                g_disp.store++;
                [e dispatchThreads:MTLSizeMake(kv_units, n, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            rope_store_done: ;

            // Chunked path: decode only (n == 1), long enough range to be
            // worth splitting, and a head_dim the partial buffer was sized
            // for. Everything else keeps the single-pass kernel, including
            // all of prefill -- which already has grid.y parallelism from the
            // batch and is the validated fast path.
            int a_t0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
            int a_span = pos - a_t0 + 1;
            if (n == 1) {
                // K and V, both read across the span this layer attends over.
                unsigned long long b = 2ull * (unsigned long long)a_span
                                     * (unsigned long long)model_kv_row_bytes(m, l);
                if (window > 0) { g_disp.kv_swa += b; g_disp.kv_layers_swa++; }
                else            { g_disp.kv_global += b; g_disp.kv_layers_global++; }
                // ~4 device round-trips of the scores over the span, per head.
                g_disp.scores_bytes += 4ull * (unsigned long long)m->n_head
                                     * (unsigned long long)a_span * sizeof(float);
            }
            int a_ov = metal_attn_chunk_override();
            int a_chunk, a_nch;
            if (a_ov == 0) { a_chunk = 0; a_nch = 0; }        // path disabled
            else if (a_ov > 0) {
                a_chunk = a_ov;
                a_nch = (a_span + a_chunk - 1) / a_chunk;
            } else {
                int want = METAL_ATTN_TARGET_GROUPS / (m->n_head > 0 ? m->n_head : 1);
                if (want < 1) want = 1;
                if (want > METAL_ATTN_MAX_CHUNKS) want = METAL_ATTN_MAX_CHUNKS;
                a_chunk = (a_span + want - 1) / want;
                // floor, not veto: a target implying tiny chunks means "use the
                // smallest chunk worth having", and short context then drops to
                // the single-pass kernel through the n_chunks < 2 test below
                if (a_chunk < METAL_ATTN_MIN_CHUNK) a_chunk = METAL_ATTN_MIN_CHUNK;
                a_nch = (a_span + a_chunk - 1) / a_chunk;
            }
            if (n == 1 && a_nch >= 2 && a_nch <= METAL_ATTN_MAX_CHUNKS &&
                hd == model_head_dim(m, 0)) {
                attn_chunk_args ca = { hd, m->n_head, n_kv, m->n_ctx, pos,
                                       (uint64_t)model_kv_byte_off(m, l),
                                       model_attn_scale(m, l), q8, window,
                                       a_chunk, a_nch,
                                       model_kv_is_ring(m, l) ? m->kv_ring : 0 };
                bool coopc = metal_attn_coop_on() && g->p_attn_chunk_coop;
                if (coopc) g_coop_dispatches++;
                [e setComputePipelineState:coopc ? g->p_attn_chunk_coop
                                                 : g->p_attn_chunk];
                [e setBuffer:g->q       offset:0 atIndex:0];
                [e setBuffer:g->kc      offset:0 atIndex:1];
                [e setBuffer:g->vc      offset:0 atIndex:2];
                [e setBuffer:g->att     offset:0 atIndex:3];
                [e setBuffer:g->att_acc offset:0 atIndex:4];
                [e setBuffer:g->att_ms  offset:0 atIndex:5];
                [e setBytes:&ca length:sizeof(ca) atIndex:6];
                g_disp.attn_chunk++;
                [e dispatchThreadgroups:MTLSizeMake(m->n_head, a_nch, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                attn_comb_args cb = { hd, m->n_head, a_nch,
                                      g->sinks[l] != nil };
                [e setComputePipelineState:g->p_attn_comb];
                [e setBuffer:g->att_acc offset:0 atIndex:0];
                [e setBuffer:g->att_ms  offset:0 atIndex:1];
                [e setBuffer:g->xb2     offset:0 atIndex:2];
                [e setBytes:&cb length:sizeof(cb) atIndex:3];
                [e setBuffer:g->sinks[l] ? g->sinks[l] : g->dummy offset:0 atIndex:4];
                g_disp.attn_chunk++;
                [e dispatchThreadgroups:MTLSizeMake(m->n_head, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                goto attn_done;
            }

            attn_args aa = { hd, m->n_head, n_kv, m->n_ctx, pos,
                             (uint64_t)model_kv_byte_off(m, l),
                             model_attn_scale(m, l), q8, window,
                             g->sinks[l] != nil,
                             q_dim, m->n_head * m->n_ctx, xdim,
                             model_kv_is_ring(m, l) ? m->kv_ring : 0 };
            // Cooperative twin at decode only: the score loop is what
            // scatters, and at n > 1 each column has its own KV range, which
            // the coop form's simdgroup-per-row split does not describe.
            bool coop = (n == 1) && metal_attn_coop_on() && g->p_attn_coop;
            if (coop) g_coop_dispatches++;
            [e setComputePipelineState:coop ? g->p_attn_coop : g->p_attn];
            [e setBuffer:g->q   offset:0 atIndex:0];
            [e setBuffer:g->kc  offset:0 atIndex:1];
            [e setBuffer:g->vc  offset:0 atIndex:2];
            [e setBuffer:g->att offset:0 atIndex:3];
            [e setBuffer:g->xb2 offset:0 atIndex:4];
            [e setBytes:&aa length:sizeof(aa) atIndex:5];
            [e setBuffer:g->sinks[l] ? g->sinks[l] : g->dummy offset:0 atIndex:6];
            // Widest power-of-two threadgroup this pipeline allows, capped
            // at the red[] scratch in the kernel. The reduction halves tpg
            // each step, so a non-power-of-two would drop lanes silently.
            NSUInteger amax = (coop ? g->p_attn_coop : g->p_attn)
                                  .maxTotalThreadsPerThreadgroup;
            if (amax > 256) amax = 256;   // measured optimum; red[] is sized to match
            NSUInteger atpg = 1;
            while (atpg * 2 <= amax) atpg *= 2;
            if (timing) {
                static bool told;
                if (!told) {
                    told = true;
                    fprintf(stderr, "metal-timing attn threadgroups=%d x %llu "
                            "threads (pipeline max %llu)\n", m->n_head,
                            (unsigned long long)atpg,
                            (unsigned long long)g->p_attn.maxTotalThreadsPerThreadgroup);
                }
            }
            g_disp.attn++;
            [e dispatchThreadgroups:MTLSizeMake(m->n_head, n, 1)
              threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
            attn_done: ;
        }

        NAN_PROBE(g->xb2, (size_t)n * xdim, "attn-out");
        if (m->attn_out_gate && ly->wq_gate)
            enc_elem_n(g, e, g->p_sigmul, g->xb2, 0, g->agate, 0,
                       q_dim_l, n, xdim, q_dim);
        enc_mv_n(g, e, m, ly->wo, g->xb2, 0, g->xb, 0,
                 q_dim_l, n_embd, g->bo[l], n, xdim, xdim);
        NAN_PROBE(g->xb, (size_t)n * xdim, "attn-wo");
        if (g->pan[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pan[l],
                          n_embd, m->post_norm_eps, n, xdim, xdim);
        if (m->resid_scale != 1.0f)  // granite muP branch scale
            enc_scale_n(g, e, g->xb, 0, n_embd, m->resid_scale, n, xdim);
        // Fusion F2a: post-attention add + the FFN's leading norm, one
        // dispatch. Guarded off wherever anything sits between the two
        // (sandwich norm, muP scale, gemma dual-branch, debug probes).
        bool f2a = n == 1 && metal_fuse_on() && g->p_add_rmsnorm &&
                   !g->pan[l] && m->resid_scale == 1.0f && !nanstage &&
                   !ly->moe_gemma;
        // Fusion F2b, decided BEFORE the FFN so F5 (sum+add) can take the
        // layers F2b leaves behind: F2b defers the post-FFN add into the
        // NEXT layer's attention norm; F5 folds it into THIS layer's
        // expert sum. Mutually exclusive by construction.
        bool f2b = n == 1 && metal_fuse_on() && g->p_add_rmsnorm &&
                   !g->pfn[l] && m->resid_scale == 1.0f && !ly->ple_gate &&
                   (ly->out_scale == 1.0f || ly->out_scale == 0.0f) &&
                   !nantrace && !nanstage && l + 1 < n_gpu_layers &&
                   n_gpu_layers == m->n_layer &&
                   !metal_front_capable(g, m, l + 1);
        bool sum_add = n == 1 && metal_fuse_on() && !f2b && ly->is_moe &&
                       !ly->moe_gemma && !g->pfn[l] &&
                       m->resid_scale == 1.0f && !ly->ple_gate &&
                       !nantrace && !nanstage;
        if (!f2a)
            enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);
        NAN_PROBE(g->x, (size_t)n * n_embd, "post-attn-resid");

        if (ly->moe_gemma || ly->is_moe) {
            if (f2a) {
                metal_fuse_announce();
                enc_add_rmsnorm(g, e, g->x, g->xb, g->xb, g->ffn_norm[l],
                                n_embd, m->rms_eps);
            }
            if (ly->moe_gemma)
                enc_gemma_moe_ffn_batch(g, e, m, ly, l, n, xdim, &cb, &e);
            else
                enc_moe_ffn_batch(g, e, m, ly, l, n, xdim, f2a, sum_add);
        } else {
            // gemma-4 E2B varies the FFN width per layer; hb/hb2 are sized
            // for the max, so pack the batch at THIS layer's width.
            int nff_l = ly->n_ff;
            if (f2a) {
                metal_fuse_announce();
                enc_add_rmsnorm(g, e, g->x, g->xb, g->xb, g->ffn_norm[l],
                                n_embd, m->rms_eps);
            } else
            enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->ffn_norm[l],
                          n_embd, m->rms_eps, n, n_embd, xdim);
            enc_mv_n(g, e, m, ly->w_gate, g->xb, 0, g->hb,  0,
                     n_embd, nff_l, nil, n, xdim, nff_l);
            enc_mv_n(g, e, m, ly->w_up,   g->xb, 0, g->hb2, 0,
                     n_embd, nff_l, nil, n, xdim, nff_l);
            // hb/hb2 are contiguous across the batch, so the activation is one
            // dispatch over the whole batch rather than n
            enc_elem(g, e, m->ffn_act == ACT_GELU ? g->p_gelu : g->p_silu,
                     g->hb, 0, g->hb2, 0, n * nff_l);
            enc_mv_n(g, e, m, ly->w_down, g->hb, 0, g->xb, 0,
                     nff_l, n_embd, nil, n, nff_l, xdim);
        }
        NAN_PROBE(g->xb, (size_t)n * xdim, "ffn-out");
        if (g->pfn[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pfn[l],
                          n_embd, m->post_norm_eps, n, xdim, xdim);
        if (m->resid_scale != 1.0f)
            enc_scale_n(g, e, g->xb, 0, n_embd, m->resid_scale, n, xdim);
        // Fusion F2b (decided above): defer this add into the next layer's
        // attention norm; F5 already folded it into the expert sum on the
        // layers it took.
        if (f2b)
            fuse_pending_add = true;
        else if (!sum_add)
        enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);
        // Order matters: the E-series branch reads the post-FFN residual of
        // EVERY token, so it has to run before any token's output scale.
        if (ly->ple_gate) enc_ple(g, e, m, ly, l, n, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            enc_scale_n(g, e, g->x, 0, n_embd, ly->out_scale, n, n_embd);
        if (nantrace) {
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (metal_scan_bad((const float *)g->x.contents,
                               n * n_embd, "resid", l))
                return NULL;
            cb = [g->queue commandBuffer];
            e = [cb computeCommandEncoder];
        }
        // Only the last position's logits are ever read (see the return
        // below), and the vocab matmul is the widest in the model — doing it
        // for every prompt token was pure waste.
        if (l == m->n_layer - 1 && n_gpu_layers == m->n_layer) {
            // Speculative verify (model_forward_batch_keep) needs EVERY
            // column's logits — the head runs per column then. The normal
            // path keeps the last-column-only economy.
            int c0 = m->spec_want_all >= n ? 0 : n - 1;
            for (int c = c0; c < n; c++) {
                NSUInteger xo = foff((size_t)c * n_embd);
                NSUInteger xbo = foff((size_t)c * xdim);
                enc_rmsnorm(g, e, g->x, xo, g->xb, xbo, g->out_norm,
                            n_embd, m->rms_eps);
                enc_mv(g, e, m, m->output, g->xb, xbo, g->logits,
                       foff((size_t)c * m->n_vocab),
                       n_embd, m->n_vocab, nil);
            }
        }
    }

#undef NAN_PROBE

    [e endEncoding];
    double t_enc1 = timing ? CFAbsoluteTimeGetCurrent() : 0;
    [cb commit];
    [cb waitUntilCompleted];
    if (timing) {
        double now = CFAbsoluteTimeGetCurrent();
        double gpu = [cb GPUEndTime] - [cb GPUStartTime];
        fprintf(stderr, "metal-timing n=%d encode=%.2fms submit+wait=%.2fms "
                "gpu=%.2fms total=%.2fms\n", n,
                (t_enc1 - t_enc0) * 1e3, (now - t_enc1) * 1e3, gpu * 1e3,
                (now - t_enc0) * 1e3);
    }
    if (metal_command_failed(cb)) {
        fprintf(stderr, "gpu: command buffer failed — falling back to CPU\n");
        return NULL;
    }
    // A binding this encoder could not place makes every number it produced
    // meaningless, including the ones written before the bad dispatch.
    if (g->bind_failed) return NULL;
    metal_moe_route_trace(g, m, n);
    metal_moe_full_trace(g, m, n, pos);
    if (metal_env_on("RUNNER_METAL_STATS")) {
        unsigned long tot = g_disp.tensor + g_disp.mm + g_disp.mv + g_disp.mvf +
                            g_disp.rmsnorm + g_disp.qknorm + g_disp.headnorm +
                            g_disp.rope + g_disp.store + g_disp.attn +
                            g_disp.attn_chunk + g_disp.elem + g_disp.moe;
        fprintf(stderr, "metal: native prompt batch n=%d command_buffers=1\n", n);
        // Per-kind census for this forward, so "which kinds scale with n"
        // can be read straight off two runs at different batch sizes rather
        // than inferred from throughput. Counters are cumulative over the
        // process; take differences across forwards.
        if (g_disp.kv_global || g_disp.kv_swa)
            fprintf(stderr, "metal-kv decode-cumulative: global %llu B over %lu "
                    "layer-dispatches | sliding %llu B over %lu | sliding share "
                    "%.1f%%\n",
                    g_disp.kv_global, g_disp.kv_layers_global,
                    g_disp.kv_swa, g_disp.kv_layers_swa,
                    100.0 * (double)g_disp.kv_swa
                        / (double)(g_disp.kv_swa + g_disp.kv_global));
        if (g_disp.scores_bytes)
            fprintf(stderr, "metal-scores decode-cumulative: %llu B (ah "
                    "round-trips) = %.1f%% of KV read\n", g_disp.scores_bytes,
                    100.0 * (double)g_disp.scores_bytes
                        / (double)(g_disp.kv_swa + g_disp.kv_global));
        fprintf(stderr, "metal-census n=%d total=%lu | tensor=%lu mm=%lu mv=%lu mvf=%lu "
                "rmsnorm=%lu qknorm=%lu headnorm=%lu rope=%lu store=%lu "
                "attn=%lu attn_chunk=%lu(coop %lu) elem=%lu moe=%lu\n",
                n, tot, g_disp.tensor, g_disp.mm, g_disp.mv, g_disp.mvf, g_disp.rmsnorm,
                g_disp.qknorm, g_disp.headnorm, g_disp.rope, g_disp.store,
                g_disp.attn, g_disp.attn_chunk, g_coop_dispatches,
                g_disp.elem, g_disp.moe);
    }
    return (float *)g->logits.contents + (size_t)(n - 1) * m->n_vocab;
}


// default-ON env switch: unset means on; "0"/"off" disables.
static bool metal_env_default_on(const char *name) {
    const char *v = getenv(name);
    return !v || !*v || (strcmp(v, "0") && strcmp(v, "off"));
}

// ---------------------------------------------------- Metal microbatch decode
//
// N server slots share one weight sweep per decode step. The projections run
// as multi-column identity matvecs (enc_mv_cols — the exact k_mv dot the solo
// path runs, per column, so the microbatch is BIT-IDENTICAL to sequential
// decode; the weight rows are simply served from cache for columns 2..N).
// Rope, KV store and attention stay one dispatch per column, because each
// column has its own position and its own slot's KV buffers — those kernels
// are the solo kernels applied to the solo data, encoder-ordered. Same twin
// discipline as CUDA's fwd_batch, minus the upload/copyback half that unified
// memory never needed.
struct gpu_batch {
    model_t     **seqs;    // borrowed
    int           n;
    gpu_t        *lead;    // whose queue/scratch encode the step
    int           n_vocab, n_embd, xdim;
    id<MTLBuffer> logits_n;   // [MODEL_BATCH_MAX][n_vocab]
};

// Dense gated transformers only, full offload, uniform engine config across
// the slots. Everything else decodes sequentially — correctness first, and
// the decline list mirrors CUDA's for the same reasons (no MoE/dual-branch
// path here, no recurrent mixer, no output gate, no per-layer embeddings).
static bool metal_batch_eligible(model_t **seqs, int n, gpu_t **lead_out) {
    if (n < 2) return false;
    if (!metal_env_default_on("RUNNER_METAL_BATCH")) return false;
    model_t *m0 = seqs[0];
    gpu_t *lead = NULL;
    for (int i = 0; i < n; i++) {
        model_t *m = seqs[i];
        if (!m || !m->gpu) return false;
        gpu_t *g = (gpu_t *)m->gpu;
        if (m->gpu_layers < m->n_layer) return false;
        if (m->n_expert > 0 || m->qwen35 || m->granite_hybrid ||
            m->nemotron_h || m->attn_out_gate || m->n_embd_ple > 0)
            return false;
        for (int l = 0; l < m->n_layer; l++)
            if (m->layers[l].moe_gemma || m->layers[l].is_moe ||
                m->layers[l].recurrent || m->layers[l].skip_mixer)
                return false;
        if (model_kv_ring_active(m)) return false;
        if (m->n_vocab != m0->n_vocab || m->n_embd != m0->n_embd ||
            m->n_ctx != m0->n_ctx || m->kv_q8 != m0->kv_q8 ||
            m->n_layer != m0->n_layer) return false;
        // every projection this walk will hand to enc_mv_cols needs the
        // identity matvec kernel for its type
        const gguf_tensor *ws0[] = { m->output, m->tok_embd };
        for (size_t k = 0; k < sizeof(ws0) / sizeof(*ws0); k++)
            if (ws0[k] && !g->p_mv[ws0[k]->type]) return false;
        for (int l = 0; l < m->n_layer; l++) {
            const layer_t *ly = &m->layers[l];
            if (!ly->wq || !ly->wo || !ly->w_gate || !ly->w_up || !ly->w_down)
                return false;
            const gguf_tensor *ws[] = { ly->wq, ly->wk, ly->wv, ly->wo,
                                        ly->w_gate, ly->w_up, ly->w_down };
            for (size_t k = 0; k < sizeof(ws) / sizeof(*ws); k++)
                if (ws[k] && !g->p_mv[ws[k]->type]) return false;
        }
        if (!lead) lead = g;
    }
    *lead_out = lead;
    return true;
}

// Unified memory: g->x and g->logits contents are host pointers, so the
// speculative verify walk can read the batch's hidden work directly. The
// forward emits per-column heads when model_forward_batch_keep asked for
// them (m->spec_want_all), and row logits are just offsets into g->logits.
bool gpu_spec_keep_ok(const model_t *m) {
    (void)m;
    return true;
}

float *gpu_spec_logits(model_t *mm, int row) {
    gpu_t *g = (gpu_t *)mm->gpu;
    if (!g || !g->logits) return NULL;
    return (float *)g->logits.contents + (size_t)row * mm->n_vocab;
}

gpu_batch *gpu_batch_create(model_t **seqs, int n) {
    gpu_t *lead = NULL;
    if (!metal_batch_eligible(seqs, n, &lead)) return NULL;
    model_t *m0 = seqs[0];
    gpu_batch *b = calloc(1, sizeof(gpu_batch));
    if (!b) return NULL;
    b->seqs = malloc(sizeof(model_t *) * (size_t)n);
    if (!b->seqs) { free(b); return NULL; }
    memcpy(b->seqs, seqs, sizeof(model_t *) * (size_t)n);
    b->n = n;
    b->lead = lead;
    b->n_vocab = m0->n_vocab;
    b->n_embd = m0->n_embd;
    b->xdim = m0->n_embd;
    for (int l = 0; l < m0->n_layer; l++)
        if (model_q_dim(m0, l) > b->xdim) b->xdim = model_q_dim(m0, l);
    b->logits_n = new_f32_scratch(lead->dev,
                                  (size_t)MODEL_BATCH_MAX * b->n_vocab);
    if (!metal_buffer_ok(b->logits_n)) {
        fprintf(stderr, "gpu: Metal batch logits allocation failed — using "
                "sequential GPU forwards\n");
        gpu_batch_free(b);
        return NULL;
    }
    fprintf(stderr, "gpu: Metal microbatch decode on for %d slots "
            "(bit-identical to sequential; RUNNER_METAL_BATCH=0 disables)\n",
            n);
    return b;
}

void gpu_batch_free(gpu_batch *b) {
    if (!b) return;
    [b->logits_n release];
    free(b->seqs);
    free(b);
}

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    if (!b || n < 1 || n > MODEL_BATCH_MAX) return false;
    if (n < 2) return false;   // one column is just a solo forward
    gpu_t *g = b->lead;
    model_t *m0 = b->seqs[0];
    int n_embd = b->n_embd;
    int xdim   = b->xdim;

    // The lead's scratch is sized by the largest n any forward has used —
    // a slot that only ever decoded has batch_cap == 1, and writing column
    // 1 into one-column buffers lands in whichever allocation the heap put
    // next (found exactly that way: x's column 1 aliased xb). Grow first.
    if (!metal_ensure_batch(g, m0, n)) return false;

    // per-column embeds (CPU-side, exactly as the solo path does)
    size_t ers = ggml_row_size(m0->tok_embd->type, n_embd);
    for (int c = 0; c < n; c++) {
        model_t *ms = b->seqs[idx[c]];
        float *xp = (float *)g->x.contents + (size_t)c * n_embd;
        dequant_row(ms->tok_embd->type,
                    (uint8_t *)ms->tok_embd->data + (size_t)tok[c] * ers,
                    xp, n_embd);
        model_embd_transform(ms, xp);
    }

    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];

    for (int l = 0; l < m0->n_layer; l++) {
        layer_t *ly = &m0->layers[l];
        int hd = model_head_dim(m0, l);
        int n_kv = model_n_head_kv(m0, l);
        int q_dim_l = model_q_dim(m0, l);
        int kv_dim_l = model_kv_dim(m0, l);
        int q_dim = m0->n_head * m0->head_dim;
        int kv_dim = m0->n_head_kv * m0->head_dim;
        int window = model_is_swa(m0, l) ? m0->swa_window : 0;
        size_t row_b = model_kv_row_bytes(m0, l);
        int q8 = m0->kv_q8;
        int kv_units = q8 ? kv_dim_l / 32 : kv_dim_l;
        bool owns_kv = model_kv_owner(m0, l) == l;

        enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->attn_norm[l],
                      n_embd, m0->rms_eps, n, n_embd, xdim);
        enc_mv_cols(g, e, ly->wq, g->xb, 0, g->q, 0,
                    n_embd, q_dim_l, g->bq[l], n, xdim, q_dim);
        if (owns_kv) {
            enc_mv_cols(g, e, ly->wk, g->xb, 0, g->kt, 0,
                        n_embd, kv_dim_l, g->bk[l], n, xdim, kv_dim);
            if (ly->wv)
                enc_mv_cols(g, e, ly->wv, g->xb, 0, g->vt, 0,
                            n_embd, kv_dim_l, g->bv[l], n, xdim, kv_dim);
        }
        if (g->qn[l])
            enc_qknorm_n(g, e, m0, g->q, 0, g->qn[l], m0->n_head, hd, n, q_dim);
        if (owns_kv) {
            if (m0->v_rmsnorm)
                enc_headnorm_n(g, e, m0, ly->wv ? g->vt : g->kt, 0,
                               g->vt, 0, nil, n_kv, hd, n, kv_dim);
            if (g->kn[l])
                enc_qknorm_n(g, e, m0, g->kt, 0, g->kn[l], n_kv, hd, n, kv_dim);
        }

        // per column: rope at ITS position, store into ITS slot's cache,
        // attend over ITS span — the solo kernels on the solo data
        for (int c = 0; c < n; c++) {
            model_t *ms = b->seqs[idx[c]];
            gpu_t *gs = (gpu_t *)ms->gpu;
            int p = pos[c];
            if (model_layer_ropes(m0, l)) {
                enc_rope_n(g, e, m0, g->q, foff((size_t)c * q_dim),
                           m0->n_head, p, l, 1, q_dim);
                if (owns_kv)
                    enc_rope_n(g, e, m0, g->kt, foff((size_t)c * kv_dim),
                               n_kv, p, l, 1, kv_dim);
            } else if (m0->attn_temp_scale != 0.0f) {
                float ts = model_attn_temp(m0, p);
                if (ts != 1.0f)
                    enc_scale(g, e, g->q, foff((size_t)c * q_dim), q_dim_l, ts);
            }
            if (owns_kv) {
                store_args sa = { kv_dim_l, q8, kv_dim, p, 0,
                                  model_kv_byte_off(ms, l), (uint64_t)row_b };
                [e setComputePipelineState:g->p_store];
                [e setBuffer:g->kt offset:foff((size_t)c * kv_dim) atIndex:0];
                [e setBuffer:g->vt offset:foff((size_t)c * kv_dim) atIndex:1];
                [e setBuffer:gs->kc offset:0 atIndex:2];
                [e setBuffer:gs->vc offset:0 atIndex:3];
                [e setBytes:&sa length:sizeof(sa) atIndex:4];
                g_disp.store++;
                [e dispatchThreads:MTLSizeMake(kv_units, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            attn_args aa = { hd, m0->n_head, n_kv, m0->n_ctx, p,
                             (uint64_t)model_kv_byte_off(ms, l),
                             model_attn_scale(m0, l), q8, window,
                             false, q_dim, m0->n_head * m0->n_ctx, xdim, 0 };
            bool coop = metal_attn_coop_on() && g->p_attn_coop;
            if (coop) g_coop_dispatches++;
            [e setComputePipelineState:coop ? g->p_attn_coop : g->p_attn];
            [e setBuffer:g->q   offset:foff((size_t)c * q_dim) atIndex:0];
            [e setBuffer:gs->kc offset:0 atIndex:1];
            [e setBuffer:gs->vc offset:0 atIndex:2];
            [e setBuffer:g->att offset:0 atIndex:3];
            [e setBuffer:g->xb2 offset:foff((size_t)c * xdim) atIndex:4];
            [e setBytes:&aa length:sizeof(aa) atIndex:5];
            [e setBuffer:g->dummy offset:0 atIndex:6];
            NSUInteger amax = (coop ? g->p_attn_coop : g->p_attn)
                                  .maxTotalThreadsPerThreadgroup;
            if (amax > 256) amax = 256;
            NSUInteger atpg = 1;
            while (atpg * 2 <= amax) atpg *= 2;
            g_disp.attn++;
            [e dispatchThreadgroups:MTLSizeMake(m0->n_head, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
        }

        enc_mv_cols(g, e, ly->wo, g->xb2, 0, g->xb, 0,
                    q_dim_l, n_embd, g->bo[l], n, xdim, xdim);
        if (g->pan[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pan[l],
                          n_embd, m0->post_norm_eps, n, xdim, xdim);
        if (m0->resid_scale != 1.0f)
            enc_scale_n(g, e, g->xb, 0, n_embd, m0->resid_scale, n, xdim);
        enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);

        int nff_l = ly->n_ff;
        enc_rmsnorm_n(g, e, g->x, 0, g->xb, 0, g->ffn_norm[l],
                      n_embd, m0->rms_eps, n, n_embd, xdim);
        enc_mv_cols(g, e, ly->w_gate, g->xb, 0, g->hb, 0,
                    n_embd, nff_l, nil, n, xdim, nff_l);
        enc_mv_cols(g, e, ly->w_up, g->xb, 0, g->hb2, 0,
                    n_embd, nff_l, nil, n, xdim, nff_l);
        enc_elem(g, e, m0->ffn_act == ACT_GELU ? g->p_gelu : g->p_silu,
                 g->hb, 0, g->hb2, 0, n * nff_l);
        enc_mv_cols(g, e, ly->w_down, g->hb, 0, g->xb, 0,
                    nff_l, n_embd, nil, n, nff_l, xdim);
        if (g->pfn[l])
            enc_rmsnorm_n(g, e, g->xb, 0, g->xb, 0, g->pfn[l],
                          n_embd, m0->post_norm_eps, n, xdim, xdim);
        if (m0->resid_scale != 1.0f)
            enc_scale_n(g, e, g->xb, 0, n_embd, m0->resid_scale, n, xdim);
        enc_elem_n(g, e, g->p_add, g->x, 0, g->xb, 0, n_embd, n, n_embd, xdim);
        if (ly->out_scale != 1.0f && ly->out_scale != 0.0f)
            enc_scale_n(g, e, g->x, 0, n_embd, ly->out_scale, n, n_embd);
    }

    // head, per column: every sequence needs its own logits
    for (int c = 0; c < n; c++) {
        enc_rmsnorm(g, e, g->x, foff((size_t)c * n_embd),
                    g->xb, foff((size_t)c * xdim), g->out_norm,
                    n_embd, m0->rms_eps);
        enc_mv(g, e, m0, m0->output, g->xb, foff((size_t)c * xdim),
               b->logits_n, foff((size_t)c * b->n_vocab),
               n_embd, b->n_vocab, nil);
    }

    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (metal_command_failed(cb)) {
        fprintf(stderr, "gpu: Metal microbatch command buffer failed — "
                "falling back to sequential decode\n");
        return false;
    }
    for (int c = 0; c < n; c++)
        out[c] = (float *)b->logits_n.contents + (size_t)c * b->n_vocab;
    return true;
}
