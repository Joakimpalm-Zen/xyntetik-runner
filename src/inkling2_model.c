// R4.48.6: Inkling 2 Mini on Runner's normal path (see inkling2_model.h).
#include "inkling2_model.h"
#include "inkling2.h"
#include "inkling2_ecache.h"
#include "inkling2_gguf.h"
#include "gpu.h"
#include "quants.h"
#include "compat.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ink2_dev_hdr hdr;   // FIRST: the device functions the engine calls through this instance (its views' and cache's dev)
    ink2g *g;
    ink2_shape s;
    ink2_weights inner, w;
    ink2_ecache *ec;
    ink2_cache c;
    bool cache_ok;
    float *rows;        // [n_batch][n_vocab]: the last chunk's rows (the keep path reads row b of the last forward)
    int rows_cap;       // rows the buffer holds
    int kept_from;      // the first position the buffer's row 0 is, for model_inkling2_row
    int kept_n;         // rows kept by the last keep_all forward
    int *last_sel;      // [n_app][T][n_used] expert selections from last backward
    int last_sel_n;     // element count of last_sel
    int n_threads;      // thread count configured at runtime init (when m->tp is NULL)
    size_t dense_bytes;   // float32 copies of the matrices (the projections, the head, the embedding): 0 on the fast route
    size_t small_bytes;   // the small tensors decoded to float32 on either route (norms, taps, queries, the router)
    size_t ec_budget;
    char route;           // 'F' the fast route, 'R' the reference (RUNNER_INKLING2_ROUTE=reference)
    bool act8;            // RUNNER_INKLING2_ACT8 active on the fast route
    bool last_ok;         // last model_inkling2_forward succeeded (distinguishes !want_logits from error)
    gpu_ink2 *gpu;        // R4.48.9: the weights on CUDA (the fast route's views, stored rows), or NULL
    const ink2_tensor **dev_of;   // [n_dev]: the view each device tensor holds, in gpu_ink2_put's order
    int n_dev;
    bool gpu_err;         // a device product failed: this instance refuses every forward from then on
    bool no_factor;       // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_NO_FACTOR=1): the device leaves out the expert factor
    int lin_fail_at;      // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_LIN_FAIL=k): the k-th device product fails; -1 none
    int n_lin;            // device products run so far (for lin_fail_at)
    long n_exp;           // G2b step 1: device expert-body calls run so far (one per application per forward)
    long exp_fail_at;     // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_EXP_FAIL=k): the k-th expert-body call fails; -1 none
    bool exp_on;          // this instance runs the expert body on the device (the hook is process-wide)
    char placed;          // the bind's decision: 'W' every view the device computes, 'M' the expert stacks on the
                          // host (--cpu-moe), 'C' the CPU; 0 when no bind was asked for
    size_t need;          // the weights' device bytes the fit counted for that placement
    struct ink2_share *share;   // the file's shared device copy this instance holds a reference to
    gpu_ink2_kv *kv;      // G2b step 2a: this instance's K/V rings on the device (the host rings then freed), or NULL
    bool attn_on;         // this instance runs the attention core on the device
    long n_attn;          // device core calls run so far (one per application per forward)
    long attn_fail_at;    // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_ATTN_FAIL=k): the k-th device core call fails; -1 none
    bool block_on;        // G2b step 2b: this instance runs each attention block whole on the device
    bool seq_block;       // the current sequence's blocks run on the device (its K/V histories are the device's)
    long n_block_calls;   // device block calls run so far (one per application per forward)
    bool msg_on;          // D1a: this instance runs the loop's message steps on the device
    long n_msg_calls;     // device message steps run so far
    long msg_fail_at;     // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_MSG_FAIL=k): the k-th device message step fails; -1 none
    bool route_on;        // D1b: this instance runs the router on the device with the expert body
    long n_route;         // device router calls run so far
    // D1d.3b: the resident decode step (on by default; TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_STEP=1 off), its post convolutions'
    // device copies (post_dev_newer: they are newer than the host's, a step ran last; the sync's staging)
    bool step_on;
    bool post_dev_newer;
    bool last_was_step;
    float *post_stage;
    long n_step;
    long n_step_try;      // steps attempted so far (for step_fail_at)
    long step_fail_at;    // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_STEP_FAIL=k): the k-th step attempt fails; -1 none
} ink2_rt;

static ink2_rt *rt_of(model_t *m) { return (ink2_rt *)m->ink2; }

// R4.48.9: a view's rows (a stack's experts one after another) and its factors
static int64_t view_rows(const ink2_tensor *t) { return t->ne[1] * t->ne[2] * t->ne[3]; }
// D1b: a view's bytes on the device and their source: the file's rows (t->q), or for a router its decoded float32 rows
// (t->data, no q; ink2g_each_view's last views), uploaded as F32
static const void *view_src(const ink2_tensor *t) { return t->q ? t->q : (const void *)t->data; }
static size_t view_row_bytes(const ink2_tensor *t) { return t->q ? t->q_row_bytes : sizeof(float) * (size_t)t->ne[0]; }
static int view_qtype(const ink2_tensor *t) { return t->q ? t->qtype : T_F32; }
static int view_scales(const ink2_tensor *t) { return t->q_scale ? (int)t->ne[2] : 0; }
typedef struct {
    ink2_rt *r; bool cpu_moe; size_t need; int n; int fail_at;
    int64_t max_in, max_out, max_out_body;   // max_out_body: the widest product but the head's (its vocabulary rows)
    bool attach;
} dev_walk;
// what goes up is what the device computes (the second read's P1): a view the handle's products take (its type's
// kernel; a factor only on NVFP4, the one kernel that applies it), not the token embedding (read row by row on the
// host, never a product), and no expert stack under --cpu-moe. A view left on the host runs its products on the CPU.
static bool dev_wanted(const dev_walk *c, const ink2_tensor *t) {
    if (!strcmp(t->name, "model.embed_tokens.weight")) return false;
    if (c->cpu_moe && strstr(t->name, ".mlp.experts.")) return false;
    if (t->q_scale && t->qtype != T_NVFP4) return false;
    // D1b: a router goes up whenever the handle has the device router, whatever this instance's TEST-ONLY NO_ROUTER
    // says (that switch is no part of the share key: every instance on one copy walks the same views)
    if (!t->q && !gpu_ink2_route_ok(c->r->gpu)) return false;
    return gpu_ink2_runs(c->r->gpu, view_qtype(t));
}
// a device allocation's bytes as the driver takes them: large ones in 2 MiB pages, small ones 512-byte aligned
static size_t dev_alloc_bytes(size_t b) {
    const size_t g = b >= ((size_t)1 << 20) ? (size_t)2 << 20 : 512;
    return (b + g - 1) / g * g;
}
static int dev_count(void *c_, const ink2_tensor *t) {
    dev_walk *c = c_;
    if (!dev_wanted(c, t)) return 0;
    c->need += dev_alloc_bytes(view_row_bytes(t) * (size_t)view_rows(t));
    if (view_scales(t)) c->need += dev_alloc_bytes(sizeof(float) * (size_t)view_scales(t));
    if (t->ne[0] > c->max_in) c->max_in = t->ne[0];
    if (t->ne[1] > c->max_out) c->max_out = t->ne[1];   // a product's rows: at most one matrix's (one expert's)
    if (strcmp(t->name, "lm_head.weight") && t->ne[1] > c->max_out_body) c->max_out_body = t->ne[1];
    c->n++;
    return 0;
}
static int dev_put(void *c_, const ink2_tensor *t) {
    dev_walk *c = c_;
    if (!dev_wanted(c, t)) return 0;
    // attach (an instance joining the file's shared device copy): the views in the same walk order as the upload that
    // made the copy (the file's order), so view k is device tensor k; nothing goes up
    if (c->attach) { if (c->r->n_dev >= c->n) return -1; c->r->dev_of[c->r->n_dev++] = t; return 0; }
    if (c->r->n_dev == c->fail_at) return -1;   // TEST-ONLY (RUNNER_INKLING2_CUDA_TEST_PUT_FAIL=k): the k-th put fails
    if (gpu_ink2_put(c->r->gpu, view_src(t), view_row_bytes(t) * (size_t)view_rows(t), t->q_scale, view_scales(t)) !=
        c->r->n_dev)
        return -1;
    c->r->dev_of[c->r->n_dev++] = t;   // not marked on the device yet: dev_mark, once every put succeeded
    return 0;
}
// the views put marked as held on the device (ink2_tensor.dev), all or none: a view marked while its instance has no
// device would send the engine's products to a device that is not there (the second read's P0)
static void dev_unmark(ink2_rt *r) {
    for (int i = 0; i < r->n_dev; i++) ink2g_set_dev(r->g, r->dev_of[i], NULL, 0);
}
static bool dev_mark(ink2_rt *r) {
    for (int i = 0; i < r->n_dev; i++)
        if (ink2g_set_dev(r->g, r->dev_of[i], r, i)) { dev_unmark(r); return false; }
    return true;
}
// G2a: the engine's product of a device-held view, on the device (gpu_ink2_lin); x gathered and y scattered here
static int dev_lin_scaled(const ink2_tensor *W, int64_t row0, int n_out, int n_in, int T, const float *x, size_t xs,
                          float *y, size_t ys, float scale) {
    ink2_rt *r = W->dev;
    if (!r->gpu) r->gpu_err = true;   // never reached while views are marked all or none; refused if it were
    if (r->gpu_err) return -1;
    if (r->no_factor && W->q_scale) scale = 1.0f;
    float *xc = (float *)x, *yc = y;
    if (xs != (size_t)n_in) {
        xc = malloc(sizeof(float) * (size_t)T * (size_t)n_in);
        if (!xc) { r->gpu_err = true; return -1; }
        for (int t = 0; t < T; t++) memcpy(xc + (size_t)t * n_in, x + (size_t)t * xs, sizeof(float) * (size_t)n_in);
    }
    if (ys != (size_t)n_out) {
        yc = malloc(sizeof(float) * (size_t)T * (size_t)n_out);
        if (!yc) { if (xc != x) free(xc); r->gpu_err = true; return -1; }
    }
    const bool ok = r->n_lin++ != r->lin_fail_at &&
                    gpu_ink2_lin(r->gpu, W->dev_i, W->qtype, W->q_row_bytes, row0, n_out, n_in, T, xc, yc, scale);
    if (ok && yc != y)
        for (int t = 0; t < T; t++) memcpy(y + (size_t)t * ys, yc + (size_t)t * n_out, sizeof(float) * (size_t)n_out);
    if (xc != x) free(xc);
    if (yc != y) free(yc);
    if (!ok) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: a product of %s failed on the device\n", W->name);
        r->gpu_err = true;
        return -1;
    }
    return 0;
}
// G2b step 1: one application's expert body on the device (gpu_ink2_experts), the host router's choice handed over
static int dev_experts(const ink2_shape *s, const ink2_tensor *t13, const ink2_tensor *t2, int T, const float *x,
                       const unsigned char *chosen, const float *wt, float *out) {
    ink2_rt *r = t13->dev;
    if (!r->gpu || t2->dev != r) r->gpu_err = true;
    if (r->gpu_err) return -1;
    if (!r->exp_on) return 1;   // this instance keeps the per-product body
    const bool inj = r->n_exp++ == r->exp_fail_at;   // TEST-ONLY: an injected failure inside the expert call
    const bool nf = r->no_factor;   // TEST-ONLY: the control leaves the expert factors out here too
    const bool ok = !inj && gpu_ink2_experts(r->gpu, t13->dev_i, t2->dev_i, t13->qtype, t2->qtype, t13->q_row_bytes,
                                     t2->q_row_bytes, s->hidden, s->ff, s->n_expert, T, x, chosen, wt,
                                     nf ? NULL : t13->q_scale, nf ? NULL : t2->q_scale, s->gate_cap, s->up_cap, out);
    if (!ok) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: the expert body of %s failed on the device\n", t13->name);
        r->gpu_err = true;
        return -1;
    }
    return 0;
}
// D1b: one application's expert layer whole on the device, the router then the body, when the router's F32 view and
// both stacks are this instance's (else the host router and dev_experts).
// D1b.1: ONLY FOR T > 1. At T == 1 (a decode step) the host router runs and dev_experts takes the body, as before D1b:
// the device router's fixed costs (two kernels, the tables' launch, the uploads, one sync) measured 0.114 ms a call
// against the host router's ~0.05-0.07 at one position, while from a chunk of positions up the device router wins
// (prefill +11%). The two routers give the same bits (B1, B2, B3), so the split changes no row.
static int dev_moe(void *dev, const ink2_moe_call *m) {
    ink2_rt *r = dev;
    if (!r->route_on || !r->exp_on || m->T == 1) return 1;
    if (r->gpu_err) return -1;
    if (!m->wr || m->wr->dev != r || m->wr->qtype != T_F32 || !m->t13 || m->t13->dev != r || !m->t2 || m->t2->dev != r)
        return 1;
    const ink2_shape *s = m->s;
    r->n_route++;
    const bool inj = r->n_exp++ == r->exp_fail_at;   // TEST-ONLY: the expert call's injected failure, as dev_experts'
    const bool nf = r->no_factor;
    const bool ok = !inj && gpu_ink2_route_experts(r->gpu, m->wr->dev_i, m->br, s->route_scale, s->n_used, m->t13->dev_i,
                                                   m->t2->dev_i, m->t13->qtype, m->t2->qtype, m->t13->q_row_bytes,
                                                   m->t2->q_row_bytes, s->hidden, s->ff, s->n_expert, m->T, m->x,
                                                   nf ? NULL : m->t13->q_scale, nf ? NULL : m->t2->q_scale, s->gate_cap,
                                                   s->up_cap, m->out, m->chosen, m->margin);
    if (!ok) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: the expert layer of %s failed on the device\n", m->t13->name);
        r->gpu_err = true;
        return -1;
    }
    return 0;
}
// G2b step 2a: one application's attention core on the device, over this instance's ring
static int dev_attn(void *dev, int app, const ink2_attn_dims *d, int T, const int *pos, const float *q,
                    const float *k, const float *v, const float *rl, const float *sk, float *att) {
    ink2_rt *r = dev;
    if (!r->attn_on) return 1;   // this instance keeps the host core (the hook is process-wide)
    if (!r->gpu || !r->kv) r->gpu_err = true;
    if (r->gpu_err) return -1;
    const bool inj = r->n_attn++ == r->attn_fail_at;   // TEST-ONLY: an injected failure inside the core's call
    const bool ok = !inj && gpu_ink2_attn(r->gpu, r->kv, app, T, pos, q, k, v, rl, sk, att, d->nh, d->nkv, d->hd,
                                          d->window, d->global, d->ext, d->scale);
    if (!ok) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: the attention core of application %d failed on the device\n", app);
        r->gpu_err = true;
        return -1;
    }
    return 0;
}
// G2b step 2b: one application's whole attention block on the device (gpu_ink2_attn_block). A sequence that begins
// with a LoRA adapter active runs 2a's path to its end (the adapter's terms are host code, and its K/V histories are
// then the host's); an adapter appearing in the middle of a sequence the device began is refused, since the host
// holds no history for it.
// D1a: one message step on the device when every view is this instance's (a view left on the host: the host path)
static int dev_msg(void *dev, const ink2_msg_call *m) {
    ink2_rt *r = dev;
    if (!r->msg_on) return 1;
    if (r->gpu_err) return -1;
    gpu_ink2_msg_role ro[3];
    for (int k = 0; k < m->n_roles; k++) {
        const ink2_tensor *t[3] = { m->wd[k], m->wf[k], m->wg[k] };
        for (int u = 0; u < 3; u++) if (!t[u] || t[u]->dev != r) return 1;
        ro[k] = (gpu_ink2_msg_role){ t[0]->dev_i, t[1]->dev_i, t[2]->dev_i, t[0]->qtype, t[1]->qtype, t[2]->qtype,
                                     1.0f, 1.0f, 1.0f };
    }
    const bool inj = r->n_msg_calls++ == r->msg_fail_at;   // TEST-ONLY: an injected failure
    if (inj || !gpu_ink2_msg_steps(r->gpu, m->n_roles, m->T, m->H, m->r, m->eps, ro, m->xs, m->msgs, m->msg_stride, m->outs)) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: a message step failed on the device\n");
        r->gpu_err = true;
        return -1;
    }
    return 0;
}
// the attention block's device descriptor from the engine's block call (its views and small tensors this instance's
// device tensors); false: one is not
static bool block_desc(ink2_rt *r, const ink2_block_call *bc, gpu_ink2_block *gb) {
    *gb = (gpu_ink2_block){ bc->H, bc->nh, bc->nkv, bc->hd, bc->K, bc->dr, bc->ext, bc->window, bc->global, bc->scale,
                            bc->qk_clip, bc->rel_clip, bc->eps, bc->log_alpha, bc->log_floor, { 0 }, { 0 }, { 0 }, { 0 } };
    for (int k = 0; k < GI2_NV; k++) {
        const ink2_tensor *t = bc->v[k];
        if (!t) { gb->vi[k] = -1; gb->vtype[k] = -1; continue; }
        if (t->dev != r) return false;
        gb->vi[k] = t->dev_i; gb->vtype[k] = t->qtype; gb->vrb[k] = t->q_row_bytes;
    }
    for (int k = 0; k < GI2_NF; k++) {
        const ink2_tensor *t = bc->f[k];
        if (!t || t->dev != r) return false;
        gb->fi[k] = t->dev_i;
    }
    return true;
}
static int dev_block(void *dev, const ink2_block_call *bc) {
    ink2_rt *r = dev;
    if (!r->block_on) return 1;
    if (r->gpu_err) return -1;
    const int p0 = bc->pos[0];
    if (bc->lora) {
        if (p0 > 0 && r->seq_block) {
            fprintf(stderr, "error: inkling2_loop: a LoRA adapter in the middle of a sequence its device began; start it "
                            "again from position 0\n");
            return -2;
        }
        r->seq_block = false;
        return 1;
    }
    if (p0 > 0 && !r->seq_block) return 1;   // a sequence begun on 2a's path continues there
    r->seq_block = true;
    gpu_ink2_block gb;
    if (!block_desc(r, bc, &gb)) { r->gpu_err = true; return -1; }   // block_on said every view is there
    r->n_block_calls++;
    const bool inj = r->n_attn++ == r->attn_fail_at;   // TEST-ONLY: the same injected failure as the core's
    const bool ok = !inj && gpu_ink2_attn_block(r->gpu, r->kv, bc->app, &gb, bc->T, bc->pos, bc->xq, bc->xk, bc->xv,
                                                bc->out);
    if (!ok) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: the attention block of application %d failed on the device\n",
                                 bc->app);
        r->gpu_err = true;
        return -1;
    }
    return 0;
}
// an application's K/V row width (its ring's rows are the host cache's: n_ctx on a full layer, window - 1 + the chunk
// on a sliding one)
static int app_c(const ink2_shape *s, int a) {
    return ink2_app_global(s, a) ? s->n_kv * s->head_dim : s->swa_n_kv * s->swa_head_dim;
}
// G2b step 2's B1: ONE device copy of the weights per file and placement per process, shared by every instance that
// asks for the same (the server loads each of its --parallel slots as its own instance). The key is the file's
// identity (path, size, index, modification time) and the placement's inputs (the --cpu-moe request, the device, -c,
// -b, the slot count, and the test-only switches that change what goes up), so two instances with different
// placements never share. The entry holds the fit's decision for all of them ('W' every view the device computes,
// 'M' the expert stacks on the host, 'C' the CPU): never a mixed server. Attach and release under g_share_mu; the
// last release (model_inkling2_free) frees the copy.
typedef struct ink2_share {
    char key[1200];
    gpu_ink2 *gpu;
    char placed;
    size_t need;
    bool exp_dev;
    bool attn_dev;   // step 2a: each instance's attention core (and its K/V rings) on the device
    size_t fit_free; // the free memory the deciding fit read (after its context, module and buffers; test-only cap)
    bool block_dev;  // step 2b: the whole attention block; its small tensors follow the views in the copy
    int n_views;     // the views in the copy (the small tensors are device tensors n_views..)
    int step_base;   // D1d.3b: the resident step's small tensors are device tensors step_base.. (-1: none up)
    int refs;
    // 'C': the sentence each instance says. Sized for the longest: share_place's do-not-fit sentence, 182 fixed bytes
    // with a device name of up to 159 (cuda_bind's dev[160]) and its numbers, at most 363 bytes with the NUL
    char why[512];
    struct ink2_share *next;
} ink2_share;
static pthread_mutex_t g_share_mu = PTHREAD_MUTEX_INITIALIZER;
static ink2_share *g_shares;
// the device bytes ONE instance holds beyond the shared weights: its K/V rings (step 2a: every application's ring at
// the host cache's rows, K and V, at the allocation granularity), when the core runs on the device. The products',
// the expert body's and the core's scratch belong to the shared copy (used under its lock).
// one slot's own device bytes: per application its K and V rings and, on the block path (hist_rows = conv_k - 1),
// its K and V convolution histories, each as the driver allocates it
static size_t dev_instance_bytes(const ink2_rt *r, bool attn_dev, int hist_rows, bool step);
// D1d.3b: the resident step's bytes on the COPY: its small F32 tensors (step_smalls' list) and its program's scratch
// (gpu_ink2_step_set's allocations: the raw rows, the normed rows, the block's inputs, the reads' rows, the logits, the
// lists), each as the driver allocates it; 0 when the plan cannot be built (the step then stays off)
static size_t step_dev_bytes(const ink2_rt *r);
// R4.48.8.5: one slot's device layout, from the shape alone, as gpu_ink2_kv_new and gpu_ink2_kv_post_init allocate
// it: the K/V rings in the cache's kind, the full applications' each its own allocation, the sliding ones packed into
// one allocation per side at GPU_INK2_RING_ALIGN-byte offsets; on the block path each application's two convolution
// histories; with the step, the post convolutions' copies (one allocation). Each allocation's bytes go through
// `round` once: the driver's rounding (dev_alloc_bytes) for what the slot takes on the device, which the load's fit
// (share_place) and --info fit count, or none for the bytes asked of the driver, which the slot's own count holds.
static size_t no_rounding(size_t b) { return b; }
static uint64_t dev_slot_layout(const ink2_shape *s, int n_ctx, int max_chunk, int kind, bool attn_dev, int hist_rows,
                                bool step, size_t (*round)(size_t)) {
    uint64_t b = step && hist_rows > 0 ? round(sizeof(float) * (size_t)s->n_app * 2 * (size_t)hist_rows * (size_t)s->hidden) : 0;
    if (!attn_dev) return b;
    size_t packed = 0;
    for (int a = 0; a < s->n_app; a++) {
        const size_t C = (size_t)ink2_app_kv_width(s, a);
        const size_t ring = ink2_kv_row_bytes((int)C, kind) * (size_t)ink2_kv_app_rows(s, a, n_ctx, max_chunk);
        if (ink2_app_global(s, a)) b += 2 * round(ring);
        else packed = (packed + GPU_INK2_RING_ALIGN - 1) / GPU_INK2_RING_ALIGN * GPU_INK2_RING_ALIGN + ring;
        if (hist_rows > 0) b += 2 * round(sizeof(float) * (size_t)hist_rows * C);
    }
    if (packed) b += 2 * round(packed);
    return b;
}
uint64_t model_inkling2_dev_slot_bytes(const ink2_shape *s, int n_ctx, int max_chunk, int kind, bool attn_dev,
                                       int hist_rows, bool step) {
    return dev_slot_layout(s, n_ctx, max_chunk, kind, attn_dev, hist_rows, step, dev_alloc_bytes);
}
uint64_t model_inkling2_dev_slot_alloc_bytes(const ink2_shape *s, int n_ctx, int max_chunk, int kind, bool attn_dev,
                                             int hist_rows, bool step) {
    return dev_slot_layout(s, n_ctx, max_chunk, kind, attn_dev, hist_rows, step, no_rounding);
}
static size_t dev_instance_bytes(const ink2_rt *r, bool attn_dev, int hist_rows, bool step) {
    return model_inkling2_dev_slot_bytes(&r->s, r->c.n_ctx, r->c.max_chunk, r->c.kv == INK2_KV_Q8 ? INK2_KV_Q8 : INK2_KV_F32,
                                         attn_dev, hist_rows, step);
}
static void share_release(ink2_rt *r) {
    if (!r->share) return;
    pthread_mutex_lock(&g_share_mu);
    ink2_share *sh = r->share;
    if (--sh->refs == 0) {
        for (ink2_share **pp = &g_shares; *pp; pp = &(*pp)->next)
            if (*pp == sh) { *pp = sh->next; break; }
        gpu_ink2_free(sh->gpu);
        free(sh);
    }
    pthread_mutex_unlock(&g_share_mu);
    r->share = NULL;
    r->gpu = NULL;
}
// G2b step 2b: each layer's small F32 attention tensors, by role (gpu.h's GI2_QS..GI2_RP), in the order the block
// needs them on the device: layer by layer, role by role (the order an attaching instance finds them in)
static const ink2_tensor *small_get(ink2_rt *r, int b, int role) {
    static const char *const nm_s[GI2_NF] = { "q_proj.norm.scale", "k_proj.sconv.conv1d.weight", "v_proj.sconv.conv1d.weight",
                                              "relative_position_bias.norm.scale", "relative_position_bias.proj" };
    static const char *const nm_g[GI2_NF] = { "q_proj.norm.scale", "k_proj.sconv.static_weight", "v_proj.sconv.static_weight",
                                              "relative_position_bias.norm.scale", "relative_position_bias.proj" };
    char name[192];
    snprintf(name, sizeof name, "model.layers.%d.self_attn.%s", b, r->s.block_global[b] ? nm_g[role] : nm_s[role]);
    const ink2_tensor *t = ink2g_get(r->g, name);
    return t && t->data ? t : NULL;
}
static size_t small_bytes(const ink2_tensor *t) { return sizeof(float) * (size_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]); }
// the block's large views by role (gpu.h's GI2_Q..GI2_O; the dynamic taps only on a full layer), for the bind's
// check that a layer's whole block is on the device
static const ink2_tensor *block_view(ink2_rt *r, int b, int role) {
    static const char *const nm[GI2_NV] = { "q_proj.linear.weight", "k_proj.linear.weight", "v_proj.linear.weight",
                                            "k_proj.sconv.weight_proj.weight", "v_proj.sconv.weight_proj.weight",
                                            "relative_position_bias.linear.weight", "sink_proj.weight",
                                            "o_proj.gate_proj.weight", "o_proj.linear.weight" };
    if ((role == GI2_KD || role == GI2_VD) && !r->s.block_global[b]) return NULL;
    char name[192];
    snprintf(name, sizeof name, "model.layers.%d.self_attn.%s", b, nm[role]);
    return ink2g_get(r->g, name);
}
// the fit and the upload for a new entry (under g_share_mu): the context, the module and the products' buffers first,
// so the free memory read counts them; then the weights the placement puts up, counted as the driver allocates them,
// plus N slots' own bytes, against the free memory with 512 MiB kept. Whole first (unless --cpu-moe was asked), then
// the expert stacks on the host, then the CPU.
// R4.48.8.5: a placement's terms, as share_place checks them against the free memory it reads after its context,
// module and reservations: the weights it would put up, the small tensors and the step's, every slot's own bytes, the
// 512 MiB kept, the reservations, and the device's free and total memory; placed 'W' whole, 'M' experts on the host,
// 'C' the CPU. A dry run (--info fit) records them and uploads nothing.
typedef struct {
    char placed;
    size_t weights, small, slots, kept, reserved, free_b, total_b;   // the last placement tried
    // the whole placement's own terms (every weight on the device), kept when a later placement is tried: what a
    // device that holds it would need besides the weights, whether or not this one does
    bool whole; size_t w_weights, w_small, w_slots, w_reserved;
} ink2_dev_terms;
static void share_place(ink2_share *sh, ink2_rt *r, bool cpu_moe_req, int n_batch, int n_ctx, int n_slots,
                        const char *dev, bool step_req, ink2_dev_terms *dry) {
    sh->placed = 'C';
    gpu_ink2 *h = gpu_ink2_bind();
    if (!h) { snprintf(sh->why, sizeof sh->why, "no CUDA device for inkling2_loop"); return; }
    r->gpu = h;   // dev_wanted asks the handle which types its products take
    const size_t T = n_batch > 0 ? (size_t)n_batch : 1, RES = 512ull << 20;
    // step 2a: the attention core on the device when the module has it (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_ATTN=1
    // keeps it on the host, a baseline); then every slot holds its own K/V rings, counted N times
    const char *na = getenv("RUNNER_INKLING2_CUDA_TEST_NO_ATTN");
    const bool attn_dev = gpu_ink2_attn_ok(h) && !(na && !strcmp(na, "1"));
    const ink2_shape *sp = &r->s;
    const int q_max = sp->n_head * sp->head_dim > sp->swa_n_head * sp->swa_head_dim ? sp->n_head * sp->head_dim
                                                                                       : sp->swa_n_head * sp->swa_head_dim;
    const int nh_max = sp->n_head > sp->swa_n_head ? sp->n_head : sp->swa_n_head;
    const int ext_max = sp->rel_extent > sp->window ? sp->rel_extent : sp->window;
    // step 2b: the whole attention block on the device when the module has its kernels (TEST-ONLY
    // RUNNER_INKLING2_CUDA_TEST_NO_BLOCK=1 keeps 2a's path, a baseline); its small F32 tensors then go up too. A
    // convolution of one tap keeps no history (the block's kernels want conv_k - 1 >= 1 rows): 2a's path
    const char *nb = getenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK");
    bool block_dev = attn_dev && sp->conv_k >= 2 && gpu_ink2_block_ok(h) && !(nb && !strcmp(nb, "1"));
    size_t small = 0;
    for (int b = 0; block_dev && b < sp->n_block; b++)
        for (int k = 0; k < GI2_NF; k++) {
            const ink2_tensor *t = small_get(r, b, k);
            if (!t) { block_dev = false; break; }
            small += dev_alloc_bytes(small_bytes(t));
        }
    if (!block_dev) small = 0;
    // D1d.3b: the resident step's small tensors and program on the copy, and per slot its post convolutions' copies
    const size_t step_b = step_req && block_dev ? step_dev_bytes(r) : 0;
    const bool step_fit = step_b > 0;
    small += step_b;
    // every slot's own rings (and on the block path its convolution histories), counted N times
    const size_t inst = (size_t)n_slots * dev_instance_bytes(r, attn_dev, block_dev ? sp->conv_k - 1 : 0, step_fit);
    // the block's buffers at a chunk of T, by the largest layer's widths
    const int c_max = sp->n_kv * sp->head_dim > sp->swa_n_kv * sp->swa_head_dim ? sp->n_kv * sp->head_dim
                                                                                   : sp->swa_n_kv * sp->swa_head_dim;
    const int nkv_max = sp->n_kv > sp->swa_n_kv ? sp->n_kv : sp->swa_n_kv;
    const char *ne = getenv("RUNNER_INKLING2_CUDA_TEST_NO_EXP");
    const char *capv = getenv("RUNNER_INKLING2_CUDA_TEST_FREE");   // TEST-ONLY: the free figure the fit reads, capped
    const char *pf = getenv("RUNNER_INKLING2_CUDA_TEST_PUT_FAIL");
    size_t need_w = 0, fr = 0;
    for (int k = cpu_moe_req ? 1 : 0; k < 2; k++) {
        const bool cm = k == 1;
        const bool exp_dev = !cm && gpu_ink2_experts_ok(h) && !(ne && !strcmp(ne, "1"));
        dev_walk c = { r, cm, 0, 0, -1, 0, 0, 0, false };
        size_t tot = 0;
        // the output buffer holds T rows of every product but the head, and at least one head row: a forward asks for
        // its last row only (INK2_LOGITS_LAST); a keep-all forward's head runs its columns in groups (gpu_ink2_lin)
        size_t y_rows = 0;
        if (!ink2g_each_view(r->g, dev_count, &c) && c.n > 0) {
            y_rows = T * (size_t)c.max_out_body;
            if ((size_t)c.max_out > y_rows) y_rows = (size_t)c.max_out;
        }
        if (c.n == 0 || y_rows == 0 ||
            !gpu_ink2_reserve(h, sizeof(float) * (T + 16) * (size_t)c.max_in, sizeof(float) * y_rows) ||   // + the
                                                                     // 16 columns of padding the batched BF16 product reads
            (exp_dev && !gpu_ink2_reserve_experts(h, (int)T, r->s.hidden, r->s.ff, r->s.n_expert, r->s.n_used)) ||
            (attn_dev && !gpu_ink2_reserve_attn(h, (int)T, q_max, nh_max, ext_max, n_ctx)) ||
            (block_dev && !gpu_ink2_reserve_block(h, (int)T, sp->hidden, c_max, nkv_max * sp->conv_k,
                                                  nh_max * sp->d_rel, q_max)) ||
            !gpu_mem_info(&fr, &tot)) {
            snprintf(sh->why, sizeof sh->why, "inkling2_loop has no weights the device computes, or no room for its "
                     "products' buffers, on %s", dev);
            break;
        }
        if (capv && *capv) { const size_t cv = (size_t)strtoull(capv, NULL, 10); if (cv < fr) fr = cv; }
        if (!cm) need_w = c.need;
        sh->fit_free = fr;
        if (dry) {   // field by field: a later placement's terms must not wipe the whole placement's
            dry->placed = 'C'; dry->weights = c.need; dry->small = small; dry->slots = inst; dry->kept = RES;
            dry->reserved = gpu_ink2_reserved_bytes(h); dry->free_b = fr; dry->total_b = tot;
            if (!cm) { dry->whole = true; dry->w_weights = c.need; dry->w_small = small; dry->w_slots = inst; dry->w_reserved = dry->reserved; }
        }
        if (dry && c.need + small + inst + RES <= fr) {   // the dry run: this placement fits; nothing goes up
            dry->placed = cm ? 'M' : 'W';
            sh->placed = dry->placed;
            gpu_ink2_free(h);
            r->gpu = NULL;
            return;
        }
        if (c.need + small + inst + RES > fr) {   // this placement does not fit: the next, or the CPU
            // the whole-device attempt's expert-body buffers go before the retry, so it reads the memory it has
            if (exp_dev) gpu_ink2_release_experts(h);
            continue;
        }
        // fits: up it goes (the views in the file's walk order; a failed put leaves the CPU, P0)
        r->dev_of = calloc((size_t)c.n, sizeof *r->dev_of);
        r->n_dev = 0;
        dev_walk u = { r, cm, 0, c.n, pf && *pf ? atoi(pf) : -1, 0, 0, 0, false };
        if (!r->dev_of || ink2g_each_view(r->g, dev_put, &u) || r->n_dev != c.n) {
            snprintf(sh->why, sizeof sh->why, "inkling2_loop's weights could not be put on %s", dev);
            free(r->dev_of); r->dev_of = NULL; r->n_dev = 0;
            break;
        }
        bool small_ok = true;   // the block's small tensors after the views, layer by layer, role by role
        for (int b = 0; block_dev && small_ok && b < sp->n_block; b++)
            for (int role = 0; small_ok && role < GI2_NF; role++) {
                const ink2_tensor *t = small_get(r, b, role);
                small_ok = t && gpu_ink2_put(h, t->data, small_bytes(t), NULL, 0) == c.n + b * GI2_NF + role;
            }
        if (!small_ok) {
            snprintf(sh->why, sizeof sh->why, "inkling2_loop's attention tensors could not be put on %s", dev);
            free(r->dev_of); r->dev_of = NULL; r->n_dev = 0;
            break;
        }
        sh->gpu = h; sh->placed = cm ? 'M' : 'W'; sh->need = c.need + small; sh->exp_dev = exp_dev; sh->attn_dev = attn_dev;
        sh->block_dev = block_dev; sh->n_views = c.n;
        if (cm && !cpu_moe_req)
            fprintf(stderr, "note: inkling2_loop's %.2f GB of weights do not fit the %.2f GB free on %s for %d slot%s; "
                            "its expert stacks stay on the host (as --cpu-moe)\n", need_w / 1e9, fr / 1e9, dev, n_slots,
                    n_slots > 1 ? "s" : "");
        r->gpu = NULL;
        return;
    }
    if (!sh->why[0])
        snprintf(sh->why, sizeof sh->why, "inkling2_loop's weights for the device do not fit the %.2f GB free on %s "
                 "for %d slot%s (after its context, module and buffers, with 0.54 GB kept), not with its expert stacks "
                 "on the host either", fr / 1e9, dev, n_slots, n_slots > 1 ? "s" : "");
    gpu_ink2_free(h);
    r->gpu = NULL;
}
// The weights the device computes on it (their stored rows), shared with the file's other instances of the same
// placement; or a sentence why not and the CPU route for every such instance. 1: on CUDA; 0: on the CPU; -1: this
// instance refuses its load (it could not join a copy its placement holds: never a CPU slot beside CUDA ones).
// ---- R4.48.9 D1d.3b: the resident decode step's model side ----
// the step's small F32 tensors in their device order: the globals (embed_norm, final_input.query, model.norm, and on a
// loop the message_input.query), then per block at its first application: its input norms (3 full, 1 sliding),
// attention_input.query (block > 0), mlp_input.query, post_attention_layernorm, the two post convolutions' conv1d, the
// router's bias. out NULL: count only; blk_at[b] (when given) the block's first index in that order (-1: no
// application). Returns the count, -1 when a tensor has no float data.
typedef struct { const float *src; size_t n; } step_small;
static int step_smalls(const ink2_rt *r, const ink2_step_plan *P, step_small *out, int *blk_at) {
    const int H = r->s.hidden;
    int k = 0;
    bool ok = true;
#define ADD(p_, n_) do { const float *p__ = (p_); if (!p__) ok = false; if (out) { out[k].src = p__; out[k].n = (n_); } k++; } while (0)
    ADD(P->embed_norm->data, (size_t)H);
    ADD(P->final_q->data, (size_t)H);
    ADD(P->norm->data, (size_t)H);
    if (P->msg_q) ADD(P->msg_q->data, (size_t)4 * H * (size_t)(r->s.loop_end - r->s.loop_start));
    for (int b = 0; b < r->s.n_block; b++) {
        int a = 0;
        while (a < P->n_app && P->app[a].block != b) a++;
        if (blk_at) blk_at[b] = a < P->n_app ? k : -1;
        if (a == P->n_app) continue;
        const ink2_step_app *A = &P->app[a];
        for (int rr = 0; rr < (A->global ? 3 : 1); rr++) ADD(A->norm[rr]->data, (size_t)H);
        if (A->read_q) ADD(A->read_q->data, (size_t)A->n_read * H);
        ADD(A->mlp_q->data, (size_t)H);
        ADD(A->post_norm->data, (size_t)H);
        ADD(A->conv[0]->data, (size_t)H * (size_t)r->s.conv_k);
        ADD(A->conv[1]->data, (size_t)H * (size_t)r->s.conv_k);
        ADD(A->br, (size_t)r->s.n_expert);
    }
#undef ADD
    return ok ? k : -1;
}
static size_t step_dev_bytes(const ink2_rt *r) {
    ink2_step_app *apps = calloc((size_t)r->s.n_app, sizeof *apps);
    ink2_step_plan P = { 0 };
    P.app = apps;
    size_t b = 0;
    if (apps && ink2_step_plan_build(&r->s, &r->w, &P) == INK2_OK) {
        const int n = step_smalls(r, &P, NULL, NULL);
        step_small *ss = n > 0 ? malloc(sizeof *ss * (size_t)n) : NULL;
        if (ss && step_smalls(r, &P, ss, NULL) == n) {
            for (int k = 0; k < n; k++) b += dev_alloc_bytes(sizeof(float) * ss[k].n);
            // the program's own allocations, as gpu_ink2_step_set makes them, EACH rounded by the driver's rule (the
            // stage-3 read's F1: a flat allowance holds only while every allocation is under the rounding unit)
            size_t sz[16];
            const int na = gpu_ink2_step_sizes(r->s.n_app, r->s.hidden, r->s.vocab_used, sz);
            for (int k = 0; k < na; k++) b += dev_alloc_bytes(sz[k]);
        }
        free(ss);
    }
    free(apps);
    return b;
}
// the step's small tensors up, as device tensors base.. (the copy new, under g_share_mu: no instance reads h->t yet)
static bool step_put(ink2_rt *r, gpu_ink2 *h, int base) {
    ink2_step_app *apps = calloc((size_t)r->s.n_app, sizeof *apps);
    ink2_step_plan P = { 0 };
    P.app = apps;
    bool ok = apps && ink2_step_plan_build(&r->s, &r->w, &P) == INK2_OK;
    const int n = ok ? step_smalls(r, &P, NULL, NULL) : -1;
    step_small *ss = n > 0 ? malloc(sizeof *ss * (size_t)n) : NULL;
    ok = ok && ss && step_smalls(r, &P, ss, NULL) == n;
    for (int k = 0; ok && k < n; k++) ok = gpu_ink2_put(h, ss[k].src, sizeof(float) * ss[k].n, NULL, 0) == base + k;
    free(ss); free(apps);
    return ok;
}
// this instance's device plan: the engine's plan by device index; false (the step declined) when any tensor it needs
// is not this instance's on the device, or a stack's type has no bounded kernel
static bool step_plan_make(ink2_rt *r, int base, gpu_ink2_step_plan *G, gpu_ink2_step_app *GA) {
    const ink2_shape *s = &r->s;
    ink2_step_app *apps = calloc((size_t)s->n_app, sizeof *apps);
    int *blk_at = malloc(sizeof(int) * (size_t)s->n_block);
    ink2_step_plan P = { 0 };
    P.app = apps;
    bool ok = apps && blk_at && ink2_step_plan_build(s, &r->w, &P) == INK2_OK && step_smalls(r, &P, NULL, blk_at) > 0;
    ok = ok && P.head->dev == r && !r->no_factor;
    const int nglob = P.msg_q ? 4 : 3;
    for (int a = 0; ok && a < s->n_app; a++) {
        const ink2_step_app *A = &P.app[a];
        gpu_ink2_step_app *g = &GA[a];
        memset(g, 0, sizeof *g);
        int i = base + blk_at[A->block];
        g->block = A->block; g->global = A->global; g->loop = A->loop; g->pass = s->app_pass[a];
        g->boundary = A->boundary; g->n_read = A->n_read;
        g->i_norm[0] = g->i_norm[1] = g->i_norm[2] = -1;
        for (int rr = 0; rr < (A->global ? 3 : 1); rr++) g->i_norm[rr] = i++;
        g->i_read_q = A->read_q ? i++ : -1;
        g->i_mlp_q = i++; g->i_post_norm = i++; g->i_conv[0] = i++; g->i_conv[1] = i++; g->i_br = i++;
        for (int rr = 0; A->loop && rr < 4; rr++) {
            const ink2_tensor *t[3] = { A->gate[rr][0], A->gate[rr][1], A->gate[rr][2] };
            for (int u = 0; u < 3; u++) if (!t[u] || t[u]->dev != r || !t[u]->q || t[u]->q_act8) ok = false;
            if (ok) g->gate[rr] = (gpu_ink2_msg_role){ t[0]->dev_i, t[1]->dev_i, t[2]->dev_i, t[0]->qtype, t[1]->qtype,
                                                       t[2]->qtype, 1.0f, 1.0f, 1.0f };
        }
        ok = ok && block_desc(r, &A->blk, &g->blk);
        ok = ok && A->router->dev == r && A->router->qtype == T_F32 && A->t13->dev == r && A->t2->dev == r &&
             (A->t13->qtype == T_NVFP4 || A->t13->qtype == T_BF16) && (A->t2->qtype == T_NVFP4 || A->t2->qtype == T_BF16);
        if (ok) {
            g->ir = A->router->dev_i; g->i13 = A->t13->dev_i; g->i2 = A->t2->dev_i;
            g->t13 = A->t13->qtype; g->t2 = A->t2->qtype; g->rb13 = A->t13->q_row_bytes; g->rb2 = A->t2->q_row_bytes;
            g->f13 = A->t13->q_scale != NULL; g->f2 = A->t2->q_scale != NULL;
        }
    }
    if (ok) {
        *G = (gpu_ink2_step_plan){ s->n_app, s->hidden, s->vocab_used, s->n_expert, s->n_used, s->ff, s->conv_k,
                                   s->loop_start, s->loop_end, s->loop_gate_rank, s->eps, s->route_scale, s->gate_cap,
                                   s->up_cap, s->final_cap, ink2_inv_sqrt(s->hidden), base, base + 1, base + 2,
                                   P.msg_q ? base + 3 : -1, P.head->dev_i, P.head->qtype, P.head->q_row_bytes, GA };
        (void)nglob;
    }
    free(apps); free(blk_at);
    return ok;
}
// the post convolutions' copies: up (the host's to the device, after a forward the step did not take) or down (the
// device's to the host, before one, when a step ran last); one transfer each way
static bool post_sync(ink2_rt *r, int up) {
    const int K1 = r->s.conv_k - 1, H = r->s.hidden;
    const size_t one = (size_t)K1 * (size_t)H;
    if (!r->post_stage || !r->kv) return false;
    if (!up && !gpu_ink2_kv_post_sync(r->kv, r->post_stage, 0)) return false;
    for (int a = 0; a < r->s.n_app; a++)
        for (int k = 0; k < 2; k++) {
            float *hs = ink2_cache_post_hist(&r->s, &r->c, a, k), *st = r->post_stage + ((size_t)a * 2 + (size_t)k) * one;
            if (!hs) return false;
            if (up) memcpy(st, hs, sizeof(float) * one); else memcpy(hs, st, sizeof(float) * one);
        }
    return !up || gpu_ink2_kv_post_sync(r->kv, r->post_stage, 1);
}
// the engine's step op: declines (1) unless this instance's step is on and its sequence's blocks are the device's; a
// decline with the device's histories newer brings them down first (the forward that runs instead reads the host's)
static int dev_step(void *dev, int p0, const float *emb_row, float *logits) {
    ink2_rt *r = dev;
    if (!r->step_on) return 1;
    if (r->gpu_err) return -1;
    if (p0 < 0 || (p0 > 0 && !r->seq_block)) {   // the prepare call, or a sequence begun on 2a's path: the host's turn
        if (r->post_dev_newer) {
            if (!post_sync(r, 0)) { r->gpu_err = true; return -1; }
            r->post_dev_newer = false;
        }
        return 1;
    }
    r->seq_block = true;
    const bool inj = r->n_step_try++ == r->step_fail_at;   // TEST-ONLY: an injected failure, as the other device ops'
    if (inj || !gpu_ink2_step(r->gpu, r->kv, p0, emb_row, logits)) {
        if (!r->gpu_err) fprintf(stderr, "error: inkling2_loop: the decode step at position %d failed on the device\n", p0);
        r->gpu_err = true;
        return -1;
    }
    r->post_dev_newer = true;
    r->last_was_step = true;
    r->n_step++;
    return 0;
}
static int cuda_bind(ink2_rt *r, model_t *m, const model_params *p) {
    if (r->c.kv == INK2_KV_RT) {   // TEST-ONLY: the round trip is the CPU routes' reference, never a device ring's
        fprintf(stderr, "note: RUNNER_INKLING2_TEST_KV_RT is the CPU routes' alone; running on the CPU\n");
        return 0;
    }
    // The fast CUDA route remains opt-in, including a resident decode step when placement permits;
    // a default GPU-allowed load still uses the CPU.
    const char *on = getenv("RUNNER_INKLING2_CUDA");
    if (!on || strcmp(on, "1")) {
        fprintf(stderr, "note: inkling2_loop runs on the CPU by default; RUNNER_INKLING2_CUDA=1 opts the fast route "
                        "into CUDA, including a resident decode step when placement permits\n");
        return 0;
    }
    if (r->route != 'F') {
        snprintf(m->route_fallback, sizeof m->route_fallback, "reference route runs on CPU");
        fprintf(stderr, "note: inkling2_loop runs on CUDA from the fast route's stored rows only; the reference route "
                        "runs on the CPU\n");
        return 0;
    }
    const char *a8 = getenv("RUNNER_INKLING2_ACT8");
    if (a8 && *a8 && strcmp(a8, "0")) {
        snprintf(m->route_fallback, sizeof m->route_fallback, "RUNNER_INKLING2_ACT8 runs on CPU");
        fprintf(stderr, "note: the 8-bit control (RUNNER_INKLING2_ACT8) is the CPU's; running on the CPU\n");
        return 0;
    }
    char dev[160] = "";
    if (!gpu_available(dev, sizeof dev)) {
        snprintf(m->route_fallback, sizeof m->route_fallback, "no CUDA device for inkling2_loop");
        fprintf(stderr, "note: no CUDA device for inkling2_loop; running on the CPU\n");
        return 0;
    }
    const int n_slots = p->n_seq > 1 ? p->n_seq : 1;
    uint64_t fsize = 0, fino = 0;
    int64_t fmt = 0, fct = 0;
    int ferr = 0;
    if (!m->path || !plat_file_identity(m->path, &fsize, &fino, &fmt, &fct, &ferr)) {
        snprintf(m->route_fallback, sizeof m->route_fallback, "file identity unknown");
        fprintf(stderr, "note: inkling2_loop's file identity is unknown; running on the CPU\n");
        return 0;
    }
    const char *pf = getenv("RUNNER_INKLING2_CUDA_TEST_PUT_FAIL"), *ne = getenv("RUNNER_INKLING2_CUDA_TEST_NO_EXP");
    const char *capv = getenv("RUNNER_INKLING2_CUDA_TEST_FREE"), *nav = getenv("RUNNER_INKLING2_CUDA_TEST_NO_ATTN");
    const char *nbv = getenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK");
    // D1d.3b: the resident decode step, ON by default (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_STEP=1 keeps every
    // position on the non-resident route, the baseline); its small tensors go up with the copy: part of the key
    const char *stv = getenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP");
    const bool step_req = !(stv && !strcmp(stv, "1"));
    char key[1200];
    snprintf(key, sizeof key, "%s|%llu|%llu|%lld|moe %d|%s|c %d|b %d|slots %d|pf %s|ne %s|cap %s|na %s|nb %s|st %d", m->path,
             (unsigned long long)fsize, (unsigned long long)fino, (long long)fmt, p->cpu_moe ? 1 : 0, dev, m->n_ctx,
             m->n_batch, n_slots, pf ? pf : "", ne ? ne : "", capv ? capv : "", nav ? nav : "", nbv ? nbv : "",
             step_req ? 1 : 0);
    pthread_mutex_lock(&g_share_mu);
    ink2_share *sh = NULL;
    for (ink2_share *q = g_shares; q; q = q->next)
        if (!strcmp(q->key, key)) { sh = q; break; }
    bool made = false;
    if (!sh) {
        sh = calloc(1, sizeof *sh);
        if (!sh) {
            pthread_mutex_unlock(&g_share_mu);
            snprintf(m->route_fallback, sizeof m->route_fallback, "out of memory binding inkling2_loop");
            fprintf(stderr, "note: out of memory binding inkling2_loop; running on the CPU\n");
            return 0;
        }
        snprintf(sh->key, sizeof sh->key, "%s", key);
        share_place(sh, r, p->cpu_moe, m->n_batch, m->n_ctx, n_slots, dev, step_req, NULL);
        // D1d.3b: the step's small tensors after the block's, while the copy is new (no instance reads it yet)
        sh->step_base = -1;
        if (step_req && sh->placed == 'W' && sh->block_dev && sh->gpu) {
            const int base = sh->n_views + r->s.n_block * GI2_NF;
            if (step_put(r, sh->gpu, base)) sh->step_base = base;
        }
        sh->next = g_shares;
        g_shares = sh;
        made = true;
    }
    sh->refs++;
    r->share = sh;
    r->placed = sh->placed;
    r->need = sh->need;
    if (sh->placed == 'C') {
        pthread_mutex_unlock(&g_share_mu);
        snprintf(m->route_fallback, sizeof m->route_fallback, "%s", sh->why[0] ? sh->why : "no CUDA placement fits");
        fprintf(stderr, "note: %s; running on the CPU%s\n", sh->why, n_slots > 1 ? " (every slot)" : "");
        return 0;
    }
    r->gpu = sh->gpu;
    if (!made) {   // join the copy: this instance's views, in the file's walk order, as device tensors 0..n-1
        const int n = sh->n_views;
        r->dev_of = calloc((size_t)n, sizeof *r->dev_of);
        r->n_dev = 0;
        dev_walk a = { r, sh->placed == 'M', 0, n, -1, 0, 0, 0, true };
        if (!r->dev_of || ink2g_each_view(r->g, dev_put, &a) || r->n_dev != n) {
            free(r->dev_of); r->dev_of = NULL; r->n_dev = 0;
            pthread_mutex_unlock(&g_share_mu);
            fprintf(stderr, "error: inkling2_loop: this instance's views do not match the device copy its placement holds "
                            "on %s; the load is refused (never a CPU instance beside CUDA ones)\n", dev);
            return -1;
        }
    }
    // the instance's own device functions, read-only, in place before any of its views is marked: a forward reads
    // them through its own views and cache only, so no other instance's bind writes what it reads
    static const ink2_dev_ops ops = { dev_lin_scaled, dev_experts, dev_attn, dev_block, dev_msg, dev_moe, dev_step };
    r->hdr.ops = &ops;
    if (!dev_mark(r)) {
        pthread_mutex_unlock(&g_share_mu);
        fprintf(stderr, "error: inkling2_loop: this instance's views could not be marked for the device; the load is "
                        "refused\n");
        return -1;
    }
    pthread_mutex_unlock(&g_share_mu);
    // step 2a: this instance's K/V rings on the device (the fit counted them for every slot), the host rings freed:
    // the device holds the only copy, so a device failure refuses (never a host core with a missing cache)
    if (sh->attn_dev) {
        int *rows = malloc(sizeof(int) * (size_t)r->s.n_app), *cw = malloc(sizeof(int) * (size_t)r->s.n_app);
        // R4.48.8.5: the rings in the cache's kind, each slot's sliding ones packed into one allocation per side
        unsigned char *pack = malloc((size_t)r->s.n_app);
        for (int a = 0; rows && cw && pack && a < r->s.n_app; a++) {
            rows[a] = r->c.rows[a]; cw[a] = app_c(&r->s, a); pack[a] = !ink2_app_global(&r->s, a);
        }
        if (r->c.kv == INK2_KV_Q8 && !gpu_ink2_kvq8_ok(r->gpu))
            fprintf(stderr, "error: inkling2_loop --kv q8: this build's CUDA module has no q8 ring kernels\n");
        r->kv = rows && cw && pack ? gpu_ink2_kv_new(r->gpu, r->s.n_app, rows, cw, sh->block_dev ? r->s.conv_k - 1 : 0,
                                                     r->c.kv == INK2_KV_Q8 ? 1 : 0, pack) : NULL;
        free(rows); free(cw); free(pack);
        if (!r->kv) {
            fprintf(stderr, "error: inkling2_loop: this instance's K/V rings could not be put on %s after the fit "
                            "counted them; the load is refused (never a CPU instance beside CUDA ones)\n", dev);
            return -1;
        }
        for (int a = 0; a < r->s.n_app; a++) {
            free(r->c.k[a]); r->c.k[a] = NULL;
            free(r->c.v[a]); r->c.v[a] = NULL;
        }
        const char *af = getenv("RUNNER_INKLING2_CUDA_TEST_ATTN_FAIL");
        r->attn_fail_at = af && *af ? atol(af) : -1;
        r->n_attn = 0;
        r->attn_on = true;
        r->c.dev = r;   // the engine hands this cache's core to dev_attn with this instance
        // step 2b: this instance's small tensors marked as the copy's device tensors n_views.., then the whole block
        // on the device when every layer's block views are on it too (an F16 file's are not: 2a's path)
        bool all = sh->block_dev;
        for (int b = 0; all && b < r->s.n_block; b++) {
            for (int k = 0; all && k < GI2_NF; k++) {
                const ink2_tensor *t = small_get(r, b, k);
                all = t && !ink2g_set_dev(r->g, t, r, sh->n_views + b * GI2_NF + k);
            }
            for (int k = 0; all && k < GI2_NV; k++) {
                const ink2_tensor *t = block_view(r, b, k);
                if ((k == GI2_KD || k == GI2_VD) && !r->s.block_global[b]) continue;
                all = t && t->dev == r;
            }
        }
        r->block_on = all;
        r->n_block_calls = 0;
        // D1a: the loop's message steps on the device (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_MSG=1 keeps them on the
        // host, the baseline; an instance's own choice, not the copy's: no part of the share key)
        const char *nm = getenv("RUNNER_INKLING2_CUDA_TEST_NO_MSG");
        r->msg_on = gpu_ink2_msg_ok(r->gpu) && !(nm && !strcmp(nm, "1"));
        r->n_msg_calls = 0;
        const char *mf = getenv("RUNNER_INKLING2_CUDA_TEST_MSG_FAIL");
        r->msg_fail_at = mf && *mf ? atol(mf) : -1;
        // D1b: the router on the device (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_ROUTER=1 keeps it on the host, the
        // baseline: an instance's own choice, no part of the share key)
        const char *nr = getenv("RUNNER_INKLING2_CUDA_TEST_NO_ROUTER");
        r->route_on = gpu_ink2_route_ok(r->gpu) && !(nr && !strcmp(nr, "1"));
        r->n_route = 0;
    }
    const char *nf = getenv("RUNNER_INKLING2_CUDA_TEST_NO_FACTOR");
    r->no_factor = nf && !strcmp(nf, "1");
    const char *lf = getenv("RUNNER_INKLING2_CUDA_TEST_LIN_FAIL");
    r->lin_fail_at = lf && *lf ? atoi(lf) : -1;
    r->n_lin = 0;
    r->n_exp = 0;
    const char *xf = getenv("RUNNER_INKLING2_CUDA_TEST_EXP_FAIL");
    r->exp_fail_at = xf && *xf ? atol(xf) : -1;
    r->exp_on = sh->exp_dev;
    // D1d.3b (on by default; TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_STEP=1 off): the resident decode step, when every op of the
    // chain is this instance's on the device and the copy holds the step's small tensors
    if (step_req && sh->step_base >= 0 && r->kv && r->block_on && r->msg_on && r->route_on && r->exp_on) {
        gpu_ink2_step_app *ga = malloc(sizeof *ga * (size_t)r->s.n_app);
        gpu_ink2_step_plan gp;
        const size_t one = (size_t)(r->s.conv_k - 1) * (size_t)r->s.hidden;
        r->post_stage = malloc(sizeof(float) * one * 2 * (size_t)r->s.n_app);
        r->step_on = ga && r->post_stage && step_plan_make(r, sh->step_base, &gp, ga) && gpu_ink2_step_set(r->gpu, &gp) &&
                     gpu_ink2_kv_post_init(r->kv, r->s.hidden, r->s.conv_k - 1);
        free(ga);
        if (!r->step_on) { free(r->post_stage); r->post_stage = NULL; }
        r->post_dev_newer = false;
        r->n_step = 0;
        r->n_step_try = 0;
        const char *sf = getenv("RUNNER_INKLING2_CUDA_TEST_STEP_FAIL");
        r->step_fail_at = sf && *sf ? atol(sf) : -1;
    }
    fprintf(stderr, "inkling2_loop: the weights on CUDA (%s): %d tensors, %.3f GB, their stored rows%s; their products "
                    "on the device%s; attention blocks and resident decode when enabled%s%s\n", dev, r->n_dev,
            gpu_ink2_bytes(r->gpu) / 1e9, made ? "" : ", shared with this file's other instances",
            sh->exp_dev ? " and each layer's expert body whole" : "",
            sh->placed == 'M' ? "; the expert stacks on the host" : "", r->no_factor ? "; TEST-ONLY: the expert factor left out" : "");
    return 1;
}

// the host's cache bytes, one figure for the runtime's line and the fit (the second read of 2a): ink2_cache_bytes,
// less the K/V rings an instance holds on the device (G2b step 2a; the host keeps the convolution histories)
static size_t host_cache_bytes(const ink2_rt *r) {
    size_t b = r->cache_ok ? ink2_cache_bytes(&r->c, &r->s) : 0;
    if (r->kv)
        for (int a = 0; a < r->s.n_app && b; a++) {
            const size_t kvb = 2 * sizeof(float) * (size_t)r->c.rows[a] * (size_t)app_c(&r->s, a);
            b = b > kvb ? b - kvb : 0;
        }
    return b;
}
// R4.48.17: the most the decoded-expert cache can hold: every expert of every layer decoded to float32 (gate_up
// [2F][H] and down [H][F] each). The cache maps its budget lazily, so it holds at most this, whatever the budget.
static uint64_t experts_decoded_bytes(const ink2_shape *s) {
    return 4ull * (uint64_t)s->n_block * (uint64_t)s->n_expert * (3ull * (uint64_t)s->ff * (uint64_t)s->hidden);
}
static uint64_t expert_cache_held(const ink2_shape *s, uint64_t budget) {
    const uint64_t cap = experts_decoded_bytes(s);
    return budget < cap ? budget : cap;
}
// RUNNER_INKLING2_ROUTE, read at each load: unset, empty or "fast" the fast route; "reference" the reference; 0 for
// anything else (refused with a sentence)
static char route_of_env(void) {
    const char *v = getenv("RUNNER_INKLING2_ROUTE");
    if (!v || !*v || !strcmp(v, "fast")) return 'F';
    if (!strcmp(v, "reference")) return 'R';
    return 0;
}
static const char *route_words(char route) {
    return route == 'F' ? "the fast CPU route (the weights in their file formats, decoded at use; float32 activations; "
                          "vectorized sums)"
                        : "the serial float32 engine on the CPU (RUNNER_INKLING2_ROUTE=reference)";
}

bool model_inkling2_bind(model_t *m, const model_params *p) {
    char why[512] = { 0 };
    ink2_rt *r = calloc(1, sizeof *r);
    if (!r) { fprintf(stderr, "error: out of memory loading an inkling2_loop model\n"); return false; }
    r->route = route_of_env();
    if (r->route && p && p->train && r->route != 'R') {
        // the LoRA backward (src/inkling2_bwd.c) reads the matrices as float32: a training load takes the reference route
        fprintf(stderr, "inkling2_loop: training runs on the reference route (the serial float32 engine), whose float32 "
                        "matrices the backward reads\n");
        r->route = 'R';
    }
    if (!r->route) {
        fprintf(stderr, "error: RUNNER_INKLING2_ROUTE is \"%s\"; it takes \"fast\" (the default) or \"reference\"\n",
                getenv("RUNNER_INKLING2_ROUTE"));
        free(r);
        return false;
    }
    r->g = ink2g_attach(&m->gf, &r->s, why, sizeof why);
    if (!r->g) {
        fprintf(stderr, "error: this inkling2_loop file is refused: %s\n", why);
        free(r);
        return false;
    }
    // RUNNER_INKLING2_ACT8=1 (TEST-ONLY, the fast route's control): every activation through 8 bits before a product
    const char *a8 = getenv("RUNNER_INKLING2_ACT8");
    r->act8 = r->route == 'F' && a8 && *a8 && strcmp(a8, "0");
    ink2g_set_route(r->g, r->route == 'F', r->act8);
    size_t decoded = 0;
    if (ink2g_decode_all(r->g, &decoded)) {
        fprintf(stderr, "error: an inkling2_loop tensor could not be decoded to float32\n");
        ink2g_close(r->g);
        free(r);
        return false;
    }
    r->dense_bytes = ink2g_matrix_bytes(r->g);
    r->small_bytes = decoded - r->dense_bytes;
    ink2g_weights(r->g, &r->inner);
    m->ink2 = r;
    snprintf(m->arch, sizeof m->arch, "inkling2_loop");
    m->ad = arch_find("inkling2_loop");
    // what the engine, the sampler and the server read from model_t; no llama-style layer is bound (n_layer 0)
    m->n_layer = 0;
    m->n_embd = r->s.hidden;
    m->n_head = r->s.n_head;
    m->n_head_kv = r->s.n_kv;
    m->head_dim = r->s.head_dim;
    m->n_ff = r->s.ff;
    m->n_vocab = r->s.vocab_used;
    m->n_ctx_train = (int)gguf_get_u32(&m->gf, "inkling2_loop.context_length", 0);
    if (m->n_ctx_train <= 0) m->n_ctx_train = 4096;
    m->n_expert = r->s.n_expert;
    m->n_expert_used = r->s.n_used;
    fprintf(stderr, "inkling2_loop: %d layers, %d applications (loop %d..%d x %d passes), %d experts (%d used), "
                    "%s; %.2f GB of float32 matrices and %.3f GB of small float32 tensors decoded\n",
            r->s.n_block, r->s.n_app, r->s.loop_start, r->s.loop_end, r->s.loop_passes, r->s.n_expert, r->s.n_used,
            route_words(r->route), r->dense_bytes / 1e9, r->small_bytes / 1e9);
    return true;
}

static size_t env_bytes(const char *k, size_t dflt) {
    const char *v = getenv(k);
    if (!v || !*v) return dflt;
    char *end = NULL;
    double x = strtod(v, &end);
    size_t mul = 1;
    if (end && (*end == 'K' || *end == 'k')) mul = 1024;
    else if (end && (*end == 'M' || *end == 'm')) mul = 1024 * 1024;
    else if (end && (*end == 'G' || *end == 'g')) mul = 1024ull * 1024 * 1024;
    return (size_t)(x * (double)mul);
}

// R4.48.20: runtime and --info fit share the house CPU batch default (clamped to n_ctx) so they cannot disagree.
static int ink2_batch_default(int n_batch, int n_ctx) {
    if (n_batch <= 0) n_batch = model_batch_default(GPU_OFF, plat_ram_bytes());
    if (n_batch > n_ctx) n_batch = n_ctx;
    return n_batch;
}

// R4.48.8.5: the K/V cache's storage from --kv. f32 is this engine's own (the default, and --kv f32); --kv q8 stores
// q8_0 rows when every head_dim is a multiple of 32. The house's other formats (f16, k8v4, fp4) are not this engine's,
// and an asked one keeps f32 with a note, as an incompatible format does elsewhere. The CUDA route keeps f32 under
// --kv q8 until its device kernels are proven (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_KVQ8=1 lifts that for the device
// gates); TEST-ONLY RUNNER_INKLING2_TEST_KV_RT=1: float32 rows holding each row's q8 round trip (gate (a)).
static const char *kv_kind_name(int kind) { return kind == INK2_KV_Q8 ? "q8" : kind == INK2_KV_RT ? "f32 round trip" : "f32"; }
// the kind for a shape, a route and the asked --kv, with the sentence a load prints as a note and the fit as a line
// (empty when there is none to say). The TEST-ONLY round trip is the runtime's alone.
static int kv_kind_pick(const ink2_shape *s, char route, const model_params *p, char *note, size_t nn) {
    note[0] = 0;
    if (p->kv_fp4 || p->kv_split || p->kv_f16) {
        snprintf(note, nn, "inkling2_loop's K/V cache is f32 or q8; --kv %s is not one of its formats, so it keeps f32",
                 p->kv_fp4 ? "fp4" : p->kv_split ? "k8v4" : "f16");
        return INK2_KV_F32;
    }
    if (!p->kv_q8) return INK2_KV_F32;
    if (!ink2_kv_kind_ok(s, INK2_KV_Q8)) {
        snprintf(note, nn, "inkling2_loop --kv q8 needs every head_dim to be a multiple of 32 (this file's are %d and %d); "
                 "it keeps f32", s->head_dim, s->swa_head_dim);
        return INK2_KV_F32;
    }
#ifdef RUNNER_GPU_CUDA
    const char *on = getenv("RUNNER_INKLING2_CUDA"), *t = getenv("RUNNER_INKLING2_CUDA_TEST_KVQ8");
    if (p->gpu_mode != GPU_OFF && on && !strcmp(on, "1") && route == 'F' && !(t && !strcmp(t, "1"))) {
        snprintf(note, nn, "inkling2_loop --kv q8 keeps f32 on the CUDA route (RUNNER_INKLING2_CUDA=1) until its device "
                 "kernels are proven; the CPU routes store q8");
        return INK2_KV_F32;
    }
#else
    (void)route;
#endif
    return INK2_KV_Q8;
}
static const char RING_NOTE[] = "inkling2_loop's sliding applications always keep a window ring (window - 1 + batch "
                                "rows); --kv-ring off changes nothing";
// a cache's K/V and convolution bytes: the figure --info fit predicts (ink2_cache_plan_bytes), from the live rows
static uint64_t cache_kv_bytes(const ink2_cache *c) {
    uint64_t n = 0;
    for (int a = 0; a < c->n_app; a++) n += 2 * (uint64_t)c->rows[a] * c->rb[a] + (uint64_t)c->conv_floats[a] * sizeof(float);
    return n;
}

bool model_inkling2_runtime(model_t *m, const model_params *p) {
    ink2_rt *r = rt_of(m);
    if (!r) return false;
    int n_ctx = p->n_ctx > 0 ? p->n_ctx : (m->n_ctx_train < 4096 ? m->n_ctx_train : 4096);
    if (n_ctx > m->n_ctx_train) {
        fprintf(stderr, "error: -c %d is past this model's training context %d\n", n_ctx, m->n_ctx_train);
        return false;
    }
    m->n_ctx = n_ctx;
    m->n_batch = ink2_batch_default(p->n_batch, n_ctx);
    m->spec_batch = m->n_batch;
    r->n_threads = p->n_threads > 0 ? p->n_threads : 1;
    char note[256];
    int kind = kv_kind_pick(&r->s, r->route, p, note, sizeof note);
    if (note[0]) fprintf(stderr, "note: %s\n", note);
    const char *rt = getenv("RUNNER_INKLING2_TEST_KV_RT");
    if (rt && !strcmp(rt, "1") && ink2_kv_kind_ok(&r->s, INK2_KV_RT)) kind = INK2_KV_RT;
    if (p->kv_ring == KV_RING_OFF) fprintf(stderr, "note: %s\n", RING_NOTE);
    if (ink2_cache_init_kind(&r->c, &r->s, n_ctx, m->n_batch, kind)) {
        fprintf(stderr, "error: the inkling2_loop cache for -c %d does not fit in memory\n", n_ctx);
        return false;
    }
    r->cache_ok = true;
    // R4.48.32: reflect the cache format this runtime actually initialized onto
    // model_t so route_collect, model_kv_name, receipts, --verify, --doctor,
    // /v1/runner/provenance, and engine model keys all read the real K/V type
    // (f32 or q8) instead of model_t's zero-initialized f16 default.
    m->kv_q8 = (r->c.kv == INK2_KV_Q8);
    m->kv_f32 = (r->c.kv != INK2_KV_Q8);
    m->kv_fp4 = false;
    m->kv_split = false;
    r->rows_cap = m->n_batch;
    // r->rows [n_batch][n_vocab] is allocated lazily on the first keep_all or RUNNER_INKLING2_ROWS forward; normal
    // prefill/decode asks only for the last row via m->logits (INK2_LOGITS_LAST).
    r->rows = NULL;
    m->logits = malloc(sizeof(float) * (size_t)m->n_vocab);
    if (!m->logits) { fprintf(stderr, "error: out of memory for the logits rows\n"); return false; }
    // the decoded-expert cache (the reference route only: the fast route reads the experts in place):
    // RUNNER_INKLING2_EXPERT_CACHE bytes (default 4 GiB; 0 decodes every expert on use)
    r->ec_budget = r->route == 'R' ? env_bytes("RUNNER_INKLING2_EXPERT_CACHE", (size_t)4 << 30) : 0;
    if (r->ec_budget) {
        ink2_ec_cfg cfg = { 1, r->ec_budget, 0, 64 };
        r->ec = ink2_ec_new(&r->inner, ink2g_fill_expert, r->g, cfg);
        if (!r->ec) { fprintf(stderr, "error: out of memory for the expert cache\n"); return false; }
        ink2_ec_weights(r->ec, &r->w);
    } else {
        r->w = r->inner;
    }
    m->tp = tpool_create(r->n_threads);
    if (!m->tp) {
        fprintf(stderr, "error: cannot create thread pool\n");
        return false;
    }
    // the device last, once -c and the host buffers are known good (a refused -c uploads nothing)
    if (p->gpu_mode != GPU_OFF && cuda_bind(r, m, p) < 0) return false;   // < 0: this slot refuses (never mixed)
    fprintf(stderr, "inkling2_loop: -c %d -b %d; cache %.2f GB (%llu bytes; K/V %s; sliding K/V for the window only), "
                    "expert cache up to %.2f GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)\n", n_ctx, m->n_batch,
            host_cache_bytes(r) / 1e9, (unsigned long long)cache_kv_bytes(&r->c), kv_kind_name(r->c.kv),
            expert_cache_held(&r->s, r->ec_budget) / 1073741824.0);
    return true;
}

// RUNNER_INKLING2_ROWS, read once for every slot
static const char *rows_dump;
static pthread_once_t rows_dump_once = PTHREAD_ONCE_INIT;
static void rows_dump_read(void) { rows_dump = getenv("RUNNER_INKLING2_ROWS"); }

static int ink2_lora_slot_cb(void *ctx, int block, int slot, const float **a, const float **b,
                             int *r, float *scale, int *n_in, int *n_out) {
    return model_lora_slot((const model_t *)ctx, block, slot, a, b, r, scale, n_in, n_out) ? 1 : 0;
}

int model_inkling2_n_block(const model_t *m) {
    const ink2_rt *r = m && m->ink2 ? (const ink2_rt *)m->ink2 : NULL;
    return r ? r->s.n_block : 0;
}

bool model_inkling2_slot_dims(const model_t *m, int layer, int slot, int *n_in, int *n_out, char *name, size_t cap) {
    const ink2_rt *r = m && m->ink2 ? (const ink2_rt *)m->ink2 : NULL;
    if (!r || layer < 0 || layer >= r->s.n_block) return false;
    if (slot < LW_Q || slot > LW_O) return false;
    int g = r->s.block_global[layer];
    int H = r->s.hidden;
    int Q = (g ? r->s.n_head : r->s.swa_n_head) * (g ? r->s.head_dim : r->s.swa_head_dim);
    int KV = (g ? r->s.n_kv : r->s.swa_n_kv) * (g ? r->s.head_dim : r->s.swa_head_dim);
    int ni = 0, no = 0;
    const char *sname = NULL;
    switch (slot) {
        case LW_Q: ni = H; no = Q;  sname = "attn_q"; break;
        case LW_K: ni = H; no = KV; sname = "attn_k"; break;
        case LW_V: ni = H; no = KV; sname = "attn_v"; break;
        case LW_O: ni = Q; no = H;  sname = "attn_output"; break;
        default: return false;
    }
    if (n_in) *n_in = ni;
    if (n_out) *n_out = no;
    if (name && cap) {
        int wn = snprintf(name, cap, "blk.%d.%s.weight", layer, sname);
        if (wn < 0 || (size_t)wn >= cap) {
            fprintf(stderr, "error: inkling2_loop tensor name 'blk.%d.%s.weight' "
                    "exceeds %zu-byte destination\n", layer, sname, cap);
            return false;
        }
    }
    return true;
}

bool model_inkling2_parts(model_t *m, const void **s_out, const void **w_out) {
    ink2_rt *r = m ? rt_of(m) : NULL;
    if (!r) return false;
    if (s_out) *s_out = &r->s;
    if (w_out) *w_out = &r->w;
    return true;
}

int model_inkling2_threads(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (m && m->tp) return tpool_size(m->tp);
    return r && r->n_threads > 0 ? r->n_threads : 1;
}

bool model_inkling2_save_sel(model_t *m, const int *sel, int n) {
    ink2_rt *r = m ? rt_of(m) : NULL;
    if (!r || !sel || n <= 0) return false;
    if (r->last_sel_n < n) {
        int *nb = realloc(r->last_sel, sizeof(int) * (size_t)n);
        if (!nb) {
            fprintf(stderr, "error: inkling2_loop: out of memory saving expert selections\n");
            return false;
        }
        r->last_sel = nb;
        r->last_sel_n = n;
    }
    memcpy(r->last_sel, sel, sizeof(int) * (size_t)n);
    return true;
}

bool model_inkling2_last_sel(const model_t *m, const int **sel_out, int *n_out) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (!r || !r->last_sel || r->last_sel_n <= 0) return false;
    if (sel_out) *sel_out = r->last_sel;
    if (n_out) *n_out = r->last_sel_n;
    return true;
}

float *model_inkling2_forward(model_t *m, const int32_t *tokens, int n, int pos, bool want_logits, bool keep_all) {
    ink2_rt *r = rt_of(m);
    if (r) r->last_ok = false;
    if (!r || !r->cache_ok || n <= 0 || pos < 0 || pos + n > m->n_ctx) {
        fprintf(stderr, "error: inkling2_loop: %d tokens at %d do not fit -c %d\n", n, pos, m->n_ctx);
        return NULL;
    }
    if (keep_all && n > r->rows_cap) return NULL;
    if (r->gpu_err) {   // a device product failed earlier: its cache holds NaN states, never CPU rows
        fprintf(stderr, "error: inkling2_loop: this instance's device failed; it computes no more rows\n");
        return NULL;
    }
    // Bind this instance's house pool (CLI's own tpool or the server's shared_pool) onto TLS for the forward.
    // Two server slots run safely without a global thread-count lock: tpool_run serializes on shared_pool, and
    // distinct pools run concurrently on their own threads with identical bit outputs.
    tpool *prev_tp = ink2_set_tpool(m->tp);
    ink2_lora lcb = { m, ink2_lora_slot_cb };
    const ink2_lora *lora = m->lora ? &lcb : NULL;
    // RUNNER_INKLING2_ROWS=FILE (TEST-ONLY, the wired path's bit-for-bit check): every forward's rows, appended
    // in call order, raw float32 [T][n_vocab], as the read harness writes them
    pthread_once(&rows_dump_once, rows_dump_read);
    const char *dump = rows_dump;
    const bool dump_on = dump && *dump;
    const bool need_all = keep_all || dump_on;
    if (need_all && !r->rows) {
        r->rows = malloc(sizeof(float) * (size_t)r->rows_cap * (size_t)m->n_vocab);
        if (!r->rows) {
            ink2_set_tpool(prev_tp);
            fprintf(stderr, "error: out of memory for the logits rows\n");
            return NULL;
        }
    }
    int last_T = 0;
    // chunks of n_batch: the engine's rows do not depend on the split (measured bit for bit)
    for (int done = 0; done < n;) {
        const int T = n - done < r->rows_cap ? n - done : r->rows_cap;
        const bool last_chunk = done + T == n;
        float *chunk_out = need_all ? r->rows
                         : (want_logits && last_chunk) ? m->logits
                         : NULL;
        const int chunk_flags = (!need_all && want_logits && last_chunk) ? INK2_LOGITS_LAST : 0;
        r->last_was_step = false;   // D1d.3b: dev_step sets it when the engine hands it this chunk
        ink2_status st = ink2_forward_chunk_lora(&r->s, &r->w, lora, &r->c, pos + done, T,
                                                 (const int *)(tokens + done), chunk_out, chunk_flags,
                                                 NULL, NULL);   // int32_t is int on every target Runner builds for
        // D1d.3b: a chunk the step did not take ran on the host's post-convolution histories: the device copies follow
        if (st == INK2_OK && r->step_on && !r->gpu_err && !r->last_was_step && !post_sync(r, 1)) r->gpu_err = true;
        if (st == INK2_OK && r->gpu_err) {   // on the flag alone: it does not depend on r->gpu
            ink2_set_tpool(prev_tp);
            fprintf(stderr, "error: inkling2_loop: the forward at position %d failed on the device\n", pos + done);
            return NULL;
        }
        if (st != INK2_OK) {
            ink2_set_tpool(prev_tp);
            fprintf(stderr, "error: inkling2_loop: the forward at position %d refused (status %d%s)\n", pos + done, (int)st,
                    st == INK2_REWIND ? ": not where the sequence ends, and this model does not rewind" : "");
            return NULL;
        }
        if (dump_on) {
            FILE *f = plat_fopen(dump, "ab");
            if (f) {
                fwrite(r->rows, sizeof(float), (size_t)T * (size_t)m->n_vocab, f);
                fclose(f);
            }
            // beside it, FILE.threads: what the pool really did for this chunk (one line a chunk)
            char side[4096];
            snprintf(side, sizeof side, "%s.threads", dump);
            FILE *ft = plat_fopen(side, "a");
            if (ft) {
                long split = 0;
                int last = 1;
                ink2_par_stats(&split, &last);
                fprintf(ft, "pos %d T %d count %d split_loops %ld last_pool %d\n", pos + done, T, ink2_get_threads(), split,
                        last);
                fclose(ft);
            }
        }
        last_T = T;
        done += T;
        if (done == n) {
            r->kept_from = keep_all ? pos + done - T : 0;
            r->kept_n = keep_all ? T : 0;
        }
    }
    ink2_set_tpool(prev_tp);
    r->last_ok = true;
    if (!want_logits) return NULL;
    if (need_all)
        memcpy(m->logits, r->rows + (size_t)(last_T - 1) * (size_t)m->n_vocab, sizeof(float) * (size_t)m->n_vocab);
    return m->logits;
}

float *model_inkling2_row(model_t *m, int b) {
    ink2_rt *r = rt_of(m);
    if (!r || !r->rows || b < 0 || b >= r->kept_n) return NULL;
    return r->rows + (size_t)b * (size_t)m->n_vocab;
}

bool model_inkling2_fit(model_t *m, inkling2_fit *f) {
    ink2_rt *r = rt_of(m);
    memset(f, 0, sizeof *f);
    if (!r) return false;
    f->dense_f32 = r->dense_bytes;
    f->small_f32 = r->small_bytes;
    f->cache = host_cache_bytes(r);
    f->expert_cache = expert_cache_held(&r->s, r->ec_budget);
    f->mapped = gguf_mapped_size(&m->gf);
    f->logits = sizeof(float) * (uint64_t)m->n_vocab * (uint64_t)((r->rows ? r->rows_cap : 0) + 1);
    return true;
}

void model_inkling2_free(model_t *m) {
    ink2_rt *r = rt_of(m);
    if (!r) return;
    if (r->cache_ok) ink2_cache_free(&r->c);
    if (r->ec) ink2_ec_free(r->ec);
    gpu_ink2_kv_free(r->kv);   // this instance's rings, before the copy whose context they live in
    r->kv = NULL;
    share_release(r);   // the file's shared device copy: freed with its last instance
    free(r->dev_of);
    free(r->post_stage);   // D1d.3b
    if (r->g) ink2g_close(r->g);
    free(r->last_sel);
    free(r->rows);
    free(r);
    m->ink2 = NULL;
}

// the stored names of the matrices (src/inkling2_gguf.c's is_matrix, by their GGUF names): what the reference route
// decodes to float32 and the fast route reads in place
static int gguf_matrix_name(const char *name) {
    if (!strcmp(name, "output.weight") || !strcmp(name, "token_embd.weight")) return 1;
    static const char *const suffix[] = { ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_k_conv_proj.weight",
        ".attn_v_conv_proj.weight", ".attn_gate.weight", ".attn_output.weight", ".attn_sink_proj.weight",
        ".attn_rel.weight", "_ctrl_down.weight", "_ctrl_forget.weight", "_ctrl_gain.weight" };
    const size_t n = strlen(name);
    for (size_t i = 0; i < sizeof suffix / sizeof *suffix; i++) {
        const size_t m = strlen(suffix[i]);
        if (n > m && !strcmp(name + n - m, suffix[i])) return 1;
    }
    return 0;
}
// R4.48.8.5: the CUDA route's placement for --info fit, as the load makes it: the file's views (ink2g_attach, the fast
// route, the small tensors decoded) in a runtime that holds no cache, then share_place's own count as a dry run (the
// device bound, its buffers reserved, the free memory read; no weight goes up). False when the file or the device
// could not be read (why says which); t->placed 'C' with why when no placement fits.
static bool fit_device(const char *path, int n_ctx, int n_batch, int slots, const model_params *mp, int kind,
                       ink2_dev_terms *t, char *why, size_t wn) {
    memset(t, 0, sizeof *t);
    t->placed = 'C';
    char dev[160] = "";
    if (!gpu_available(dev, sizeof dev)) { snprintf(why, wn, "no CUDA device for inkling2_loop"); return true; }
    gguf_file gf;
    if (!gguf_open(&gf, path)) { snprintf(why, wn, "the file could not be opened"); return false; }
    ink2_rt *r = calloc(1, sizeof *r);
    bool ok = r != NULL;
    if (ok) {
        r->route = 'F';
        r->g = ink2g_attach(&gf, &r->s, why, wn);
        ok = r->g != NULL;
    }
    size_t decoded = 0;
    if (ok) {
        ink2g_set_route(r->g, 1, 0);
        ok = ink2g_decode_all(r->g, &decoded) == 0;
        if (!ok) snprintf(why, wn, "an inkling2_loop tensor could not be decoded to float32");
    }
    if (ok) {
        ink2g_weights(r->g, &r->inner);
        r->w = r->inner;
        r->c.n_ctx = n_ctx; r->c.max_chunk = n_batch; r->c.kv = kind;   // the slot bytes' inputs (dev_instance_bytes)
        const char *stv = getenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP");
        ink2_share sh;
        memset(&sh, 0, sizeof sh);
        share_place(&sh, r, mp->cpu_moe, n_batch, n_ctx, slots, dev, !(stv && !strcmp(stv, "1")), t);
        if (t->placed == 'C') snprintf(why, wn, "%s", sh.why[0] ? sh.why : "no placement fits");
    }
    if (r && r->g) ink2g_close(r->g);
    free(r);
    gguf_close(&gf);
    return ok;
}
static const char *placed_words(char p) {
    return p == 'W' ? "every weight on the device" : p == 'M' ? "the expert stacks on the host (--cpu-moe)" : "the CPU";
}

int model_inkling2_fit_print(gguf_file *g, const char *path, int n_ctx_want, const model_params *mp, int n_seq) {
    int n_batch = mp->n_batch;
    const int slots = n_seq > 1 ? n_seq : 1;
    ink2_shape s;
    char why[512] = { 0 };
    if (ink2g_shape(g, &s, why, sizeof why)) {
        printf("fit: %s\n  verdict       WILL NOT LOAD: %s\n", path, why);
        return 1;
    }
    int train = (int)gguf_get_u32(g, "inkling2_loop.context_length", 4096);
    int n_ctx = n_ctx_want > 0 ? n_ctx_want : (train < 4096 ? train : 4096);
    n_batch = ink2_batch_default(n_batch, n_ctx);
    // the dense float32 tensors the load decodes: every tensor but the expert stacks and their factors, the head at
    // its unpadded rows
    const char route = route_of_env();
    if (!route) {
        printf("fit: %s\n  verdict       WILL NOT LOAD: RUNNER_INKLING2_ROUTE takes \"fast\" or \"reference\"\n", path);
        return 1;
    }
    uint64_t dense = 0, small = 0;   // the matrices' float32 (the reference route), the small tensors' (both)
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor *t = &g->tensors[i];
        if (strstr(t->name, "_exps.")) continue;
        uint64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        if (!strcmp(t->name, "output.weight") && t->ne[1] > (uint64_t)s.vocab_used) n = t->ne[0] * (uint64_t)s.vocab_used;
        if (gguf_matrix_name(t->name)) dense += 4 * n;
        else small += 4 * n;
    }
    if (route == 'F') dense = 0;
    // the engine's cache (ink2_cache_init_kind): K and V per application in the kind --kv asks, the sliding ones for
    // the window and the batch, plus each application's four convolution histories; one per slot
    char note[256];
    const int kind = kv_kind_pick(&s, route, mp, note, sizeof note);
    const uint64_t cache = (uint64_t)slots * ink2_cache_plan_bytes(&s, n_ctx, n_batch, kind);
    int n_sliding = 0;
    for (int a = 0; a < s.n_app; a++) n_sliding += !ink2_app_global(&s, a);
    const uint64_t ec = route == 'R' ? expert_cache_held(&s, env_bytes("RUNNER_INKLING2_EXPERT_CACHE", (size_t)4 << 30)) : 0;
    uint64_t logits = 4ull * (uint64_t)s.vocab_used * (uint64_t)(n_batch + 1);
    uint64_t mapped = gguf_mapped_size(g);
    if (!mapped) for (uint64_t i = 0; i < g->n_tensors; i++) mapped += g->tensors[i].nbytes;
    uint64_t total = dense + small + cache + ec + logits + mapped;
    uint64_t avail = plat_ram_available_bytes();
    printf("fit: %s\n", path);
    printf("  arch          inkling2_loop, %d layers, %d applications, %d experts (%d used)\n",
           s.n_block, s.n_app, s.n_expert, s.n_used);
    printf("  route         %s\n", route_words(route));
    printf("  context       %d (trained %d), batch %d\n", n_ctx, train, n_batch);
    printf("  file mapped   %.2f GiB (%s)\n", mapped / 1073741824.0,
           route == 'F' ? "every weight is read from it" : "the experts are read from it");
    if (route == 'R') printf("  dense f32     %.2f GiB (the matrices, decoded once)\n", dense / 1073741824.0);
    printf("  small f32     %.3f GiB (norms, convolution taps, queries, scales, the router; decoded once)\n",
           small / 1073741824.0);
    printf("  cache         %.2f GiB (%llu bytes; K/V %s per application, the %d sliding ones for window %d + batch; "
           "convolutions%s)\n", cache / 1073741824.0, (unsigned long long)cache, kv_kind_name(kind), n_sliding, s.window,
           slots > 1 ? "; every slot's" : "");
    if (slots > 1) printf("  slots         %d, each its own cache\n", slots);
    if (note[0]) printf("  note          %s\n", note);
    if (mp->kv_ring == KV_RING_OFF) printf("  note          %s\n", RING_NOTE);
    if (route == 'R') printf("  expert cache  up to %.2f GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)\n", ec / 1073741824.0);
    printf("  logits        %.2f GiB\n", logits / 1073741824.0);
    printf("  total         %.2f GiB of RAM (%.2f GB)\n", total / 1073741824.0, total / 1e9);
    if (avail)
        printf("  verdict       %s (%.2f GiB available)\n", total <= avail ? "FITS" : "DOES NOT FIT", avail / 1073741824.0);
    else
        printf("  verdict       unknown (available RAM unreadable)\n");
    // R4.48.8.5: the CUDA route, as its load would place it (a dry run of share_place), with every term the load
    // checks; the non-weight line is everything on the device but the weights
    const char *on = getenv("RUNNER_INKLING2_CUDA");
    if (mp->gpu_mode == GPU_OFF || !on || strcmp(on, "1")) {
        if (mp->gpu_mode != GPU_OFF) printf("  device        the CPU (RUNNER_INKLING2_CUDA=1 opts the fast route into CUDA)\n");
        return avail && total > avail ? 1 : 0;
    }
    if (route != 'F') { printf("  device        the CPU (the CUDA route takes the fast route's rows only)\n"); return 0; }
    ink2_dev_terms t;
    char dwhy[512] = "";
    if (!fit_device(path, n_ctx, n_batch, slots, mp, kind, &t, dwhy, sizeof dwhy)) {
        printf("  device        unknown: %s\n", dwhy);
        return 1;
    }
    if (t.placed == 'C' && !t.total_b) { printf("  device        the CPU: %s\n", dwhy); return 0; }
    const size_t rest = t.reserved + t.small + t.slots + t.kept;
    printf("  device        CUDA route: %s%s%s\n", placed_words(t.placed), t.placed == 'C' ? ": " : "", t.placed == 'C' ? dwhy : "");
    printf("  dev weights   %.2f GiB (%zu bytes, as the driver allocates them)\n", t.weights / 1073741824.0, t.weights);
    printf("  dev buffers   %.2f GiB (%zu bytes: the products', experts', attention's and block's reservations)\n",
           t.reserved / 1073741824.0, t.reserved);
    printf("  dev small     %.2f GiB (%zu bytes: the attention's small tensors and the resident step's)\n",
           t.small / 1073741824.0, t.small);
    printf("  dev slots     %.2f GiB (%zu bytes: %d slot%s' K/V rings in %s and histories)\n", t.slots / 1073741824.0,
           t.slots, slots, slots > 1 ? "s" : "", kv_kind_name(kind));
    printf("  dev kept      %.2f GiB (%zu bytes)\n", t.kept / 1073741824.0, t.kept);
    printf("  non-weight    %.2f GiB (%zu bytes: everything above but the weights)\n", rest / 1073741824.0, rest);
    if (t.whole) {
        const size_t wrest = t.w_reserved + t.w_small + t.w_slots + t.kept;
        printf("  whole         %.2f GiB of weights (%zu bytes) and %.2f GiB besides them (non-weight %zu bytes): every "
               "weight on the device, %s\n", t.w_weights / 1073741824.0, t.w_weights, wrest / 1073741824.0, wrest,
               t.placed == 'W' ? "the placement above" : "what a device holding it would need");
    }
    printf("  dev free      %.2f GiB of %.2f GiB (after the context, the module and the buffers)\n",
           t.free_b / 1073741824.0, t.total_b / 1073741824.0);
    const char *dv = t.placed == 'W' ? "FITS" : "DOES NOT FIT";
    // the generic vocabulary: FITS WITH --kv q8 when the f32 cache leaves the weights off the device and q8 does not
    model_params alt = *mp;
    char anote[256];
    alt.kv_q8 = kind != INK2_KV_Q8;
    alt.kv_fp4 = alt.kv_split = alt.kv_f16 = false;
    const int akind = kv_kind_pick(&s, route, &alt, anote, sizeof anote);
    if (akind != kind) {
        ink2_dev_terms ta;
        char awhy[512] = "";
        if (fit_device(path, n_ctx, n_batch, slots, mp, akind, &ta, awhy, sizeof awhy)) {
            if (kind == INK2_KV_Q8 && t.placed == 'W') dv = ta.placed == 'W' ? "FITS" : "FITS WITH --kv q8";
            if (kind == INK2_KV_F32 && t.placed != 'W' && ta.placed == 'W') dv = "FITS WITH --kv q8";
        }
    }
    printf("  dev verdict   %s\n", dv);
    return (avail && total > avail) || t.placed == 'C' || (t.placed == 'M' && !mp->cpu_moe) ? 1 : 0;
}

char model_inkling2_route(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r ? r->route : 0;
}

// R4.48.9: the instance's weights are on CUDA
bool model_inkling2_cuda(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->gpu;
}
bool model_inkling2_act8(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->act8 && !r->gpu;
}
bool model_inkling2_last_ok(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->last_ok;
}
const char *model_inkling2_route_detail(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (!r || !r->gpu) return "";
    static _Thread_local char buf[128];
    snprintf(buf, sizeof buf, "placed=%c attn=%s exp=%s msg=%s router=%s step=%s",
             r->placed ? r->placed : 'W',
             r->block_on ? "block" : r->attn_on ? "core" : "host",
             r->exp_on ? "on" : "off",
             r->msg_on ? "on" : "off",
             r->route_on ? "on" : "off",
             r->step_on ? "on" : "off");
    return buf;
}
size_t model_inkling2_cuda_fit_free(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->share ? r->share->fit_free : 0;
}
size_t model_inkling2_cuda_lin_out_bytes(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->gpu ? gpu_ink2_lin_out_bytes(r->gpu) : 0;
}
size_t model_inkling2_cuda_slot_fit_bytes(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (!r || !r->share || !r->gpu) return 0;
    return dev_instance_bytes(r, r->share->attn_dev, r->share->block_dev ? r->s.conv_k - 1 : 0, r->share->step_base >= 0);
}
size_t model_inkling2_cuda_block_buf_bytes(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->gpu ? gpu_ink2_block_bytes(r->gpu) : 0;
}
long model_inkling2_cuda_block_calls(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->block_on ? r->n_block_calls : -1;
}
long model_inkling2_cuda_msg_calls(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->msg_on ? r->n_msg_calls : -1;
}
long model_inkling2_cuda_route_calls(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->route_on ? r->n_route : -1;
}
long model_inkling2_cuda_step_calls(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->step_on ? r->n_step : -1;
}
bool model_inkling2_cuda_step_bytes(const model_t *m, size_t out[5]) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (!r || !r->step_on || !r->share || r->share->step_base < 0 || !out) return false;
    out[0] = step_dev_bytes(r);
    // as allocated: the small tensors from step_base (their count by the same list), then the program's allocations
    ink2_step_app *apps = calloc((size_t)r->s.n_app, sizeof *apps);
    ink2_step_plan P = { 0 };
    P.app = apps;
    const int ns = apps && ink2_step_plan_build(&r->s, &r->w, &P) == INK2_OK ? step_smalls(r, &P, NULL, NULL) : -1;
    free(apps);
    if (ns <= 0) return false;
    size_t real = 0;
    for (int k = 0; k < ns; k++) real += dev_alloc_bytes(gpu_ink2_test_tensor_bytes(r->share->gpu, r->share->step_base + k));
    size_t got[16], sz[16];
    const int na = gpu_ink2_test_step_allocs(r->gpu, got), nf = gpu_ink2_step_sizes(r->s.n_app, r->s.hidden, r->s.vocab_used, sz);
    bool same = na == nf && na > 0;
    for (int k = 0; k < na; k++) {
        real += dev_alloc_bytes(got[k]);
        if (k < nf) same = same && (k == 0 ? got[k] <= sz[k] : got[k] == sz[k]);
    }
    out[1] = real;
    const int hist = r->share->block_dev ? r->s.conv_k - 1 : 0;
    out[2] = dev_instance_bytes(r, r->share->attn_dev, hist, true) - dev_instance_bytes(r, r->share->attn_dev, hist, false);
    out[3] = dev_alloc_bytes(gpu_ink2_kv_post_bytes(r->kv));
    out[4] = same;
    return true;
}
long model_inkling2_cuda_attn_calls(const model_t *m, size_t *kv_bytes) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (kv_bytes) *kv_bytes = r ? gpu_ink2_kv_bytes(r->kv) : 0;
    return r && r->attn_on ? r->n_attn : -1;
}
bool model_inkling2_cuda_placement(const model_t *m, char *placed, size_t *need, const void **handle) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (placed) *placed = r ? r->placed : 0;
    if (need) *need = r ? r->need : 0;
    if (handle) *handle = r ? (const void *)r->gpu : NULL;
    return r && r->gpu;
}
void model_inkling2_cuda_calls(const model_t *m, long *n_products, long *n_expert_bodies) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (n_products) *n_products = r ? r->n_lin : 0;
    if (n_expert_bodies) *n_expert_bodies = r ? r->n_exp : 0;
}
bool model_inkling2_cuda_holds(const model_t *m, const char *name) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    if (!r || !r->gpu || !name) return false;
    for (int i = 0; i < r->n_dev; i++)
        if (!strcmp(r->dev_of[i]->name, name)) return true;
    return false;
}
// TEST-ONLY: every device tensor copied back and decoded (dequant_row, row by row) against the host's decode of the
// view it was put from, bit for bit; a stack's factors compared apart. False: no device, or a copy failed.
bool model_inkling2_cuda_readback(model_t *m, int *n_tensors, int *n_differ, char *first, size_t first_cap) {
    ink2_rt *r = m ? rt_of(m) : NULL;
    if (n_tensors) *n_tensors = 0;
    if (n_differ) *n_differ = 0;
    if (first && first_cap) first[0] = 0;
    if (!r || !r->gpu) return false;
    int bad = 0;
    for (int i = 0; i < r->n_dev; i++) {
        const ink2_tensor *t = r->dev_of[i];
        const int64_t nr = view_rows(t), nc = t->ne[0];
        const size_t rb = view_row_bytes(t), bytes = rb * (size_t)nr;   // D1b: a router's float32 rows too
        const int qt = view_qtype(t);
        const unsigned char *src = view_src(t);
        const int ns = view_scales(t);
        unsigned char *q = malloc(bytes);
        float *sc = ns ? malloc(sizeof(float) * (size_t)ns) : NULL, *a = malloc(sizeof(float) * (size_t)nc),
              *b = malloc(sizeof(float) * (size_t)nc);
        bool ok = q && a && b && (!ns || sc) && gpu_ink2_get(r->gpu, i, q, bytes, sc, ns);
        if (!ok) { free(q); free(sc); free(a); free(b); return false; }
        bool same = !ns || !memcmp(sc, t->q_scale, sizeof(float) * (size_t)ns);
        for (int64_t row = 0; same && row < nr; row++) {
            dequant_row(qt, q + rb * (size_t)row, a, (int)nc);
            dequant_row(qt, src + rb * (size_t)row, b, (int)nc);
            same = !memcmp(a, b, sizeof(float) * (size_t)nc);
        }
        if (!same && !bad++ && first && first_cap) snprintf(first, first_cap, "%s", t->name);
        free(q); free(sc); free(a); free(b);
    }
    if (n_tensors) *n_tensors = r->n_dev;
    if (n_differ) *n_differ = bad;
    return true;
}
