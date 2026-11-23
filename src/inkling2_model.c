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
} ink2_rt;

static ink2_rt *rt_of(model_t *m) { return (ink2_rt *)m->ink2; }

// R4.48.9: a view's rows (a stack's experts one after another) and its factors
static int64_t view_rows(const ink2_tensor *t) { return t->ne[1] * t->ne[2] * t->ne[3]; }
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
    return gpu_ink2_runs(c->r->gpu, t->qtype);
}
// a device allocation's bytes as the driver takes them: large ones in 2 MiB pages, small ones 512-byte aligned
static size_t dev_alloc_bytes(size_t b) {
    const size_t g = b >= ((size_t)1 << 20) ? (size_t)2 << 20 : 512;
    return (b + g - 1) / g * g;
}
static int dev_count(void *c_, const ink2_tensor *t) {
    dev_walk *c = c_;
    if (!dev_wanted(c, t)) return 0;
    c->need += dev_alloc_bytes(t->q_row_bytes * (size_t)view_rows(t));
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
    if (gpu_ink2_put(c->r->gpu, t->q, t->q_row_bytes * (size_t)view_rows(t), t->q_scale, view_scales(t)) != c->r->n_dev)
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
    gpu_ink2_block gb = { bc->H, bc->nh, bc->nkv, bc->hd, bc->K, bc->dr, bc->ext, bc->window, bc->global, bc->scale,
                          bc->qk_clip, bc->rel_clip, bc->eps, bc->log_alpha, bc->log_floor, { 0 }, { 0 }, { 0 }, { 0 } };
    for (int k = 0; k < GI2_NV; k++) {
        const ink2_tensor *t = bc->v[k];
        if (!t) { gb.vi[k] = -1; gb.vtype[k] = -1; continue; }
        if (t->dev != r) { r->gpu_err = true; return -1; }   // block_on said every view is there
        gb.vi[k] = t->dev_i; gb.vtype[k] = t->qtype; gb.vrb[k] = t->q_row_bytes;
    }
    for (int k = 0; k < GI2_NF; k++) {
        const ink2_tensor *t = bc->f[k];
        if (!t || t->dev != r) { r->gpu_err = true; return -1; }
        gb.fi[k] = t->dev_i;
    }
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
    int refs;
    char why[200];   // 'C': the sentence each instance says
    struct ink2_share *next;
} ink2_share;
static pthread_mutex_t g_share_mu = PTHREAD_MUTEX_INITIALIZER;
static ink2_share *g_shares;
// the device bytes ONE instance holds beyond the shared weights: its K/V rings (step 2a: every application's ring at
// the host cache's rows, K and V, at the allocation granularity), when the core runs on the device. The products',
// the expert body's and the core's scratch belong to the shared copy (used under its lock).
// one slot's own device bytes: per application its K and V rings and, on the block path (hist_rows = conv_k - 1),
// its K and V convolution histories, each as the driver allocates it
static size_t dev_instance_bytes(const ink2_rt *r, bool attn_dev, int hist_rows) {
    if (!attn_dev) return 0;
    size_t b = 0;
    for (int a = 0; a < r->s.n_app; a++) {
        const size_t C = (size_t)app_c(&r->s, a);
        b += 2 * dev_alloc_bytes(sizeof(float) * (size_t)r->c.rows[a] * C);
        if (hist_rows > 0) b += 2 * dev_alloc_bytes(sizeof(float) * (size_t)hist_rows * C);
    }
    return b;
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
static void share_place(ink2_share *sh, ink2_rt *r, bool cpu_moe_req, int n_batch, int n_ctx, int n_slots,
                        const char *dev) {
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
    // every slot's own rings (and on the block path its convolution histories), counted N times
    const size_t inst = (size_t)n_slots * dev_instance_bytes(r, attn_dev, block_dev ? sp->conv_k - 1 : 0);
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
            !gpu_ink2_reserve(h, sizeof(float) * T * (size_t)c.max_in, sizeof(float) * y_rows) ||
            (exp_dev && !gpu_ink2_reserve_experts(h, (int)T, r->s.hidden, r->s.ff, r->s.n_expert)) ||
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
static int cuda_bind(ink2_rt *r, const model_t *m, const model_params *p) {
    // until the whole CUDA forward is built (R4.48.9 G2b) the route is opt-in: G2a runs only the products on the
    // device, so a default GPU load keeps the CPU
    const char *on = getenv("RUNNER_INKLING2_CUDA");
    if (!on || strcmp(on, "1")) {
        fprintf(stderr, "note: inkling2_loop runs on the CPU (its CUDA route is not complete; RUNNER_INKLING2_CUDA=1 "
                        "runs its products on the device, the rest of the forward on the CPU)\n");
        return 0;
    }
    if (r->route != 'F') {
        fprintf(stderr, "note: inkling2_loop runs on CUDA from the fast route's stored rows only; the reference route "
                        "runs on the CPU\n");
        return 0;
    }
    const char *a8 = getenv("RUNNER_INKLING2_ACT8");
    if (a8 && *a8 && strcmp(a8, "0")) {
        fprintf(stderr, "note: the 8-bit control (RUNNER_INKLING2_ACT8) is the CPU's; running on the CPU\n");
        return 0;
    }
    char dev[160] = "";
    if (!gpu_available(dev, sizeof dev)) {
        fprintf(stderr, "note: no CUDA device for inkling2_loop; running on the CPU\n");
        return 0;
    }
    const int n_slots = p->n_seq > 1 ? p->n_seq : 1;
    uint64_t fsize = 0, fino = 0;
    int64_t fmt = 0, fct = 0;
    int ferr = 0;
    if (!m->path || !plat_file_identity(m->path, &fsize, &fino, &fmt, &fct, &ferr)) {
        fprintf(stderr, "note: inkling2_loop's file identity is unknown; running on the CPU\n");
        return 0;
    }
    const char *pf = getenv("RUNNER_INKLING2_CUDA_TEST_PUT_FAIL"), *ne = getenv("RUNNER_INKLING2_CUDA_TEST_NO_EXP");
    const char *capv = getenv("RUNNER_INKLING2_CUDA_TEST_FREE"), *nav = getenv("RUNNER_INKLING2_CUDA_TEST_NO_ATTN");
    const char *nbv = getenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK");
    char key[1200];
    snprintf(key, sizeof key, "%s|%llu|%llu|%lld|moe %d|%s|c %d|b %d|slots %d|pf %s|ne %s|cap %s|na %s|nb %s", m->path,
             (unsigned long long)fsize, (unsigned long long)fino, (long long)fmt, p->cpu_moe ? 1 : 0, dev, m->n_ctx,
             m->n_batch, n_slots, pf ? pf : "", ne ? ne : "", capv ? capv : "", nav ? nav : "", nbv ? nbv : "");
    pthread_mutex_lock(&g_share_mu);
    ink2_share *sh = NULL;
    for (ink2_share *q = g_shares; q; q = q->next)
        if (!strcmp(q->key, key)) { sh = q; break; }
    bool made = false;
    if (!sh) {
        sh = calloc(1, sizeof *sh);
        if (!sh) { pthread_mutex_unlock(&g_share_mu); fprintf(stderr, "note: out of memory binding inkling2_loop; running on the CPU\n"); return 0; }
        snprintf(sh->key, sizeof sh->key, "%s", key);
        share_place(sh, r, p->cpu_moe, m->n_batch, m->n_ctx, n_slots, dev);
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
    static const ink2_dev_ops ops = { dev_lin_scaled, dev_experts, dev_attn, dev_block };
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
        for (int a = 0; rows && cw && a < r->s.n_app; a++) { rows[a] = r->c.rows[a]; cw[a] = app_c(&r->s, a); }
        r->kv = rows && cw ? gpu_ink2_kv_new(r->gpu, r->s.n_app, rows, cw, sh->block_dev ? r->s.conv_k - 1 : 0) : NULL;
        free(rows); free(cw);
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
    fprintf(stderr, "inkling2_loop: the weights on CUDA (%s): %d tensors, %.3f GB, their stored rows%s; their products "
                    "on the device%s, the rest of the forward on the CPU (R4.48.9 G2b)%s%s\n", dev, r->n_dev,
            gpu_ink2_bytes(r->gpu) / 1e9, made ? "" : ", shared with this file's other instances",
            sh->exp_dev ? " and each layer's expert body whole (the router on the host)" : "",
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
    ink2g_set_route(r->g, r->route == 'F', a8 && *a8 && strcmp(a8, "0"));
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
    if (ink2_cache_init_window(&r->c, &r->s, n_ctx, m->n_batch)) {
        fprintf(stderr, "error: the inkling2_loop cache for -c %d does not fit in memory\n", n_ctx);
        return false;
    }
    r->cache_ok = true;
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
    fprintf(stderr, "inkling2_loop: -c %d -b %d; cache %.2f GB (sliding K/V for the window only), expert cache up to "
                    "%.2f GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)\n", n_ctx, m->n_batch,
            host_cache_bytes(r) / 1e9, expert_cache_held(&r->s, r->ec_budget) / 1073741824.0);
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
        ink2_status st = ink2_forward_chunk_lora(&r->s, &r->w, lora, &r->c, pos + done, T,
                                                 (const int *)(tokens + done), chunk_out, chunk_flags,
                                                 NULL, NULL);   // int32_t is int on every target Runner builds for
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
int model_inkling2_fit_print(gguf_file *g, const char *path, int n_ctx_want, int n_batch) {
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
    // the engine's cache under ink2_cache_init_window: K and V per application, the sliding ones for the window and
    // the batch, plus each application's four convolution histories
    uint64_t cache = 0;
    for (int a = 0; a < s.n_app; a++) {
        int gl = ink2_app_global(&s, a);
        uint64_t kv = (uint64_t)(gl ? s.n_kv * s.head_dim : s.swa_n_kv * s.swa_head_dim);
        uint64_t rows = (uint64_t)n_ctx;
        if (!gl && (uint64_t)s.window - 1 + (uint64_t)n_batch < rows) rows = (uint64_t)s.window - 1 + (uint64_t)n_batch;
        uint64_t wmax = kv > (uint64_t)s.hidden ? kv : (uint64_t)s.hidden;
        cache += 2 * rows * kv * 4 + 4ull * (uint64_t)(s.conv_k - 1) * wmax * 4;
    }
    const uint64_t ec = route == 'R' ? expert_cache_held(&s, env_bytes("RUNNER_INKLING2_EXPERT_CACHE", (size_t)4 << 30)) : 0;
    uint64_t logits = 4ull * (uint64_t)s.vocab_used * (uint64_t)(n_batch + 1);
    uint64_t mapped = gguf_mapped_size(g);
    if (!mapped) for (uint64_t i = 0; i < g->n_tensors; i++) mapped += g->tensors[i].nbytes;
    uint64_t total = dense + small + cache + ec + logits + mapped;
    uint64_t avail = plat_ram_available_bytes();
    printf("fit: %s\n", path);
    printf("  arch          inkling2_loop, %d layers, %d applications, %d experts (%d used); CPU only (no GPU path)\n",
           s.n_block, s.n_app, s.n_expert, s.n_used);
    printf("  route         %s\n", route_words(route));
    printf("  context       %d (trained %d), batch %d\n", n_ctx, train, n_batch);
    printf("  file mapped   %.2f GiB (%s)\n", mapped / 1073741824.0,
           route == 'F' ? "every weight is read from it" : "the experts are read from it");
    if (route == 'R') printf("  dense f32     %.2f GiB (the matrices, decoded once)\n", dense / 1073741824.0);
    printf("  small f32     %.3f GiB (norms, convolution taps, queries, scales, the router; decoded once)\n",
           small / 1073741824.0);
    printf("  cache         %.2f GiB (K/V per application, the %d sliding ones for window %d + batch; convolutions)\n",
           cache / 1073741824.0, s.n_app - 8 > 0 ? s.n_app - 8 : 0, s.window);
    if (route == 'R') printf("  expert cache  up to %.2f GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)\n", ec / 1073741824.0);
    printf("  logits        %.2f GiB\n", logits / 1073741824.0);
    printf("  total         %.2f GiB of RAM (%.2f GB)\n", total / 1073741824.0, total / 1e9);
    if (avail)
        printf("  verdict       %s (%.2f GiB available)\n", total <= avail ? "FITS" : "DOES NOT FIT", avail / 1073741824.0);
    else
        printf("  verdict       unknown (available RAM unreadable)\n");
    return avail && total > avail ? 1 : 0;
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
    return dev_instance_bytes(r, r->share->attn_dev, r->share->block_dev ? r->s.conv_k - 1 : 0);
}
size_t model_inkling2_cuda_block_buf_bytes(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->gpu ? gpu_ink2_block_bytes(r->gpu) : 0;
}
long model_inkling2_cuda_block_calls(const model_t *m) {
    const ink2_rt *r = m ? (const ink2_rt *)m->ink2 : NULL;
    return r && r->block_on ? r->n_block_calls : -1;
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
        const size_t bytes = t->q_row_bytes * (size_t)nr;
        const int ns = view_scales(t);
        unsigned char *q = malloc(bytes);
        float *sc = ns ? malloc(sizeof(float) * (size_t)ns) : NULL, *a = malloc(sizeof(float) * (size_t)nc),
              *b = malloc(sizeof(float) * (size_t)nc);
        bool ok = q && a && b && (!ns || sc) && gpu_ink2_get(r->gpu, i, q, bytes, sc, ns);
        if (!ok) { free(q); free(sc); free(a); free(b); return false; }
        bool same = !ns || !memcmp(sc, t->q_scale, sizeof(float) * (size_t)ns);
        for (int64_t row = 0; same && row < nr; row++) {
            dequant_row(t->qtype, q + t->q_row_bytes * (size_t)row, a, (int)nc);
            dequant_row(t->qtype, (const unsigned char *)t->q + t->q_row_bytes * (size_t)row, b, (int)nc);
            same = !memcmp(a, b, sizeof(float) * (size_t)nc);
        }
        if (!same && !bad++ && first && first_cap) snprintf(first, first_cap, "%s", t->name);
        free(q); free(sc); free(a); free(b);
    }
    if (n_tensors) *n_tensors = r->n_dev;
    if (n_differ) *n_differ = bad;
    return true;
}
